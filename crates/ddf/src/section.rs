//! Sections of the exterior bundles over the simplicial manifold.
//!
//! A [`Section`] is the discrete-geometry notion of a field: a section of
//! $Lambda^k T^* M$ (covariant: a differential form) or $Lambda^k T M$
//! (contravariant: a multivector field) over the simplicial manifold $M$ --
//! the piecewise-affine object the mesh *is*, as opposed to whatever smooth
//! manifold it may be approximating. It is evaluated at a [`MeshPoint`] -- a
//! cell together with barycentric coordinates -- and its value is expressed in
//! the **reference frame of that cell's chart**, so it needs no embedding and
//! no global coordinate system. Sections therefore work verbatim on a purely
//! metric (Regge) manifold, where no global coordinate exists at all.
//!
//! The flat, mesh-independent [`CoordField`]s of the `exterior` crate connect
//! to this world through the functor whose direction the [`Variance`] fixes:
//!
//! - covariant: a coordinate form **pulls back** onto the manifold along the
//!   cell chart ([`Pullback`], canonical and metric-free);
//! - contravariant: the direction reverses, so a multivector field on the
//!   manifold is what pushes forward into ambient space instead.
//!
//! The type system enforces this: [`Pullback`] implements [`Section`] only for
//! [`Covariant`]. The opposite direction, sampling a section back into ambient
//! coordinates ([`Sampler`]), is not canonical for forms -- it extends the
//! value by zero on the normal space through the chart pseudo-inverse -- and is
//! confined to visualization and I/O.

use crate::whitney::interpolant::WhitneyInterpolant;

use {
  common::{gramian::RiemannianMetric, linalg::nalgebra::VectorView},
  exterior::{
    field::CoordField, Contravariant, Covariant, Dim, ExteriorElement, ExteriorGrade, MultiForm,
    MultiVector, Variance,
  },
  manifold::{
    geometry::{
      coord::{locate::PointLocator, mesh::MeshCoords, simplex::SimplexRefExt},
      metric::Geometry,
    },
    point::MeshPoint,
    topology::complex::Complex,
  },
};

/// A section of the exterior bundle $Lambda^k T^(*) M$ over the simplicial
/// manifold: a differential form when [`Covariant`], a multivector field when
/// [`Contravariant`].
///
/// The value at a [`MeshPoint`] is expressed in the reference frame of the
/// containing cell's chart, hence lives in $Lambda^k (RR^n)$ for the intrinsic
/// dimension $n$ of the manifold -- never in an ambient space.
///
/// Sections need not be continuous across cells: the Whitney forms have only
/// the tangential continuity their conformity requires. What all sections here
/// do share is that the quantities extracted from them -- the integral over a
/// face in the de Rham map, the $L^2$ inner product over a cell -- are
/// chart-independent.
pub trait Section<V: Variance> {
  /// The dimension of the simplicial manifold, which is that of the cell
  /// charts.
  fn dim(&self) -> Dim;
  fn grade(&self) -> ExteriorGrade;
  fn at(&self, point: &MeshPoint) -> ExteriorElement<V>;
}

/// The pullback of a coordinate differential form onto the manifold along the
/// cell charts, $omega |-> phi_K^* omega$.
///
/// The canonical way a mesh-independent form becomes a field on the mesh:
/// evaluate at the global image of the mesh point and pull the value back
/// through the chart's linear part. Metric-free, and only for [`Covariant`]
/// fields -- pullback is the contravariant action of $Lambda^k$, so the
/// functor runs this way and no other.
pub struct Pullback<'a, F> {
  field: &'a F,
  topology: &'a Complex,
  coords: &'a MeshCoords,
}

impl<'a, F: CoordField<Covariant>> Pullback<'a, F> {
  pub fn new(field: &'a F, topology: &'a Complex, coords: &'a MeshCoords) -> Self {
    assert_eq!(
      field.dim(),
      coords.dim(),
      "Field lives in the ambient space."
    );
    Self {
      field,
      topology,
      coords,
    }
  }
}

impl<F: CoordField<Covariant>> Section<Covariant> for Pullback<'_, F> {
  fn dim(&self) -> Dim {
    self.topology.dim()
  }
  fn grade(&self) -> ExteriorGrade {
    self.field.grade()
  }
  fn at(&self, point: &MeshPoint) -> MultiForm {
    let chart = point.cell.handle(self.topology).coord_simplex(self.coords);
    let global = chart.bary2global(point.bary());
    self.field.at(&global).pullback(&chart.linear_transform())
  }
}

/// Pull a coordinate form onto the manifold: `f.pullback_on(&topology, &coords)`.
pub trait CoordFieldExt: Sized + CoordField<Covariant> {
  fn pullback_on<'a>(
    &'a self,
    topology: &'a Complex,
    coords: &'a MeshCoords,
  ) -> Pullback<'a, Self> {
    Pullback::new(self, topology, coords)
  }
}
impl<F: CoordField<Covariant>> CoordFieldExt for F {}

/// The ambient-coordinate sampling of a section: the inverse road, taken
/// only for visualization and I/O.
///
/// Locates the global point in the mesh, evaluates the field in the cell
/// chart, and extends the reference-frame value to the ambient frame by
/// pulling it back along the chart pseudo-inverse $A^+$ -- i.e. by declaring
/// it zero on the normal space of the cell. That choice is metric-dependent
/// (it is the Moore-Penrose one for the Euclidean ambient metric) and hence
/// *not* canonical, which is exactly why it may not sit in the core path:
/// nothing in assembly or discretization is allowed to need it.
///
/// Without a [`PointLocator`] attached, locating a point is a linear scan over
/// all cells; with one it is logarithmic, which is what makes grid sampling
/// affordable.
pub struct Sampler<'a, F> {
  field: &'a F,
  topology: &'a Complex,
  coords: &'a MeshCoords,
  locator: Option<&'a PointLocator>,
}

impl<'a, F: Section<Covariant>> Sampler<'a, F> {
  pub fn new(field: &'a F, topology: &'a Complex, coords: &'a MeshCoords) -> Self {
    Self {
      field,
      topology,
      coords,
      locator: None,
    }
  }
  /// Attach a prebuilt locator, making [`locate`](Self::locate) logarithmic
  /// instead of a linear scan.
  pub fn with_locator(mut self, locator: &'a PointLocator) -> Self {
    self.locator = Some(locator);
    self
  }

  /// The point of the manifold at a global coordinate; `None` outside the mesh.
  pub fn locate<'b>(&self, coord: impl Into<VectorView<'b>>) -> Option<MeshPoint> {
    let coord = coord.into();
    match self.locator {
      Some(locator) => locator.locate(coord),
      None => self
        .coords
        .find_cell_containing(self.topology, coord)
        .map(|cell| {
          MeshPoint::new(
            cell.idx(),
            cell.coord_simplex(self.coords).global2bary(coord),
          )
        }),
    }
  }

  /// The value at a mesh point, expressed in the ambient frame.
  pub fn at_point(&self, point: &MeshPoint) -> MultiForm {
    let chart = point.cell.handle(self.topology).coord_simplex(self.coords);
    self.field.at(point).pullback(&chart.inv_linear_transform())
  }

  /// The value at a global coordinate, in the ambient frame;
  /// `None` outside the mesh.
  pub fn at_global<'b>(&self, coord: impl Into<VectorView<'b>>) -> Option<MultiForm> {
    self.locate(coord).map(|point| self.at_point(&point))
  }
}

/// Sample a section in ambient coordinates: `f.sampled_on(&topology, &coords)`.
pub trait SectionExt: Sized + Section<Covariant> {
  fn sampled_on<'a>(&'a self, topology: &'a Complex, coords: &'a MeshCoords) -> Sampler<'a, Self> {
    Sampler::new(self, topology, coords)
  }
}
impl<F: Section<Covariant>> SectionExt for F {}

/// The pointwise wedge $alpha wedge beta$ of two fields of the same variance.
///
/// Metric-free, like the wedge on values: the combinator is lazy, the algebra
/// happens at evaluation.
pub struct Wedge<A, B> {
  left: A,
  right: B,
}
impl<A, B> Wedge<A, B> {
  pub fn new(left: A, right: B) -> Self {
    Self { left, right }
  }
}
impl<V: Variance, A: Section<V>, B: Section<V>> Section<V> for Wedge<A, B> {
  fn dim(&self) -> Dim {
    self.left.dim()
  }
  fn grade(&self) -> ExteriorGrade {
    self.left.grade() + self.right.grade()
  }
  fn at(&self, point: &MeshPoint) -> ExteriorElement<V> {
    self.left.at(point).wedge(&self.right.at(point))
  }
}

/// A pointwise metric operation on a field, measured by the metric of the cell
/// the point lies in.
///
/// The metric enters a field only here: [`Pullback`], [`Wedge`] and the de Rham
/// map are metric-free, and the [`Geometry`] appears exactly where the
/// mathematics demands it.
pub struct MetricOp<'a, F, G> {
  field: F,
  topology: &'a Complex,
  geometry: &'a G,
}
impl<'a, F, G: Geometry> MetricOp<'a, F, G> {
  fn cell_metric(&self, point: &MeshPoint) -> RiemannianMetric {
    self.geometry.cell_metric(point.cell.handle(self.topology))
  }
}

/// The musical isomorphism $sharp$ applied pointwise: a differential form
/// becomes a multivector field, on fully equal footing.
pub struct Sharp<'a, F, G>(MetricOp<'a, F, G>);
impl<F: Section<Covariant>, G: Geometry> Section<Contravariant> for Sharp<'_, F, G> {
  fn dim(&self) -> Dim {
    self.0.field.dim()
  }
  fn grade(&self) -> ExteriorGrade {
    self.0.field.grade()
  }
  fn at(&self, point: &MeshPoint) -> MultiVector {
    self.0.field.at(point).sharp(&self.0.cell_metric(point))
  }
}

/// The musical isomorphism $flat$ applied pointwise: a multivector field
/// becomes a differential form.
pub struct Flat<'a, F, G>(MetricOp<'a, F, G>);
impl<F: Section<Contravariant>, G: Geometry> Section<Covariant> for Flat<'_, F, G> {
  fn dim(&self) -> Dim {
    self.0.field.dim()
  }
  fn grade(&self) -> ExteriorGrade {
    self.0.field.grade()
  }
  fn at(&self, point: &MeshPoint) -> MultiForm {
    self.0.field.at(point).flat(&self.0.cell_metric(point))
  }
}

/// The Hodge star $star: Lambda^k -> Lambda^(n-k)$ applied pointwise,
/// preserving the variance.
pub struct HodgeStar<'a, F, G>(MetricOp<'a, F, G>);
impl<V: Variance, F: Section<V>, G: Geometry> Section<V> for HodgeStar<'_, F, G> {
  fn dim(&self) -> Dim {
    self.0.field.dim()
  }
  fn grade(&self) -> ExteriorGrade {
    self.0.field.dim() - self.0.field.grade()
  }
  fn at(&self, point: &MeshPoint) -> ExteriorElement<V> {
    self
      .0
      .field
      .at(point)
      .hodge_star(&self.0.cell_metric(point))
  }
}

/// The pointwise combinators, in method position: `omega.sharp(&topology, &geometry)`.
pub trait SectionOps<V: Variance>: Sized + Section<V> {
  fn wedge<B: Section<V>>(self, other: B) -> Wedge<Self, B> {
    Wedge::new(self, other)
  }
  fn hodge_star<'a, G: Geometry>(
    self,
    topology: &'a Complex,
    geometry: &'a G,
  ) -> HodgeStar<'a, Self, G> {
    HodgeStar(MetricOp {
      field: self,
      topology,
      geometry,
    })
  }
}
impl<V: Variance, F: Section<V>> SectionOps<V> for F {}

/// The musicals, which change the variance and so cannot sit on the
/// variance-generic [`SectionOps`].
pub trait SharpOp: Sized + Section<Covariant> {
  fn sharp<'a, G: Geometry>(self, topology: &'a Complex, geometry: &'a G) -> Sharp<'a, Self, G> {
    Sharp(MetricOp {
      field: self,
      topology,
      geometry,
    })
  }
}
impl<F: Section<Covariant>> SharpOp for F {}

pub trait FlatOp: Sized + Section<Contravariant> {
  fn flat<'a, G: Geometry>(self, topology: &'a Complex, geometry: &'a G) -> Flat<'a, Self, G> {
    Flat(MetricOp {
      field: self,
      topology,
      geometry,
    })
  }
}
impl<F: Section<Contravariant>> FlatOp for F {}

/// The Whitney interpolation of a cochain, as a section of the manifold.
impl Section<Covariant> for WhitneyInterpolant<'_> {
  fn dim(&self) -> Dim {
    self.complex().dim()
  }
  fn grade(&self) -> ExteriorGrade {
    self.cochain().grade()
  }
  fn at(&self, point: &MeshPoint) -> MultiForm {
    self.eval(point)
  }
}

#[cfg(test)]
mod test {
  use super::*;

  use crate::derham::derham_map;

  use {
    common::linalg::nalgebra::Vector,
    exterior::field::DiffFormClosure,
    manifold::{gen::cartesian::CartesianMeshInfo, geometry::coord::locate::PointLocator},
  };

  use approx::assert_relative_eq;

  /// A grid of sample points, kept off cell faces and triangulation diagonals.
  fn probe_points(dim: Dim, samples: usize) -> impl Iterator<Item = Vector> {
    let phase = [0.017, 0.113, 0.237];
    (0..samples.pow(dim as u32)).map(move |flat| {
      Vector::from_iterator(
        dim,
        (0..dim).map(|d| {
          let i = flat / samples.pow(d as u32) % samples;
          0.05 + 0.9 * (i as f64 + phase[d]) / samples as f64
        }),
      )
    })
  }

  /// The locator-accelerated sampling agrees exactly with the linear-scan
  /// fallback: same cell, same reconstructed value.
  #[test]
  fn located_sampling_matches_scan() {
    for dim in 1..=3 {
      let (topology, coords) = CartesianMeshInfo::new_unit(dim, 3).compute_coord_complex();
      let locator = PointLocator::new(&topology, &coords);

      let field = DiffFormClosure::one_form(|p| p.clone_owned(), dim);
      let cochain = derham_map(&field.pullback_on(&topology, &coords), &topology, 2);
      let whitney = WhitneyInterpolant::new(cochain, &topology);

      let scan = whitney.sampled_on(&topology, &coords);
      let fast = whitney
        .sampled_on(&topology, &coords)
        .with_locator(&locator);

      for x in probe_points(dim, 4) {
        let a = scan.at_global(&x).unwrap();
        let b = fast.at_global(&x).unwrap();
        assert_relative_eq!(a.coeffs(), b.coeffs(), epsilon = 1e-12);
      }
    }
  }

  /// Pulling a coordinate form onto the mesh and sampling it back in ambient
  /// coordinates is the identity, on a mesh of full intrinsic dimension.
  ///
  /// There the chart is invertible, so the pseudo-inverse *is* the inverse and
  /// $((A^+)^* compose A^*) omega = omega$ -- the round trip through the
  /// manifold loses nothing.
  #[test]
  fn pullback_then_sample_is_identity() {
    for dim in 1..=3 {
      let (topology, coords) = CartesianMeshInfo::new_unit(dim, 3).compute_coord_complex();
      let locator = PointLocator::new(&topology, &coords);

      let field = DiffFormClosure::one_form(
        |p| Vector::from_iterator(p.len(), p.iter().map(|x| (2.0 * x).sin())),
        dim,
      );
      let pulled = field.pullback_on(&topology, &coords);
      let sampled = pulled.sampled_on(&topology, &coords).with_locator(&locator);

      for x in probe_points(dim, 4) {
        assert_relative_eq!(
          sampled.at_global(&x).unwrap().coeffs(),
          field.at(&x).coeffs(),
          epsilon = 1e-12
        );
      }
    }
  }

  /// $star star = (-1)^(k(n-k))$ holds pointwise on the field level, with the
  /// cell metric supplied by the [`Geometry`].
  #[test]
  fn hodge_star_field_involution() {
    use common::combo::Sign;
    use manifold::point::MeshPoint;

    for dim in 1..=3 {
      let (topology, coords) = CartesianMeshInfo::new_unit(dim, 2).compute_coord_complex();
      let lengths = coords.to_edge_lengths(&topology);

      for grade in 0..=dim {
        let ndofs = topology.nsimplices(grade);
        let cochain = crate::cochain::Cochain::new(
          grade,
          Vector::from_iterator(ndofs, (0..ndofs).map(|i| (i % 5) as f64 - 2.0)),
        );
        let whitney = WhitneyInterpolant::new(cochain, &topology);
        let star_star = WhitneyInterpolant::new(whitney.cochain().clone(), &topology)
          .hodge_star(&topology, &lengths)
          .hodge_star(&topology, &lengths);

        let sign = Sign::from_parity(grade * (dim - grade));
        for cell in topology.cells().handle_iter() {
          let point = MeshPoint::barycenter(cell.idx());
          assert_relative_eq!(
            star_star.at(&point).coeffs(),
            &(sign.as_f64() * whitney.at(&point)).coeffs(),
            epsilon = 1e-12
          );
        }
      }
    }
  }
}
