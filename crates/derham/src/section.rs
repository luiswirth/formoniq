//! Sections of the exterior bundles over the simplicial manifold.
//!
//! A [`Section`] is the discrete-geometry notion of a field: a section of
//! $Lambda^k T^* M$ (covariant: a differential form) or $Lambda^k T M$
//! (contravariant: a multivector field) over the simplicial manifold $M$,
//! the piecewise-affine object the mesh is, as opposed to whatever smooth
//! manifold it may be approximating. It is evaluated at a [`MeshPoint`], a
//! cell together with barycentric coordinates, and its value is expressed in
//! the reference frame of that cell's chart, so it needs no embedding and
//! no global coordinate system. Sections therefore work verbatim on a purely
//! metric (Regge) manifold, where no global coordinate exists at all.
//!
//! The mesh-independent [`CoordField`]s of the `glatt` crate, analytic
//! data on the smooth manifold $M$, connect to this world through the functor
//! whose direction the variance of the values fixes:
//!
//! - covariant: a coordinate form pulls back onto the mesh along the
//!   composite of the cell parametrization and the continuum chart
//!   ([`Pullback`], canonical and metric-free);
//! - contravariant: the direction reverses, so a multivector field on the
//!   manifold is what pushes forward into ambient space instead.
//!
//! Variance is carried by the slots of the value, so [`Pullback`] rejects a
//! contravariant field where it once failed to compile. The opposite direction,
//! sampling a section back into ambient coordinates ([`Sampler`]), is not
//! canonical for forms, it extends the value by zero on the normal space
//! through the chart pseudo-inverse, and is confined to visualization and
//! I/O.

use crate::interpolate::interpolant::WhitneyInterpolant;
use metric::tensor::TensorExt;

use {
  coorder::{Ambient, CoordSpace, Coords},
  glatt::{field::CoordField, parametrization::Parametrization},
  metric::Metric,
  multialgebra::{Dim, ExteriorGrade, Tensor},
  regge::{
    coord::{CoordRef, locate::PointLocator, mesh::MeshCoords, simplex::SimplexRefExt},
    lengths::mesh::MeshLengthsSq,
  },
  simplicial::linalg::Vector,
  simplicial::{
    atlas::MeshPoint,
    topology::{complex::Complex, orientation::Orientation},
  },
};

/// A section of the exterior bundle $Lambda^k T^(*) M$ over the simplicial
/// manifold: a differential form when its values are covariant, a multivector
/// field when they are contravariant.
///
/// The value at a [`MeshPoint`] is expressed in the reference frame of the
/// containing cell's chart, hence lives in $Lambda^k (RR^n)$ for the intrinsic
/// dimension $n$ of the manifold, never in an ambient space.
///
/// Sections need not be continuous across cells: the Whitney forms have only
/// the tangential continuity their conformity requires. What all sections here
/// do share is that the quantities extracted from them, the integral over a
/// face in the de Rham map, the $L^2$ inner product over a cell, are
/// chart-independent.
pub trait Section {
  /// The dimension of the simplicial manifold, which is that of the cell
  /// charts.
  fn dim(&self) -> Dim;
  fn grade(&self) -> ExteriorGrade;
  fn at(&self, point: &MeshPoint) -> Tensor;
}

/// The pullback of a continuum differential form onto the mesh, $omega |->
/// (chi compose psi_K)^* omega$.
///
/// A form $omega$ on a chart domain $Omega$ of the continuum $M$ reaches a cell
/// $K$ of the simplicial mesh through the composite
///
/// $$ hat(K) ->^(psi_K) RR^N ->^chi Omega, qquad chi = phi^(-1) compose r, $$
///
/// the affine parametrization $psi_K$ of the cell followed by the continuum
/// chart $chi$, the ordinary transition-map pattern, but spanning $M_h$ and
/// $M$ instead of two patches of one manifold. Evaluate $omega$ at
/// $u = chi(psi_K(lambda))$ and pull the value back through the composite
/// differential $dif chi dot dif psi_K$. Metric-free, and only for covariant
/// fields, pullback is the contravariant action of $Lambda^k$, so the functor
/// runs this way and no other, and a contravariant value is rejected rather
/// than silently transported backwards.
///
/// The construction is a bona fide pullback of a form along a smooth map, and
/// $(R s)_sigma = integral_(r(sigma)) omega_M$ is the exact integral of the true
/// form over the curved image of the face. What it is not is exact on $M$,
/// and the inexactness is the $M_h != M$ gap made quantitative, a domain gap
/// ($r(sigma)$ is not the geodesic simplex, $O(h^2)$ under the orthogonal
/// projection) and a metric gap (the $L^2$ pairings downstream use the chord
/// metric $g_h$, not $g_M$, also $O(h^2)$). Both are inherent to a linear mesh
/// and distinct from the exact structure inside $M_h$.
///
/// The flat case is the value $Omega = RR^N$, $phi = id$: then $chi$ is the
/// identity and the composite collapses to $psi_K$ alone, taken with no chart
/// solve. That is exactly what `pullback_on` is.
pub struct Pullback<'a, F, S: CoordSpace = Ambient> {
  field: &'a F,
  topology: &'a Complex,
  coords: &'a MeshCoords,
  chart_map: ContinuumChartMap<'a, S>,
}

/// The chart map $chi$ of the continuum $M$: how a mesh point reaches the
/// chart domain $Omega$.
///
/// Not a [`Chart`](simplicial::atlas::Chart), which is a cell of the simplicial
/// manifold $M_h$. Both are charts, of the two manifolds the
/// [`Pullback`] spans, and the name says which.
enum ContinuumChartMap<'a, S: CoordSpace> {
  /// The flat case: $Omega = RR^N$, $phi = id$, so $chi = id$ and the ambient
  /// image of the mesh point is the domain point. No chart solve.
  Identity,
  /// The curved case: $chi = phi^(-1) compose r$, evaluated by the
  /// [`Parametrization`]. `vertex_omega` caches $chi$ at every mesh vertex, so an
  /// interior point can seed its Gauss-Newton from the barycentric interpolation
  /// of its cell's vertices rather than a cold start.
  Through {
    param: &'a Parametrization<S>,
    vertex_omega: Vec<Coords<S>>,
  },
}

impl<'a, F: CoordField<Ambient>> Pullback<'a, F, Ambient> {
  /// The flat pullback: the field's domain is the ambient space.
  pub fn identity(field: &'a F, topology: &'a Complex, coords: &'a MeshCoords) -> Self {
    assert_eq!(
      field.dim(),
      coords.dim(),
      "Field lives in the ambient space."
    );
    Self {
      field,
      topology,
      coords,
      chart_map: ContinuumChartMap::Identity,
    }
  }
}

impl<'a, S: CoordSpace, F: CoordField<S>> Pullback<'a, F, S> {
  /// The curved pullback: the field lives on a chart domain of the continuum,
  /// reached through `param`.
  pub fn through(
    field: &'a F,
    topology: &'a Complex,
    coords: &'a MeshCoords,
    param: &'a Parametrization<S>,
  ) -> Self {
    assert_eq!(
      field.dim(),
      param.dim(),
      "Field lives on the parametrization's domain."
    );
    let vertex_omega = (0..coords.nvertices())
      .map(|v| {
        let vc = coords.coord(v).to_coords();
        param.chart(&vc, &param.seed_at(&vc))
      })
      .collect();
    Self {
      field,
      topology,
      coords,
      chart_map: ContinuumChartMap::Through {
        param,
        vertex_omega,
      },
    }
  }
}

impl<S: CoordSpace, F: CoordField<S>> Section for Pullback<'_, F, S> {
  fn dim(&self) -> Dim {
    self.topology.dim()
  }
  fn grade(&self) -> ExteriorGrade {
    self.field.grade()
  }
  fn at(&self, point: &MeshPoint) -> Tensor {
    let cell = point.chart(self.topology);
    let parametrization = cell.coord_simplex(self.coords);
    let global = parametrization.bary2global(point.bary());
    match &self.chart_map {
      // `Identity` is only ever built at `S = Ambient`, where the ambient image
      // is the domain point. The relabel is the sanctioned unchecked entry.
      ContinuumChartMap::Identity => {
        let u = Coords::<S>::new(global.into_vector());
        self
          .field
          .at(&u)
          .pullback(&parametrization.linear_transform())
      }
      ContinuumChartMap::Through {
        param,
        vertex_omega,
      } => {
        let seed: Vector = cell
          .simplex()
          .iter()
          .zip(point.bary().iter())
          .map(|(v, &w)| w * vertex_omega[v].view())
          .sum();
        let u = param.chart(&global, &Coords::new(seed));
        let composite = param.chart_differential(&u) * parametrization.linear_transform();
        self.field.at(&u).pullback(&composite)
      }
    }
  }
}

/// Pull a continuum form onto the mesh:
/// `f.pullback_on(&topology, &coords)` (flat) or
/// `f.pullback_through(&topology, &coords, &param)` (curved).
pub trait CoordFieldExt<S: CoordSpace = Ambient>: Sized + CoordField<S> {
  /// Pull the form onto the mesh along a continuum [`Parametrization`].
  fn pullback_through<'a>(
    &'a self,
    topology: &'a Complex,
    coords: &'a MeshCoords,
    param: &'a Parametrization<S>,
  ) -> Pullback<'a, Self, S> {
    Pullback::through(self, topology, coords, param)
  }

  /// Pull the form onto the mesh in the flat case: the identity special case of
  /// [`pullback_through`](Self::pullback_through), where the continuum is $RR^N$
  /// and its chart is the identity.
  fn pullback_on<'a>(
    &'a self,
    topology: &'a Complex,
    coords: &'a MeshCoords,
  ) -> Pullback<'a, Self, Ambient>
  where
    Self: CoordField<Ambient>,
  {
    Pullback::identity(self, topology, coords)
  }
}
impl<S: CoordSpace, F: CoordField<S>> CoordFieldExt<S> for F {}

/// The ambient-coordinate sampling of a section: the inverse road, taken
/// only for visualization and I/O.
///
/// Locates the global point in the mesh, evaluates the field in the cell
/// chart, and extends the reference-frame value to the ambient frame by
/// pulling it back along the chart pseudo-inverse $A^+$, i.e. by declaring
/// it zero on the normal space of the cell. That choice is metric-dependent
/// (it is the Moore-Penrose one for the Euclidean ambient metric) and hence
/// not canonical, which is exactly why it may not sit in the core path:
/// nothing in assembly or discretization is allowed to need it.
///
/// Without a [`PointLocator`] attached, locating a point is a linear scan over
/// all cells. With one it is logarithmic, which is what makes grid sampling
/// affordable.
pub struct Sampler<'a, F> {
  field: &'a F,
  topology: &'a Complex,
  coords: &'a MeshCoords,
  locator: Option<&'a PointLocator>,
}

impl<'a, F: Section> Sampler<'a, F> {
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
  pub fn locate<'b>(&self, coord: impl Into<CoordRef<'b>>) -> Option<MeshPoint> {
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
  pub fn at_point(&self, point: &MeshPoint) -> Tensor {
    let parametrization = point.chart(self.topology).coord_simplex(self.coords);
    self
      .field
      .at(point)
      .pullback(&parametrization.inv_linear_transform())
  }

  /// The value at a global coordinate, in the ambient frame;
  /// `None` outside the mesh.
  pub fn at_global<'b>(&self, coord: impl Into<CoordRef<'b>>) -> Option<Tensor> {
    self.locate(coord).map(|point| self.at_point(&point))
  }
}

/// Sample a section in ambient coordinates: `f.sampled_on(&topology, &coords)`.
pub trait SectionExt: Sized + Section {
  fn sampled_on<'a>(&'a self, topology: &'a Complex, coords: &'a MeshCoords) -> Sampler<'a, Self> {
    Sampler::new(self, topology, coords)
  }
}
impl<F: Section> SectionExt for F {}

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
impl<A: Section, B: Section> Section for Wedge<A, B> {
  fn dim(&self) -> Dim {
    self.left.dim()
  }
  fn grade(&self) -> ExteriorGrade {
    self.left.grade() + self.right.grade()
  }
  fn at(&self, point: &MeshPoint) -> Tensor {
    self.left.at(point).wedge(&self.right.at(point))
  }
}

/// A pointwise metric operation on a field, measured by the metric of the cell
/// the point lies in.
///
/// The metric enters a field only here: [`Pullback`], [`Wedge`] and the de Rham
/// map are metric-free, and the geometry ([`MeshLengthsSq`], the intrinsic
/// primitive) appears exactly where the mathematics demands it.
pub struct MetricOp<'a, F> {
  field: F,
  topology: &'a Complex,
  geometry: &'a MeshLengthsSq,
}
impl<F> MetricOp<'_, F> {
  fn cell_metric(&self, point: &MeshPoint) -> Metric {
    self.geometry.cell_metric(point.chart(self.topology))
  }
}

/// The musical isomorphism applied pointwise, raising a differential form to a
/// multivector field or lowering one to a form.
///
/// $sharp$ and $flat$ are one map, not two: the variance of each slot decides
/// which way that slot travels, and with it whether the metric enters as $g$ or
/// as $g^(-1)$ (invariant 4). A field of mixed variance is raised and lowered
/// slot by slot in the one pass, which is what makes them the same operation
/// rather than a pair to choose between.
pub struct Musical<'a, F>(MetricOp<'a, F>);
impl<F: Section> Section for Musical<'_, F> {
  fn dim(&self) -> Dim {
    self.0.field.dim()
  }
  fn grade(&self) -> ExteriorGrade {
    self.0.field.grade()
  }
  fn at(&self, point: &MeshPoint) -> Tensor {
    self.0.field.at(point).musical(&self.0.cell_metric(point))
  }
}

/// The Hodge star $star: Lambda^k -> Lambda^(n-k)$ applied pointwise,
/// preserving the variance.
///
/// Takes the coherent [`Orientation`], and must. The star is defined against a
/// volume form, so a cell-by-cell star reads each cell's colex orientation,
/// which is a gauge unrelated to its neighbors': the field it builds flips sign
/// across every facet where colex disagrees with the manifold. Holding an
/// `&Orientation` is the proof the mesh is orientable at all, so a caller that
/// cannot get one has no star to apply.
pub struct HodgeStar<'a, F> {
  op: MetricOp<'a, F>,
  orientation: &'a Orientation,
}
impl<F: Section> Section for HodgeStar<'_, F> {
  fn dim(&self) -> Dim {
    self.op.field.dim()
  }
  fn grade(&self) -> ExteriorGrade {
    self.op.field.dim() - self.op.field.grade()
  }
  fn at(&self, point: &MeshPoint) -> Tensor {
    let chart = point.chart(self.op.topology);
    self
      .op
      .field
      .at(point)
      .star(&self.op.cell_metric(point), self.orientation.sign(chart))
  }
}

/// The pointwise combinators, in method position: `omega.musical(&topology, &geometry)`.
pub trait SectionOps: Sized + Section {
  fn wedge<B: Section>(self, other: B) -> Wedge<Self, B> {
    Wedge::new(self, other)
  }
  fn musical<'a>(self, topology: &'a Complex, geometry: &'a MeshLengthsSq) -> Musical<'a, Self> {
    Musical(MetricOp {
      field: self,
      topology,
      geometry,
    })
  }
  fn hodge_star<'a>(
    self,
    topology: &'a Complex,
    geometry: &'a MeshLengthsSq,
    orientation: &'a Orientation,
  ) -> HodgeStar<'a, Self> {
    HodgeStar {
      op: MetricOp {
        field: self,
        topology,
        geometry,
      },
      orientation,
    }
  }
}
impl<F: Section> SectionOps for F {}

/// The Whitney interpolation of a cochain, as a section of the manifold.
impl Section for WhitneyInterpolant<'_> {
  fn dim(&self) -> Dim {
    self.complex().dim()
  }
  fn grade(&self) -> ExteriorGrade {
    self.cochain().grade()
  }
  fn at(&self, point: &MeshPoint) -> Tensor {
    self.eval(point)
  }
}
