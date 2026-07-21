//! Embeddings of the flat quotients of [`super::quotient`] into Euclidean
//! space.
//!
//! These exist for visualization and for I/O, never for the core path: the
//! engine consumes the intrinsic [`MeshLengthsSq`][crate::geometry::metric::mesh::MeshLengthsSq]
//! the quotient itself
//! produces (invariant 2), and an embedding is a *second*, independent object.
//! Where the embedding is not isometric the two are genuinely different
//! manifolds -- the coordinates carry curvature the flat quotient does not --
//! so deriving edge lengths from these coordinates gives a mesh whose spectra
//! and convergence rates are not the flat ones. Do not cross the two.
//!
//! Two constructions live here, and they do not unify.
//!
//! The first is the **equivariant embedding** [`equivariant`], which is
//! general: it realizes the quotient in $RR^m$ by mapping each identified axis
//! to a harmonic of its own deck transformation, so the map descends to the
//! quotient by construction. A periodic axis becomes a circle factor, costing
//! two ambient dimensions; a *reflected* axis is split into its even and odd
//! parts under the reflection, the odd part being carried by the half-angle
//! frame $(cos(theta\/2), sin(theta\/2))$, which reverses sign after one turn
//! and so realizes the twist. With every axis periodic this is the Clifford
//! torus $T^d subset RR^(2d)$, and it is then a genuine **isometry**: the flat
//! quotient's own geometry, exactly. Any reflected axis makes it curved.
//!
//! The second is the pair of **surfaces of revolution** [`donut_r3`] and
//! [`moebius_r3`], which are not instances of the first: they are the
//! two-dimensional pictures that fit in $RR^3$, and they exist only because
//! that is where a viewer draws. Neither is isometric, and neither generalizes.
//! There is no $RR^3$ companion for the Klein bottle because none exists; it
//! renders from its $RR^4$ equivariant embedding by projection.

use std::f64::consts::TAU;

use crate::{
  Dim,
  geometry::coord::mesh::MeshCoords,
  linalg::{Matrix, Vector},
  mesher::quotient::{FlatQuotient, Identification},
};

/// The equivariant embedding of a flat quotient, the general construction.
///
/// **Isometric exactly when no axis is reflected**, i.e. on the flat torus and
/// its open-axis relatives, where it is the Clifford embedding
/// $T^d arrow.r.hook RR^(2d)$. A twist forces the half-angle frame, which
/// bends the fiber as it goes around, so the twisted members are curved
/// realizations of a flat manifold. [`is_isometric`] answers this for a given
/// quotient.
///
/// `radius_slack` scales the distance the twisted axis's circle keeps from its
/// own axis of revolution. It must exceed `1.0` for the embedding to be
/// injective; the reflected fiber sweeps a tube of the natural radius, and a
/// circle that small would pass through itself.
///
/// Panics if an axis is reflected by more than one twisted axis: the two
/// half-angle frames would have to share one fiber coordinate, which needs the
/// full representation-theoretic decomposition of the deck group rather than
/// this per-axis one.
pub fn equivariant(quotient: &FlatQuotient, radius_slack: f64) -> MeshCoords {
  assert!(
    radius_slack > 1.0,
    "The revolution radius must exceed the fiber it sweeps, or the embedding self-intersects."
  );
  let dim = quotient.dim();
  let ids = quotient.identifications();

  let reflector = reflectors(quotient);
  let emitters: Vec<Emitter> = (0..dim)
    .filter(|&axis| reflector[axis].is_none())
    .map(|axis| match ids[axis] {
      Identification::Open => Emitter::Line(axis),
      _ => Emitter::Circle {
        axis,
        fiber: (0..dim).filter(|&j| reflector[j] == Some(axis)).collect(),
      },
    })
    .collect();

  let ambient: usize = emitters.iter().map(|e| e.ndims()).sum();
  let mut matrix = Matrix::zeros(ambient, quotient.nvertices());
  for vertex in 0..quotient.nvertices() {
    let cart = quotient.vertex_cart_idx(vertex);
    let mut row = 0;
    for emitter in &emitters {
      emitter.emit(quotient, &cart, radius_slack, &mut |value| {
        matrix[(row, vertex)] = value;
        row += 1;
      });
    }
  }
  MeshCoords::new(matrix)
}

/// Whether [`equivariant`] realizes this quotient isometrically, i.e. whether
/// its induced edge lengths are the quotient's own.
///
/// True exactly when no axis is reflected. The half-angle frame a twist
/// requires rotates the fiber as it sweeps, and that rotation is curvature the
/// flat quotient does not have.
pub fn is_isometric(quotient: &FlatQuotient) -> bool {
  quotient
    .identifications()
    .iter()
    .all(|id| !matches!(id, Identification::Twisted(_)))
}

/// Which twisted axis reflects each axis, if any.
fn reflectors(quotient: &FlatQuotient) -> Vec<Option<Dim>> {
  let mut reflector = vec![None; quotient.dim()];
  for (axis, id) in quotient.identifications().iter().enumerate() {
    if let Identification::Twisted(reflected) = id {
      for &j in reflected {
        assert!(
          reflector[j].is_none(),
          "Axis {j} is reflected by two twisted axes; that needs the full \
           representation of the deck group, not a per-axis frame."
        );
        reflector[j] = Some(axis);
      }
    }
  }
  reflector
}

/// One axis's contribution to the ambient coordinates.
enum Emitter {
  /// An unidentified axis, carried as itself: one dimension, isometrically.
  Line(Dim),
  /// An identified axis as a circle, together with the fiber axes its gluing
  /// reflects. Two dimensions for the circle, two more per reflected axis.
  Circle { axis: Dim, fiber: Vec<Dim> },
}

impl Emitter {
  fn ndims(&self) -> usize {
    match self {
      Self::Line(_) => 1,
      Self::Circle { fiber, .. } => 2 + 2 * fiber.len(),
    }
  }

  fn emit(
    &self,
    quotient: &FlatQuotient,
    cart: &[usize],
    radius_slack: f64,
    out: &mut impl FnMut(f64),
  ) {
    match self {
      Self::Line(axis) => out(position(quotient, cart, *axis)),
      Self::Circle { axis, fiber } => {
        let angle = TAU * fraction(quotient, cart, *axis);
        // The even parts ride the radius of revolution: invariant under the
        // reflection, so they descend to the quotient untouched. The odd parts
        // ride the half-angle frame, which reverses after one turn and is
        // therefore exactly the twist.
        let parts: Vec<(f64, f64)> = fiber
          .iter()
          .map(|&j| reflected_parts(quotient, cart, j))
          .collect();
        let natural = radius(quotient, *axis);
        let radius = natural * radius_slack + parts.iter().map(|&(even, _)| even).sum::<f64>();
        out(radius * angle.cos());
        out(radius * angle.sin());
        for (_, odd) in parts {
          out(odd * (angle / 2.0).cos());
          out(odd * (angle / 2.0).sin());
        }
      }
    }
  }
}

/// The even and odd parts of a reflected axis under its reflection.
///
/// A periodic axis is a circle, whose reflection $theta |-> -theta$ splits it
/// into $cos$ and $sin$; an open axis is an interval, reflected about its
/// midpoint, so it is purely odd once centred.
fn reflected_parts(quotient: &FlatQuotient, cart: &[usize], axis: Dim) -> (f64, f64) {
  if quotient.identifications()[axis].is_closed() {
    let angle = TAU * fraction(quotient, cart, axis);
    let radius = radius(quotient, axis);
    (radius * angle.cos(), radius * angle.sin())
  } else {
    let half = quotient.side_lengths()[axis] / 2.0;
    (0.0, position(quotient, cart, axis) - half)
  }
}

/// The radius of the circle an identified axis of period $L$ becomes.
///
/// **Chord-matched, not arc-matched.** What is embedded is the simplicial
/// manifold, not the smooth one: its vertices lie on the circle but its edges
/// are the straight chords between them, and a chord is shorter than the arc it
/// subtends. Taking the smooth radius $L \/ 2 pi$ would inscribe a polygon of
/// perimeter $2 pi R sin(pi \/ n) \/ (pi \/ n) < L$, isometric only as
/// $n -> oo$. Solving $2 R sin(pi \/ n) = L \/ n$ instead makes each axis step
/// exactly the length the flat quotient assigns it,
/// $R = L \/ (2 n sin(pi \/ n))$, and the Kuhn diagonals follow because the
/// circle factors are mutually orthogonal, so their squared chords simply add.
///
/// It converges to $L \/ 2 pi$ from above as the mesh refines.
fn radius(quotient: &FlatQuotient, axis: Dim) -> f64 {
  let n = quotient.ncells_per_axis()[axis] as f64;
  quotient.side_lengths()[axis] / (2.0 * n * (std::f64::consts::PI / n).sin())
}

/// The position along an axis within the fundamental domain.
fn position(quotient: &FlatQuotient, cart: &[usize], axis: Dim) -> f64 {
  fraction(quotient, cart, axis) * quotient.side_lengths()[axis]
}

/// The position along an axis as a fraction of its period.
fn fraction(quotient: &FlatQuotient, cart: &[usize], axis: Dim) -> f64 {
  cart[axis] as f64 / quotient.ncells_per_axis()[axis] as f64
}

/// The torus of revolution in $RR^3$ -- the donut -- for a two-dimensional
/// flat torus.
///
/// $(u, v) |-> ((R + r cos v) cos u, (R + r cos v) sin u, r sin v)$, with $u$
/// the first axis and $v$ the second.
///
/// **Not isometric, and not an instance of [`equivariant`].** This is a surface
/// of revolution, a construction of its own that happens to be topologically
/// $T^2$; it carries genuine Gaussian curvature, positive on the outer rim and
/// negative on the inner one, where the flat torus has none. A flat torus
/// admits no isometric embedding in $RR^3$ at all, which is why the isometric
/// one needs $RR^4$. Two-dimensional by nature: revolution does not generalize.
pub fn donut_r3(quotient: &FlatQuotient, tube_ratio: f64) -> MeshCoords {
  assert_eq!(quotient.dim(), 2, "The donut is a surface of revolution.");
  assert!(
    quotient
      .identifications()
      .iter()
      .all(|id| *id == Identification::Periodic),
    "The donut realizes the flat torus: both axes must be periodic."
  );
  assert!(
    (0.0..1.0).contains(&tube_ratio),
    "The tube must be thinner than the revolution radius, or the surface self-intersects."
  );

  let major = radius(quotient, 0);
  let minor = major * tube_ratio;
  revolve(quotient, |cart| {
    let u = TAU * fraction(quotient, cart, 0);
    let v = TAU * fraction(quotient, cart, 1);
    let r = major + minor * v.cos();
    [r * u.cos(), r * u.sin(), minor * v.sin()]
  })
}

/// The familiar Möbius strip in $RR^3$: a segment swept once around a circle
/// while rotating by half a turn.
///
/// $(u, y) |-> ((R + y cos(u\/2)) cos u, (R + y cos(u\/2)) sin u,
/// y sin(u\/2))$, with $u$ the twisted axis and $y$ the open fiber, centred.
///
/// **Not isometric.** The pure equivariant Möbius band needs $RR^4$ (a circle
/// and a half-angle frame, two dimensions each); this one folds into $RR^3$ by
/// borrowing the revolution trick, at the cost of the geometry. Flat Möbius
/// bands do embed in $RR^3$ (Wunderlich), but not by any expression as simple
/// as this, and the isometric case is served by [`equivariant`] instead.
pub fn moebius_r3(quotient: &FlatQuotient, radius_slack: f64) -> MeshCoords {
  assert_eq!(
    quotient.identifications(),
    [Identification::Twisted(vec![1]), Identification::Open],
    "The strip is axis 0 twisted about an open fiber axis 1."
  );
  assert!(
    radius_slack > 1.0,
    "The strip would pass through its own axis."
  );

  let width = quotient.side_lengths()[1];
  let major = radius(quotient, 0) * radius_slack;
  revolve(quotient, |cart| {
    let u = TAU * fraction(quotient, cart, 0);
    let y = position(quotient, cart, 1) - width / 2.0;
    let r = major + y * (u / 2.0).cos();
    [r * u.cos(), r * u.sin(), y * (u / 2.0).sin()]
  })
}

fn revolve(quotient: &FlatQuotient, point: impl Fn(&[usize]) -> [f64; 3]) -> MeshCoords {
  let mut matrix = Matrix::zeros(3, quotient.nvertices());
  for vertex in 0..quotient.nvertices() {
    let coords = point(&quotient.vertex_cart_idx(vertex));
    matrix.set_column(vertex, &Vector::from_column_slice(&coords));
  }
  MeshCoords::new(matrix)
}

#[cfg(test)]
mod test {
  use super::{donut_r3, equivariant, is_isometric, moebius_r3};
  use crate::{
    linalg::Vector,
    mesher::quotient::{FlatQuotient, Identification},
  };

  /// The Clifford embedding is an **isometry**: the edge lengths it induces are
  /// the flat quotient's own, in every dimension.
  ///
  /// This is the law that ties the two constructions together, and it checks
  /// both the generator's intrinsic geometry and the coordinate-to-lengths
  /// bridge against each other, neither standing in for the other.
  #[test]
  fn the_clifford_embedding_is_isometric() {
    for dim in 1..=3 {
      let quotient = FlatQuotient::unit_torus(dim, 4);
      assert!(is_isometric(&quotient));

      let (complex, intrinsic) = quotient.triangulate();
      let coords = equivariant(&quotient, 1.0 + f64::EPSILON);
      assert_eq!(coords.dim(), 2 * dim);

      let induced = coords.to_edge_lengths_sq(&complex);
      for (a, b) in intrinsic.iter().zip(induced.iter()) {
        assert!((a - b).abs() < 1e-9, "dim {dim}: {a} vs {b}");
      }
    }
  }

  /// An open axis is carried as itself, so a slab (no identification at all) is
  /// embedded isometrically too, in $RR^d$ rather than $RR^(2d)$. The
  /// degenerate end of the family, where the quotient is the grid.
  #[test]
  fn an_unidentified_slab_embeds_isometrically_in_its_own_dimension() {
    let quotient = FlatQuotient::new(
      Vector::from_element(2, 1.0),
      vec![Identification::Open, Identification::Open],
      3,
    );
    let (complex, intrinsic) = quotient.triangulate();
    let coords = equivariant(&quotient, 2.0);
    assert_eq!(coords.dim(), 2);

    let induced = coords.to_edge_lengths_sq(&complex);
    for (a, b) in intrinsic.iter().zip(induced.iter()) {
      assert!((a - b).abs() < 1e-9);
    }
  }

  /// The twisted quotients land where the mathematics says they must: the
  /// Möbius band and the Klein bottle both in $RR^4$, neither isometrically.
  /// The Klein bottle has no $RR^3$ embedding at all, so the ambient count is
  /// not a matter of taste.
  #[test]
  fn twisted_quotients_need_four_dimensions_and_are_not_isometric() {
    for quotient in [
      FlatQuotient::moebius(1.0, 0.4, 4),
      FlatQuotient::klein(Vector::from_element(2, 1.0), 4),
    ] {
      assert!(!is_isometric(&quotient));

      let (complex, intrinsic) = quotient.triangulate();
      let coords = equivariant(&quotient, 3.0);
      assert_eq!(coords.dim(), 4);

      let induced = coords.to_edge_lengths_sq(&complex);
      assert!(
        intrinsic
          .iter()
          .zip(induced.iter())
          .any(|(a, b)| (a - b).abs() > 1e-6),
        "a twisted embedding is curved, so it cannot reproduce the flat lengths"
      );
      // Curved, but still a faithful realization: no edge collapses.
      assert!(induced.iter().all(|&l| l > 0.0));
    }
  }

  /// The embedding descends to the quotient: identified vertices are one
  /// vertex, so the coordinates are single-valued, and distinct vertices stay
  /// distinct. This is what "equivariant" buys, and it is the property the
  /// half-angle frame exists to provide.
  #[test]
  fn the_embedding_separates_the_vertices() {
    for quotient in [
      FlatQuotient::unit_torus(2, 4),
      FlatQuotient::moebius(1.0, 0.4, 4),
      FlatQuotient::klein(Vector::from_element(2, 1.0), 4),
    ] {
      let coords = equivariant(&quotient, 3.0);
      let matrix = coords.matrix();
      for i in 0..quotient.nvertices() {
        for j in (i + 1)..quotient.nvertices() {
          let separation = (matrix.column(i) - matrix.column(j)).norm();
          assert!(separation > 1e-6, "vertices {i} and {j} coincide");
        }
      }
    }
  }

  /// The $RR^3$ pictures are three-dimensional, injective, and -- the point
  /// worth asserting -- **not** isometric: their induced lengths are a
  /// different manifold from the flat quotient that produced the topology.
  #[test]
  fn the_r3_pictures_are_curved() {
    let torus = FlatQuotient::unit_torus(2, 4);
    let strip = FlatQuotient::moebius(1.0, 0.4, 4);
    let cases = [
      (&torus, donut_r3(&torus, 0.4)),
      (&strip, moebius_r3(&strip, 2.0)),
    ];
    for (quotient, coords) in cases {
      assert_eq!(coords.dim(), 3);
      let (complex, intrinsic) = quotient.triangulate();
      let induced = coords.to_edge_lengths_sq(&complex);
      assert!(induced.iter().all(|&l| l > 0.0));
      assert!(
        intrinsic
          .iter()
          .zip(induced.iter())
          .any(|(a, b)| (a - b).abs() > 1e-6),
        "an RR^3 realization of a flat surface is curved"
      );
    }
  }
}
