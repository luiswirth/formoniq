//! [`SimplexLengthsSq`]: the Regge representation of a cell's metric, on
//! any signature. `from_metric`/`metric` round-trip, the edge squares are a
//! basis of $"Sym"^2$, and geometry restricts to a face by index alone.

use approx::assert_relative_eq;
use metric::{CausalType, Metric};
use multialgebra::tensor::pairing;
use multialgebra::{Factor, Variance};
use multiindex::{Dim, combinations};
use regge::lengths::LengthsSq;
use regge::lengths::simplex::{SimplexLengthsSq, unit_edge_squares};
use simplicial::linalg::Matrix;
use simplicial::topology::simplex::{edge_index, nedges};

/// A metric with distinct entries throughout: an equal-entry probe hides a
/// wrong weight in a basis change.
fn probe_metric(dim: usize) -> Metric {
  let a = Matrix::from_fn(dim, dim, |i, j| ((3 * i + 7 * j) % 5) as f64 / 5.0);
  Metric::new(
    Variance::Covariant,
    a.transpose() * &a + Matrix::identity(dim, dim),
  )
}

/// from_metric and metric are inverse, on every
/// signature, the flat models pulled back to non-diagonal form included.
/// The Regge representation loses nothing of a pseudo-Riemannian metric.
#[test]
fn metric_tensor_roundtrip() {
  for dim in (1..=4usize).map(Dim::from) {
    let lengths_sq = SimplexLengthsSq::unit(dim);
    let roundtrip = SimplexLengthsSq::from_metric(&lengths_sq.metric());
    assert_relative_eq!(lengths_sq.vector(), roundtrip.vector(), epsilon = 1e-12);

    for q in 0..=dim.index() {
      let j = Matrix::from_fn(dim.index(), dim.index(), |i, jj| {
        if i == jj {
          1.0
        } else if i > jj {
          ((2 * i + 3 * jj) % 4) as f64 / 8.0
        } else {
          0.0
        }
      });
      let g = Metric::pseudo_euclidean(dim.index() - q, q).pullback(&j);
      let regge = SimplexLengthsSq::from_metric(&g);
      assert_relative_eq!(regge.metric().matrix(), g.matrix(), epsilon = 1e-12);
      assert_eq!(regge.metric().signature(), (dim.index() - q, q));
    }
  }
}

/// Degeneracy is measured against the simplex's own scale, so a uniform
/// scaling leaves it alone. Collapsing two vertices onto each other, which
/// makes two rows of the distance matrix agree, trips it at every scale.
#[test]
fn degeneracy_is_scale_invariant() {
  for dim in (2..=4usize).map(Dim::from) {
    for scale in [1e-4, 1.0, 1e4] {
      let mut lengths = SimplexLengthsSq::unit(dim);
      *lengths.vector_mut() *= scale * scale;
      assert!(!lengths.is_degenerate());

      lengths.vector_mut()[edge_index(1, 2)] = 0.0;
      assert!(lengths.is_degenerate());
    }
  }
}

/// The checked constructor decides the hypothesis it is named for: it accepts
/// the unit simplex and rejects the same lengths with two vertices collapsed.
///
/// Both directions, so that a constructor returning `Some` unconditionally
/// would fail. Non-degeneracy, not definiteness: the Minkowski simplex is
/// accepted too, and it is not Euclidean-realizable.
#[test]
fn new_checked_decides_non_degeneracy() {
  for dim in (2..=4usize).map(Dim::from) {
    let unit = SimplexLengthsSq::unit(dim);
    assert!(unit.is_valid());
    assert!(SimplexLengthsSq::new_checked(unit.vector().clone(), dim).is_some());

    let lorentzian = SimplexLengthsSq::from_metric(&Metric::minkowski(dim.index()));
    assert!(!lorentzian.is_coordinate_realizable());
    assert!(SimplexLengthsSq::new_checked(lorentzian.into_vector(), dim).is_some());

    let mut collapsed = unit.into_vector();
    collapsed[edge_index(1, 2)] = 0.0;
    assert!(SimplexLengthsSq::new_checked(collapsed, dim).is_none());
  }
}

/// The causal trichotomy of Regge edges on a Minkowski cell: the reference
/// simplex measured with $eta$ has its time edge timelike, its space edges
/// spacelike, and the volume is the reference volume, $|det eta| = 1$.
#[test]
fn minkowski_regge_edges() {
  for dim in (2..=4usize).map(Dim::from) {
    let regge = SimplexLengthsSq::from_metric(&Metric::minkowski(dim.index()));
    // Edge 0-1 is the time axis $e_0$.
    assert_eq!(regge.causal_type(edge_index(0, 1)), CausalType::Timelike);
    // Edge 0-2 is the space axis $e_1$.
    assert_eq!(regge.causal_type(edge_index(0, 2)), CausalType::Spacelike);
    // Edge 1-2 is $e_1 - e_0$ with $norm^2_eta = 1 - 1 = 0$: lightlike.
    assert_eq!(regge.causal_type(edge_index(1, 2)), CausalType::Null);

    assert!(!regge.is_coordinate_realizable());
    assert_relative_eq!(
      regge.vol(),
      SimplexLengthsSq::unit(dim).vol(),
      epsilon = 1e-12
    );
  }
}

/// The edge squares are a basis of $"Sym"^2$: as many as its dimension,
/// and independent. The count is $binom(n+1,2) = n(n+1)\/2 = dim "Sym"^2(RR^n)$,
/// so a rank check is what separates "the right number" from "a basis".
#[test]
fn the_edge_squares_are_a_basis_of_sym2() {
  for dim in (0..=4usize).map(Dim::from) {
    let squares = unit_edge_squares(dim);
    let sym2 = Factor::symmetric(2).multidim(dim);
    assert_eq!(squares.len(), nedges(dim));
    assert_eq!(squares.len(), sym2);

    // The point simplex has no edges and a zero-dimensional Sym^2, so the
    // empty family is its basis: the count above is the whole statement and
    // there is no rank to take.
    if sym2 > 0 {
      let components = Matrix::from_fn(sym2, squares.len(), |i, e| squares[e].components()[i]);
      assert_eq!(
        components.rank(1e-9),
        sym2,
        "the squares must be independent"
      );
    }
  }
}

/// Squared edge lengths are the components of the metric in the basis dual to
/// the edge squares: $s_e = angle.l g, u_e dot.circle u_e angle.r$.
///
/// The polarization identity of [`SimplexLengthsSq::metric`] and
/// [`SimplexLengthsSq::from_metric`] is that change of basis, and this is
/// what says so rather than asserting it in prose. Swept over every signature,
/// since the pairing is metric-free and so must hold on all of them.
#[test]
fn the_squared_lengths_are_the_metric_paired_with_the_edge_squares() {
  for dim in (1..=4usize).map(Dim::from) {
    for q in 0..=dim.index() {
      let metric = Metric::pseudo_euclidean(dim.index() - q, q);
      let lengths = SimplexLengthsSq::from_metric(&metric);
      for (iedge, square) in unit_edge_squares(dim).iter().enumerate() {
        assert_relative_eq!(
          lengths[iedge],
          pairing(&metric.tensor(), square),
          epsilon = 1e-12
        );
      }
    }
  }
}

/// Restricting the geometry to a face is an index selection in the edge
/// basis: the face's squared lengths are the parent's at the face's own edge
/// indices, with nothing computed.
///
/// This is why geometry is defined on every simplex and not only on the cells
/// (invariant 2). In the cartesian frame the same restriction is a projection
/// $J^top g J$; here the two agree, which is the statement that the edge basis
/// is the one adapted to the face lattice.
#[test]
fn restricting_to_a_face_selects_edge_components() {
  for dim in (1..=4usize).map(Dim::from) {
    let metric = probe_metric(dim.index());
    let lengths = SimplexLengthsSq::from_metric(&metric);

    for face in combinations(dim.index() + 1, dim.index()) {
      // The face's own squared lengths, read off the parent by index alone.
      let selected: Vec<f64> = combinations(face.card(), 2)
        .map(|pair| {
          lengths[edge_index(
            face.index_at(pair.index_at(0)),
            face.index_at(pair.index_at(1)),
          )]
        })
        .collect();
      let face_lengths = SimplexLengthsSq::new(selected.into(), dim.index() - 1);

      // The same restriction done the cartesian way: pull the metric back
      // along the inclusion of the face's spanning vectors.
      let inclusion = Matrix::from_fn(dim.index(), dim.index() - 1, |i, k| {
        let (base, other) = (face.index_at(0), face.index_at(k + 1));
        let mut column = 0.0;
        if other == i + 1 {
          column += 1.0;
        }
        if base == i + 1 {
          column -= 1.0;
        }
        column
      });
      let pulled = SimplexLengthsSq::from_metric(&metric.pullback(&inclusion));

      assert_relative_eq!(face_lengths.vector(), pulled.vector(), epsilon = 1e-12);
    }
  }
}
