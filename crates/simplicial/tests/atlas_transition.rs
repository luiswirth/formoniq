//! Transitions between charts: the identity on a chart with itself, the
//! cocycle law across three, and agreement on a shared face.

use approx::assert_relative_eq;
use multialgebra::{ExteriorGrade, Tensor, Vector, exterior_dim};
use simplicial::Dim;
use simplicial::atlas::bundle::{FaceTrace, face_tangent_blade};
use simplicial::atlas::{Chart, ChartExt, MeshPoint, barycenter_bary};
use simplicial::linalg::Matrix;
use simplicial::mesher::grid::CartesianTopology;
use simplicial::topology::complex::Complex;
use simplicial::topology::handle::SimplexRef;

/// The transition of a chart with itself is the identity map.
#[test]
fn self_transition_is_the_identity() {
  for dim in (1..=3usize).map(Dim::from) {
    let complex = Complex::unit(dim);
    let cell = complex.cells().handle_iter().next().unwrap();

    let transition = cell.transition_to(cell);
    assert!(transition.is_identity());
    assert_relative_eq!(
      transition.bary_map(),
      &Matrix::identity((dim + 1).index(), (dim + 1).index())
    );
    assert_relative_eq!(
      transition.differential(),
      Matrix::identity(dim.index(), dim.index())
    );

    let point = MeshPoint::barycenter(cell.idx());
    let mapped = transition.apply(&point).unwrap();
    assert_eq!(mapped, point);
  }
}

/// Every pair of adjacent cells, with the barycenter of the facet they share:
/// the setting in which a transition is defined.
fn adjacent_pairs(complex: &Complex) -> Vec<(Chart<'_>, Chart<'_>, MeshPoint)> {
  let dim = complex.dim();
  let mut pairs = Vec::new();
  for facet in complex.skeleton(dim - 1).handle_iter() {
    let cells: Vec<_> = facet.cells().collect();
    for (i, &source) in cells.iter().enumerate() {
      for &target in &cells[i + 1..] {
        let positions = facet.simplex().relative_to(source.simplex());
        let point = source.point_on_face(&positions, &barycenter_bary(dim - 1));
        pairs.push((source, target, point));
      }
    }
  }
  pairs
}

/// A point of the overlap, carried into the neighboring chart and back, is
/// the point one started with: the transitions of an atlas are invertible on
/// the overlap, and the two directions are mutually inverse.
#[test]
fn transition_roundtrip_on_the_overlap() {
  for dim in (1..=3usize).map(Dim::from) {
    let complex = CartesianTopology::cube(dim, 2).triangulate();

    for (source, target, point) in adjacent_pairs(&complex) {
      let transition = source.transition_to(target);
      let there = transition.apply(&point).expect("point is on the overlap");
      assert_eq!(there.cell_idx(), target.idx());

      let back = transition.inverse().apply(&there).unwrap();
      assert_eq!(back.cell_idx(), source.idx());
      assert_relative_eq!(back.bary().view(), point.bary().view(), epsilon = 1e-12);
    }
  }
}

/// Off the overlap there is no transition: a point in the interior of a cell
/// has no representation in any other chart.
#[test]
fn no_transition_off_the_overlap() {
  for dim in (1..=3usize).map(Dim::from) {
    let complex = CartesianTopology::cube(dim, 2).triangulate();

    for (source, target, _) in adjacent_pairs(&complex) {
      let interior = source.barycenter();
      assert!(source.transition_to(target).apply(&interior).is_none());
    }
  }
}

/// $psi_(K'' K') compose psi_(K' K) = psi_(K'' K)$: the cocycle condition, on
/// the triple overlap where all three charts see the point.
///
/// This is the coherence law of an atlas, the statement that the charts
/// describe one manifold and not three.
#[test]
fn transition_cocycle() {
  for dim in (2..=3usize).map(Dim::from) {
    let complex = CartesianTopology::cube(dim, 2).triangulate();

    // A vertex of the mesh lies in the overlap of every cell around it.
    for vertex in complex.vertices().handle_iter() {
      let cells: Vec<_> = vertex.cells().collect();
      for &first in &cells {
        let positions = vertex.simplex().relative_to(first.simplex());
        let point = first.point_on_face(&positions, &barycenter_bary(Dim::new(0)));

        for &second in &cells {
          for &third in &cells {
            let direct = first.transition_to(third).apply(&point).unwrap();
            let composed = first
              .transition_to(second)
              .apply(&point)
              .and_then(|mid| second.transition_to(third).apply(&mid))
              .unwrap();
            assert_eq!(direct, composed);
          }
        }
      }
    }
  }
}

/// An arbitrary, nowhere-vanishing form of the given shape.
fn test_form(dim: Dim, grade: ExteriorGrade) -> Tensor {
  let n = exterior_dim(dim, grade);
  Tensor::multiform(
    Vector::from_iterator(n, (0..n).map(|i| 0.7 * (i as f64) - 1.3)),
    dim,
    grade,
  )
}

/// Every face of dimension at least one, with the charts that contain it:
/// the setting in which the fibers over an overlap can be compared.
fn shared_faces(complex: &Complex) -> Vec<(SimplexRef<'_>, Vec<Chart<'_>>)> {
  let mut faces = Vec::new();
  for face_dim in Dim::ONE.range_to_inclusive(complex.dim()) {
    for face in complex.skeleton(face_dim).handle_iter() {
      let cells = face.cells().collect();
      faces.push((face, cells));
    }
  }
  faces
}

/// The tangent blade of a shared face transforms by $Lambda^d (dif psi)$: the
/// pushforward of the blade computed in one chart is the blade computed in
/// the other.
///
/// The vector side of the agreement, and the sharpest form of it: the blade
/// spans the whole of $Lambda^d (T tau)$, so the transition is pinned on the
/// tangential part with nothing left over.
#[test]
fn tangent_blade_transforms_by_the_transition_differential() {
  for dim in (2..=3usize).map(Dim::from) {
    let complex = CartesianTopology::cube(dim, 2).triangulate();

    for (face, cells) in shared_faces(&complex) {
      for (i, &source) in cells.iter().enumerate() {
        for &target in &cells[i + 1..] {
          let here = face_tangent_blade(dim, &face.simplex().relative_to(source.simplex()));
          let there = face_tangent_blade(dim, &face.simplex().relative_to(target.simplex()));

          assert_relative_eq!(
            source.transition_to(target).pushforward(&here).components(),
            there.components(),
            epsilon = 1e-12
          );
        }
      }
    }
  }
}

/// Two charts sharing a face agree on the tangential part of a fiber value:
/// $tr_tau (psi^* omega) = tr_tau omega$, the traces taken in the respective
/// charts.
///
/// The form side of the same fact, and the one that makes an integral over a
/// face well defined regardless of which adjacent chart computes it.
///
/// The equality is not vacuous, and the test says so: the two chart
/// representations $psi^* omega$ and $omega$ genuinely differ, and it is only
/// after tracing that they agree.
#[test]
fn charts_agree_on_the_tangential_part() {
  let mut disagreements = 0;
  for dim in (2..=3usize).map(Dim::from) {
    let complex = CartesianTopology::cube(dim, 2).triangulate();

    for (face, cells) in shared_faces(&complex) {
      let face_dim = face.dim();
      for (i, &source) in cells.iter().enumerate() {
        for &target in &cells[i + 1..] {
          let transition = source.transition_to(target);
          let here = face.simplex().relative_to(source.simplex());
          let there = face.simplex().relative_to(target.simplex());

          for grade in Dim::ZERO.range_to_inclusive(face_dim) {
            let value = test_form(dim, grade);
            let pulled = transition.pullback(&value);

            assert_relative_eq!(
              FaceTrace::new(dim, &here, grade)
                .apply(&pulled)
                .components(),
              FaceTrace::new(dim, &there, grade)
                .apply(&value)
                .components(),
              epsilon = 1e-12
            );

            if !pulled.eq_epsilon(&value, 1e-9) {
              disagreements += 1;
            }
          }
        }
      }
    }
  }
  assert!(
    disagreements > 0,
    "The charts must genuinely disagree off the tangential part."
  );
}

/// The cocycle law on the fibers:
/// $psi_(K' K)^* compose psi_(K'' K')^* = psi_(K'' K)^*$, traced onto the
/// face all three charts share. Pullbacks compose contravariantly, so a route
/// through a third chart is the same map as the direct one.
///
/// The law holds where the composite means anything, which is the triple
/// overlap and no further. Off it the two routes are affine extensions of maps
/// that describe nothing, and the test asserts that they do differ there: the
/// untraced values disagree, which is what it looks like for the
/// non-tangential part of a fiber value to be an artifact of the route rather
/// than data of the manifold.
///
/// This is the fiber-level counterpart of the cocycle on points, and it is
/// what makes a tangential quantity well defined over the whole manifold
/// rather than merely between neighbors.
#[test]
fn fiber_cocycle_on_the_triple_overlap() {
  let mut disagreements = 0;
  for dim in (2..=3usize).map(Dim::from) {
    let complex = CartesianTopology::cube(dim, 2).triangulate();

    for (face, cells) in shared_faces(&complex) {
      for &first in &cells {
        let positions = face.simplex().relative_to(first.simplex());
        for &second in &cells {
          for &third in &cells {
            let (a, b) = (first.transition_to(second), second.transition_to(third));
            let c = first.transition_to(third);

            for grade in Dim::ZERO.range_to_inclusive(face.dim()) {
              let value = test_form(dim, grade);
              let routed = a.pullback(&b.pullback(&value));
              let direct = c.pullback(&value);
              let trace = FaceTrace::new(dim, &positions, grade);

              assert_relative_eq!(
                trace.apply(&routed).components(),
                trace.apply(&direct).components(),
                epsilon = 1e-12
              );

              if !routed.eq_epsilon(&direct, 1e-9) {
                disagreements += 1;
              }
            }
          }
        }
      }
    }
  }
  assert!(
    disagreements > 0,
    "The two routes must differ off the triple overlap."
  );
}
