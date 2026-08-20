//! [`MeshLengthsSq`]: the local length scale under refinement, per-cell
//! non-degeneracy, and the trace law a face's metric obeys against any
//! containing cell's.

use multiindex::Dim;
use regge::lengths::mesh::MeshLengthsSq;
use regge::mesher::cartesian::CartesianGrid;
use regge::mesher::sphere::mesh_sphere_surface;
use simplicial::topology::data::SkeletonData;
use simplicial::topology::simplex::edge_index;

/// The mesh's local length scale halves when every edge is split: it tracks
/// the refinement, which is exactly what distinguishes it from the extent of
/// an embedding, which does not move at all.
#[test]
fn mean_width_halves_under_subdivision() {
  let mut previous: Option<f64> = None;
  for subdivisions in 1..=4 {
    let (topology, coords) = mesh_sphere_surface(subdivisions);
    let mean = coords.to_edge_lengths_sq(&topology).mesh_width_mean();
    if let Some(previous) = previous {
      let ratio = mean / previous;
      assert!(
        (ratio - 0.5).abs() < 0.05,
        "subdivision {subdivisions}: edge length scaled by {ratio}, expected ~0.5"
      );
    }
    previous = Some(mean);
  }
}

/// The checked constructor decides per-cell non-degeneracy: it accepts the
/// geometry a grid embedding induces and rejects the same geometry with one
/// edge collapsed.
///
/// Both directions, so that a constructor returning `Some` unconditionally
/// would fail. The hypothesis is relational, which is why the complex enters
/// here and not at [`MeshLengthsSq::new`]: the same vector is a geometry or
/// not depending on which cells are asked about.
///
/// The perturbation identifies two vertices of one cell, which makes two rows
/// of its distance matrix agree and its Cayley-Menger determinant vanish.
/// Merely zeroing one edge would not do: it leaves a length set no Euclidean
/// configuration realizes, whose determinant is negative rather than zero, so
/// the simplex is non-degenerate of indefinite signature and rightly
/// accepted. Non-degeneracy is the hypothesis, realizability is not.
#[test]
fn new_checked_decides_cell_non_degeneracy() {
  for dim in (1..=3usize).map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let lengths_sq = coords.to_edge_lengths_sq(&topology);
    assert!(lengths_sq.is_valid(&topology));

    let vector = lengths_sq.into_vector();
    assert!(MeshLengthsSq::new_checked(vector.clone(), &topology).is_some());

    let cell = topology.cells().handle_iter().next().unwrap();
    let edge_kidxs: Vec<_> = cell.get().edges().map(|edge| edge.kidx()).collect();
    let mut collapsed = vector.clone();
    collapsed[edge_kidxs[edge_index(0, 1)]] = 0.0;
    for other in 2..=dim.index() {
      collapsed[edge_kidxs[edge_index(1, other)]] = vector[edge_kidxs[edge_index(0, other)]];
    }
    assert!(MeshLengthsSq::new_checked(collapsed, &topology).is_none());
  }
}

/// A point cloud has no edges and so no local length. The caller is told so
/// rather than dividing by a count of zero.
#[test]
fn a_mesh_without_edges_has_no_local_length() {
  assert_eq!(MeshLengthsSq::unit(Dim::new(0)).mesh_width_mean(), 0.0);
}

/// Coordinates and squared edge lengths read uniformly as data on simplices:
/// coords (grade 0) return a column view, squared lengths (grade 1) a
/// scalar ref.
#[test]
fn geometry_as_simplex_data() {
  let (topology, coords) = CartesianGrid::new_unit(Dim::new(2), 2).triangulate();
  let lengths_sq = coords.to_edge_lengths_sq(&topology);

  assert_eq!(SkeletonData::grade(&coords), 0);
  assert_eq!(SkeletonData::grade(&lengths_sq), 1);

  for vertex in topology.vertices().handle_iter() {
    assert_eq!(coords.at_ref(vertex.get()), coords.coord(vertex.kidx()));
  }
  for edge in topology.edges().handle_iter() {
    let [vi, vj] = edge.simplex().clone().try_into().unwrap();
    let expected = (coords.coord(vj) - coords.coord(vi)).norm_squared();
    assert_eq!(*lengths_sq.at_ref(edge.get()), expected);
  }
}

/// The trace law [`MeshLengthsSq::simplex_metric`] rests on: a face's
/// intrinsic metric is the pullback of any containing cell's along the
/// inclusion of tangent spaces, $g_sigma = B^T g_K B$.
///
/// This is the tangential-tangential trace of the Regge field, and it is
/// what makes the edge lengths rather than the per-cell Gramians the
/// primitive: the trace exists at every grade, so a face carries a metric
/// with no containing cell consulted. Swept over every dimension, every
/// grade and every face, the top grade included, where the inclusion is the
/// identity.
#[test]
fn subsimplex_metric_is_restriction_of_cell_metric() {
  for dim in 1..=4 {
    let dim = Dim::new(dim);
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let lengths = coords.to_edge_lengths_sq(&topology);

    for cell in topology.cells().handle_iter() {
      let cell_simplex = cell.get().simplex().clone();
      let cell_metric = lengths.cell_metric(cell);

      for grade in 1..=dim.index() {
        for face in cell.get().faces(grade) {
          let positions = face.simplex().relative_to(&cell_simplex);

          // Column $a$ is the face's tangent vector $u_(a+1) - u_0$ read in
          // the cell's basis $e_i = v_(i+1) - v_0$, where the apex $v_0$
          // contributes nothing because $e_(-1) = 0$.
          let mut inclusion = simplicial::linalg::Matrix::zeros(dim.index(), grade);
          let apex = positions.index_at(0);
          for a in 0..grade {
            let head = positions.index_at(a + 1);
            if head > 0 {
              inclusion[(head - 1, a)] += 1.0;
            }
            if apex > 0 {
              inclusion[(apex - 1, a)] -= 1.0;
            }
          }

          let restricted = cell_metric.pullback(&inclusion);
          let intrinsic = lengths.simplex_metric(face);
          approx::assert_relative_eq!(intrinsic.matrix(), restricted.matrix(), epsilon = 1e-12);
        }
      }
    }
  }
}
