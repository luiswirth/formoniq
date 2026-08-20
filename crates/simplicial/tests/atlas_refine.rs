//! The Freudenthal reference pattern: it composes like a semigroup, its
//! children partition the reference volume, and its vertices are the lattice.

use simplicial::Dim;
use simplicial::atlas::{unit_lattice, unit_refinement, unit_simplex_volume};
use simplicial::linalg::Vector;

/// Freudenthal subdivision composes: refining an ordered simplex $R$-fold
/// and then $R'$-fold again is the $R R'$-fold refinement, cell for cell.
///
/// $ "refine"_(R') compose "refine"_R = "refine"_(R R') $
///
/// The semigroup law of the reference pattern, and the reason a refinement
/// tower stays inside the Kuhn family: every child is similar to its parent,
/// in every dimension, at every level. It holds only when each child is
/// refined in the order this pattern emits its corners, that order carries
/// Freudenthal's type, and it is the whole of what a child inherits. Sorting a
/// child's vertices instead (by a global numbering, say) reproduces the
/// pattern at the first level and drifts out of the family after, into a
/// growing number of congruence classes.
#[test]
fn refinement_composes_on_ordered_simplices() {
  /// The children of an ordered simplex, each in the pattern's own corner
  /// order: the parent's vertices read through the lattice weights.
  fn refine_ordered(vertices: &[Vector], refinement: usize) -> Vec<Vec<Vector>> {
    let dim = vertices.len() - 1;
    let pattern = unit_refinement(dim, refinement);
    pattern
      .children()
      .iter()
      .map(|child| {
        child
          .iter()
          .map(|&corner| {
            pattern.vertices()[corner]
              .iter()
              .enumerate()
              .fold(Vector::zeros(dim), |point, (i, &weight)| {
                point + (weight as f64 / refinement as f64) * &vertices[i]
              })
          })
          .collect()
      })
      .collect()
  }

  /// The cells as sorted vertex coordinates, quantized: the mesh's identity,
  /// independent of the order the children were produced in.
  fn mesh(cells: &[Vec<Vector>]) -> Vec<Vec<Vec<i64>>> {
    let mut cells: Vec<Vec<Vec<i64>>> = cells
      .iter()
      .map(|cell| {
        let mut vertices: Vec<Vec<i64>> = cell
          .iter()
          .map(|v| v.iter().map(|x| (x * 1e9).round() as i64).collect())
          .collect();
        vertices.sort();
        vertices
      })
      .collect();
    cells.sort();
    cells
  }

  for dim in (1..=4usize).map(Dim::from) {
    // The Kuhn simplex of the unit cube: the chain of partial sums of the axes.
    let mut corner = Vector::zeros(dim.index());
    let mut kuhn = vec![corner.clone()];
    for axis in 0..dim.index() {
      corner[axis] = 1.0;
      kuhn.push(corner.clone());
    }
    // And a sheared image of it. The subdivision is defined by barycentric
    // weights, so it commutes with any affine map: the law is affine, not a
    // property of the Kuhn simplex, and therefore holds on an arbitrary mesh.
    // What is special to Kuhn is similarity of the children, an affine map
    // preserves the composition but not the shape classes.
    let skewed: Vec<Vector> = kuhn
      .iter()
      .map(|v| {
        let mut w = v.clone();
        for axis in 0..dim.index() {
          w[axis] += 0.3 * (axis + 1) as f64 * v[(axis + 1) % dim.index()] + 0.1 * v[axis];
        }
        w
      })
      .collect();

    for base in [&kuhn, &skewed] {
      for refinement in 1..=3 {
        let tower: Vec<Vec<Vector>> = refine_ordered(base, refinement)
          .iter()
          .flat_map(|child| refine_ordered(child, refinement))
          .collect();
        assert_eq!(
          mesh(&tower),
          mesh(&refine_ordered(base, refinement * refinement)),
          "dim {dim}: refining twice by {refinement} must equal refining once by {}",
          refinement * refinement
        );
      }
    }
  }
}

/// The edgewise subdivision has $R^n$ children, its vertices are exactly the
/// lattice $L_R^n$, and every child is a nondegenerate simplex.
#[test]
fn children_and_vertices() {
  for dim in (0..=4usize).map(Dim::from) {
    for r in 1..=3 {
      let sub = unit_refinement(dim, r);
      assert_eq!(sub.nchildren(), r.pow(dim.index() as u32));

      let lattice: Vec<Vec<usize>> = unit_lattice(dim, r).collect();
      assert_eq!(sub.vertices(), lattice.as_slice());

      for child in sub.children() {
        assert_eq!(child.len(), dim + 1);
        let mut corners = child.to_vec();
        corners.sort_unstable();
        corners.dedup();
        assert_eq!(corners.len(), child.len(), "corners must be distinct");
      }
    }
  }
}

/// The child volumes partition the reference cell: in the affine (metric-free)
/// reference frame each child has volume $1 \/ (R^n n!)$, and they sum to the
/// reference volume $1 \/ n!$. Equivalently every child is congruent to the
/// $R^(-n)$-scaled reference cell. Read through the child's realization in the
/// parent frame, vertex order is immaterial here, the volume being
/// order-invariant.
#[test]
fn volume_partition() {
  for dim in (0..=4usize).map(Dim::from) {
    for r in 1..=3 {
      let sub = unit_refinement(dim, r);
      let total: f64 = (0..sub.nchildren())
        .map(|c| sub.child_local_simplex(c).vol())
        .sum();
      approx::assert_relative_eq!(total, unit_simplex_volume(dim), epsilon = 1e-12);
      for c in 0..sub.nchildren() {
        approx::assert_relative_eq!(
          sub.child_local_simplex(c).vol(),
          unit_simplex_volume(dim) / (r.pow(dim.index() as u32) as f64),
          epsilon = 1e-12
        );
      }
    }
  }
}
