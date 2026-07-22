//! The reference subdivision pattern of a chart.
//!
//! Refining a mesh is refining every chart the same way, up to the labelling of
//! its vertices -- the atlas philosophy applied to $h$-refinement. So the whole
//! combinatorics lives once on the reference cell, as a function of the
//! dimension and the refinement level alone, and a mesh is refined by relabelling
//! this one pattern onto every cell (see
//! [`Complex::refine`](crate::topology::complex::Complex::refine)).
//!
//! The pattern is the **edgewise (Freudenthal) subdivision** of the reference
//! $n$-simplex into $R^n$ children, the canonical dimension-general
//! generalization of the 2D red refinement (Bank) and its 3D analogue (Bey).
//! It is affine and metric-free: a statement about the barycentric lattice, not
//! about any geometry the cell may carry.
//!
//! # Construction
//!
//! The reference simplex maps affinely and bijectively onto the *order simplex*
//! $Delta = { t in RR^n : 1 >= t_1 >= dots.h.c >= t_n >= 0 }$ by the cumulative
//! barycentric coordinates $t_j = sum_(i >= j) lambda_i$ -- itself a Kuhn
//! simplex of the unit cube. Scaling by $R$ and intersecting the integer
//! Freudenthal (Kuhn) triangulation of $ZZ^n$ with $R Delta$ gives exactly the
//! $R^n$ children. A Kuhn simplex is a base point $b in ZZ^n$ and a permutation
//! $pi in S_n$, with vertices $w^0 = b$, $w^k = w^(k-1) + e_(pi_k)$; it is a
//! child iff every vertex lies in $R Delta$, i.e. satisfies
//! $R >= s_1 >= dots.h.c >= s_n >= 0$.
//!
//! # Conformity
//!
//! Because the construction is driven only by the order of the cell's vertices,
//! its restriction to a face depends only on that face's vertices and their
//! order -- not on the opposite vertex. Two cells sharing a face carry the same
//! global order on it (simplices store vertices increasingly), so they subdivide
//! the shared face identically, and the refined mesh is conforming. This is why
//! the reference pattern is expressed on lattice points keyed by the vertices
//! they are supported on: the key is what a shared point agrees on across
//! charts.

use super::{Bary, LocalCartesian, SimplexCoords, bary2local};
use crate::Dim;
use crate::linalg::{Matrix, Vector};

use multiindex::{Composition, Permutation, cartesian};

use std::collections::HashMap;

/// The reference edgewise subdivision of the $n$-simplex into $R^n$ children,
/// as combinatorics on the barycentric lattice $L_R^n$.
///
/// A function of `(dim, refinement)` alone. Its vertices are exactly the
/// lattice points [`ref_lattice`](super::ref_lattice) produces, and a child is
/// the list of colex ranks of its $n+1$ corners among them.
#[derive(Debug, Clone)]
pub struct ReferenceRefinement {
  dim: Dim,
  refinement: usize,
  /// Every lattice point of $L_R^n$, colex-ordered: each an integer
  /// barycentric numerator vector $k$ ($"len" = "dim" + 1$, $sum_i k_i = R$).
  vertices: Vec<Vec<usize>>,
  /// Each child, as the ranks into `vertices` of its $n+1$ corners, with the
  /// zeroth being the child's reference base vertex.
  children: Vec<Vec<usize>>,
}

/// The reference edgewise subdivision of the $n$-simplex at refinement $R >= 1$.
pub fn ref_refinement(dim: Dim, refinement: usize) -> ReferenceRefinement {
  assert!(refinement >= 1, "A refinement is at least one.");

  let vertices: Vec<Vec<usize>> = Composition::all((dim + 1).index(), refinement)
    .map(Composition::into_parts)
    .collect();
  let rank: HashMap<&Vec<usize>, usize> =
    vertices.iter().enumerate().map(|(i, k)| (k, i)).collect();

  let children: Vec<Vec<usize>> = if dim == 0 {
    // The point is its own only child: $R^0 = 1$.
    vec![vec![0]]
  } else {
    kuhn_children(dim, refinement)
      .map(|child| {
        child
          .into_iter()
          .map(|k| rank[&cube_to_lattice(&k, refinement)])
          .collect()
      })
      .collect()
  };

  assert_eq!(
    children.len(),
    refinement.pow(dim.index() as u32),
    "edgewise subdivision must yield R^n children"
  );

  ReferenceRefinement {
    dim,
    refinement,
    vertices,
    children,
  }
}

/// The Kuhn simplices of the scaled order simplex $R Delta$, each as its $n + 1$
/// cube-coordinate vertices $s in {0, dots, R}^n$ (descending).
fn kuhn_children(dim: Dim, refinement: usize) -> impl Iterator<Item = Vec<Vec<usize>>> {
  let r = refinement;
  // A vertex lies in $R Delta$ iff $R >= s_1 >= ... >= s_n >= 0$.
  let in_region = move |s: &[usize]| s[0] <= r && s.windows(2).all(|w| w[0] >= w[1]);

  let n = dim.index();
  cartesian::grid(r + 1, n).flat_map(move |base| {
    Permutation::all(n).filter_map(move |perm| {
      let mut w = base.clone();
      let mut child = Vec::with_capacity((dim + 1).index());
      child.push(w.clone());
      for axis in perm.iter() {
        w[axis] += 1;
        child.push(w.clone());
      }
      child.iter().all(|s| in_region(s)).then_some(child)
    })
  })
}

/// A cube-coordinate vertex $s$ (descending, in $[0, R]^n$) as the lattice
/// point $k in L_R^(n)$ it labels: $k_0 = R - s_1$, $k_j = s_j - s_(j+1)$,
/// $k_n = s_n$.
fn cube_to_lattice(s: &[usize], refinement: usize) -> Vec<usize> {
  let n = s.len();
  let mut k = Vec::with_capacity(n + 1);
  k.push(refinement - s[0]);
  for j in 1..n {
    k.push(s[j - 1] - s[j]);
  }
  k.push(s[n - 1]);
  k
}

impl ReferenceRefinement {
  pub fn dim(&self) -> Dim {
    self.dim
  }
  pub fn refinement(&self) -> usize {
    self.refinement
  }
  /// The lattice points that are the vertices of the subdivision, colex-ordered.
  pub fn vertices(&self) -> &[Vec<usize>] {
    &self.vertices
  }
  /// The children, each as the ranks of its corners into [`Self::vertices`].
  pub fn children(&self) -> &[Vec<usize>] {
    &self.children
  }
  pub fn nchildren(&self) -> usize {
    self.children.len()
  }

  /// The parent-local (cartesian) coordinates of the `ivertex`-th lattice
  /// vertex: $x_i = lambda_(i+1)$, its position in the parent chart's frame.
  pub fn vertex_local(&self, ivertex: usize) -> Vector {
    bary2local(&self.vertex_bary(ivertex)).into_vector()
  }

  /// The barycentric coordinates $lambda = k \/ R$ of the `ivertex`-th lattice
  /// vertex.
  pub fn vertex_bary(&self, ivertex: usize) -> Bary {
    let scale = (self.refinement as f64).recip();
    Bary::new(Vector::from_iterator(
      (self.dim + 1).index(),
      self.vertices[ivertex].iter().map(|&k| k as f64 * scale),
    ))
  }

  /// The `ichild`-th child realized in the parent chart's local frame, its
  /// corners in reference order. Pure reference data -- a function of
  /// `(dim, refinement)` alone. Its [`linear_transform`] is the reference-order
  /// child Jacobian and its [`vol`] the child's share of the reference volume;
  /// the per-cell refinement path ([`Complex::refine`]) rebuilds this same
  /// realization in the *sorted* global vertex order, the only ordering a stored
  /// cell's metric reads (see there).
  ///
  /// [`linear_transform`]: SimplexCoords::linear_transform
  /// [`vol`]: SimplexCoords::vol
  /// [`Complex::refine`]: crate::topology::complex::Complex::refine
  pub fn child_local_simplex(&self, ichild: usize) -> SimplexCoords<LocalCartesian> {
    let cols: Vec<Vector> = self.children[ichild]
      .iter()
      .map(|&c| self.vertex_local(c))
      .collect();
    SimplexCoords::new(Matrix::from_columns(&cols))
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::Dim;
  use crate::atlas::{ref_lattice, refsimp_vol};

  /// Freudenthal subdivision *composes*: refining an ordered simplex $R$-fold
  /// and then $R'$-fold again is the $R R'$-fold refinement, cell for cell.
  ///
  /// $ "refine"_(R') compose "refine"_R = "refine"_(R R') $
  ///
  /// The semigroup law of the reference pattern, and the reason a refinement
  /// tower stays inside the Kuhn family: every child is similar to its parent,
  /// in every dimension, at every level. It holds only when each child is
  /// refined in the order this pattern emits its corners -- that order carries
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
      let pattern = ref_refinement(dim.into(), refinement);
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
      // What *is* special to Kuhn is similarity of the children -- an affine map
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
        let sub = ref_refinement(dim, r);
        assert_eq!(sub.nchildren(), r.pow(dim.index() as u32));

        let lattice: Vec<Vec<usize>> = ref_lattice(dim, r).collect();
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
  /// parent frame -- vertex order is immaterial here, the volume being
  /// order-invariant.
  #[test]
  fn volume_partition() {
    for dim in (0..=4usize).map(Dim::from) {
      for r in 1..=3 {
        let sub = ref_refinement(dim, r);
        let total: f64 = (0..sub.nchildren())
          .map(|c| sub.child_local_simplex(c).vol())
          .sum();
        approx::assert_relative_eq!(total, refsimp_vol(dim), epsilon = 1e-12);
        for c in 0..sub.nchildren() {
          approx::assert_relative_eq!(
            sub.child_local_simplex(c).vol(),
            refsimp_vol(dim) / (r.pow(dim.index() as u32) as f64),
            epsilon = 1e-12
          );
        }
      }
    }
  }
}
