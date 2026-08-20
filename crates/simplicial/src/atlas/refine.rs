//! The reference subdivision pattern of a chart.
//!
//! Refining a mesh is refining every chart the same way, up to the labeling of
//! its vertices, the atlas philosophy applied to $h$-refinement. So the whole
//! combinatorics lives once on the reference cell, as a function of the
//! dimension and the refinement level alone, and a mesh is refined by relabeling
//! this one pattern onto every cell (see
//! [`Complex::refine`](crate::topology::complex::Complex::refine)).
//!
//! The pattern is the edgewise (Freudenthal) subdivision of the reference
//! $n$-simplex into $R^n$ children, the canonical dimension-general
//! generalization of the 2D red refinement (Bank) and its 3D analogue (Bey).
//! It is affine and metric-free: a statement about the barycentric lattice, not
//! about any geometry the cell may carry.
//!
//! # Construction
//!
//! The unit simplex maps affinely and bijectively onto the order simplex
//! $Delta = { t in RR^n : 1 >= t_1 >= dots.h.c >= t_n >= 0 }$ by the cumulative
//! barycentric coordinates $t_j = sum_(i >= j) lambda_i$, itself a Kuhn
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
//! order, not on the opposite vertex. Two cells sharing a face carry the same
//! global order on it (simplices store vertices increasingly), so they subdivide
//! the shared face identically, and the refined mesh is conforming. This is why
//! the reference pattern is expressed on lattice points keyed by the vertices
//! they are supported on: the key is what a shared point agrees on across
//! charts.

use super::{Bary, LocalCartesian, SimplexCoords, bary2local, lattice_bary};
use crate::Dim;
use crate::linalg::{Matrix, Vector};

use multiindex::{Composition, Permutation, Radix};

use std::collections::HashMap;

/// The reference edgewise subdivision of the $n$-simplex into $R^n$ children,
/// as combinatorics on the barycentric lattice $L_R^n$.
///
/// A function of `(dim, refinement)` alone. Its vertices are exactly the
/// lattice points [`unit_lattice`](super::unit_lattice) produces, and a child is
/// the list of colex ranks of its $n+1$ corners among them.
#[derive(Debug, Clone)]
pub struct UnitRefinement {
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
pub fn unit_refinement(dim: impl Into<Dim>, refinement: usize) -> UnitRefinement {
  let dim = dim.into();
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

  UnitRefinement {
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
  Radix::uniform(r + 1, n).all().flat_map(move |base| {
    Permutation::all(n).filter_map(move |perm| {
      let mut w = base.clone();
      let mut child = Vec::with_capacity((dim + 1).index());
      child.push(w.to_vec());
      for axis in perm.iter() {
        w[axis] += 1;
        child.push(w.to_vec());
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

impl UnitRefinement {
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
    lattice_bary(&self.vertices[ivertex], self.refinement)
  }

  /// The `ichild`-th child realized in the parent chart's local frame, its
  /// corners in reference order. Pure reference data, a function of
  /// `(dim, refinement)` alone. Its [`linear_transform`] is the reference-order
  /// child Jacobian and its [`vol`] the child's share of the reference volume;
  /// the per-cell refinement path ([`Complex::refine`]) rebuilds this same
  /// realization in the sorted global vertex order, the only ordering a stored
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
