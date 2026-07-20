//! Flat-torus generation in arbitrary dimension: the periodic quotient of the
//! Cartesian mesh.
//!
//! The flat torus $T^d = RR^d \/ (L_0 ZZ times dots.c times L_(d-1) ZZ)$ is the
//! Cartesian grid of [`super::cartesian`] with opposite faces identified: the
//! high face of each axis is glued to its low face. The gluing is *purely
//! topological* -- a relabelling of vertices, $c_i |-> c_i mod n$ per axis --
//! so the piecewise-flat geometry is untouched and no coordinates are involved.
//! The seam edges have the same lengths as the interior ones, and the result is
//! delivered as [`MeshLengthsSq`], the intrinsic Regge primitive (invariant 2),
//! not an embedding: a torus has no flat realization in $RR^d$, and it needs
//! none.
//!
//! It is the closed, flat, dimension-agnostic test manifold: $M_h = M$ exactly
//! (so refinement introduces no geometric error), boundaryless (no essential
//! boundary conditions to entangle with a convergence rate), and with the
//! nontrivial cohomology of the $d$-torus -- Betti numbers
//! $b_k = binom(d, k)$ -- so it exercises the *full* mixed Hodge--Laplace
//! problem, harmonic sector included, at every intermediate grade.

use itertools::Itertools;
use multiindex::cartesian::{cartesian2linear, linear2cartesian};

use crate::{
  gen::cartesian::CartesianGrid,
  geometry::metric::mesh::MeshLengthsSq,
  linalg::Vector,
  topology::{
    complex::Complex, ordering::CellOrdering, simplex::Simplex, skeleton::Skeleton, VertexIdx,
  },
  Dim,
};

/// The flat torus $T^d$ as a uniform Kuhn-triangulated periodic grid:
/// `ncells_axis` cells per axis, each period identified with the next.
///
/// The Kuhn triangulation tiles every box identically (a fixed corner, one
/// simplex per axis permutation), so the pattern matches across the periodic
/// seam and the quotient is conforming.
pub struct FlatTorus {
  /// The period $L_i$ of each axis: the side length of one fundamental domain.
  side_lengths: Vector,
  ncells_axis: usize,
}

impl FlatTorus {
  /// A torus of the given per-axis periods. `ncells_axis` must be at least `3`:
  /// two cells per periodic axis would glue a cell onto itself (the circle
  /// needs three edges to be a simplicial complex), which
  /// [`crate::topology::skeleton::Skeleton`] silently deduplicates
  /// into a degenerate mesh.
  pub fn new(side_lengths: Vector, ncells_axis: usize) -> Self {
    assert!(
      ncells_axis >= 3,
      "A flat torus needs at least 3 cells per periodic axis."
    );
    Self {
      side_lengths,
      ncells_axis,
    }
  }
  /// The unit torus $[0, 1)^d$ with equal periods.
  pub fn new_unit(dim: Dim, ncells_axis: usize) -> Self {
    Self::new(Vector::from_element(dim, 1.0), ncells_axis)
  }

  pub fn dim(&self) -> Dim {
    self.side_lengths.len()
  }
  pub fn ncells_axis(&self) -> usize {
    self.ncells_axis
  }
  /// The number of vertices after identification: $n^d$, one per cell of the
  /// quotient grid.
  pub fn nvertices(&self) -> usize {
    self.ncells_axis.pow(self.dim() as u32)
  }

  /// The topology and the Regge geometry of the torus: the periodic complex and
  /// its signed squared edge lengths.
  ///
  /// No coordinates: the flat torus admits no isometric embedding in $RR^d$, so
  /// the intrinsic geometry is the only faithful one. This is invariant 2 with
  /// nothing to fall back on.
  pub fn triangulate(&self) -> (Complex, MeshLengthsSq) {
    let (complex, lengths, _) = self.triangulate_ordered();
    (complex, lengths)
  }

  /// As [`FlatTorus::triangulate`], also returning the Kuhn chain order each
  /// cell was built in, which identification and the colex sort would otherwise
  /// discard.
  ///
  /// `None` if that order is not face-consistent. The tiling matches across the
  /// periodic seam, so it is expected to be; the check is what says so rather
  /// than an assumption.
  pub fn triangulate_ordered(&self) -> (Complex, MeshLengthsSq, Option<CellOrdering>) {
    let words = self.cell_words();
    let complex = Complex::from_cells(Skeleton::new(
      words
        .iter()
        .map(|word| Simplex::from_word(word.clone()).1)
        .collect(),
    ));
    let lengths = self.edge_lengths_sq(&complex);
    let ordering = (words.len() == complex.cells().len()).then(|| {
      let ordering = CellOrdering::new(&complex, words);
      ordering.is_face_consistent(&complex).then_some(ordering)
    });
    (complex, lengths, ordering.flatten())
  }

  /// The quotient vertex of a grid vertex: reduce each Cartesian coordinate
  /// modulo `ncells_axis`, identifying the high face of every axis with the
  /// low one.
  fn reduce_vertex(&self, grid_vertex: usize) -> usize {
    let n = self.ncells_axis;
    let cart = linear2cartesian(grid_vertex, n + 1, self.dim());
    let reduced = cart.iter().map(|&c| c % n).collect_vec();
    cartesian2linear(&reduced, n)
  }

  /// Each cell as the identified image of its Kuhn chain, in chain order.
  ///
  /// Reduction permutes the vertices out of ascending order, which is exactly
  /// the ordering datum a colex sort would destroy.
  fn cell_words(&self) -> Vec<Vec<VertexIdx>> {
    let grid = CartesianGrid::new_unit(self.dim(), self.ncells_axis);
    grid
      .cell_skeleton()
      .into_iter()
      .map(|simplex| {
        simplex
          .vertices
          .iter()
          .map(|&v| self.reduce_vertex(v))
          .collect()
      })
      .collect()
  }

  /// The signed squared length of every edge, read off the flat geometry: an
  /// edge steps by $plus.minus 1$ per axis (a unit box's edges and Kuhn
  /// diagonals), and across the seam the periodic representative recovers the
  /// same $plus.minus 1$. Squared length is $sum_i (d_i L_i \/ n)^2$.
  fn edge_lengths_sq(&self, complex: &Complex) -> MeshLengthsSq {
    let n = self.ncells_axis;
    let dim = self.dim();
    let spacing = &self.side_lengths / n as f64;

    let edges = complex.edges();
    let mut lengths_sq = Vector::zeros(edges.len());
    for (iedge, edge) in edges.handle_iter().enumerate() {
      let (vi, vj) = edge.endpoints();
      let ci = linear2cartesian(vi.kidx(), n, dim);
      let cj = linear2cartesian(vj.kidx(), n, dim);
      lengths_sq[iedge] = (0..dim)
        .map(|a| {
          let mut d = cj[a] as isize - ci[a] as isize;
          // Minimal periodic representative: the wrap-around step n-1 is the
          // step -1 of the neighbouring period.
          if d > (n as isize) / 2 {
            d -= n as isize;
          } else if d < -(n as isize) / 2 {
            d += n as isize;
          }
          let length = d as f64 * spacing[a];
          length * length
        })
        .sum();
    }
    MeshLengthsSq::new(lengths_sq, complex)
  }
}

#[cfg(test)]
mod test {
  use super::FlatTorus;
  use multiindex::binomial;

  /// The generator's Kuhn chain order survives identification as a
  /// face-consistent ordering, and refining in it keeps the torus self-similar:
  /// one shape class, on a closed manifold with no boundary to anchor anything.
  ///
  /// The tiling matches across the periodic seam, which is what makes the
  /// quotient conforming; this is that statement made checkable.
  #[test]
  fn the_chain_order_survives_identification() {
    use crate::geometry::metric::mesh::MeshLengthsSq;
    use crate::topology::complex::Complex;

    fn shape_classes(complex: &Complex, lengths: &MeshLengthsSq) -> usize {
      let mut classes: Vec<Vec<u64>> = complex
        .cells()
        .handle_iter()
        .map(|cell| {
          let mut ls: Vec<f64> = lengths
            .simplex_lengths_sq(*cell)
            .vector()
            .iter()
            .copied()
            .collect();
          ls.sort_by(|a, b| a.partial_cmp(b).unwrap());
          let max = *ls.last().unwrap();
          ls.iter().map(|l| (l / max * 1e6).round() as u64).collect()
        })
        .collect();
      classes.sort_unstable();
      classes.dedup();
      classes.len()
    }

    for dim in 1..=3 {
      let (complex, lengths, ordering) = FlatTorus::new_unit(dim, 3).triangulate_ordered();
      let ordering = ordering.expect("the tiling matches across the seam");
      assert_eq!(shape_classes(&complex, &lengths), 1);

      let sub = complex.refine_with(&ordering, 2);
      let fine_lengths = lengths.refine(&sub, &complex);
      assert_eq!(shape_classes(sub.complex(), &fine_lengths), 1);
    }
  }

  /// The flat torus is closed (no boundary) and carries the cohomology of
  /// $T^d$: Betti numbers $b_k = binom(d, k)$, Euler characteristic $0$, for
  /// every dimension.
  #[test]
  fn torus_topology() {
    for dim in 1..=3 {
      let (complex, lengths) = FlatTorus::new_unit(dim, 3).triangulate();

      assert!(!complex.has_boundary(), "dim {dim}: torus is boundaryless");
      assert_eq!(complex.nsimplices(0), 3usize.pow(dim as u32));

      let betti = complex.betti_numbers();
      let expected = (0..=dim).map(|k| binomial(dim, k)).collect::<Vec<_>>();
      assert_eq!(betti, expected, "dim {dim}: Betti numbers of T^d");
      assert_eq!(complex.euler_characteristic(), 0);

      // The geometry is flat and uniform: every edge is spacelike, and the
      // shortest edges are the axis steps of length 1/n.
      assert!(lengths.iter().all(|&s| s > 0.0));
      assert!((lengths.mesh_width_min() - 1.0 / 3.0).abs() < 1e-12);
    }
  }

  /// Uniform refinement of the torus is topological: the refined mesh stays
  /// closed and carries the same cohomology as the coarse one.
  #[test]
  fn torus_refined_topology() {
    let (coarse, _) = FlatTorus::new_unit(2, 3).triangulate();
    for refinement in 1..=2 {
      let fine = coarse.refine(refinement).into_complex();
      assert!(!fine.has_boundary());
      assert_eq!(fine.betti_numbers(), vec![1, 2, 1]);
      assert_eq!(fine.euler_characteristic(), 0);
    }
  }
}
