//! Flat-quotient generation in arbitrary dimension: the Cartesian mesh with
//! its opposite faces identified, axis by axis.
//!
//! A flat quotient is $RR^d \/ Gamma$ for a group $Gamma$ acting by
//! isometries of the Kuhn-triangulated grid. Each axis carries one
//! [`Identification`], and the whole family, flat tori, Möbius bands, Klein
//! bottles, orientable twisted tori, is that one construction with different
//! per-axis choices. There is no separate torus generator and no separate
//! Möbius generator. There is one quotient with a flag per axis.
//!
//! The gluing is purely topological: a relabeling of vertices, so the
//! piecewise-flat geometry is untouched and no coordinates are involved. The
//! seam edges have the same lengths as the interior ones, and the result is
//! delivered as [`MeshLengthsSq`], the intrinsic Regge primitive (invariant 2).
//! Most of these manifolds have no isometric realization in $RR^d$ and need
//! none. The optional embeddings of [`super::quotient_embed`] are for
//! visualization and are a different, curved manifold wherever they are not
//! isometric.
//!
//! These are the closed, flat, dimension-agnostic test manifolds: $M_h = M$
//! exactly (so refinement introduces no geometric error), and with cohomology
//! rich enough to exercise the full mixed Hodge--Laplace problem, harmonic
//! sector included, at every intermediate grade. The twisted members add the
//! non-orientable case, which is how invariant 6, that no assembly, solve or
//! homology computation may depend on a coherent orientation, becomes
//! checkable rather than merely asserted.

use itertools::Itertools;
use multiindex::Radix;

use crate::lengths::mesh::MeshLengthsSq;
use crate::mesher::cartesian::CartesianGrid;
use crate::mesher::quasi_uniform_counts;
use simplicial::{
  Dim,
  linalg::Vector,
  topology::{
    VertexIdx, complex::Complex, ordering::CellOrdering, simplex::Simplex, skeleton::Skeleton,
  },
};

/// How the two opposite faces of one axis are glued.
///
/// The gluing is always by an isometry of the transverse lattice, which is what
/// keeps every quotient in the family flat.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Identification {
  /// Not glued: the axis keeps its two boundary faces.
  Open,
  /// Glued by pure translation, $x_i |-> x_i + L_i$. The circle factor.
  Periodic,
  /// Glued by translation composed with a reflection of the listed transverse
  /// axes: $x_i |-> x_i + L_i$ together with $x_j |-> -x_j$ for each listed
  /// $j$.
  ///
  /// Walking once around the axis returns to the starting point with those
  /// coordinates reversed. The parity of the list is the orientability: an
  /// odd number of reflections has determinant $-1$, so the quotient is
  /// non-orientable (Möbius band, Klein bottle). An even number is a rotation
  /// and the quotient stays orientable (a twisted torus).
  Twisted(Vec<usize>),
}

impl Identification {
  /// Whether the axis is glued at all, hence whether its two faces are
  /// identified rather than left as boundary.
  pub fn is_closed(&self) -> bool {
    !matches!(self, Self::Open)
  }
  /// The fewest cells an axis with this identification may carry.
  ///
  /// A closed axis needs three: with two, a cell is glued onto itself, the
  /// skeleton dedups the pair and the mesh degenerates. It is the simplicial
  /// circle needing three edges, one dimension up.
  pub fn min_cells(&self) -> usize {
    if self.is_closed() { 3 } else { 1 }
  }
  /// The transverse axes the gluing reflects, empty unless the axis is
  /// twisted. This, and not which variant the identification is, is what
  /// decides orientability and whether an embedding of it can be isometric.
  pub fn reflected_axes(&self) -> &[usize] {
    match self {
      Self::Twisted(axes) => axes,
      _ => &[],
    }
  }
}

/// A flat quotient of the uniform Kuhn-triangulated grid: `ncells_axis` cells
/// per axis, each axis identified as its [`Identification`] says.
///
/// The Kuhn triangulation tiles every box identically (a fixed corner, one
/// simplex per axis permutation), and the tiling it induces on a box face
/// is what a seam has to match, so every quotient in the family is conforming.
///
/// The tiling of the box interior is a weaker matter, and a reflection does not
/// preserve it: mirroring an axis exchanges the diagonal. That costs the Kuhn
/// chain ordering on a twisted seam, not the conformity, see
/// [`FlatQuotient::triangulate_ordered`].
pub struct FlatQuotient {
  /// The period $L_i$ of each axis: the side length of one fundamental domain.
  side_lengths: Vector,
  identifications: Vec<Identification>,
  /// Cells per axis, independently: the periods of a quotient are rarely equal
  /// (a Möbius band is long and narrow), and one count over unequal periods
  /// makes the cells as anisotropic as the fundamental domain.
  ncells: Vec<usize>,
}

impl FlatQuotient {
  /// A quotient with the given per-axis periods and identifications.
  ///
  /// `ncells_axis` must be at least `3` on a closed axis: two cells would glue
  /// a cell onto itself (the circle needs three edges to be a simplicial
  /// complex), which [`Skeleton`] silently deduplicates into a degenerate mesh.
  pub fn new(
    side_lengths: Vector,
    identifications: Vec<Identification>,
    ncells_axis: usize,
  ) -> Self {
    let dim = side_lengths.len();
    Self::new_anisotropic(side_lengths, identifications, vec![ncells_axis; dim])
  }

  /// A quotient with an independent cell count per axis.
  ///
  /// Every closed axis needs at least `3` cells. An open one needs `1`.
  pub fn new_anisotropic(
    side_lengths: Vector,
    identifications: Vec<Identification>,
    ncells: Vec<usize>,
  ) -> Self {
    let dim = side_lengths.len();
    assert_eq!(ncells.len(), dim, "One cell count per axis is required.");
    assert_eq!(
      identifications.len(),
      dim,
      "One identification per axis is required."
    );
    for (axis, id) in identifications.iter().enumerate() {
      assert!(
        !id.reflected_axes().contains(&axis),
        "Axis {axis} cannot reflect itself: the gluing would not be an involution."
      );
      assert!(
        id.reflected_axes().iter().all(|&j| j < dim),
        "A reflected axis must be an axis of the grid."
      );
    }
    for (axis, id) in identifications.iter().enumerate() {
      let floor = id.min_cells();
      assert!(
        ncells[axis] >= floor,
        "Axis {axis} needs at least {floor} cells; a closed axis with two would \
         glue a cell onto itself."
      );
    }
    Self {
      side_lengths,
      identifications,
      ncells,
    }
  }

  /// A quotient whose cells are as near equilateral as the counts allow
  /// ([`quasi_uniform_counts`]), raising each axis to the cells its
  /// identification needs.
  ///
  /// This is the constructor to reach for whenever the periods differ, which is
  /// most of the family: a Möbius band is a long strip, and giving its
  /// circumference and its width the same count meshes it into slivers whose
  /// aspect ratio is the ratio of the two periods.
  pub fn quasi_uniform(
    side_lengths: Vector,
    identifications: Vec<Identification>,
    ncells_longest: usize,
  ) -> Self {
    let ncells = quasi_uniform_counts(&side_lengths, ncells_longest)
      .into_iter()
      .zip(&identifications)
      .map(|(ncells, id)| ncells.max(id.min_cells()))
      .collect();
    Self::new_anisotropic(side_lengths, identifications, ncells)
  }

  /// The flat torus $T^d = RR^d \/ (L_0 ZZ times dots.c times L_(d-1) ZZ)$:
  /// every axis periodic.
  ///
  /// Closed, boundaryless, orientable, with the cohomology of the $d$-torus,
  /// Betti numbers $b_k = binom(d, k)$.
  pub fn torus(side_lengths: Vector, ncells_axis: usize) -> Self {
    let dim = side_lengths.len();
    Self::new(
      side_lengths,
      vec![Identification::Periodic; dim],
      ncells_axis,
    )
  }

  /// The unit torus $[0, 1)^d$ with equal periods.
  pub fn unit_torus(dim: impl Into<Dim>, ncells_axis: usize) -> Self {
    Self::torus(Vector::from_element(dim.into().index(), 1.0), ncells_axis)
  }

  /// The Möbius band: axis 0 twisted, reflecting the open fiber axis 1.
  ///
  /// The smallest non-orientable surface. It has a boundary, the single
  /// circle traversing the open axis twice.
  /// `ncells_longest` is the resolution of the longer period: the two are
  /// discretized quasi-uniformly, so a long narrow band gets cells that are
  /// near equilateral rather than slivers of its aspect ratio.
  pub fn moebius(circumference: f64, width: f64, ncells_longest: usize) -> Self {
    Self::quasi_uniform(
      Vector::from_column_slice(&[circumference, width]),
      vec![Identification::Twisted(vec![1]), Identification::Open],
      ncells_longest,
    )
  }

  /// The Klein bottle: axis 0 twisted, reflecting the periodic axis 1.
  ///
  /// Closed and non-orientable. Over $RR$ its Betti numbers are
  /// $b_0 = b_1 = 1$, $b_2 = 0$: the $ZZ_2$ torsion of $H_1$ is invisible to
  /// real coefficients, and a closed non-orientable surface carries no
  /// fundamental class, which is exactly why $b_2$ vanishes.
  pub fn klein(side_lengths: Vector, ncells_axis: usize) -> Self {
    assert_eq!(side_lengths.len(), 2, "The Klein bottle is a surface.");
    Self::new(
      side_lengths,
      vec![Identification::Twisted(vec![1]), Identification::Periodic],
      ncells_axis,
    )
  }

  pub fn dim(&self) -> Dim {
    self.side_lengths.len().into()
  }
  /// The cell count of each axis.
  pub fn ncells_per_axis(&self) -> &[usize] {
    &self.ncells
  }
  pub fn side_lengths(&self) -> &Vector {
    &self.side_lengths
  }
  pub fn identifications(&self) -> &[Identification] {
    &self.identifications
  }

  /// Whether every reflection is applied an even number of times around every
  /// seam, i.e. whether the deck group lies in $"SO"(d)$.
  ///
  /// A sufficient condition for orientability, not a necessary one, and it is
  /// the cheap combinatorial reading of the identification rather than a
  /// statement about the assembled complex. The authority on the mesh itself is
  /// [`Complex::orientation`], which returns `None` exactly when no coherent
  /// orientation exists.
  pub fn is_orientation_preserving(&self) -> bool {
    self
      .identifications
      .iter()
      .all(|id| id.reflected_axes().len() % 2 == 0)
  }

  /// The number of distinct values each axis coordinate takes after
  /// identification: `ncells_axis` on a closed axis, one more on an open one,
  /// whose far face survives.
  fn shape(&self) -> Radix {
    self
      .identifications
      .iter()
      .enumerate()
      .map(|(axis, id)| {
        if id.is_closed() {
          self.ncells[axis]
        } else {
          self.ncells[axis] + 1
        }
      })
      .collect()
  }

  /// The shape of the covering vertex grid, before identification: one more
  /// vertex than cells along every axis, closed or not.
  fn grid_shape(&self) -> Radix {
    self.ncells.iter().map(|&n| n + 1).collect()
  }

  /// The number of vertices after identification.
  pub fn nvertices(&self) -> usize {
    self.shape().count()
  }

  /// The topology and the Regge geometry of the quotient: the identified
  /// complex and its signed squared edge lengths.
  ///
  /// No coordinates: most of these manifolds admit no isometric embedding in
  /// $RR^d$, so the intrinsic geometry is the only faithful one. This is
  /// invariant 2 with nothing to fall back on.
  pub fn triangulate(&self) -> (Complex, MeshLengthsSq) {
    let (complex, lengths, _) = self.triangulate_ordered();
    (complex, lengths)
  }

  /// As [`FlatQuotient::triangulate`], also returning the Kuhn chain order each
  /// cell was built in, which identification and the colex sort would otherwise
  /// discard.
  ///
  /// `None` if that order is not face-consistent, which is the honest answer
  /// for every reflecting identification: the Kuhn triangulation of a box is
  /// not reflection-invariant, so the two sides of a twisted seam emit
  /// incompatible chain orders on the face they share. Translational
  /// identifications keep it. Refinement of a twisted quotient therefore goes
  /// through the colex ordering, losing only the guarantee that a refinement
  /// tower stays self-similar; recovering that needs a reflection-invariant
  /// triangulation of the box, not a repair of this one.
  pub fn triangulate_ordered(&self) -> (Complex, MeshLengthsSq, Option<CellOrdering>) {
    let words = self.cell_words();
    let complex = Complex::from_cells(Skeleton::new(
      words
        .iter()
        .map(|word| Simplex::from_word(word.clone()).1)
        .collect(),
    ));
    let lengths = self.edge_lengths_sq(&complex);
    let ordering = CellOrdering::try_new(&complex, words)
      .filter(|ordering| ordering.is_face_consistent(&complex));
    (complex, lengths, ordering)
  }

  /// The quotient vertex of a grid vertex: fold each axis coordinate back into
  /// the fundamental domain, applying the reflection that a twisted axis's
  /// gluing carries.
  ///
  /// The grid spans one period per axis, so each seam is crossed at most once
  /// and the wraps of distinct axes are independent.
  fn reduce_vertex(&self, grid_vertex: usize) -> usize {
    let grid_shape = self.grid_shape();
    let mut cart = grid_shape.delinearize(grid_vertex);

    let wrapped = (0..self.dim().index())
      .filter(|&axis| self.identifications[axis].is_closed() && cart[axis] == self.ncells[axis])
      .collect_vec();
    for &axis in &wrapped {
      for &reflected in self.identifications[axis].reflected_axes() {
        cart[reflected] = self.ncells[reflected] - cart[reflected];
      }
    }
    for (axis, coord) in cart.iter_mut().enumerate() {
      if self.identifications[axis].is_closed() {
        *coord %= self.ncells[axis];
      }
    }
    self.shape().linearize(&cart)
  }

  /// Each cell as the identified image of its Kuhn chain, in chain order.
  ///
  /// Reduction permutes the vertices out of ascending order, which is exactly
  /// the ordering datum a colex sort would destroy.
  fn cell_words(&self) -> Vec<Vec<VertexIdx>> {
    self
      .grid()
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

  /// The signed squared length of every edge, read off the flat geometry of the
  /// unidentified grid.
  ///
  /// Measuring upstairs is what makes this total over every identification: the
  /// displacement of an edge is unambiguous before the quotient, whereas the
  /// coordinate difference of two identified representatives is not, a
  /// reflecting seam sends a step to its mirror image, and no minimal-
  /// representative rule downstairs recovers it. That every cell containing an
  /// edge agrees on its length is precisely the statement that the gluing was
  /// by an isometry, and it is asserted rather than assumed.
  fn edge_lengths_sq(&self, complex: &Complex) -> MeshLengthsSq {
    let dim = self.dim();
    let spacing = Vector::from_iterator(
      dim.index(),
      self
        .side_lengths
        .iter()
        .zip(&self.ncells)
        .map(|(&side, &n)| side / n as f64),
    );
    let grid_shape = self.grid_shape();

    let edges = complex.skeleton_raw(Dim::ONE);
    let mut lengths_sq = Vector::from_element(edges.len(), f64::NAN);

    for cell in self.grid().cell_skeleton() {
      for [&vi, &vj] in cell.vertices.iter().array_combinations() {
        let ci = grid_shape.delinearize(vi);
        let cj = grid_shape.delinearize(vj);
        let length_sq = (0..dim.index())
          .map(|a| {
            let step = (cj[a] as isize - ci[a] as isize) as f64 * spacing[a];
            step * step
          })
          .sum::<f64>();

        let edge = Simplex::from_word(vec![self.reduce_vertex(vi), self.reduce_vertex(vj)]).1;
        let iedge = edges.kidx_by_simplex(&edge);
        let known = lengths_sq[iedge];
        assert!(
          known.is_nan() || (known - length_sq).abs() <= 1e-12 * length_sq,
          "The identification is not by an isometry: an edge inherits two lengths."
        );
        lengths_sq[iedge] = length_sq;
      }
    }
    assert!(
      lengths_sq.iter().all(|l| !l.is_nan()),
      "Every quotient edge is the image of a grid edge."
    );
    // The grid is non-degenerate and the identification is by an isometry,
    // asserted above, so every quotient cell inherits a grid cell's metric.
    MeshLengthsSq::new(lengths_sq)
  }

  /// The unidentified grid the quotient is built from: the fundamental domain
  /// as a box, at this quotient's per-axis resolution.
  fn grid(&self) -> CartesianGrid {
    CartesianGrid::new_anisotropic(
      Vector::zeros(self.dim().index()),
      self.side_lengths.clone(),
      self.ncells.clone(),
    )
  }

  /// The cartesian multi-index of a quotient vertex, its position in the
  /// fundamental domain, in units of the grid spacing.
  pub fn vertex_cart_idx(&self, vertex: VertexIdx) -> Vec<usize> {
    self.shape().delinearize(vertex).to_vec()
  }
}
