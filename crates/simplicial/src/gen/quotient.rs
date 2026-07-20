//! Flat-quotient generation in arbitrary dimension: the Cartesian mesh with
//! its opposite faces identified, axis by axis.
//!
//! A *flat quotient* is $RR^d \/ Gamma$ for a group $Gamma$ acting by
//! isometries of the Kuhn-triangulated grid. Each axis carries one
//! [`Identification`], and the whole family -- flat tori, Möbius bands, Klein
//! bottles, orientable twisted tori -- is that one construction with different
//! per-axis choices. There is no separate torus generator and no separate
//! Möbius generator; there is one quotient with a flag per axis.
//!
//! The gluing is *purely topological*: a relabelling of vertices, so the
//! piecewise-flat geometry is untouched and no coordinates are involved. The
//! seam edges have the same lengths as the interior ones, and the result is
//! delivered as [`MeshLengthsSq`], the intrinsic Regge primitive (invariant 2).
//! Most of these manifolds have no isometric realization in $RR^d$ and need
//! none; the optional embeddings of [`super::quotient_embed`] are for
//! visualization and are a *different*, curved manifold wherever they are not
//! isometric.
//!
//! These are the closed, flat, dimension-agnostic test manifolds: $M_h = M$
//! exactly (so refinement introduces no geometric error), and with cohomology
//! rich enough to exercise the *full* mixed Hodge--Laplace problem, harmonic
//! sector included, at every intermediate grade. The twisted members add the
//! non-orientable case, which is how invariant 6 -- that no assembly, solve or
//! homology computation may depend on a coherent orientation -- becomes
//! checkable rather than merely asserted.

use itertools::Itertools;
use multiindex::cartesian::{cartesian2linear_mixed, linear2cartesian_mixed};

use crate::{
  gen::cartesian::CartesianGrid,
  geometry::metric::mesh::MeshLengthsSq,
  linalg::Vector,
  topology::{
    complex::Complex, ordering::CellOrdering, simplex::Simplex, skeleton::Skeleton, VertexIdx,
  },
  Dim,
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
  /// Glued by translation composed with a *reflection* of the listed transverse
  /// axes: $x_i |-> x_i + L_i$ together with $x_j |-> -x_j$ for each listed
  /// $j$.
  ///
  /// Walking once around the axis returns to the starting point with those
  /// coordinates reversed. The **parity of the list is the orientability**: an
  /// odd number of reflections has determinant $-1$, so the quotient is
  /// non-orientable (Möbius band, Klein bottle); an even number is a rotation
  /// and the quotient stays orientable (a twisted torus).
  Twisted(Vec<Dim>),
}

impl Identification {
  /// Whether the axis is glued at all, hence whether its two faces are
  /// identified rather than left as boundary.
  pub fn is_closed(&self) -> bool {
    !matches!(self, Self::Open)
  }
  fn reflected_axes(&self) -> &[Dim] {
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
/// simplex per axis permutation), and the tiling it induces on a box *face*
/// is what a seam has to match, so every quotient in the family is conforming.
///
/// The tiling of the box interior is a weaker matter, and a reflection does not
/// preserve it: mirroring an axis exchanges the diagonal. That costs the Kuhn
/// chain ordering on a twisted seam, not the conformity -- see
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
  /// Every *closed* axis needs at least `3` cells; an open one needs `1`.
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
      let floor = if id.is_closed() { 3 } else { 1 };
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

  /// A quotient whose cells are as near equilateral as the counts allow: the
  /// count on each axis is scaled by that axis's period, so the spacing is
  /// quasi-uniform. `ncells_longest` fixes the resolution of the longest axis.
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
    let longest = side_lengths.iter().copied().fold(0.0_f64, f64::max);
    let ncells = side_lengths
      .iter()
      .zip(&identifications)
      .map(|(&side, id)| {
        let floor = if id.is_closed() { 3 } else { 1 };
        ((side / longest * ncells_longest as f64).round() as usize).max(floor)
      })
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
  pub fn unit_torus(dim: Dim, ncells_axis: usize) -> Self {
    Self::torus(Vector::from_element(dim, 1.0), ncells_axis)
  }

  /// The Möbius band: axis 0 twisted, reflecting the *open* fiber axis 1.
  ///
  /// The smallest non-orientable surface. It has a boundary -- the single
  /// circle traversing the open axis twice.
  /// `ncells_longest` is the resolution of the *longer* period: the two are
  /// discretized quasi-uniformly, so a long narrow band gets cells that are
  /// near equilateral rather than slivers of its aspect ratio.
  pub fn moebius(circumference: f64, width: f64, ncells_longest: usize) -> Self {
    Self::quasi_uniform(
      Vector::from_column_slice(&[circumference, width]),
      vec![Identification::Twisted(vec![1]), Identification::Open],
      ncells_longest,
    )
  }

  /// The Klein bottle: axis 0 twisted, reflecting the *periodic* axis 1.
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
    self.side_lengths.len()
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
  /// A *sufficient* condition for orientability, not a necessary one, and it is
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
  fn radices(&self) -> Vec<usize> {
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

  /// The number of vertices after identification.
  pub fn nvertices(&self) -> usize {
    self.radices().iter().product()
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
  /// for every *reflecting* identification: the Kuhn triangulation of a box is
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
    let ordering = (words.len() == complex.cells().len()).then(|| {
      let ordering = CellOrdering::new(&complex, words);
      ordering.is_face_consistent(&complex).then_some(ordering)
    });
    (complex, lengths, ordering.flatten())
  }

  /// The quotient vertex of a grid vertex: fold each axis coordinate back into
  /// the fundamental domain, applying the reflection that a twisted axis's
  /// gluing carries.
  ///
  /// The grid spans one period per axis, so each seam is crossed at most once
  /// and the wraps of distinct axes are independent.
  fn reduce_vertex(&self, grid_vertex: usize) -> usize {
    let grid_radices = self.ncells.iter().map(|&n| n + 1).collect_vec();
    let mut cart = linear2cartesian_mixed(grid_vertex, &grid_radices);

    let wrapped = (0..self.dim())
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
    cartesian2linear_mixed(&cart, &self.radices())
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
  /// *unidentified* grid.
  ///
  /// Measuring upstairs is what makes this total over every identification: the
  /// displacement of an edge is unambiguous before the quotient, whereas the
  /// coordinate difference of two identified representatives is not -- a
  /// reflecting seam sends a step to its mirror image, and no minimal-
  /// representative rule downstairs recovers it. That every cell containing an
  /// edge agrees on its length is precisely the statement that the gluing was
  /// by an isometry, and it is asserted rather than assumed.
  fn edge_lengths_sq(&self, complex: &Complex) -> MeshLengthsSq {
    let dim = self.dim();
    let spacing = Vector::from_iterator(
      dim,
      self
        .side_lengths
        .iter()
        .zip(&self.ncells)
        .map(|(&side, &n)| side / n as f64),
    );
    let grid_radices = self.ncells.iter().map(|&n| n + 1).collect_vec();

    let edges = complex.skeleton_raw(1);
    let mut lengths_sq = Vector::from_element(edges.len(), f64::NAN);

    for cell in self.grid().cell_skeleton() {
      for [&vi, &vj] in cell.vertices.iter().array_combinations() {
        let ci = linear2cartesian_mixed(vi, &grid_radices);
        let cj = linear2cartesian_mixed(vj, &grid_radices);
        let length_sq = (0..dim)
          .map(|a| {
            let step = (cj[a] as isize - ci[a] as isize) as f64 * spacing[a];
            step * step
          })
          .sum::<f64>();

        let edge = Simplex::from_word(vec![self.reduce_vertex(vi), self.reduce_vertex(vj)]).1;
        let iedge = edges.kidx_by_simplex(&edge);
        let known = lengths_sq[iedge];
        assert!(
          known.is_nan() || (known - length_sq).abs() <= 1e-12 * length_sq.max(1.0),
          "The identification is not by an isometry: an edge inherits two lengths."
        );
        lengths_sq[iedge] = length_sq;
      }
    }
    assert!(
      lengths_sq.iter().all(|l| !l.is_nan()),
      "Every quotient edge is the image of a grid edge."
    );
    MeshLengthsSq::new(lengths_sq, complex)
  }

  /// The unidentified grid the quotient is built from: the fundamental domain
  /// as a box, at this quotient's per-axis resolution.
  fn grid(&self) -> CartesianGrid {
    CartesianGrid::new_anisotropic(
      Vector::zeros(self.dim()),
      self.side_lengths.clone(),
      self.ncells.clone(),
    )
  }

  /// The cartesian multi-index of a quotient vertex, its position in the
  /// fundamental domain, in units of the grid spacing.
  pub fn vertex_cart_idx(&self, vertex: VertexIdx) -> Vec<usize> {
    linear2cartesian_mixed(vertex, &self.radices())
  }
}

#[cfg(test)]
mod test {
  use super::{FlatQuotient, Identification};
  use crate::{geometry::metric::mesh::MeshLengthsSq, linalg::Vector, topology::complex::Complex};
  use multiindex::binomial;

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

  /// A *translational* identification preserves the generator's Kuhn chain
  /// order as a face-consistent ordering, and refining in it keeps the quotient
  /// self-similar: one shape class. Translation carries the Kuhn tiling of one
  /// box onto the tiling of the next, chain order included, so the seam is
  /// indistinguishable from the interior.
  #[test]
  fn the_chain_order_survives_a_translational_identification() {
    for dim in 1..=3 {
      let (complex, lengths, ordering) = FlatQuotient::unit_torus(dim, 3).triangulate_ordered();
      let ordering = ordering.expect("the tiling matches across the seam");
      assert_eq!(shape_classes(&complex, &lengths), 1);

      let sub = complex.refine_with(&ordering, 2);
      let fine_lengths = lengths.refine(&sub, &complex);
      assert_eq!(shape_classes(sub.complex(), &fine_lengths), 1);
    }
  }

  /// A *reflecting* identification does not, and the generator says so rather
  /// than handing back an ordering that lies.
  ///
  /// The Kuhn triangulation of a box is not reflection-invariant: mirroring an
  /// axis exchanges the diagonal, so the two sides of a twisted seam emit
  /// incompatible chain orders on the face they share. The quotient is still
  /// *conforming* -- the exchanged diagonal is interior to a box, never on a
  /// shared face, which is why the topology above is correct -- but invariant 7
  /// asks for more than conformity, and the Kuhn order cannot supply it here.
  /// Refinement still works through the colex ordering; what is lost is the
  /// guarantee that a refinement tower stays self-similar.
  ///
  /// Recovering it needs a reflection-invariant triangulation of the box, not a
  /// repair of this one.
  #[test]
  fn a_reflecting_identification_admits_no_kuhn_chain_order() {
    for quotient in [
      FlatQuotient::moebius(1.0, 1.0, 3),
      FlatQuotient::klein(Vector::from_element(2, 1.0), 3),
    ] {
      let (_, _, ordering) = quotient.triangulate_ordered();
      assert!(
        ordering.is_none(),
        "a reflecting seam cannot be face-consistent in the Kuhn order"
      );
    }
  }

  /// Quasi-uniform resolution bounds the cell aspect ratio *independently of
  /// the fundamental domain's own aspect ratio*, which one shared cell count
  /// cannot: a Möbius band 16 times longer than it is wide, meshed with one
  /// count, has edges 16 times longer one way than the other, and the shape
  /// regularity every FEM error constant depends on degrades with it.
  ///
  /// The bound is on the *spacing*, so it is a statement about the geometry and
  /// not about the counts.
  #[test]
  fn quasi_uniform_resolution_bounds_the_aspect_ratio() {
    fn edge_length_spread(quotient: &FlatQuotient) -> f64 {
      let (_, lengths) = quotient.triangulate();
      let (mut min, mut max) = (f64::MAX, 0.0_f64);
      for &l in lengths.iter() {
        let l = l.sqrt();
        min = min.min(l);
        max = max.max(l);
      }
      max / min
    }

    let (circumference, width) = (16.0, 1.0);
    let ids = || vec![Identification::Twisted(vec![1]), Identification::Open];
    let sides = || Vector::from_column_slice(&[circumference, width]);

    let uniform = FlatQuotient::new_anisotropic(sides(), ids(), vec![16, 16]);
    let quasi = FlatQuotient::quasi_uniform(sides(), ids(), 16);

    // One count over unequal periods reproduces the domain's own aspect ratio.
    assert!(
      edge_length_spread(&uniform) > circumference / width,
      "a shared count meshes a long strip into slivers"
    );
    // Scaling the counts by the periods leaves only the Kuhn diagonal, whose
    // length is $sqrt(2)$ times an axis step: the regular cell's own spread.
    let spread = edge_length_spread(&quasi);
    assert!(
      spread < 1.5,
      "quasi-uniform cells should differ only by the Kuhn diagonal, got {spread}"
    );
    assert_eq!(quasi.ncells_per_axis(), [16, 1]);
  }

  /// The flat torus is closed (no boundary) and carries the cohomology of
  /// $T^d$: Betti numbers $b_k = binom(d, k)$, Euler characteristic $0$, for
  /// every dimension.
  #[test]
  fn torus_topology() {
    for dim in 1..=3 {
      let (complex, lengths) = FlatQuotient::unit_torus(dim, 3).triangulate();

      assert!(!complex.has_boundary(), "dim {dim}: torus is boundaryless");
      assert_eq!(complex.nsimplices(0), 3usize.pow(dim as u32));

      let betti = complex.betti_numbers();
      let expected = (0..=dim).map(|k| binomial(dim, k)).collect::<Vec<_>>();
      assert_eq!(betti, expected, "dim {dim}: Betti numbers of T^d");
      assert_eq!(complex.euler_characteristic(), 0);
      assert!(
        complex.orientation().is_some(),
        "dim {dim}: T^d is orientable"
      );

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
    let (coarse, _) = FlatQuotient::unit_torus(2, 3).triangulate();
    for refinement in 1..=2 {
      let fine = coarse.refine(refinement).into_complex();
      assert!(!fine.has_boundary());
      assert_eq!(fine.betti_numbers(), vec![1, 2, 1]);
      assert_eq!(fine.euler_characteristic(), 0);
    }
  }

  /// The Möbius band: a non-orientable surface with boundary, homotopy
  /// equivalent to its core circle, so $b_0 = b_1 = 1$, $b_2 = 0$ and
  /// $chi = 0$.
  #[test]
  fn moebius_topology() {
    for ncells in 3..=5 {
      let (complex, lengths) = FlatQuotient::moebius(1.0, 1.0, ncells).triangulate();

      assert!(complex.has_boundary(), "the band has a boundary circle");
      assert_eq!(complex.betti_numbers(), vec![1, 1, 0]);
      assert_eq!(complex.euler_characteristic(), 0);
      assert!(
        complex.orientation().is_none(),
        "the Möbius band is non-orientable"
      );
      assert!(lengths.iter().all(|&s| s > 0.0));
    }
  }

  /// The Klein bottle: closed and non-orientable, so no fundamental class and
  /// $b_2 = 0$. Over $RR$ the $ZZ_2$ torsion of $H_1$ is invisible, leaving
  /// $b_0 = b_1 = 1$ and $chi = 0$.
  #[test]
  fn klein_topology() {
    for ncells in 3..=5 {
      let (complex, _) = FlatQuotient::klein(Vector::from_element(2, 1.0), ncells).triangulate();

      assert!(!complex.has_boundary(), "the Klein bottle is closed");
      assert_eq!(complex.betti_numbers(), vec![1, 1, 0]);
      assert_eq!(complex.euler_characteristic(), 0);
      assert!(
        complex.orientation().is_none(),
        "the Klein bottle is non-orientable"
      );
    }
  }

  /// The parity of the reflections is the orientability: reflecting *two*
  /// transverse axes is a rotation, so the twisted 3-torus it glues stays
  /// orientable and closed, with the Euler characteristic of any closed
  /// odd-dimensional manifold.
  ///
  /// This is the control for [`moebius_topology`] and [`klein_topology`]: it
  /// isolates *non-orientability* from *being twisted at all*.
  #[test]
  fn even_reflection_count_stays_orientable() {
    let twisted = FlatQuotient::new(
      Vector::from_element(3, 1.0),
      vec![
        Identification::Twisted(vec![1, 2]),
        Identification::Periodic,
        Identification::Periodic,
      ],
      3,
    );
    assert!(twisted.is_orientation_preserving());

    let (complex, lengths) = twisted.triangulate();
    assert!(!complex.has_boundary());
    assert!(complex.orientation().is_some());
    assert_eq!(complex.euler_characteristic(), 0);
    assert!(lengths.iter().all(|&s| s > 0.0));
  }

  /// Every identification leaves the geometry flat and uniform: the quotient is
  /// a relabelling, so its edge lengths are exactly the grid's, seam included.
  #[test]
  fn identification_does_not_move_the_geometry() {
    let reference = {
      let (_, lengths) = FlatQuotient::new(
        Vector::from_element(2, 1.0),
        vec![Identification::Open, Identification::Open],
        4,
      )
      .triangulate();
      let mut ls: Vec<u64> = lengths.iter().map(|l| (l * 1e9).round() as u64).collect();
      ls.sort_unstable();
      ls.dedup();
      ls
    };
    for quotient in [
      FlatQuotient::unit_torus(2, 4),
      FlatQuotient::moebius(1.0, 1.0, 4),
      FlatQuotient::klein(Vector::from_element(2, 1.0), 4),
    ] {
      let (_, lengths) = quotient.triangulate();
      let mut ls: Vec<u64> = lengths.iter().map(|l| (l * 1e9).round() as u64).collect();
      ls.sort_unstable();
      ls.dedup();
      assert_eq!(ls, reference, "the seam introduced a new length");
    }
  }
}
