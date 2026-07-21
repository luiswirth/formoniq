//! Transporting geometry across a [`Subdivision`].
//!
//! A [`Subdivision`] carries only topology and affine provenance; the geometry
//! is followed here, on the geometry side of the topology/geometry split. Every
//! child cell is an affine subcell of a flat parent, so its geometry is an exact
//! *pullback* of the parent's -- no approximation is introduced by refining, on
//! either representation:
//!
//! - **Intrinsic** ([`Subdivision::refine_gramians`]): each child's metric is
//!   the parent metric pulled back along the child's Jacobian. This is the
//!   coordinate-free refinement, and the primitive the extrinsic case must
//!   agree with.
//! - **Extrinsic** ([`MeshCoords::refine`]): each new vertex is placed by the
//!   affine combination of coarse vertices recorded in its
//!   [`VertexBirth`](crate::topology::refine::VertexBirth). An embedding is not
//!   needed to refine -- it is refined only because visualization and I/O want
//!   one -- and the metric it induces equals the intrinsic refinement, which is
//!   the law that ties the two.

use crate::{
  geometry::{coord::mesh::MeshCoords, metric::CellGramians, metric::mesh::MeshLengthsSq},
  linalg::Vector,
  topology::{complex::Complex, refine::Subdivision},
};

impl Subdivision {
  /// Refine per-cell metrics: each child carries the pullback of its parent
  /// cell's metric along the child's affine Jacobian. Exact and coordinate-free
  /// -- the intrinsic refinement of any geometry, once reduced to its per-cell
  /// metrics ([`CellGramians`]).
  pub fn refine_gramians(&self, coarse: &CellGramians) -> CellGramians {
    let metrics = self
      .children()
      .values()
      .iter()
      .map(|child| coarse.metrics()[child.parent].pullback(&child.jacobian))
      .collect();
    CellGramians::new(self.complex().dim(), metrics)
  }
}

impl MeshLengthsSq {
  /// Refine intrinsic Regge geometry: the refined squared edge lengths of the flat
  /// subdivision. Routed through the metric primitive
  /// ([`Subdivision::refine_gramians`]) rather than reimplemented -- coarse
  /// lengths give per-cell metrics, those are pulled back onto the children, and
  /// the fine metrics are read back as edge lengths. Exact; refinement of a flat
  /// cell introduces no geometric error.
  pub fn refine(&self, sub: &Subdivision, coarse: &Complex) -> MeshLengthsSq {
    let coarse_g = CellGramians::from_lengths(coarse, self);
    sub
      .refine_gramians(&coarse_g)
      .to_edge_lengths_sq(sub.complex())
  }
}

impl MeshCoords {
  /// Refine an embedding: the coarse vertices keep their coordinates and label,
  /// and each new vertex is the affine combination of coarse vertices its
  /// [`VertexBirth`](crate::topology::refine::VertexBirth) records. Extrinsic,
  /// for I/O and visualization; the intrinsic refinement is
  /// [`Subdivision::refine_gramians`].
  pub fn refine(&self, sub: &Subdivision) -> MeshCoords {
    assert_eq!(
      self.nvertices(),
      sub.ncoarse_vertices(),
      "coordinates must match the coarse mesh being refined"
    );
    let ambient = self.dim();
    let mut matrix = crate::linalg::Matrix::zeros(ambient, sub.nvertices());
    matrix
      .view_range_mut(.., 0..sub.ncoarse_vertices())
      .copy_from(self.matrix());
    for (i, birth) in sub.new_births().iter().enumerate() {
      let mut col = Vector::zeros(ambient);
      for &(v, w) in &birth.combination {
        col += w * self.matrix().column(v);
      }
      matrix.set_column(sub.ncoarse_vertices() + i, &col);
    }
    MeshCoords::with_ambient(matrix, self.ambient().clone())
  }
}

#[cfg(test)]
mod test {
  use crate::mesher::cartesian::CartesianGrid;
  use crate::topology::complex::Complex;

  fn signature(complex: &Complex, lengths: &MeshLengthsSq) -> Vec<Vec<u64>> {
    let mut cells: Vec<Vec<u64>> = complex
      .cells()
      .handle_iter()
      .map(|cell| {
        let mut ls: Vec<u64> = lengths
          .simplex_lengths_sq(*cell)
          .vector()
          .iter()
          .map(|&l| (l * 1e9).round() as u64)
          .collect();
        ls.sort_unstable();
        ls
      })
      .collect();
    cells.sort_unstable();
    cells
  }

  use crate::geometry::cell_volume;
  use crate::geometry::metric::mesh::MeshLengthsSq;

  /// Refining a Kuhn-triangulated grid reproduces the finer grid the generator
  /// would have built: $"refine"("grid"(n), r) tilde.equiv "grid"(n r)$, up to
  /// vertex relabelling. Freudenthal subdivision of a Kuhn cube *is* the Kuhn
  /// triangulation of its $2^n$ subcubes, so the general refinement algorithm
  /// loses none of the regularity of the structured generator -- in every
  /// dimension, not just the 2D case where red refinement happens to be
  /// self-similar. Congruence is tested on the intrinsic invariant: the multiset
  /// over cells of each cell's sorted squared edge lengths.
  #[test]
  fn refine_reproduces_the_generator_family() {
    for dim in 1..=3 {
      for (n, r) in [(1, 2), (2, 2), (1, 3), (2, 3), (1, 4)] {
        let (coarse, coarse_coords) = CartesianGrid::new_unit(dim, n).triangulate();
        let sub = coarse.refine(r);
        let refined = sub.complex();
        let refined_lengths = coarse_coords
          .to_edge_lengths_sq(&coarse)
          .refine(&sub, &coarse);

        let (direct, direct_coords) = CartesianGrid::new_unit(dim, n * r).triangulate();
        let direct_lengths = direct_coords.to_edge_lengths_sq(&direct);

        assert_eq!(refined.nsimplices(dim), direct.nsimplices(dim));
        assert_eq!(refined.vertices().len(), direct.vertices().len());
        assert_eq!(
          signature(refined, &refined_lengths),
          signature(&direct, &direct_lengths),
          "dim {dim}, grid({n}) refined by {r} is not congruent to grid({})",
          n * r
        );
      }
    }
  }

  /// The refined cells partition the measure of the coarse ones: refinement
  /// moves no volume, on the intrinsic representation. Each child also carries
  /// exactly $1 \/ R^n$ of its parent's volume.
  #[test]
  fn measure_partition() {
    for dim in 1..=3 {
      let (coarse, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      let coarse_g = coords.to_cell_gramians(&coarse);
      let coarse_vol: f64 = coarse_g.metrics().iter().map(cell_volume).sum();

      for r in 1..=3 {
        let sub = coarse.refine(r);
        let fine_g = sub.refine_gramians(&coarse_g);
        let fine_vol: f64 = fine_g.metrics().iter().map(cell_volume).sum();
        approx::assert_relative_eq!(fine_vol, coarse_vol, epsilon = 1e-12);

        // Per child: parent volume shared equally among its R^n children.
        let scale = (r.pow(dim as u32) as f64).recip();
        for child in sub.children().values() {
          let child_vol = cell_volume(&coarse_g.metrics()[child.parent].pullback(&child.jacobian));
          let parent_vol = cell_volume(&coarse_g.metrics()[child.parent]);
          approx::assert_relative_eq!(child_vol, parent_vol * scale, epsilon = 1e-12);
        }
      }
    }
  }

  /// Intrinsic equals extrinsic: refining the embedding and then inducing its
  /// metric gives the same per-cell metric as pulling the coarse metric back
  /// intrinsically. The one test that certifies the coordinate-free transport --
  /// it says the flat-cell subdivision loses nothing.
  #[test]
  fn intrinsic_equals_extrinsic() {
    for dim in 1..=3 {
      let (coarse, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      let coarse_g = coords.to_cell_gramians(&coarse);

      for r in 1..=3 {
        let sub = coarse.refine(r);
        let intrinsic = sub.refine_gramians(&coarse_g);

        let fine_coords = coords.refine(&sub);
        let extrinsic = fine_coords.to_cell_gramians(sub.complex());

        for (a, b) in intrinsic.metrics().iter().zip(extrinsic.metrics().iter()) {
          approx::assert_relative_eq!(
            a.vector_gramian().matrix(),
            b.vector_gramian().matrix(),
            epsilon = 1e-12
          );
        }
      }
    }
  }

  /// The Regge (edge-length) refinement agrees with the embedded one: refining
  /// lengths intrinsically and refining coordinates then measuring their edges
  /// give the same lengths. Certifies the metric-only path against the extrinsic
  /// one, edge by edge.
  #[test]
  fn lengths_match_coords() {
    use crate::topology::data::SkeletonData;
    for dim in 1..=3 {
      let (coarse, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      let coarse_lengths = coords.to_edge_lengths_sq(&coarse);

      for r in 1..=3 {
        let sub = coarse.refine(r);
        let intrinsic = coarse_lengths.refine(&sub, &coarse);
        let extrinsic = coords.refine(&sub).to_edge_lengths_sq(sub.complex());

        assert_eq!(intrinsic.len(), extrinsic.len());
        for (a, b) in intrinsic.iter().zip(extrinsic.iter()) {
          approx::assert_relative_eq!(a, b, epsilon = 1e-12);
        }
      }
    }
  }

  /// A refinement *tower* built on the inherited ordering is the single
  /// refinement of the product: refining twice by $R$ gives the same mesh as
  /// once by $R^2$, cells and geometry alike.
  ///
  /// $ "refine"_R compose "refine"_R = "refine"_(R^2) $
  ///
  /// The mesh-level statement of the atlas law, and the reason a convergence
  /// sweep may now be built level on level rather than from the base each time:
  /// every cell stays similar to the coarse cell it descends from. Refining in
  /// the colex ordering instead re-derives each child's order by sorting, which
  /// agrees with the pattern only at the first level and drifts after -- into a
  /// growing number of congruence classes above dimension two.
  #[test]
  fn a_tower_on_the_inherited_ordering_is_the_product_refinement() {
    use crate::topology::ordering::CellOrdering;

    for dim in 1..=3 {
      let (coarse, coords) = CartesianGrid::new_unit(dim, 1).triangulate();
      let lengths = coords.to_edge_lengths_sq(&coarse);

      for r in 2..=3 {
        let once = coarse.refine(r * r);
        let once_lengths = lengths.refine(&once, &coarse);

        let first = coarse.refine_with(&CellOrdering::colex(&coarse), r);
        let first_lengths = lengths.refine(&first, &coarse);
        let second = first.complex().refine_with(first.ordering(), r);
        let second_lengths = first_lengths.refine(&second, first.complex());

        assert_eq!(
          signature(second.complex(), &second_lengths),
          signature(once.complex(), &once_lengths),
          "dim {dim}: a tower of two {r}-fold refinements must be the {}-fold one",
          r * r
        );
      }
    }
  }

  /// Every cell of a tower is similar to the cell it came from: one similarity
  /// class, at every level, in every dimension.
  ///
  /// The property the ordering exists to preserve, stated where it is visible.
  /// Shape alone -- scale is divided out -- so it is a statement about mesh
  /// quality rather than about which mesh was built.
  #[test]
  fn a_tower_stays_self_similar() {
    use crate::topology::ordering::CellOrdering;

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

    for dim in 1..=4 {
      let (coarse, coords) = CartesianGrid::new_unit(dim, 1).triangulate();
      let mut lengths = coords.to_edge_lengths_sq(&coarse);
      let mut complex = coarse;
      let mut ordering = CellOrdering::colex(&complex);

      // Two levels already exhibit the drift the ordering prevents (the colex
      // tower leaves one class at level two); the top dimension is capped there
      // because a third level is ~10^5 cells for no further statement.
      let levels = if dim <= 3 { 3 } else { 2 };
      for level in 1..=levels {
        let sub = complex.refine_with(&ordering, 2);
        lengths = lengths.refine(&sub, &complex);
        ordering = sub.ordering().clone();
        complex = sub.into_complex();
        assert_eq!(
          shape_classes(&complex, &lengths),
          1,
          "dim {dim}, level {level}: a tower must stay self-similar"
        );
      }
    }
  }
}
