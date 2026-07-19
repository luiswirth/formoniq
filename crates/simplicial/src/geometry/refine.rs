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
  geometry::{coord::mesh::MeshCoords, metric::mesh::MeshLengthsSq, metric::CellGramians},
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
    let coarse_g = CellGramians::from_geometry(coarse, self);
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
  use crate::gen::cartesian::CartesianMeshInfo;
  use crate::geometry::{cell_volume, metric::CellGramians};

  /// The refined cells partition the measure of the coarse ones: refinement
  /// moves no volume, on the intrinsic representation. Each child also carries
  /// exactly $1 \/ R^n$ of its parent's volume.
  #[test]
  fn measure_partition() {
    for dim in 1..=3 {
      let (coarse, coords) = CartesianMeshInfo::new_unit(dim, 2).compute_coord_complex();
      let coarse_g = CellGramians::from_geometry(&coarse, &coords);
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
      let (coarse, coords) = CartesianMeshInfo::new_unit(dim, 2).compute_coord_complex();
      let coarse_g = CellGramians::from_geometry(&coarse, &coords);

      for r in 1..=3 {
        let sub = coarse.refine(r);
        let intrinsic = sub.refine_gramians(&coarse_g);

        let fine_coords = coords.refine(&sub);
        let extrinsic = CellGramians::from_geometry(sub.complex(), &fine_coords);

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
      let (coarse, coords) = CartesianMeshInfo::new_unit(dim, 2).compute_coord_complex();
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
}
