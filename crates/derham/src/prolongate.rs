//! Cochain prolongation across a uniform refinement.
//!
//! Under uniform refinement the coarse Whitney space is a subspace of the fine
//! one -- every coarse Whitney form is, on each fine cell, an affine form and so
//! lies in $P^-_1 Lambda^k$ of the refined mesh. The inclusion has a canonical
//! matrix on cochains, the prolongation
//! $P: C^k("coarse") -> C^k("fine")$, obtained by expressing the coarse form in
//! the fine degrees of freedom:
//!
//! $ (P c)_sigma = R_sigma (W c) = integral_sigma W("coarse") c $
//!
//! for each fine simplex $sigma$ -- the coarse Whitney interpolant re-sampled by
//! the de Rham map of the fine complex. It is **metric-free**: every fine
//! simplex sits inside a coarse cell with barycentric coordinates fixed by the
//! [`Subdivision`] provenance alone (invariant 2), and the de Rham integral is
//! itself metric-free.
//!
//! The governing laws are executable:
//!
//! - commuting: $dif("fine") compose P = P compose dif("coarse")$
//!   (test `prolongation_is_a_cochain_map`);
//! - nesting / consistency: $P = R("fine") compose W("coarse")$, which is the
//!   definition, checked against the direct de Rham map (test
//!   `prolongation_is_the_resampled_interpolant`);
//! - identity: $R = 1$ gives $P = "id"$ (test `identity_refinement_prolongs_trivially`).
//!
//! It is what refinement is otherwise short of: comparing coarse and fine
//! discrete solutions without an analytic reference, and the intergrid transfer
//! of FEEC multigrid and auxiliary-space preconditioning.

use crate::{
  cochain::Cochain, interpolate::interpolant::WhitneyInterpolant, project::derham_map,
  section::Section,
};

use {
  exterior::{Covariant, Dim, ExteriorGrade, MultiForm},
  simplicial::{
    atlas::{Bary, MeshPoint},
    linalg::{Matrix, Vector},
    topology::{VertexIdx, complex::Complex, handle::SimplexIdx, refine::Subdivision},
  },
};

/// The prolongation $P c$ of a coarse cochain onto the refined complex: the
/// coarse Whitney interpolant, re-sampled by the fine de Rham map.
///
/// `coarse` is a cochain on `coarse_complex`, and `subdivision` its refinement;
/// the result is a cochain of the same grade on `subdivision.complex()`. The
/// coarse Whitney forms are affine on every fine cell, so degree-1 quadrature
/// integrates them exactly and $P$ is computed with no discretization error of
/// its own.
///
/// Metric-free: the whole map is fixed by the affine provenance of the
/// refinement, no geometry consulted.
pub fn prolongate(
  coarse: &Cochain,
  coarse_complex: &Complex,
  subdivision: &Subdivision,
) -> Cochain {
  let section = ProlongedWhitney::new(coarse, coarse_complex, subdivision);
  derham_map(&section, subdivision.complex(), 1)
}

/// The coarse Whitney interpolant seen as a section of the *fine* manifold: at a
/// fine mesh point it maps into the coarse cell that contains the point,
/// evaluates the coarse form there, and expresses the value in the fine cell's
/// reference frame.
struct ProlongedWhitney<'a> {
  coarse: WhitneyInterpolant<'a>,
  subdivision: &'a Subdivision,
  /// Per fine cell (by kidx): the coarse parent, and the affine map from the
  /// fine cell's barycentric coordinates to the parent's -- the columns are the
  /// parent-barycentric coordinates of the fine cell's (sorted) vertices, so
  /// `parent_bary = bary_map * fine_bary`.
  provenance: Vec<(SimplexIdx, Matrix)>,
}

impl<'a> ProlongedWhitney<'a> {
  fn new(coarse: &Cochain, coarse_complex: &'a Complex, subdivision: &'a Subdivision) -> Self {
    let dim = coarse_complex.dim();
    let ncoarse = subdivision.ncoarse_vertices();
    let births = subdivision.new_births();
    let children = subdivision.children();
    let fine = subdivision.complex();

    let provenance = fine
      .cells()
      .handle_iter()
      .map(|cell| {
        let parent = SimplexIdx::new(dim, children[cell.kidx()].parent);
        let parent_verts = &parent.handle(coarse_complex).simplex().vertices;
        // Local index of a coarse global vertex within the parent cell; the
        // provenance guarantees every vertex referenced here is one of them.
        let slot = |v: VertexIdx| {
          parent_verts
            .binary_search(&v)
            .expect("a fine vertex lies in its parent's vertex set")
        };

        // Column i: the parent-barycentric coordinates of the fine cell's i-th
        // (sorted) vertex. A coarse vertex keeps its label and is a corner of
        // the parent; a new vertex carries its affine birth over coarse
        // vertices, all of which are the parent's.
        let mut bary_map = Matrix::zeros(dim + 1, dim + 1);
        for (i, &v) in cell.simplex().vertices.iter().enumerate() {
          if v < ncoarse {
            bary_map[(slot(v), i)] = 1.0;
          } else {
            for &(g, w) in &births[v - ncoarse].combination {
              bary_map[(slot(g), i)] = w;
            }
          }
        }
        (parent, bary_map)
      })
      .collect();

    Self {
      coarse: WhitneyInterpolant::new(coarse.clone(), coarse_complex),
      subdivision,
      provenance,
    }
  }
}

impl Section<Covariant> for ProlongedWhitney<'_> {
  fn dim(&self) -> Dim {
    self.subdivision.complex().dim()
  }
  fn grade(&self) -> ExteriorGrade {
    self.coarse.cochain().grade()
  }
  fn at(&self, point: &MeshPoint) -> MultiForm {
    let fine_cell = point.cell_idx().kidx;
    let (parent, bary_map) = &self.provenance[fine_cell];

    let parent_bary: Vector = bary_map * point.bary().view();
    let value = self
      .coarse
      .eval(&MeshPoint::new(*parent, Bary::new(parent_bary)));

    // The coarse value lives in the parent's reference frame; the child's
    // Jacobian maps the fine reference frame into the parent's, so pulling back
    // along it expresses the value in the fine cell's frame, as the fine de Rham
    // integral over `point`'s simplex requires.
    value.pullback(&self.subdivision.children()[fine_cell].jacobian)
  }
}

#[cfg(test)]
mod test {
  use super::*;

  use crate::section::{Sampler, SectionExt};

  use simplicial::{
    geometry::coord::{locate::PointLocator, mesh::MeshCoords, simplex::SimplexRefExt},
    mesher::cartesian::CartesianGrid,
  };

  use approx::assert_relative_eq;

  /// A cochain with distinct, non-degenerate entries on every DOF of a grade.
  fn probe_cochain(complex: &Complex, grade: ExteriorGrade) -> Cochain {
    let ndofs = complex.nsimplices(grade);
    Cochain::new(
      grade,
      Vector::from_iterator(ndofs, (0..ndofs).map(|i| 0.7 * i as f64 - 1.3)),
    )
  }

  /// $P = R("fine") compose W("coarse")$: the prolongation is exactly the coarse
  /// Whitney form re-sampled on the fine complex.
  ///
  /// The definition made a theorem, checked against an independent route: rather
  /// than the affine provenance `prolongate` rides, this evaluates the coarse
  /// interpolant through an *embedding* -- the fine mesh point placed in ambient
  /// coordinates, located back in the coarse mesh, sampled there, and pulled
  /// into the fine cell's frame. Refinement is affine so the two agree exactly,
  /// but the two share no code, so a bug in the provenance path cannot hide.
  #[test]
  fn prolongation_is_the_resampled_interpolant() {
    for dim in 1..=3 {
      let (coarse, coarse_coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      for r in 1..=3 {
        let sub = coarse.refine(r);
        let fine = sub.complex();
        let fine_coords = coarse_coords.refine(&sub);
        let locator = PointLocator::new(&coarse, &coarse_coords);

        for grade in 0..=dim {
          let c = probe_cochain(&coarse, grade);
          let prolonged = prolongate(&c, &coarse, &sub);

          let interpolant = WhitneyInterpolant::new(c, &coarse);
          let sampler = interpolant
            .sampled_on(&coarse, &coarse_coords)
            .with_locator(&locator);
          let resampled = ResampledViaEmbedding {
            sampler: &sampler,
            fine,
            fine_coords: &fine_coords,
            grade,
          };
          let direct = derham_map(&resampled, fine, 1);

          assert_relative_eq!(prolonged.coeffs(), direct.coeffs(), epsilon = 1e-10);
        }
      }
    }
  }

  /// $R = 1$ gives $P = "id"$: the identity refinement prolongs a cochain to
  /// itself, coarse vertices keeping their labels so the DOFs line up.
  #[test]
  fn identity_refinement_prolongs_trivially() {
    for dim in 0..=3 {
      let (coarse, _) = CartesianGrid::new_unit(dim, 2).triangulate();
      let sub = coarse.refine(1);
      for grade in 0..=dim {
        let c = probe_cochain(&coarse, grade);
        let prolonged = prolongate(&c, &coarse, &sub);
        assert_relative_eq!(prolonged.coeffs(), c.coeffs(), epsilon = 1e-10);
      }
    }
  }

  /// $dif("fine") compose P = P compose dif("coarse")$: prolongation is a
  /// cochain map.
  ///
  /// The coarse and fine exterior derivatives are the coboundary operators of
  /// their complexes, and $P$ intertwines them -- the Whitney space nesting
  /// respects the de Rham differential, since $W$ and $R$ each do.
  #[test]
  fn prolongation_is_a_cochain_map() {
    for dim in 1..=3 {
      let (coarse, _) = CartesianGrid::new_unit(dim, 2).triangulate();
      for r in 1..=3 {
        let sub = coarse.refine(r);
        let fine = sub.complex();
        for grade in 0..dim {
          let c = probe_cochain(&coarse, grade);

          let dif_then_prolong = prolongate(&c.dif(&coarse), &coarse, &sub);
          let prolong_then_dif = prolongate(&c, &coarse, &sub).dif(fine);

          assert_eq!(dif_then_prolong.grade(), prolong_then_dif.grade());
          assert_relative_eq!(
            dif_then_prolong.coeffs(),
            prolong_then_dif.coeffs(),
            epsilon = 1e-10
          );
        }
      }
    }
  }

  /// The coarse Whitney interpolant re-sampled on the fine mesh through an
  /// embedding: the reference route for $R("fine") compose W("coarse")$, sharing
  /// no code with the affine provenance `prolongate` rides.
  ///
  /// A fine mesh point is placed in ambient coordinates through the fine cell's
  /// parametrization, the coarse form is sampled there (`Sampler` locates the
  /// coarse cell and returns the value in the ambient frame), and the value is
  /// pulled back into the fine cell's reference frame.
  struct ResampledViaEmbedding<'a> {
    sampler: &'a Sampler<'a, WhitneyInterpolant<'a>>,
    fine: &'a Complex,
    fine_coords: &'a MeshCoords,
    grade: ExteriorGrade,
  }
  impl Section<Covariant> for ResampledViaEmbedding<'_> {
    fn dim(&self) -> Dim {
      self.fine.dim()
    }
    fn grade(&self) -> ExteriorGrade {
      self.grade
    }
    fn at(&self, point: &MeshPoint) -> MultiForm {
      let param = point.chart(self.fine).coord_simplex(self.fine_coords);
      let global = param.bary2global(point.bary());
      let ambient = self
        .sampler
        .at_global(&global)
        .expect("fine point is in the coarse mesh");
      ambient.pullback(&param.linear_transform())
    }
  }
}
