//! Whitney prolongation onto a refinement: it agrees with resampling the
//! coarse interpolant through an embedding, is the identity at $R = 1$, and
//! is a cochain map.

use approx::assert_relative_eq;
use derham::Cochain;
use derham::interpolate::interpolant::WhitneyInterpolant;
use derham::project::derham_map;
use derham::prolongate::prolongate;
use derham::section::{Sampler, Section, SectionExt};
use multialgebra::{ExteriorGrade, Tensor};
use multiindex::Dim;
use regge::coord::locate::PointLocator;
use regge::coord::mesh::MeshCoords;
use regge::coord::simplex::SimplexRefExt;
use regge::mesher::cartesian::CartesianGrid;
use simplicial::atlas::MeshPoint;
use simplicial::linalg::Vector;
use simplicial::topology::complex::Complex;

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
/// interpolant through an embedding, the fine mesh point placed in ambient
/// coordinates, located back in the coarse mesh, sampled there, and pulled
/// into the fine cell's frame. Refinement is affine so the two agree exactly,
/// but the two share no code, so a bug in the provenance path cannot hide.
#[test]
fn prolongation_is_the_resampled_interpolant() {
  for dim in (1..=3).into_iter().map(Dim::from) {
    let (coarse, coarse_coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    for r in 1..=3 {
      let sub = coarse.refine(r);
      let fine = sub.complex();
      let fine_coords = coarse_coords.refine(&sub);
      let locator = PointLocator::new(&coarse, &coarse_coords);

      for grade in dim.range_inclusive() {
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
  for dim in (0..=3).into_iter().map(Dim::from) {
    let (coarse, _) = CartesianGrid::new_unit(dim, 2).triangulate();
    let sub = coarse.refine(1);
    for grade in dim.range_inclusive() {
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
/// their complexes, and $P$ intertwines them, the Whitney space nesting
/// respects the de Rham differential, since $W$ and $R$ each do.
#[test]
fn prolongation_is_a_cochain_map() {
  for dim in (1..=3).into_iter().map(Dim::from) {
    let (coarse, _) = CartesianGrid::new_unit(dim, 2).triangulate();
    for r in 1..=3 {
      let sub = coarse.refine(r);
      let fine = sub.complex();
      for grade in dim.range() {
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
impl Section for ResampledViaEmbedding<'_> {
  fn dim(&self) -> Dim {
    self.fine.dim()
  }
  fn grade(&self) -> ExteriorGrade {
    self.grade
  }
  fn at(&self, point: &MeshPoint) -> Tensor {
    let param = point.chart(self.fine).coord_simplex(self.fine_coords);
    let global = param.bary2global(point.bary());
    let ambient = self
      .sampler
      .at_global(&global)
      .expect("fine point is in the coarse mesh");
    ambient.pullback(&param.linear_transform())
  }
}
