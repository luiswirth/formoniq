use crate::{operators::FeFunction, whitney::WhitneyForm};

use exterior::dense::{ExteriorField, MultiForm};
use geometry::coord::{
  manifold::{EmbeddedComplex, SimplexHandleExt},
  CoordRef,
};
use topology::complex::dim::DimInfoProvider;

pub fn evaluate_fe_function_at_coord<'a>(
  coord: impl Into<CoordRef<'a>>,
  fe: &FeFunction,
  mesh: &EmbeddedComplex,
) -> MultiForm {
  let coord = coord.into();

  let dim = mesh.dim_embedded();
  assert_eq!(coord.len(), dim);
  let grade = fe.dim.dim(mesh.dim_intrinsic());

  // Find facet that contains coord.
  // WARN: very slow and inefficent
  let Some(facet) = mesh
    .topology()
    .facets()
    .iter()
    .find(|facet| facet.coord_simplex(mesh.coords()).is_coord_inside(coord))
  else {
    return MultiForm::zero(dim, grade);
  };

  let mut fe_value = MultiForm::zero(mesh.dim_intrinsic(), grade);
  for dof_simp in facet.subsimps(grade) {
    let local_dof_simp = facet
      .simplex_set()
      .global_to_local_subsimp(dof_simp.simplex_set());

    let dof_value = fe[dof_simp]
      * WhitneyForm::new(facet.coord_simplex(mesh.coords()), local_dof_simp).at_point(coord);
    fe_value += dof_value;
  }

  fe_value
}

pub fn evaluate_fe_function_at_facet_barycenters(
  fe: &FeFunction,
  mesh: &EmbeddedComplex,
) -> Vec<MultiForm> {
  let grade = fe.dim.dim(mesh.dim_intrinsic());

  mesh
    .topology()
    .facets()
    .iter()
    .map(|facet| {
      let mut value = MultiForm::zero(mesh.dim_embedded(), grade);
      for dof_simp in facet.subsimps(grade) {
        let local_dof_simp = facet
          .simplex_set()
          .global_to_local_subsimp(dof_simp.simplex_set());

        let barycenter = facet.coord_simplex(mesh.coords()).barycenter();

        let dof_value = fe[dof_simp]
          * WhitneyForm::new(facet.coord_simplex(mesh.coords()), local_dof_simp)
            .at_point(&barycenter);
        value += dof_value;
      }
      value
    })
    .collect()
}

pub fn evaluate_fe_function_at_dof_barycenters(
  fe: &FeFunction,
  mesh: &EmbeddedComplex,
) -> Vec<MultiForm> {
  let grade = fe.dim.dim(mesh.dim_intrinsic());

  mesh
    .topology()
    .skeleton(fe.dim)
    .iter()
    .map(|dof_simp| {
      let cofacet = dof_simp.cofacets().next().unwrap();

      let mut value = MultiForm::zero(mesh.dim_embedded(), grade);
      let local_dof_simp = cofacet
        .simplex_set()
        .global_to_local_subsimp(dof_simp.simplex_set());

      let barycenter = dof_simp.coord_simplex(mesh.coords()).barycenter();

      let dof_value = fe[dof_simp]
        * WhitneyForm::new(cofacet.coord_simplex(mesh.coords()), local_dof_simp)
          .at_point(&barycenter);
      value += dof_value;
      value
    })
    .collect()
}
