extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use exterior::{
  dense::{DifferentialFormClosure, ExteriorField, MultiForm},
  manifold::discretize_form_on_mesh,
};
use formoniq::{operators::FeFunction, problems::hodge_laplace, whitney::WhitneyForm};
use geometry::coord::{
  manifold::{cartesian::CartesianMesh, CoordComplex, CoordSimplex, SimplexHandleExt},
  CoordRef,
};
use topology::complex::dim::DimInfoProvider;

fn main() {
  let dim = 2;
  let nboxes_per_dim = 10;
  let box_mesh = CartesianMesh::new_unit(dim, nboxes_per_dim);
  let coord_mesh = box_mesh.compute_coord_manifold().into_coord_complex();
  let (mesh, _) = coord_mesh.clone().into_metric_complex();

  let form_grade = 1;
  let source = Box::new(move |x: CoordRef| MultiForm::new(x.into(), dim, form_grade));
  let source = DifferentialFormClosure::new(source, dim, form_grade);

  let source = discretize_form_on_mesh(&source, &coord_mesh);

  let (_sigma, u) = hodge_laplace::solve_hodge_laplace_source(&mesh, form_grade, source);

  for facet in mesh.topology().facets().iter() {
    let coord_facet =
      CoordSimplex::from_simplex_and_coords(facet.simplex_set(), coord_mesh.coords());
    let coord = coord_facet.barycenter();

    let approx_u = evaluate_fe_function_at_coord(&coord, &u, &coord_mesh).into_coeffs();

    let x = coord[0];
    let y = coord[1];
    let exact_ux = -x.powi(3) / 6.0 + x / 2.0 * (y.powi(2) - y);
    let exact_uy = -y.powi(3) / 6.0 + y / 2.0 * (x.powi(2) - x);
    let exact_u = na::DVector::from_row_slice(&[exact_ux, exact_uy]);

    println!("approx: {approx_u}");
    println!("exact: {exact_u}");
  }
}

pub fn evaluate_fe_function_at_coord<'a>(
  coord: impl Into<CoordRef<'a>>,
  fe: &FeFunction,
  mesh: &CoordComplex,
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
  let dof_simps = facet.subsimps(grade);
  for dof_simp in dof_simps {
    let local_dof_simp = facet
      .simplex_set()
      .global_to_local_subsimp(dof_simp.simplex_set());

    let dof_value = fe[dof_simp]
      * WhitneyForm::new(facet.coord_simplex(mesh.coords()), local_dof_simp).at_point(coord);
    fe_value += dof_value;
  }

  fe_value
}
