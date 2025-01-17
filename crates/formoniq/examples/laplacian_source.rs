extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use exterior::{
  dense::{DifferentialFormClosure, MultiForm},
  manifold::discretize_form_on_mesh,
};
use formoniq::{fe::evaluate_fe_function_at_coord, problems::hodge_laplace};
use geometry::coord::{
  manifold::{cartesian::CartesianMesh, CoordSimplex},
  CoordRef,
};

fn main() {
  let dim = 2;
  let nboxes_per_dim = 10;
  let box_mesh = CartesianMesh::new_unit(dim, nboxes_per_dim);
  let coord_mesh = box_mesh.compute_coord_manifold().into_coord_complex();
  let mesh = coord_mesh.to_metric_complex();

  let form_grade = 1;
  let source = Box::new(move |x: CoordRef| MultiForm::new(x.into(), dim, form_grade));
  let source = DifferentialFormClosure::new(source, dim, form_grade);

  let source = discretize_form_on_mesh(&source, &coord_mesh);

  let (_sigma, u, _p) = hodge_laplace::solve_hodge_laplace_source(&mesh, form_grade, source);

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
