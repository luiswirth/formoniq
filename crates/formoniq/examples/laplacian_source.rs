extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use {
  exterior::{field::DifferentialFormClosure, MultiForm},
  formoniq::problems::hodge_laplace,
  manifold::{gen::cartesian::CartesianMeshInfo, geometry::coord::CoordRef},
  whitney::discretize_form_on_mesh,
};

use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let path = "out/laplacian_source";
  let _ = fs::remove_dir_all(path);
  fs::create_dir_all(path).unwrap();

  let dim = 2;
  let nboxes_per_dim = 3;
  let box_mesh = CartesianMeshInfo::new_unit(dim, nboxes_per_dim);
  let (topology, coords) = box_mesh.compute_coord_complex();
  let metric = coords.to_edge_lengths(&topology);

  manifold::io::save_coords_to_file(&coords, format!("{path}/coords.txt"))?;
  manifold::io::save_cells_to_file(&topology, format!("{path}/cells.txt"))?;

  let form_grade = 1;
  let source = Box::new(move |x: CoordRef| {
    let r = x - na::dvector![0.5, 0.5];
    MultiForm::from_grade1(r)
  });
  let source = DifferentialFormClosure::new(source, dim, form_grade);
  let source = discretize_form_on_mesh(&source, &topology, &coords);

  formoniq::io::save_evaluations_to_file(
    &source,
    &topology,
    &coords,
    format!("{path}/evaluations.txt"),
  )?;

  let (_sigma, u, _p) =
    hodge_laplace::solve_hodge_laplace_source(&topology, &metric, form_grade, source);

  let analytic_form = |coord: CoordRef| {
    let x = coord[0];
    let y = coord[1];
    let exact_ux = -x.powi(3) / 6.0 + x / 2.0 * (y.powi(2) - y);
    let exact_uy = -y.powi(3) / 6.0 + y / 2.0 * (x.powi(2) - x);
    let exact_u = na::DVector::from_row_slice(&[exact_ux, exact_uy]);
    MultiForm::from_grade1(exact_u)
  };
  let analytic_form = DifferentialFormClosure::new(Box::new(analytic_form), dim, form_grade);
  let analytic_cochain = discretize_form_on_mesh(&analytic_form, &topology, &coords);

  for (&approx, &exact) in u.coeffs().iter().zip(analytic_cochain.coeffs().iter()) {
    println!("approx={approx}, exact={exact}");
  }
  let diff = analytic_cochain - u;
  let norm = diff.coeffs().norm();
  println!("{norm}");

  Ok(())
}
