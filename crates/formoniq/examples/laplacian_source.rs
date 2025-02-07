extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use std::{
  fs::{self, File},
  io::BufWriter,
};

use exterior::{
  dense::{DifferentialFormClosure, MultiForm},
  manifold::discretize_form_on_mesh,
};
use formoniq::{
  fe::{evaluate_fe_function_facets_vertices, write_evaluations},
  problems::hodge_laplace,
};
use geometry::coord::{manifold::cartesian::CartesianMeshInfo, write_coords, CoordRef};
use topology::simplex::write_simplicies;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let path = "out/laplacian_source";
  let _ = fs::remove_dir_all(path);
  fs::create_dir_all(path).unwrap();

  let dim = 2;
  let nboxes_per_dim = 3;
  let box_mesh = CartesianMeshInfo::new_unit(dim, nboxes_per_dim);
  let (topology, coords) = box_mesh.compute_coord_complex();
  let metric = coords.to_edge_lengths(&topology);

  let file = File::create(format!("{path}/coords.txt"))?;
  let writer = BufWriter::new(file);
  write_coords(writer, &coords)?;

  let file = File::create(format!("{path}/facets.txt"))?;
  let writer = BufWriter::new(file);
  write_simplicies(writer, topology.facets().set_iter())?;

  let form_grade = 1;
  let source = Box::new(move |x: CoordRef| {
    let r = x - na::dvector![0.5, 0.5];
    MultiForm::from_grade1(r)
  });

  //let source = Box::new(move |x: CoordRef| MultiForm::from_grade1(na::dvector![1.0, 0.0]));
  let source = DifferentialFormClosure::new(source, dim, form_grade);
  let source = discretize_form_on_mesh(&source, &topology, &coords);

  let facet_evals = evaluate_fe_function_facets_vertices(&source, &topology, &coords);

  let file = File::create(format!("{path}/evaluations.txt"))?;
  let writer = BufWriter::new(file);
  write_evaluations(writer, &facet_evals)?;

  //let ndofs = mesh.topology().skeleton(form_grade).len();
  //let source = na::DVector::from_element(ndofs, 1.0);
  //let source = FeFunction::new(form_grade, source);

  let (_sigma, u, _p) =
    hodge_laplace::solve_hodge_laplace_source(&topology, &metric, form_grade, source);

  //let ndofs = mesh.topology().skeleton(form_grade).len();
  //let mut u = na::DVector::zeros(ndofs);
  //u[ndofs / 2 + 1] = 1.0;
  //let u = FeFunction::new(form_grade, u);

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
