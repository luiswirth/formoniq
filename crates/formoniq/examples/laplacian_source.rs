extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use {
  common::util::algebraic_convergence_rate,
  exterior::{field::DifferentialFormClosure, MultiForm},
  formoniq::{fe::l2_norm, problems::hodge_laplace},
  manifold::{gen::cartesian::CartesianMeshInfo, geometry::coord::CoordRef},
  std::f64::consts::PI,
  whitney::discretize_form_on_mesh,
};

use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let path = "out/laplacian_source";
  let _ = fs::remove_dir_all(path);
  fs::create_dir_all(path).unwrap();

  let dim = 2;

  let mut errors = Vec::new();
  for refinement in 0..=10 {
    let nboxes_per_dim = 2usize.pow(refinement);
    let box_mesh = CartesianMeshInfo::new_unit_scaled(dim, nboxes_per_dim, PI);
    let (topology, coords) = box_mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);

    manifold::io::save_coords_to_file(&coords, format!("{path}/coords.txt"))?;
    manifold::io::save_cells_to_file(&topology, format!("{path}/cells.txt"))?;

    let form_grade = 1;

    let analytic_sol = |p: CoordRef| {
      let mut comps = na::DVector::zeros(p.len());
      for i in 0..p.len() {
        comps[i] = p[i].sin().powi(2) * p.remove_row(i).map(|a| a.cos()).product();
      }
      MultiForm::from_grade1(comps)
    };

    let scalar_lapl_closure = |selected: f64, others: &[f64]| {
      let mut scalar_lapl = 0.0;
      let product: f64 = others.iter().map(|a| a.cos()).product();
      scalar_lapl += 2.0 * (2.0 * selected).cos() * product;
      scalar_lapl -= others.len() as f64 * selected.sin().powi(2) * product;
      scalar_lapl
    };

    let analytic_laplacian = Box::new(move |p: CoordRef| {
      let mut comps = na::DVector::zeros(p.len());
      for i in 0..p.len() {
        let selected = p[i];
        let others = p.remove_row(i);
        comps[i] = -scalar_lapl_closure(selected, others.as_slice());
      }
      MultiForm::from_grade1(comps)
    });

    let source = DifferentialFormClosure::new(analytic_laplacian, dim, form_grade);
    let source = discretize_form_on_mesh(&source, &topology, &coords);

    let (_sigma, u, _p) =
      hodge_laplace::solve_hodge_laplace_source(&topology, &metric, form_grade, source);

    let analytic_form = DifferentialFormClosure::new(Box::new(analytic_sol), dim, form_grade);
    let analytic_cochain = discretize_form_on_mesh(&analytic_form, &topology, &coords);

    //for (&approx, &exact) in u.coeffs().iter().zip(analytic_cochain.coeffs().iter()) {
    //  println!("approx={approx}, exact={exact}");
    //}
    let diff = analytic_cochain - u;
    let l2_norm = l2_norm(&diff, &topology, &metric);
    println!("{} {l2_norm}", refinement);

    let conv_rate = |errors: &[f64], curr: f64| {
      errors
        .last()
        .map(|&prev| algebraic_convergence_rate(curr, prev))
        .unwrap_or(f64::INFINITY)
    };
    let conv_rate = conv_rate(&errors, l2_norm);
    errors.push(l2_norm);

    println!("conv_rate={conv_rate}");
  }

  Ok(())
}
