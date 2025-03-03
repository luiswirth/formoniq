extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use {
  common::util::algebraic_convergence_rate,
  exterior::{field::DifferentialFormClosure, MultiForm},
  formoniq::{fe::l2_norm, problems::hodge_laplace},
  manifold::{gen::cartesian::CartesianMeshInfo, geometry::coord::CoordRef},
  std::f64::consts::PI,
  whitney::cochain::discretize_form_on_mesh,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let dim = 2;
  let form_grade = 1;

  let exact_solution = |p: CoordRef| {
    let comps = (0..p.len()).map(|i| {
      let prod = p.remove_row(i).map(|a| a.cos()).product();
      p[i].sin().powi(2) * prod
    });
    MultiForm::from_grade1(na::DVector::from_iterator(p.len(), comps))
  };
  let laplacian = |p: CoordRef| {
    let comps = (0..p.len()).map(|i| {
      let prod: f64 = p.remove_row(i).map(|a| a.cos()).product();
      -(2.0 * (2.0 * p[i]).cos() - (p.len() - 1) as f64 * p[i].sin().powi(2)) * prod
    });
    MultiForm::from_grade1(na::DVector::from_iterator(p.len(), comps))
  };

  let laplacian = DifferentialFormClosure::new(Box::new(laplacian), dim, form_grade);
  let exact_solution = DifferentialFormClosure::new(Box::new(exact_solution), dim, form_grade);

  let mut errors = Vec::new();
  for refinement in 0..=10 {
    let nboxes_per_dim = 2usize.pow(refinement);
    let box_mesh = CartesianMeshInfo::new_unit_scaled(dim, nboxes_per_dim, PI);
    let (topology, coords) = box_mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);

    let laplacian = discretize_form_on_mesh(&laplacian, &topology, &coords);
    let exact_solution = discretize_form_on_mesh(&exact_solution, &topology, &coords);

    let (_sigma, u, _p) =
      hodge_laplace::solve_hodge_laplace_source(&topology, &metric, form_grade, laplacian);

    let diff = exact_solution - u;
    let l2_norm = l2_norm(&diff, &topology, &metric);

    let conv_rate = |errors: &[f64], curr: f64| {
      errors
        .last()
        .map(|&prev| algebraic_convergence_rate(curr, prev))
        .unwrap_or(f64::INFINITY)
    };
    let conv_rate = conv_rate(&errors, l2_norm);
    errors.push(l2_norm);

    println!("refinement={refinement} | L2_error={l2_norm:<7.2e} | conv_rate={conv_rate:>5.2}");
  }

  Ok(())
}
