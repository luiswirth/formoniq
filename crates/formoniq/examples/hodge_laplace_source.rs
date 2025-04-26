use {
  common::{linalg::nalgebra::Vector, util::algebraic_convergence_rate},
  ddf::cochain::cochain_projection,
  exterior::field::DiffFormClosure,
  formoniq::fe::fe_l2_error,
  formoniq::problems::hodge_laplace,
  manifold::gen::cartesian::CartesianMeshInfo,
};

use std::{f64::consts::PI, fs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let path = "out/laplacian_source";
  let _ = fs::remove_dir_all(path);
  fs::create_dir_all(path).unwrap();

  let dim = 2;
  let homology_dim = 0;

  let solution_exact = DiffFormClosure::one_form(
    |p| {
      Vector::from_iterator(
        p.len(),
        (0..p.len()).map(|i| {
          let prod = p.remove_row(i).map(|a| a.cos()).product();
          p[i].sin().powi(2) * prod
        }),
      )
    },
    dim,
  );

  let laplacian_exact = DiffFormClosure::one_form(
    |p| {
      Vector::from_iterator(
        p.len(),
        (0..p.len()).map(|i| {
          let prod: f64 = p.remove_row(i).map(|a| a.cos()).product();
          -(2.0 * (2.0 * p[i]).cos() - (p.len() - 1) as f64 * p[i].sin().powi(2)) * prod
        }),
      )
    },
    dim,
  );

  let mut errors_l2 = Vec::new();
  for irefine in 0..=10 {
    let refine_path = &format!("{path}/refine{irefine}");
    fs::create_dir_all(refine_path).unwrap();

    let nboxes_per_dim = 2usize.pow(irefine);
    let box_mesh = CartesianMeshInfo::new_unit_scaled(dim, nboxes_per_dim, PI);
    let (topology, coords) = box_mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);

    let laplacian_cochain = cochain_projection(&laplacian_exact, &topology, &coords, None);
    let exact_u_cochain = cochain_projection(&solution_exact, &topology, &coords, None);

    let (_, galsol, _) = hodge_laplace::solve_hodge_laplace_source(
      &topology,
      &metric,
      laplacian_cochain,
      homology_dim,
    );

    manifold::io::save_skeleton_to_file(&topology, dim, format!("{refine_path}/cells.skel"))?;
    manifold::io::save_skeleton_to_file(&topology, 1, format!("{refine_path}/edges.skel"))?;
    manifold::io::save_coords_to_file(&coords, format!("{refine_path}/vertices.coords"))?;

    ddf::io::save_cochain_to_file(&exact_u_cochain, format!("{refine_path}/exact.cochain"))?;
    ddf::io::save_cochain_to_file(&galsol, format!("{refine_path}/fe.cochain"))?;

    let conv_rate = |errors: &[f64], curr: f64| {
      errors
        .last()
        .map(|&prev| algebraic_convergence_rate(curr, prev))
        .unwrap_or(f64::INFINITY)
    };

    let error_l2 = fe_l2_error(&galsol, &solution_exact, &topology, &coords);
    let conv_rate_l2 = conv_rate(&errors_l2, error_l2);
    errors_l2.push(error_l2);

    // TODO: H1 convergence
    //let dif_galsol = galsol.dif(&topology);
    //let error_h1 = fe_l2_error(&dif_galsol, &dif_solution_exact, &topology, &coords);
    //let conv_rate_h1 = conv_rate(&errors_h1, error_h1);
    //errors_h1.push(error_h1);

    println!("refinement={irefine} | L2_error={error_l2:<7.2e} | conv_rate={conv_rate_l2:>5.2}");
  }

  Ok(())
}
