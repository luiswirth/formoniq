use {
  common::{linalg::nalgebra::Vector, util::algebraic_convergence_rate},
  ddf::cochain::cochain_projection,
  exterior::field::DiffFormClosure,
  formoniq::{fe::l2_norm, problems::hodge_laplace},
  manifold::gen::cartesian::CartesianMeshInfo,
};

use std::{f64::consts::PI, fs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let path = "out/laplacian_source";
  let _ = fs::remove_dir_all(path);
  fs::create_dir_all(path).unwrap();

  let dim = 2;

  let exact_solution = DiffFormClosure::one_form(
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

  let laplacian = DiffFormClosure::one_form(
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

  let mut errors = Vec::new();
  for irefine in 0..=10 {
    let refine_path = &format!("{path}/refine{irefine}");
    fs::create_dir_all(refine_path).unwrap();

    let nboxes_per_dim = 2usize.pow(irefine);
    let box_mesh = CartesianMeshInfo::new_unit_scaled(dim, nboxes_per_dim, PI);
    let (topology, coords) = box_mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);

    let laplacian = cochain_projection(&laplacian, &topology, &coords);
    let exact_u = cochain_projection(&exact_solution, &topology, &coords);

    let (_sigma, u, _p) = hodge_laplace::solve_hodge_laplace_source(&topology, &metric, laplacian);

    manifold::io::save_skeleton_to_file(&topology, dim, format!("{refine_path}/cells.skel"))?;
    manifold::io::save_skeleton_to_file(&topology, 1, format!("{refine_path}/edges.skel"))?;
    manifold::io::save_coords_to_file(&coords, format!("{refine_path}/vertices.coords"))?;

    ddf::io::save_cochain_to_file(&exact_u, format!("{refine_path}/exact.cochain"))?;
    ddf::io::save_cochain_to_file(&u, format!("{refine_path}/fe.cochain"))?;

    let diff = exact_u - u;
    let l2_norm = l2_norm(&diff, &topology, &metric);

    let conv_rate = |errors: &[f64], curr: f64| {
      errors
        .last()
        .map(|&prev| algebraic_convergence_rate(curr, prev))
        .unwrap_or(f64::INFINITY)
    };
    let conv_rate = conv_rate(&errors, l2_norm);
    errors.push(l2_norm);

    println!("refinement={irefine} | L2_error={l2_norm:<7.2e} | conv_rate={conv_rate:>5.2}");
  }

  Ok(())
}
