use {
  common::{linalg::nalgebra::Vector, util::algebraic_convergence_rate},
  exterior::{field::DiffFormClosure, ExteriorElement},
  formoniq::{
    assemble::assemble_galvec, fe::fe_l2_error, operators::SourceElVec, problems::hodge_laplace,
  },
  manifold::{gen::cartesian::CartesianMeshInfo, geometry::coord::CoordRef},
};

use std::{f64::consts::PI, fs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
  tracing_subscriber::fmt::init();
  let path = "out/laplacian_source";
  let _ = fs::remove_dir_all(path);
  fs::create_dir_all(path).unwrap();

  let grade = 1;
  let homology_dim = 0;

  for dim in 2_usize..=3 {
    println!("Solving Hodge-Laplace in {dim}d.");

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

    let dif_solution_exact = DiffFormClosure::new(
      Box::new(move |p: CoordRef| {
        let dim = p.len();
        let ncomponents = if dim > 1 { dim * (dim - 1) / 2 } else { 0 };
        let mut components = Vec::with_capacity(ncomponents);

        let sin_p: Vec<_> = p.iter().map(|&pi| pi.sin()).collect();
        let cos_p: Vec<_> = p.iter().map(|&pi| pi.cos()).collect();

        for k in 0..dim {
          for i in (k + 1)..dim {
            let mut prod_cos = 1.0;
            #[allow(clippy::needless_range_loop)]
            for j in 0..dim {
              if j != i && j != k {
                prod_cos *= cos_p[j];
              }
            }
            let coeff = prod_cos * sin_p[i] * sin_p[k] * (sin_p[k] - sin_p[i]);
            components.push(coeff);
          }
        }
        ExteriorElement::new(components.into(), dim, 2)
      }),
      dim,
      2,
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

    println!(
      "| {:>2} | {:8} | {:>7} | {:>8} | {:>7} |",
      "k", "L2 err", "L2 conv", "Hd err", "Hd conv",
    );

    let mut errors_l2 = Vec::new();
    let mut errors_hd = Vec::new();
    for irefine in 0..=(15 / dim as u32) {
      let refine_path = &format!("{path}/refine{irefine}");
      fs::create_dir_all(refine_path).unwrap();

      let nboxes_per_dim = 2usize.pow(irefine);
      let box_mesh = CartesianMeshInfo::new_unit_scaled(dim, nboxes_per_dim, PI);
      let (topology, coords) = box_mesh.compute_coord_complex();
      let metric = coords.to_edge_lengths(&topology);

      let source_data = assemble_galvec(
        &topology,
        &metric,
        SourceElVec::new(&laplacian_exact, &coords, None),
      );

      let (_, galsol, _) = hodge_laplace::solve_hodge_laplace_source(
        &topology,
        &metric,
        source_data,
        grade,
        homology_dim,
      );

      let conv_rate = |errors: &[f64], curr: f64| {
        errors
          .last()
          .map(|&prev| algebraic_convergence_rate(curr, prev))
          .unwrap_or(f64::INFINITY)
      };

      let error_l2 = fe_l2_error(&galsol, &solution_exact, &topology, &coords);
      let conv_rate_l2 = conv_rate(&errors_l2, error_l2);
      errors_l2.push(error_l2);

      let dif_galsol = galsol.dif(&topology);
      let error_hd = fe_l2_error(&dif_galsol, &dif_solution_exact, &topology, &coords);
      let conv_rate_hd = conv_rate(&errors_hd, error_hd);
      errors_hd.push(error_hd);

      println!(
        "| {:>2} | {:<8.2e} | {:>7.2} | {:<8.2e} | {:>7.2} |",
        irefine, error_l2, conv_rate_l2, error_hd, conv_rate_hd
      );
    }
  }

  Ok(())
}
