//! Hodge-Laplace eigenvalue problem.
//!
//! By default, a generic sweep over every dimension $0 <= n <= 3$, every form
//! grade $0 <= k <= n$, and both boundary conditions on the flat box $[0, pi]^n$:
//! the lowest eigenvalues of the mixed Hodge Laplacian, refined until their
//! algebraic self-convergence rate shows. The count of zero eigenvalues is the
//! harmonic dimension — absolute $b_k (K)$ ($1$ at grade $0$) versus relative
//! $b_k (K, diff K)$ ($1$ at top grade) on the contractible box, an exact
//! topological anchor and a staging of Poincaré--Lefschetz duality.
//!
//! Run with `-i` / `--interactive` to instead load an external mesh (`.msh` or
//! `.obj`) and read the grade and eigenvalue count from stdin — the way to put
//! the solver on a curved domain such as the sphere, where the grade-0 spectrum
//! is $ell(ell + 1)$.
//!
//! Run by hand; read the spectra off the tables.

#[path = "util/mod.rs"]
mod util;

use {
  common::util::algebraic_convergence_rate,
  formoniq::{problems::hodge_laplace, whitney_complex::WhitneyComplex},
  manifold::gen::cartesian::CartesianMeshInfo,
  util::{report, BoundaryCondition},
};

use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  tracing_subscriber::fmt::init();

  let interactive = std::env::args()
    .nth(1)
    .is_some_and(|arg| arg == "-i" || arg == "--interactive");

  if interactive {
    interactive_mesh()
  } else {
    box_sweep();
    Ok(())
  }
}

/// The default: the mixed Hodge-Laplace spectrum on the flat box, swept over all
/// dimensions and grades, with each eigenvalue's self-convergence rate.
fn box_sweep() {
  let neigen = 6;
  // Dimension 0 is the base case: the box is a single point, the de Rham complex
  // is $RR$ in degree 0, the Hodge Laplacian is the zero operator, and the whole
  // spectrum is ${0}$ with multiplicity $b_0 = 1$. It carries no convergence (a
  // point does not refine), but it runs on exactly the same code.
  for dim in 0..=3 {
    for grade in 0..=dim {
      // Both boundary conditions: absolute (natural / Neumann) on the full
      // Whitney complex, relative (essential / Dirichlet) on its
      // boundary-vanishing subcomplex. The zero-eigenvalue count is the harmonic
      // dimension — $b_k (K)$ for absolute, $b_k (K, diff K)$ for relative — so
      // the harmonic mode sits at grade $0$ for absolute and top grade for
      // relative, staging Poincaré--Lefschetz duality. On the point ($n = 0$)
      // the boundary is empty and the two coincide.
      for bc in [BoundaryCondition::Absolute, BoundaryCondition::Relative] {
        let relative = bc == BoundaryCondition::Relative;
        println!(
          "\nHodge-Laplace spectrum — dim {dim}, grade {grade}, {}",
          bc.label()
        );
        println!(
          "| {:>2} | {:>7} | lowest {neigen} eigenvalues (self-conv rate)",
          "r", "ncells"
        );

        // Every grade solves a dense generalized eigenproblem, $O(N^3)$ in the
        // mixed-system size $N$, so refine only while that stays affordable and
        // stop at the first level that would exceed the budget. In 3D this leaves
        // just a couple of levels; the source example, on the sparse LU, carries
        // the finer convergence story.
        const MAX_DOFS: usize = 1200;

        // History of the lowest eigenvalues per level, for the three-point
        // Richardson estimate of each eigenvalue's convergence order.
        let mut history: Vec<Vec<f64>> = Vec::new();
        let mut prev_ndofs = 0;
        for irefine in 0u32..=8 {
          let nboxes_per_dim = 2usize.pow(irefine);
          let box_mesh = CartesianMeshInfo::new_unit_scaled(dim, nboxes_per_dim, PI);
          let (topology, coords) = box_mesh.compute_coord_complex();
          let metric = coords.to_edge_lengths(&topology);
          let whitney = WhitneyComplex::new(&topology, &metric);

          let ndofs = whitney.ndofs(grade)
            + if grade > 0 {
              whitney.ndofs(grade - 1)
            } else {
              0
            };
          // Stop once the dense solve would exceed the budget, or once refinement
          // no longer grows the mesh — a 0-manifold is a single point and does not
          // subdivide, so it has exactly one level.
          if !history.is_empty() && (ndofs > MAX_DOFS || ndofs == prev_ndofs) {
            break;
          }
          prev_ndofs = ndofs;

          let (eigenvals, _, _) = if relative {
            hodge_laplace::solve_hodge_laplace_evp(&whitney.relative(), grade, neigen)
          } else {
            hodge_laplace::solve_hodge_laplace_evp(&whitney, grade, neigen)
          };
          // The coarsest levels can carry fewer DOFs than the requested count; the
          // eigensolver pads the rest with non-finite values.
          let eigenvals: Vec<f64> = eigenvals
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .collect();

          // Richardson: with three successive levels the ratio of consecutive
          // differences estimates the order without knowing the exact eigenvalue.
          // FEEC eigenvalues converge as $O(h^2)$, so expect a rate near 2.
          let rates: Option<Vec<f64>> = match history.as_slice() {
            [.., older, newer] => Some(
              (0..eigenvals.len().min(newer.len()).min(older.len()))
                .map(|i| {
                  let (d_old, d_new) =
                    ((newer[i] - older[i]).abs(), (eigenvals[i] - newer[i]).abs());
                  algebraic_convergence_rate(d_new, d_old)
                })
                .collect(),
            ),
            _ => None,
          };

          let ncells = topology.cells().len();
          print!("| {irefine:>2} | {ncells:>7} |");
          for (i, &lambda) in eigenvals.iter().enumerate() {
            // A near-zero eigenvalue is a harmonic mode: FEEC captures it exactly,
            // so a "convergence rate" would only compare solver noise — mark it,
            // and the coarsest levels with no prior triple, absent.
            let rate = rates
              .as_ref()
              .and_then(|rates| rates.get(i).copied())
              .filter(|_| lambda.abs() > 1e-6);
            print!(" {:>7}({:>6})", report::eigval(lambda), report::rate(rate));
          }
          println!();

          history.push(eigenvals);
        }
      }
    }
  }
}

/// Load an external mesh and solve the eigenproblem on it, reading the grade and
/// eigenvalue count from stdin. The route to a curved domain: the solver is
/// intrinsic, so the mesh's own edge lengths are all it consumes.
fn interactive_mesh() -> Result<(), Box<dyn std::error::Error>> {
  let prompt = |msg: &str| -> Result<String, Box<dyn std::error::Error>> {
    println!("{msg}");
    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    Ok(line.trim().to_string())
  };

  let path = std::path::PathBuf::from(prompt("Enter mesh file path (.msh or .obj).")?);
  let (base_surface, topology, coords) = match path.extension().and_then(|e| e.to_str()) {
    Some("msh") => {
      let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&std::fs::read(path)?);
      (None, topology, coords)
    }
    Some("obj") => {
      let surface = manifold::io::blender::from_obj_string(&String::from_utf8(std::fs::read(path)?)?);
      let (topology, coords) = surface.clone().into_coord_complex();
      (Some(surface), topology, coords)
    }
    _ => return Err("Unknown or missing file extension.".into()),
  };
  let metric = coords.to_edge_lengths(&topology);

  let grade: usize = prompt("Enter exterior grade.")?.parse()?;
  let neigen: usize = prompt("Enter number of eigenvalues.")?.parse()?;

  let (eigenvals, _, eigen_us) =
    hodge_laplace::solve_hodge_laplace_evp(&WhitneyComplex::new(&topology, &metric), grade, neigen);
  for (i, &lambda) in eigenvals.iter().enumerate() {
    println!("eigenvalue {i}: {lambda:.4}");
  }

  if let (Some(surface), 0) = (base_surface, grade) {
    let out_dir = "out/hodge_laplace_evp";
    std::fs::create_dir_all(out_dir)?;

    let do_animation = prompt("Generate breathing animation? (y/n)")?.to_lowercase() == "y";

    for (i, &lambda) in eigenvals.iter().enumerate() {
      let eigenfunc = eigen_us.column(i);

      let mut mode_surface = surface.clone();
      for (ivertex, mut cart_pos) in mode_surface.vertex_coords_mut().coord_iter_mut().enumerate() {
        cart_pos *= eigenfunc[ivertex];
      }
      std::fs::write(
        format!("{out_dir}/eigenmode{i}.obj"),
        manifold::io::blender::to_obj_string(&mode_surface),
      )?;

      if do_animation {
        let omega = lambda.abs().sqrt();
        let fps = 30;
        let duration = if omega > 1e-6 {
          std::f64::consts::TAU / omega
        } else {
          1.0
        };
        let nframes = (duration * fps as f64).ceil() as usize;
        let times: Vec<_> = (0..=nframes).map(|frame| frame as f64 / fps as f64).collect();

        let displacements: Vec<_> = times
          .iter()
          .map(|&t| (eigenfunc * (omega * t).cos()).into_owned())
          .collect();

        manifold::io::blender::write_displacement_animation(
          format!("{out_dir}/breathing{i}.mdd"),
          &surface,
          &displacements,
          times.iter().copied(),
        );
      }
    }
  }

  Ok(())
}

