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
//! Run with `-i` / `--interactive` to instead load an external mesh (`.msh`)
//! and read the grade and eigenvalue count from stdin — the way to put
//! the solver on a curved domain such as the sphere, where the grade-0 spectrum
//! is $ell(ell + 1)$.
//!
//! Run by hand; read the spectra off the tables.

#[path = "util/mod.rs"]
mod util;

use {
  formoniq::{problems::elliptic, whitney_complex::WhitneyComplex},
  simplicial::gen::cartesian::CartesianGrid,
  util::{algebraic_convergence_rate, report, BoundaryCondition},
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
  // Both boundary conditions: absolute (natural / Neumann) on the full
  // Whitney complex, relative (essential / Dirichlet) on its
  // boundary-vanishing subcomplex. The zero-eigenvalue count is the harmonic
  // dimension — $b_k (K)$ for absolute, $b_k (K, diff K)$ for relative — so
  // the harmonic mode sits at grade $0$ for absolute and top grade for
  // relative, staging Poincaré--Lefschetz duality. On the point ($n = 0$)
  // the boundary is empty and the two coincide.
  const BCS: [BoundaryCondition; 2] = [BoundaryCondition::Absolute, BoundaryCondition::Relative];

  for dim in 0..=3 {
    for grade in 0..=dim {
      // The eigensolve is a sparse shift-invert Lanczos, so the budget is set
      // by wall-clock patience for a hand-run sweep, not by a dense O(N^3)
      // ceiling.
      const MAX_DOFS: usize = 20_000;

      // Rows are gathered per BC and printed as grouped tables afterwards, so
      // the mesh and Whitney complex — independent of `bc` — are built once
      // per refinement and shared by both BCs, not duplicated.
      // History of the lowest eigenvalues per level, for the three-point
      // Richardson estimate of each eigenvalue's convergence order.
      let mut history: [Vec<Vec<f64>>; 2] = [const { Vec::new() }; 2];
      let mut rows: [Vec<String>; 2] = [const { Vec::new() }; 2];
      let mut prev_ndofs = 0;
      // Freudenthal refinement of one coarse box generates the h-family: nested
      // meshes, and the mesh-agnostic path rather than the structured generator.
      // Each level refines the base $R$-fold rather than iterating the previous
      // level, because Freudenthal children are not similar to their parent above
      // dimension two — a single $R$-fold refinement of a Kuhn box reproduces the
      // generator's grid cell for cell, an iterated one does not.
      let (base_topology, base_coords) = CartesianGrid::new_unit_scaled(dim, 1, PI).triangulate();
      let base_metric = base_coords.to_edge_lengths_sq(&base_topology);
      for irefine in 0u32..=8 {
        let sub = base_topology.refine(2usize.pow(irefine));
        let metric = base_metric.refine(&sub, &base_topology);
        let topology = sub.into_complex();
        let whitney = WhitneyComplex::new(&topology, &metric);

        let ndofs = whitney.ndofs(grade)
          + if grade > 0 {
            whitney.ndofs(grade - 1)
          } else {
            0
          };
        // Stop once the solve would exceed the budget, or once refinement
        // no longer grows the mesh — a 0-manifold is a single point and does not
        // subdivide, so it has exactly one level.
        if !history[0].is_empty() && (ndofs > MAX_DOFS || ndofs == prev_ndofs) {
          break;
        }
        prev_ndofs = ndofs;

        let ncells = topology.cells().len();
        for (i, bc) in BCS.into_iter().enumerate() {
          let relative = bc == BoundaryCondition::Relative;
          let (eigenvals, _, _) = if relative {
            elliptic::solve_evp(&whitney.relative(), grade, neigen).unwrap()
          } else {
            elliptic::solve_evp(&whitney, grade, neigen).unwrap()
          };
          // The coarsest levels can carry fewer DOFs than the requested count,
          // and then the eigensolver returns fewer pairs.
          let eigenvals: Vec<f64> = eigenvals.iter().copied().collect();

          // Richardson: with three successive levels the ratio of consecutive
          // differences estimates the order without knowing the exact eigenvalue.
          // FEEC eigenvalues converge as $O(h^2)$, so expect a rate near 2.
          let rates: Option<Vec<f64>> = match history[i].as_slice() {
            [.., older, newer] => Some(
              (0..eigenvals.len().min(newer.len()).min(older.len()))
                .map(|j| {
                  let (d_old, d_new) =
                    ((newer[j] - older[j]).abs(), (eigenvals[j] - newer[j]).abs());
                  algebraic_convergence_rate(d_new, d_old)
                })
                .collect(),
            ),
            _ => None,
          };

          let mut row = format!("| {irefine:>2} | {ncells:>7} |");
          for (j, &lambda) in eigenvals.iter().enumerate() {
            // A near-zero eigenvalue is a harmonic mode: FEEC captures it exactly,
            // so a "convergence rate" would only compare solver noise — mark it,
            // and the coarsest levels with no prior triple, absent.
            let rate = rates
              .as_ref()
              .and_then(|rates| rates.get(j).copied())
              .filter(|_| lambda.abs() > 1e-6);
            row.push_str(&format!(
              " {:>7}({:>6})",
              report::eigval(lambda),
              report::rate(rate)
            ));
          }
          rows[i].push(row);

          history[i].push(eigenvals);
        }
      }

      for (i, bc) in BCS.into_iter().enumerate() {
        println!(
          "\nHodge-Laplace spectrum — dim {dim}, grade {grade}, {}",
          bc.label()
        );
        println!(
          "| {:>2} | {:>7} | lowest {neigen} eigenvalues (self-conv rate)",
          "r", "ncells"
        );
        for row in &rows[i] {
          println!("{row}");
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

  let path = std::path::PathBuf::from(prompt("Enter mesh file path (.msh).")?);
  let (topology, coords) = match path.extension().and_then(|e| e.to_str()) {
    Some("msh") => simplicial::io::gmsh::gmsh2coord_complex(&std::fs::read(path)?),
    _ => return Err("Unknown or missing file extension.".into()),
  };
  let metric = coords.to_edge_lengths_sq(&topology);

  let grade: usize = prompt("Enter exterior grade.")?.parse()?;
  let neigen: usize = prompt("Enter number of eigenvalues.")?.parse()?;

  let (eigenvals, _, _) =
    elliptic::solve_evp(&WhitneyComplex::new(&topology, &metric), grade, neigen)?;
  for (i, &lambda) in eigenvals.iter().enumerate() {
    println!("eigenvalue {i}: {lambda:.4}");
  }

  Ok(())
}
