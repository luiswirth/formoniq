//! Hodge-Laplace eigenvalue problem.
//!
//! By default, a generic sweep over every dimension $0 <= n <= 3$, every form
//! grade $0 <= k <= n$, and three cases: the contractible box $[0, pi]^n$ under
//! each of its two boundary conditions, and the closed flat torus, which has no
//! boundary and hence only one problem. The lowest eigenvalues of the mixed
//! Hodge Laplacian, refined until their algebraic self-convergence rate shows.
//!
//! The count of zero eigenvalues is the harmonic dimension, an exact topological
//! prediction the discrete problem reproduces at every resolution: absolute
//! $b_k (K)$ ($1$ at grade $0$) against relative $b_k (K, diff K)$ ($1$ at top
//! grade) on the box, staging Poincaré--Lefschetz duality, and $b_k (T^d) =
//! binom(d, k)$ on the torus — the only case where it is neither $0$ nor $1$,
//! and so the only one that tests the harmonic sector rather than anchoring it.
//! A row whose count disagrees is marked `!`.
//!
//! One loop body serves all three. A closed manifold's boundary subcomplex is
//! empty, so the relative problem there *is* the absolute one: the torus is not
//! a case the code excludes but one it cannot tell apart.
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
  util::{algebraic_convergence_rate, report, BoundaryCondition, Manifold},
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
    sweep();
    Ok(())
  }
}

/// The default: the mixed Hodge-Laplace spectrum swept over all dimensions,
/// grades and manifolds, with each eigenvalue's self-convergence rate.
fn sweep() {
  let neigen = 6;
  // Dimension 0 is the base case: the box is a single point, the de Rham complex
  // is $RR$ in degree 0, the Hodge Laplacian is the zero operator, and the whole
  // spectrum is ${0}$ with multiplicity $b_0 = 1$. It carries no convergence (a
  // point does not refine), but it runs on exactly the same code.
  //
  // Three cases, one loop body. The box carries a boundary and hence the two
  // dual conditions: absolute (natural / Neumann) on the full Whitney complex,
  // relative (essential / Dirichlet) on its boundary-vanishing subcomplex, so
  // the harmonic mode sits at grade $0$ for one and top grade for the other,
  // staging Poincare--Lefschetz duality. The torus is closed, so its boundary
  // subcomplex is empty and the two conditions *are* the same problem -- not a
  // case the code excludes but one it cannot tell apart, which is the totality
  // claim rather than a special path. What the torus adds is a harmonic sector
  // that is genuinely large: $b_k (T^d) = binom(d, k)$, against $0$ or $1$ on the
  // contractible box, so it is the only case here that tests the harmonic part
  // of the mixed solve rather than merely anchoring it.
  let cases: [(Manifold, BoundaryCondition); 3] = [
    (Manifold::Box, BoundaryCondition::Absolute),
    (Manifold::Box, BoundaryCondition::Relative),
    (Manifold::Torus, BoundaryCondition::Absolute),
  ];

  for dim in 0..=3 {
    for grade in 0..=dim {
      // The eigensolve is a sparse shift-invert Lanczos, so the budget is set
      // by wall-clock patience for a hand-run sweep, not by a dense O(N^3)
      // ceiling.
      const MAX_DOFS: usize = 20_000;

      // Each case owns its mesh, since the manifolds differ; the loop body that
      // refines and solves does not.
      for &(manifold, bc) in &cases {
        // A torus needs a dimension to be a torus: $T^0$ is a point, already
        // covered by the box's base case.
        if manifold == Manifold::Torus && dim == 0 {
          continue;
        }
        let harmonic_dim = manifold.harmonic_dim(dim, grade, bc);
        // History of the lowest eigenvalues per level, for the three-point
        // Richardson estimate of each eigenvalue's convergence order.
        let mut history: Vec<Vec<f64>> = Vec::new();
        let mut rows: Vec<String> = Vec::new();
        let mut prev_ndofs = 0;

        let (mut topology, mut metric, mut ordering) = manifold.coarse_mesh(dim, PI);
        for irefine in 0u32..=8 {
          if irefine > 0 {
            let sub = topology.refine_with(&ordering, 2);
            metric = metric.refine(&sub, &topology);
            ordering = sub.ordering().clone();
            topology = sub.into_complex();
          }
          let whitney = WhitneyComplex::new(&topology, &metric);

          let ndofs = whitney.ndofs(grade)
            + if grade > 0 {
              whitney.ndofs(grade - 1)
            } else {
              0
            };
          // Stop once the solve would exceed the budget, or once refinement no
          // longer grows the mesh — a 0-manifold is a single point and does not
          // subdivide, so it has exactly one level.
          if !history.is_empty() && (ndofs > MAX_DOFS || ndofs == prev_ndofs) {
            break;
          }
          prev_ndofs = ndofs;

          let ncells = topology.cells().len();
          // The relative problem is posed on the boundary-vanishing subcomplex.
          // On a closed manifold that subcomplex is the whole complex, so this
          // branch is not a case distinction the torus escapes — it is one that
          // collapses on it.
          let (eigenvals, _, _) = if bc == BoundaryCondition::Relative {
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
          let rates: Option<Vec<f64>> = match history.as_slice() {
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

          // The harmonic dimension is exact, not asymptotic: FEEC reproduces it
          // at every resolution, so a mismatch is a defect rather than a coarse
          // mesh. Flagged in the row instead of asserted, since the examples are
          // read by hand.
          let nzero = eigenvals.iter().filter(|l| l.abs() <= 1e-6).count();
          let harmonic_flag = if nzero == harmonic_dim.min(eigenvals.len()) {
            ' '
          } else {
            '!'
          };

          let mut row = format!("|{harmonic_flag}{irefine:>2} | {ncells:>7} |");
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
          rows.push(row);

          history.push(eigenvals);
        }

        println!(
          "\nHodge-Laplace spectrum — {} dim {dim}, grade {grade}, {} — harmonic dim {harmonic_dim}",
          manifold.label(),
          bc.label()
        );
        println!(
          "| {:>2} | {:>7} | lowest {neigen} eigenvalues (self-conv rate)",
          "r", "ncells"
        );
        for row in &rows {
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
