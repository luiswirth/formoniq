//! Hodge-Laplace eigenvalue convergence in the *polynomial degree*.
//!
//! The same mixed eigenvalue problem as `evp`, on the trimmed complex
//! $P^-_r Lambda^k$ at $r = 1, 2, 3$.
//!
//! Down a column, the rate: FEEC eigenvalues converge as $O(h^(2r))$, so expect
//! roughly $2$, $4$ and $6$. Across columns at comparable dof counts, the error.
//!
//! The domain is the box $[0, pi]^n$ under the absolute boundary condition.
//! Comparing against the exact eigenvalue rather than a Richardson
//! extrapolation matters: at $r = 3$ the error reaches roundoff within a couple
//! of refinements, which a self-convergence estimate cannot tell from a bug.
//!
//! Run by hand; read the rates off the tables.

#[path = "util/mod.rs"]
mod util;

use {
  formoniq::{
    problems::elliptic, trimmed_complex::TrimmedComplex, whitney_complex::HilbertComplex,
  },
  simplicial::{mesher::cartesian::CartesianGrid, topology::ordering::CellOrdering},
  util::algebraic_convergence_rate,
};

/// The lowest nonzero Hodge-Laplace eigenvalue on the box $[0, pi]^n$ under the
/// absolute boundary condition, at grade $k$.
///
/// The Hodge Laplacian on the box splits into scalar Laplacians, one per
/// component $dif x^I$, the absolute condition putting a Dirichlet factor on
/// each direction in $I$ and a Neumann factor on each direction outside it. An
/// eigenvalue is $sum_i m_i^2$ with $m_i >= 1$ for $i in I$ and $m_i >= 0$
/// otherwise, so the smallest is $k$.
///
/// At grade $0$ that gives $0$, the harmonic mode rather than a Laplace
/// eigenvalue since the box is contractible, and the smallest nonzero one is
/// $1$. Hence the maximum.
fn lowest_nonzero_eigenvalue(grade: usize) -> f64 {
  grade.max(1) as f64
}

use std::f64::consts::PI;

/// Past this the shift-invert solve stops being a thing to run by hand.
const MAX_DOFS: usize = 30_000;
/// The degrees to compare. Three is enough to see a rate triple.
const DEGREES: [usize; 3] = [1, 2, 3];

fn main() {
  tracing_subscriber::fmt::init();

  for dim in 1..=3 {
    for grade in 0..=dim {
      let exact = lowest_nonzero_eigenvalue(grade);
      println!(
        "\nHodge-Laplace eigenvalue — dim {dim}, grade {grade}, absolute — exact λ = {exact}"
      );
      println!("|  r |  ncells |   ndofs |         λ |     err | rate |");

      for degree in DEGREES {
        let (mut topology, coords) = CartesianGrid::new_unit_scaled(dim, 1, PI).triangulate();
        let mut metric = coords.to_edge_lengths_sq(&topology);
        let mut ordering = CellOrdering::colex(&topology);

        let mut previous: Option<f64> = None;
        for irefine in 0u32..=6 {
          if irefine > 0 {
            let sub = topology.refine_with(&ordering, 2);
            metric = metric.refine(&sub, &topology);
            ordering = sub.ordering().clone();
            topology = sub.into_complex();
          }

          let complex = TrimmedComplex::new(&topology, &metric, degree);
          let ndofs = complex.ndofs(grade)
            + if grade > 0 {
              complex.ndofs(grade - 1)
            } else {
              0
            };
          if irefine > 0 && ndofs > MAX_DOFS {
            break;
          }

          // Past the harmonic sector, whose eigenvalues are exactly zero.
          let harmonic = usize::from(grade == 0);
          // A coarse level can carry fewer dofs than eigenpairs asked for.
          let Ok((eigenvals, _, _)) = elliptic::solve_evp(&complex, grade, harmonic + 3) else {
            continue;
          };
          let Some(&lambda) = eigenvals.iter().nth(harmonic) else {
            continue;
          };

          let error = (lambda - exact).abs();
          let rate = previous.map(|previous| algebraic_convergence_rate(error, previous));
          previous = Some(error);

          println!(
            "| {degree:2} | {:7} | {ndofs:7} | {lambda:9.5} | {error:7.1e} | {} |",
            topology.cells().len(),
            rate.map_or("   — ".to_string(), |rate| format!("{rate:5.2}"))
          );
        }
      }
    }
  }
}
