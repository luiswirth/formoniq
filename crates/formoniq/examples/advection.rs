//! Linear advection $diff_t omega + cal(L)_v omega = 0$ on the flat box, run
//! with the *central* (unstabilized) discretization, to measure what it does
//! wrong.
//!
//! Cartan's formula $cal(L)_v = iota_v dif + dif iota_v$ makes one operator out
//! of the classical pair: at $k = 0$ it is the advective form $v dot nabla f$,
//! at $k = n$ the conservation form $nabla dot (v rho)$, and in between the
//! transport of circulation and of flux. The loop below runs every grade of
//! every dimension through the same code.
//!
//! Two things are read off, and neither is a convergence rate, because a smooth
//! resolved solution converges perfectly well here and would say nothing:
//!
//! - **$L^2$ history.** A constant $v$ on a flat domain is divergence-free and
//!   Killing, so exact transport conserves $norm(omega)_(L^2)$. Gauss-Legendre
//!   is non-dissipative, so any drift is the space discretization's: growth is
//!   instability, decay is numerical diffusion.
//! - **Range violation at grade 0.** Transport moves values around and creates
//!   none, so $max omega_h$ must not exceed $max omega_0$. The overshoot is the
//!   oscillation, in the units of the transported quantity.
//!
//! The velocity is a genuine constant field: the sharp of a constant $1$-form
//! pulled back onto the mesh, so the metric enters at the musical isomorphism
//! where it belongs and $iota_v$ stays metric-free. Constant *components in
//! every reference frame* would be a different and discontinuous field.
//!
//! No boundary condition is imposed, so the run stops before the bump reaches
//! the boundary; the reported window is transport in the interior alone.
//!
//! What comes out is that the drift is *structural in the grade*, and the
//! antisymmetry defect $integral_(diff K) inner(omega, eta) iota_v vol$ says
//! why. It vanishes wherever the facet terms of neighboring cells cancel, and
//! that happens at both ends of the grade range and nowhere between:
//!
//! - At $k = 0$ the Whitney forms are the continuous hat functions, so
//!   $inner(omega, eta)$ is single-valued on a facet and the two sides cancel.
//! - At $k = n$ they are constant per cell, so $inner(omega, eta)$ comes out of
//!   the integral and what is left is $integral_(diff K) iota_v vol =
//!   integral_K div v = 0$.
//! - In between only the *tangential* part is single-valued, the normal part
//!   jumps, and nothing cancels.
//!
//! So the classical two, advective and conservation form, are the two grades
//! that are stable for free, and the transport of circulation and flux between
//! them is where an unstabilized scheme actually fails.

use {
  derham::{
    project::derham_map,
    section::{CoordFieldExt, SharpOp},
  },
  exterior::{ExteriorElement, MultiForm},
  formoniq::problems::advection::{Transport, assemble_transport, solve_transport},
  glatt::field::DiffFormClosure,
  simplicial::{Dim, linalg::Vector, mesher::cartesian::CartesianGrid},
};

fn main() {
  println!("Linear advection, central scheme. Unit box, a bump carried 0.2 at unit speed.");
  println!(
    "{:>3} {:>7} {:>5} {:>11} {:>12} {:>12}",
    "dim", "ncells", "grade", "|w|_L2 t=0", "drift %", "overshoot"
  );

  for dim in (1..=3).map(Dim::from) {
    let ncells = match dim.index() {
      1 => 64,
      2 => 24,
      _ => 8,
    };
    let (topology, coords) = CartesianGrid::new_unit(dim, ncells).triangulate();
    let geometry = coords.to_edge_lengths_sq(&topology);

    // A constant vector field of unit speed, reached as the sharp of a
    // constant 1-form, so the travel distance is the elapsed time.
    let raw: Vec<f64> = (0..dim.index()).map(|i| 1.0 / (1.0 + i as f64)).collect();
    let speed = raw.iter().map(|c| c * c).sum::<f64>().sqrt();
    let direction: Vec<f64> = raw.iter().map(|c| c / speed).collect();
    let velocity_form = DiffFormClosure::new(
      {
        let direction = direction.clone();
        move |_x| MultiForm::line(Vector::from_vec(direction.clone()))
      },
      dim,
      Dim::ONE,
    );
    let velocity = velocity_form
      .pullback_on(&topology, &coords)
      .sharp(&topology, &geometry);

    // A bump at 0.3 travelling 0.2, so it stays well inside the box.
    let travel = 0.2;
    let bump = |x: &Vector| {
      let r2: f64 = x.iter().map(|xi| (xi - 0.3).powi(2)).sum();
      (-r2 / (2.0 * 0.12f64.powi(2))).exp()
    };

    for grade in dim.range_inclusive() {
      let ncomponents = exterior::exterior_dim(dim, grade);
      let field = DiffFormClosure::new(
        move |x| {
          let coeffs = Vector::from_element(ncomponents, bump(x.vector()));
          ExteriorElement::new(coeffs, dim, grade)
        },
        dim,
        grade,
      );
      let initial = derham_map(&field.pullback_on(&topology, &coords), &topology, 3);

      // A quarter of a cell per step, so the time error stays under the space
      // error and the drift reported is the space discretization's.
      let h = 1.0 / ncells as f64;
      let nsteps = (travel / (0.25 * h)).ceil() as usize;
      let dt = travel / nsteps as f64;
      let transport = Transport {
        grade,
        velocity: &velocity,
        quad_degree: 2,
      };
      let solution = solve_transport(&topology, &geometry, &transport, nsteps, dt, &initial);

      let (mass, _) = assemble_transport(&topology, &geometry, &transport);
      let l2 = |c: &derham::cochain::Cochain| {
        formoniq::linalg::quadratic_form_sparse(&mass, c.coeffs()).sqrt()
      };

      let initial_norm = l2(&solution[0]);
      let final_norm = l2(solution.last().unwrap());
      let drift = 100.0 * (final_norm - initial_norm) / initial_norm;

      let start_max = solution[0].coeffs().max();
      let end_max = solution.last().unwrap().coeffs().max();
      let overshoot = (end_max - start_max).max(0.0);

      println!(
        "{:>3} {:>7} {:>5} {:>11.4e} {:>+12.4} {:>12.4e}",
        dim.index(),
        ncells,
        grade.index(),
        initial_norm,
        drift,
        overshoot
      );
    }
  }
}
