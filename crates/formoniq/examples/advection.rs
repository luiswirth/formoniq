//! Linear advection $diff_t omega + cal(L)_v omega = 0$ on the flat box, with
//! the central discretization, to measure what it does wrong.
//!
//! Cartan's $cal(L)_v = iota_v dif + dif iota_v$ makes one operator out of the
//! classical pair: advective form at $k = 0$, conservation form at $k = n$, the
//! transport of circulation and flux between. Every grade of every dimension
//! runs through the same code.
//!
//! Not a convergence rate, which a smooth resolved solution passes while
//! oscillating. Instead the $L^2$ drift, zero under exact transport of a
//! divergence-free field and left alone by non-dissipative Gauss-Legendre, and
//! the range violation at grade 0, which transport cannot produce.
//!
//! The velocity is a genuine constant field, the sharp of a constant $1$-form
//! pulled back onto the mesh; constant components in every reference frame
//! would be a different and discontinuous field. No boundary condition is
//! imposed, so the bump stops short of the boundary.
//!
//! The drift turns out to be structural in the grade. The antisymmetry defect
//! $integral_(diff K) inner(omega, eta) iota_v vol$ vanishes wherever
//! neighboring facet terms cancel, which happens at both ends and nowhere
//! between: the shape functions are continuous at $k = 0$, and constant per
//! cell at $k = n$, where what is left is $integral_K div v$. So the classical
//! two are stable for free, and the transport of circulation and flux between
//! them is where an unstabilized scheme fails.

use {
  derham::{
    project::derham_map,
    section::{CoordFieldExt, SharpOp},
  },
  formoniq::problems::advection::{Transport, assemble_transport, solve_transport},
  glatt::field::DiffFormClosure,
  multialgebra::{Tensor, Variance},
  regge::mesher::cartesian::CartesianGrid,
  simplicial::{Dim, linalg::Vector},
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
        move |_x| Tensor::line(Vector::from_vec(direction.clone()), Variance::Covariant)
      },
      dim,
      Dim::ONE,
    );
    let velocity = velocity_form
      .pullback_on(&topology, &coords)
      .sharp(&topology, &geometry);

    // A bump at 0.3 traveling 0.2, so it stays well inside the box.
    let travel = 0.2;
    let bump = |x: &Vector| {
      let r2: f64 = x.iter().map(|xi| (xi - 0.3).powi(2)).sum();
      (-r2 / (2.0 * 0.12f64.powi(2))).exp()
    };

    for grade in dim.range_inclusive() {
      let ncomponents = multialgebra::exterior_dim(dim, grade);
      let field = DiffFormClosure::new(
        move |x| {
          let coeffs = Vector::from_element(ncomponents, bump(x.vector()));
          Tensor::multiform(coeffs, dim, grade)
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
