//! Linear advection: transport of a differential form along a prescribed
//! vector field.

use crate::{
  assemble::assemble_galmat,
  operators::{HodgeMassElmat, LieDerivativeElmat},
  time::{LinearIrk, Tableau},
};

/// How the transport is posed: the space it acts on and the field it follows.
pub struct Transport<'a, V> {
  pub grade: ExteriorGrade,
  pub velocity: &'a V,
  /// Exact for a velocity of polynomial degree $p$ at $2 + p$; the boundary
  /// integrand of two affine shape functions is the binding side.
  pub quad_degree: usize,
}

use derham::{Cochain, section::Section};
use multialgebra::ExteriorGrade;
use regge::lengths::mesh::MeshLengthsSq;
use simplicial::{
  linalg::{CsrMatrix, Vector},
  topology::complex::Complex,
};

/// The mass matrix and the discrete Lie derivative of the semidiscrete
/// transport system $M dot(u) = -A u$.
///
/// $A$ is the central discretization: conservative, and dispersive for it.
/// See [`LieDerivativeElmat`].
pub fn assemble_transport<V: Sync + Section>(
  topology: &Complex,
  geometry: &MeshLengthsSq,
  transport: &Transport<V>,
) -> (CsrMatrix, CsrMatrix) {
  let mass = assemble_galmat(
    topology,
    geometry,
    HodgeMassElmat::new(topology.dim(), transport.grade),
  );
  let lie = assemble_galmat(
    topology,
    geometry,
    LieDerivativeElmat::new(transport.velocity, transport.grade, transport.quad_degree),
  );
  (CsrMatrix::from(&mass), CsrMatrix::from(&lie))
}

/// $diff_t omega + cal(L)_v omega = 0$ on Whitney $k$-forms of any grade,
/// stepped with Gauss-Legendre.
///
/// Gauss-Legendre and not Radau: it is non-dissipative, so it neither damps nor
/// amplifies what the space discretization does, and the $L^2$ history of the
/// solution then reports on $A$ alone. A stiffly accurate rule would hide
/// exactly the oscillation this is built to expose.
///
/// No boundary condition is imposed. On a mesh with boundary the operator's own
/// facet terms act there, which is an outflow-like condition and not a
/// prescribed inflow, so a solution transported into the boundary is not
/// meaningful. Run this on a closed manifold, or stop before the feature
/// reaches the boundary.
pub fn solve_transport<V: Sync + Section>(
  topology: &Complex,
  geometry: &MeshLengthsSq,
  transport: &Transport<V>,
  nsteps: usize,
  dt: f64,
  initial: &Cochain,
) -> Vec<Cochain> {
  let grade = transport.grade;
  let (mass, lie) = assemble_transport(topology, geometry, transport);
  let irk = LinearIrk::new(Tableau::gauss_legendre(2), &mass, -lie, dt);

  let mut u = initial.coeffs().clone();
  let mut solution = Vec::with_capacity(nsteps + 1);
  solution.push(initial.clone());
  for istep in 0..nsteps {
    u = irk.step(&u, istep as f64 * dt, |_| Vector::zeros(u.len()));
    solution.push(Cochain::new(grade, u.clone()));
  }
  solution
}
