//! Boundary conditions for the Whitney complex.
//!
//! Essential (Dirichlet) conditions constrain the trace $"tr" u = g$:
//! homogeneous ones restrict to the relative complex (the kernel of the
//! trace), inhomogeneous ones are reduced to homogeneous by the affine
//! lifting $u = hat(g) + E u_0$ with the zero-extension $hat(g) = "tr"^T g$.
//!
//! Natural (Neumann) conditions add a boundary load to the right-hand side
//! ([`neumann_load`]); homogeneous ones are "do nothing". Robin conditions
//! add the boundary mass ([`boundary_mass`]). Mixed conditions partition the
//! boundary facets and combine the two ([`WhitneyComplex::relative_to`]).
//!
//! [`WhitneyComplex::relative_to`]: crate::whitney_complex::WhitneyComplex::relative_to

use crate::{
  assemble::assemble_galvec,
  operators::SourceElVec,
  whitney_complex::{BoundaryWhitneyComplex, RelativeWhitneyComplex},
};

use {
  common::linalg::{
    faer::FaerCholesky,
    nalgebra::{CsrMatrix, Vector},
  },
  ddf::{cochain::Cochain, section::Section},
  exterior::{Covariant, ExteriorGrade},
  manifold::atlas::SimplexQuadRule,
};

/// A Galerkin system with essential boundary conditions imposed by affine
/// lifting:
///
/// $u = hat(g) + E u_0, quad E^T A E med u_0 = E^T (f - A hat(g)), quad hat(g) = "tr"^T g$
///
/// Holds the factorization of the constrained system $E^T A E$, so repeated
/// solves (time stepping) pay the lifting only once per right-hand side.
pub struct LiftedSystem {
  system: CsrMatrix,
  inclusion: CsrMatrix,
  lift: Vector,
  cholesky: FaerCholesky,
  grade: usize,
}

impl LiftedSystem {
  /// `system` and later right-hand sides are those of the unconstrained
  /// full space; `boundary_values` is a cochain on the trace complex.
  pub fn new(
    relative: &RelativeWhitneyComplex,
    boundary: &BoundaryWhitneyComplex,
    system: CsrMatrix,
    boundary_values: &Cochain,
  ) -> Self {
    let grade = boundary_values.grade();
    let lift = boundary.extend_cochain(boundary_values).into_coeffs();
    let inclusion = relative.inclusion(grade);
    let system_relative = inclusion.transpose() * &system * &inclusion;
    let cholesky = FaerCholesky::new(system_relative);
    Self {
      system,
      inclusion,
      lift,
      cholesky,
      grade,
    }
  }

  pub fn solve(&self, rhs: &Vector) -> Cochain {
    let rhs_relative = self.inclusion.transpose() * (rhs - &self.system * &self.lift);
    let solution_relative = self.cholesky.solve(&rhs_relative);
    Cochain::new(self.grade, &self.inclusion * solution_relative + &self.lift)
  }
}

/// Solve the s.p.d. Galerkin system $A u = f$ subject to the essential
/// boundary condition $"tr" u = g$. See [`LiftedSystem`].
pub fn solve_with_essential_bc(
  relative: &RelativeWhitneyComplex,
  boundary: &BoundaryWhitneyComplex,
  system: CsrMatrix,
  rhs: &Vector,
  boundary_values: &Cochain,
) -> Cochain {
  LiftedSystem::new(relative, boundary, system, boundary_values).solve(rhs)
}

/// The natural (Neumann) boundary load
/// $[integral_(diff K) angle.l "tr" W_sigma, h angle.r vol_(diff K)]_sigma$,
/// to be added to the right-hand side of the unconstrained system.
///
/// The data $h$ is a $k$-form field on the boundary manifold $diff K$ -- not
/// on the parent mesh -- since the load is an integral over $diff K$ and only
/// the trace of $h$ enters it. A form given in the ambient coordinates of the
/// parent reaches it through the pullback adapter against the trace coords:
///
/// ```ignore
/// let boundary_coords = boundary.boundary_complex().trace_coords(&coords);
/// let h = flux.pullback_on(boundary.topology(), &boundary_coords);
/// ```
///
/// For the scalar Laplacian ($k = 0$) the data is the outward flux
/// $h = diff u \/ diff n$. Homogeneous natural conditions need no call.
pub fn neumann_load(
  boundary: &BoundaryWhitneyComplex,
  data: &(impl Section<Covariant> + Sync),
  qr: Option<SimplexQuadRule>,
) -> Vector {
  assert_eq!(data.dim(), boundary.topology().dim());
  let elvec = SourceElVec::new(data, qr);
  let load = assemble_galvec(boundary.topology(), boundary.geometry(), elvec);
  boundary.trace(data.grade()).transpose() * load
}

/// The boundary mass matrix $"tr"^T M_(diff K)^k "tr"$ of the bilinear form
/// $integral_(diff K) angle.l "tr" u, "tr" v angle.r vol_(diff K)$:
/// scaled by the impedance $alpha$ and added to the system matrix, this
/// imposes a Robin condition.
pub fn boundary_mass(boundary: &BoundaryWhitneyComplex, grade: ExteriorGrade) -> CsrMatrix {
  let trace = boundary.trace(grade);
  let mass = CsrMatrix::from(&boundary.whitney_complex().mass(grade));
  trace.transpose() * mass * trace
}
