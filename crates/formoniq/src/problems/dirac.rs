//! Maxwell's equations as the Hodge–Dirac evolution on the *whole* de Rham
//! complex, in the 3+1 split.
//!
//! One first-order operator collapses the two Maxwell field equations into one.
//! The Hodge–Dirac operator
//!
//! $ sans(D) = dif - delta: Lambda^bullet -> Lambda^bullet, quad
//!   Lambda^bullet = plus.circle.big_(k=0)^n Lambda^k, $
//!
//! is the Dirac operator of the de Rham complex: metric-free $dif$ raising grade,
//! its formal adjoint $delta$ lowering it, acting on the *full* mixed-grade
//! field at once. It is the square root of the Hodge–Laplacian,
//!
//! $ sans(D)^2 = (dif - delta)^2 = -(dif delta + delta dif) = -Delta, $
//!
//! the two cross terms $dif^2 = delta^2 = 0$ vanishing by nilpotency; and it is
//! skew-adjoint, $sans(D)^* = delta - dif = -sans(D)$, so the first-order flow
//!
//! $ diff_t u = sans(D) u = (dif - delta) u $
//!
//! is norm-preserving and its solutions solve the wave equation
//! $diff_(t t) u = -Delta u$ grade by grade. This is Maxwell. Writing the field
//! as $u = E + B$ with the electric $E in Lambda^1$ and the magnetic
//! $B in Lambda^2$, the middle two grades of $diff_t u = sans(D) u$ are exactly
//!
//! $ diff_t E = -delta B quad & "(Ampère, source-free)" \
//!   diff_t B = dif E     quad & "(Faraday)", $
//!
//! while the extremal grades are the two Gauss laws: the grade-0 component
//! $diff_t u_0 = -delta E$ is the electric constraint $delta E = 0$ (no charge),
//! the grade-$n$ component $diff_t u_n = dif B$ the magnetic constraint
//! $dif B = 0$ (no monopoles). The four classical equations are the four grades
//! of one Dirac evolution. The *field* itself is not mixed-grade --- it is
//! $E + B$, living in grades 1 and 2 (a single Faraday 2-form before the split);
//! the whole graded space is carried because $sans(D)$ is grade-mixing, so it maps the field out into the neighbouring grades,
//! where the divergence/Gauss parts land.
//!
//! # The sign, and the canonical Hodge–Dirac operator
//!
//! The textbook Hodge–Dirac operator is $dif + delta$, which is *self*-adjoint
//! with $(dif + delta)^2 = +Delta$. A first-order energy-conserving flow on a
//! Riemannian (positive-definite) slice needs a *skew*-adjoint generator, so the
//! evolution operator is the skew $dif - delta$, with $sans(D)^2 = -Delta$. This
//! is not a lesser cousin of $dif + delta$: the two are the same operator,
//! conjugate by the diagonal unitary $Q = i^"deg"$ (grade $k |-> i^k$), since
//! $Q dif Q^(-1) = i dif$ and $Q delta Q^(-1) = -i delta$ give
//! $Q (dif - delta) Q^(-1) = i (dif + delta)$. Under $psi = i^"deg" u$ the real
//! flow $diff_t u = (dif - delta) u$ becomes the complex Schrödinger form
//! $diff_t psi = i (dif + delta) psi$ --- the dimension-general Riemann–Silberstein
//! ($E + i B$) picture. The real form is used here because it keeps the fields
//! real and grade $1 = E$, grade $2 = B$ direct, in any dimension. The genuinely
//! self-adjoint $dif + delta$ with $+Delta$ and no $i$ exists only *without* a
//! split: the 4D Lorentzian $sans(D) F = J$, where the signature makes it
//! hyperbolic on its own. That covariant operator is [`HodgeDirac::assemble_selfadjoint`],
//! and [`solve_dirac_source`] solves its static source problem on a spacetime
//! mesh -- the same blocks as the evolution form, differing in exactly the one
//! sign the conjugation predicts.
//!
//! # Discretization
//!
//! In FEEC $dif$ is the exact coboundary $D_k$ on Whitney cochains, but $delta$
//! is metric and lives only weakly, through the Galerkin Hodge masses $M_k$. The
//! weak form of $diff_t u = (dif - delta) u$, tested grade by grade with
//! $angle.l delta u, v angle.r = angle.l u, dif v angle.r$, is the linear system
//!
//! $ M dot(u) = A u, quad M = plus.circle.big_k M_k, quad
//!   A = mat(
//!     0, -D_0^T M_1, , ;
//!     M_1 D_0, 0, -D_1^T M_2, ;
//!     , M_2 D_1, 0, dots.down;
//!     , , dots.down, ), $
//!
//! with $M$ the SPD block-diagonal mass and $A$ *skew-symmetric by construction*:
//! the super-diagonal blocks are minus the transposes of the sub-diagonal ones,
//! $(D_(k-1)^T M_k) = (M_k D_(k-1))^T$. The strong operator $M^(-1) A = dif -
//! delta_h$ with the weak codifferential $delta_h = M_k^(-1) D_k^T M_(k+1)$ is
//! the discrete Hodge–Dirac operator, and $dif compose dif = 0$ makes it square
//! to the discrete Hodge–Laplacian exactly: $(M^(-1) A)^2 = -Delta_h$, the
//! grade-shifting-by-two terms cancelling by nilpotency.
//!
//! Because $A + A^T = 0$, the quadratic energy $H = 1/2 thin u^T M u = 1/2
//! norm(u)_(L^2)^2$ is a conserved invariant of the semi-discrete flow, and the
//! Gauss–Legendre integrator [`solve_dirac`] conserves it *to roundoff* --- no
//! drift, only the physical sloshing of energy between the electric grade 1 and
//! magnetic grade 2. The boundary condition is the choice of complex: essential
//! (perfect electric conductor, $"tr" u = 0$ on every grade) on the
//! [`RelativeWhitneyComplex`], natural on the full [`WhitneyComplex`] --- the
//! same code either way, since the skew structure is algebraic and survives
//! both.
//!
//! [`WhitneyComplex`]: crate::whitney_complex::WhitneyComplex
//! [`RelativeWhitneyComplex`]: crate::whitney_complex::RelativeWhitneyComplex

use crate::{
  linalg::{faer::FaerLu, quadratic_form_sparse},
  time::{Leapfrog, LinearIrk, Tableau},
  whitney_complex::{HilbertComplex, RelativeWhitneyComplex},
};

use derham::cochain::Cochain;
use exterior::ExteriorGrade;
use simplicial::{
  Dim,
  linalg::{CooMatrix, CsrMatrix, Vector},
};

/// A field on the full de Rham complex: one cochain per grade,
/// $u = (u_0, dots, u_n) in plus.circle.big_k C^k$.
///
/// The state the Hodge–Dirac operator evolves. In the Maxwell reading the
/// electric field is `grade(1)`, the magnetic flux `grade(2)`; the extremal
/// grades carry the two Gauss constraints ($delta E$ in grade 0, $dif B$ in
/// grade $n$), which stay negligible for a physical field.
#[derive(Clone)]
pub struct MixedField {
  /// Slot $k$ holds the $k$-cochain, for $k = 0, dots, n$.
  grades: Vec<Cochain>,
}
impl MixedField {
  /// Assemble from one cochain per grade, `grades[k]` a $k$-cochain.
  pub fn new(grades: Vec<Cochain>) -> Self {
    for (k, c) in grades.iter().enumerate() {
      assert_eq!(c.grade(), k, "slot k must hold a k-cochain");
    }
    Self { grades }
  }

  /// The zero field with the DOF layout of `complex`.
  pub fn zeros<C: HilbertComplex>(complex: &C) -> Self {
    let grades = (0..=complex.dim())
      .map(|k| Cochain::new(k, Vector::zeros(complex.ndofs(k))))
      .collect();
    Self { grades }
  }

  /// A field with a single grade populated, the others zero --- e.g. an electric
  /// field at `grade = 1` or a magnetic flux at `grade = 2` in the Maxwell
  /// reading.
  pub fn from_grade<C: HilbertComplex>(complex: &C, u: Cochain) -> Self {
    let k = u.grade();
    assert_eq!(u.len(), complex.ndofs(k), "grade k cochain has wrong ndofs");
    let mut field = Self::zeros(complex);
    field.grades[k] = u;
    field
  }

  pub fn dim(&self) -> Dim {
    self.grades.len() - 1
  }
  pub fn grade(&self, k: ExteriorGrade) -> &Cochain {
    &self.grades[k]
  }
  pub fn into_grades(self) -> Vec<Cochain> {
    self.grades
  }
}

/// The discrete Hodge–Dirac operator on a Hilbert complex, assembled as the
/// two flat global matrices its consumers share: the block-diagonal mass
/// $M = plus.circle.big_k M_k$ and the grade-coupling $A$, weak $dif$ on the
/// sub-diagonal, weak $delta$ on the super-diagonal.
///
/// The operator comes in its two signs, one assembly parameterized by the
/// other ([`Self::assemble`] / [`Self::assemble_selfadjoint`]): the *skew*
/// $dif - delta$ ($A + A^T = 0$), the energy-conserving generator of the 3+1
/// evolution $M dot(u) = A u$ on a Riemannian slice, and the *self-adjoint*
/// canonical $dif + delta$ ($A = A^T$), the covariant static operator of
/// $sans(D) u = f$ -- on a Lorentzian spacetime mesh (indefinite $M$) this is
/// the hyperbolic spacetime-Maxwell operator, no split and no
/// time integrator involved. Same blocks, same code; the sign of the
/// super-diagonal is the entire difference.
///
/// Total at the degenerate grades without a special case: grade $0$ has no
/// sub-diagonal coupling (no $Lambda^(-1)$), grade $n$ no super-diagonal one
/// (no $Lambda^(n+1)$), and the tridiagonal loop simply never emits those
/// blocks. Works on both the full and the relative complex through the
/// [`HilbertComplex`] trait --- the boundary condition is the choice of complex.
pub struct HodgeDirac {
  /// Per-grade DOF offsets into the flat field vector; length $n + 2$, the last
  /// entry the total DOF count.
  offsets: Vec<usize>,
  /// The per-grade Galerkin masses $M_k$, kept for per-grade energies.
  masses: Vec<CsrMatrix>,
  /// $M = plus.circle.big_k M_k$: block-diagonal, symmetric; positive
  /// definite exactly when the geometry is Riemannian.
  mass_block: CsrMatrix,
  /// $A$: block-tridiagonal; skew-symmetric for $dif - delta$, symmetric for
  /// $dif + delta$.
  op: CsrMatrix,
}
impl HodgeDirac {
  /// The skew evolution generator $sans(D) = dif - delta$, $A + A^T = 0$:
  /// the 3+1 form on a Riemannian slice, driving $M dot(u) = A u$.
  pub fn assemble<C: HilbertComplex>(complex: &C) -> Self {
    Self::assemble_signed(complex, -1.0)
  }

  /// The canonical self-adjoint Hodge–Dirac operator
  /// $sans(D) = dif + delta$, $A = A^T$: the covariant form, which on a
  /// Lorentzian spacetime is hyperbolic through the signature alone
  /// ($sans(D)^2 = Delta$ is the d'Alembertian there) and needs no 3+1 split.
  /// Consumed by [`solve_dirac_source`].
  pub fn assemble_selfadjoint<C: HilbertComplex>(complex: &C) -> Self {
    Self::assemble_signed(complex, 1.0)
  }

  /// One assembly for both signs: the super-diagonal block is
  /// `delta_sign` times the transpose of the sub-diagonal one, making
  /// $A^T = "delta_sign" dot A$ exact by construction.
  fn assemble_signed<C: HilbertComplex>(complex: &C, delta_sign: f64) -> Self {
    let dim = complex.dim();

    let masses: Vec<CsrMatrix> = (0..=dim)
      .map(|k| CsrMatrix::from(&complex.mass(k)))
      .collect();
    // The coboundaries $D_k: C^k -> C^(k+1)$, one per interior coupling.
    let difs: Vec<CsrMatrix> = (0..dim).map(|k| complex.dif(k)).collect();

    let mut offsets = Vec::with_capacity(dim + 2);
    let mut acc = 0;
    for mass in &masses {
      offsets.push(acc);
      acc += mass.nrows();
    }
    offsets.push(acc);
    let total = acc;

    // $M$: the masses on the block diagonal.
    let mut mass_block = CooMatrix::new(total, total);
    for (k, mass) in masses.iter().enumerate() {
      let off = offsets[k];
      for (r, c, &v) in mass.triplet_iter() {
        mass_block.push(off + r, off + c, v);
      }
    }

    // $A$: the sub-diagonal block $(k, k-1)$ is $U_k = M_k D_(k-1)$ (weak $dif$),
    // the super-diagonal block $(k-1, k)$ is `delta_sign` times its transpose
    // $U_k^T = D_(k-1)^T M_k$ (weak $delta$). Emitting both from the same
    // triplet makes the symmetry class exact by construction.
    let mut op = CooMatrix::new(total, total);
    for k in 1..=dim {
      let u_k = &masses[k] * &difs[k - 1];
      let (row, col) = (offsets[k], offsets[k - 1]);
      for (r, c, &v) in u_k.triplet_iter() {
        op.push(row + r, col + c, v);
        op.push(col + c, row + r, delta_sign * v);
      }
    }

    Self {
      offsets,
      masses,
      mass_block: CsrMatrix::from(&mass_block),
      op: CsrMatrix::from(&op),
    }
  }

  pub fn dim(&self) -> Dim {
    self.offsets.len() - 2
  }
  pub fn ndofs_total(&self) -> usize {
    *self.offsets.last().unwrap()
  }
  /// $M = plus.circle.big_k M_k$: the block-diagonal mass carrying the
  /// $L^2 Lambda^bullet$ pairing (indefinite on a Lorentzian geometry).
  pub fn mass_block(&self) -> &CsrMatrix {
    &self.mass_block
  }
  /// The grade-coupling operator $A$: skew for $dif - delta$, symmetric for
  /// $dif + delta$.
  pub fn op(&self) -> &CsrMatrix {
    &self.op
  }

  /// Pack a field's per-grade coefficients into the flat vector, in grade order.
  /// The field must already live in this complex's DOFs.
  pub fn flatten(&self, field: &MixedField) -> Vector {
    let mut y = Vector::zeros(self.ndofs_total());
    for k in 0..=self.dim() {
      let (off, n) = (self.offsets[k], self.offsets[k + 1] - self.offsets[k]);
      y.rows_mut(off, n).copy_from(field.grade(k).coeffs());
    }
    y
  }

  /// Unpack the flat vector back into a per-grade field.
  pub fn unflatten(&self, y: &Vector) -> MixedField {
    let grades = (0..=self.dim())
      .map(|k| {
        let (off, n) = (self.offsets[k], self.offsets[k + 1] - self.offsets[k]);
        Cochain::new(k, y.rows(off, n).into_owned())
      })
      .collect();
    MixedField::new(grades)
  }

  /// The total energy $1/2 thin u^T M u = 1/2 norm(u)_(L^2)^2$, the conserved
  /// invariant of the Hodge–Dirac flow (the electromagnetic energy in the
  /// Maxwell reading). On a Lorentzian geometry $M$ is indefinite and this is
  /// the signed $L^2$ pairing, not a norm.
  pub fn energy(&self, field: &MixedField) -> f64 {
    0.5 * quadratic_form_sparse(&self.mass_block, &self.flatten(field))
  }

  /// The energy $1/2 norm(u_k)_(L^2)^2 = 1/2 thin u_k^T M_k u_k$ carried by a
  /// single grade. In the Maxwell reading this is the electric energy at
  /// $k = 1$, the magnetic at $k = 2$, and the Gauss-constraint residuals at the
  /// extremal grades.
  pub fn grade_energy(&self, field: &MixedField, grade: ExteriorGrade) -> f64 {
    0.5 * quadratic_form_sparse(&self.masses[grade], field.grade(grade).coeffs())
  }

  /// The grade-parity 2-coloring of the DOFs (`true` on odd grades): the
  /// partition under which $A$ is block-antidiagonal, since $dif$ and $delta$
  /// shift grade by one and so never couple two grades of the same parity. This
  /// is the coloring [`Leapfrog`] consumes to run the explicit Yee-style split.
  pub fn grade_parity_coloring(&self) -> Vec<bool> {
    let mut color = vec![false; self.ndofs_total()];
    for k in (1..=self.dim()).step_by(2) {
      color[self.offsets[k]..self.offsets[k + 1]].fill(true);
    }
    color
  }
}

/// Restrict an ambient field to `complex`'s DOFs, grade by grade ($E_k^T$): the
/// identity on the full complex, a restriction to interior DOFs on the relative
/// one.
fn restrict_field<C: HilbertComplex>(complex: &C, f: &MixedField) -> MixedField {
  MixedField::new(
    (0..=complex.dim())
      .map(|k| Cochain::new(k, complex.inclusion(k).transpose() * f.grade(k).coeffs()))
      .collect(),
  )
}

/// Extend a field on `complex`'s DOFs back to the ambient Whitney space grade by
/// grade ($E_k$), by zero on the constrained boundary.
fn extend_field<C: HilbertComplex>(complex: &C, f: &MixedField) -> MixedField {
  MixedField::new(
    (0..=complex.dim())
      .map(|k| Cochain::new(k, &complex.inclusion(k) * f.grade(k).coeffs()))
      .collect(),
  )
}

/// Evolve Maxwell's equations as the Hodge–Dirac flow $diff_t u = (dif - delta)
/// u$ on the full de Rham complex, by Gauss–Legendre collocation.
///
/// The semi-discrete system $M dot(u) = A u$ has SPD $M$ and skew $A$, a linear
/// Hamiltonian system whose quadratic energy $1/2 thin u^T M u$ Gauss–Legendre
/// conserves *exactly* --- to roundoff, not merely bounded. `times` is assumed
/// evenly spaced (the stage system is factored once). The boundary condition is
/// the `complex`: essential (PEC) on the [`RelativeWhitneyComplex`], natural on
/// the full [`WhitneyComplex`]. `initial` is given in the ambient Whitney space
/// per grade, restricted to this complex's DOFs internally and the returned
/// states extended back, so the caller is oblivious to the boundary condition.
///
/// [`WhitneyComplex`]: crate::whitney_complex::WhitneyComplex
/// [`RelativeWhitneyComplex`]: crate::whitney_complex::RelativeWhitneyComplex
pub fn solve_dirac<C: HilbertComplex>(
  complex: &C,
  times: &[f64],
  initial: MixedField,
) -> Vec<MixedField> {
  let dirac = HodgeDirac::assemble(complex);

  let dt = times.windows(2).next().map_or(0.0, |w| w[1] - w[0]);
  let irk = LinearIrk::new(
    Tableau::gauss_legendre(2),
    &dirac.mass_block,
    dirac.op.clone(),
    dt,
  );

  let mut y = dirac.flatten(&restrict_field(complex, &initial));
  let mut solution = Vec::with_capacity(times.len());
  solution.push(extend_field(complex, &dirac.unflatten(&y)));
  for t01 in times.windows(2) {
    let [t0, _t1] = t01 else { unreachable!() };
    y = irk.step(&y, *t0, |_| Vector::zeros(dirac.ndofs_total()));
    solution.push(extend_field(complex, &dirac.unflatten(&y)));
  }
  solution
}

/// Evolve the Hodge–Dirac Maxwell flow by the explicit Yee-style leapfrog
/// instead of [`solve_dirac`]'s implicit Gauss–Legendre: the grades are
/// 2-colored by parity (the split under which $A$ is block-antidiagonal) and
/// stepped by the symplectic [`Leapfrog`].
///
/// Cheaper per step --- only the two color-block masses are factored, never the
/// coupled operator --- but only *conditionally* stable: `times` must resolve the
/// CFL limit, $dif t lt.eq h_min \/ c$, or the staggered energy loses positive
/// definiteness and the scheme blows up. Boundary conditions and the ambient
/// restrict/extend are exactly as in [`solve_dirac`].
pub fn solve_dirac_leapfrog<C: HilbertComplex>(
  complex: &C,
  times: &[f64],
  initial: MixedField,
) -> Vec<MixedField> {
  let dirac = HodgeDirac::assemble(complex);
  let color = dirac.grade_parity_coloring();

  let dt = times.windows(2).next().map_or(0.0, |w| w[1] - w[0]);
  let leapfrog = Leapfrog::new(&dirac.mass_block, &dirac.op, &color, dt);

  let mut y = dirac.flatten(&restrict_field(complex, &initial));
  let mut solution = Vec::with_capacity(times.len());
  solution.push(extend_field(complex, &dirac.unflatten(&y)));
  for _ in times.windows(2) {
    y = leapfrog.step(&y);
    solution.push(extend_field(complex, &dirac.unflatten(&y)));
  }
  solution
}

/// Solve the static covariant Hodge–Dirac source problem
///
/// $ (sans(D) + m) u = f, quad sans(D) = dif + delta, $
///
/// on the full de Rham complex, with essential boundary values imposed by
/// affine lifting. This is the spacetime form of the equation: on a Lorentzian
/// mesh it is the massive Hodge–Dirac equation with mass $m$ (squaring to
/// Klein–Gordon, $sans(D)^2 = Delta$ the d'Alembertian), Maxwell with sources
/// the middle grades of the massless case. No time integrator appears -- on a
/// spacetime mesh, time is one of the mesh directions and causality lives in
/// the signature of the metric, not in a stepping loop.
///
/// The weak form is symmetric on any signature,
/// $angle.l (dif + delta) u, v angle.r = angle.l u, (dif + delta) v angle.r$,
/// so the assembled system $A + m M$ is symmetric (indefinite on a Lorentzian
/// geometry, where $M$ itself is), and sparse LU solves it uniformly.
///
/// `load` holds the Galerkin load functionals per grade,
/// $b_k \[sigma\] = integral_M inner(f_k, W_sigma) vol$, in the ambient
/// (full-complex) DOF layout. `boundary_values` is an ambient field whose
/// constrained DOFs carry the essential data $"tr" u = "tr" g$ (a full-mesh
/// interpolant of a manufactured solution does); the lifting
/// $u = hat(g) + E u_0$, $E^T (A + m M) E thin u_0 = E^T (b - (A + m M) hat(g))$
/// reduces to the homogeneous problem on `relative`'s interior DOFs, and the
/// returned field is the lifted ambient solution.
///
/// # The massless kernel
///
/// At $m = 0$ the relative operator is *singular*, and not by accident: its
/// kernel is the space of relative harmonic fields, which Poincaré--Lefschetz
/// duality pins to $H^n (M, diff M) tilde.equiv H_0 (M) = RR$ on a connected
/// mesh --- one dimension, in the top grade, on any geometry and any signature.
/// The Lorentzian signature does not remove it: it is topological, not
/// spectral. [`top_harmonic`] writes that mode down in closed form, and this
/// solve deflates it, returning the unique solution orthogonal to it. Grades
/// $k < n$ are untouched, so a field living below the top grade --- the
/// Faraday 2-form of Maxwell for $n > 2$ --- is unique outright.
pub fn solve_dirac_source(
  relative: &RelativeWhitneyComplex,
  mass_term: f64,
  load: &MixedField,
  boundary_values: &MixedField,
) -> MixedField {
  let full = relative.full();
  let dirac = HodgeDirac::assemble_selfadjoint(&full);
  let n_full = dirac.ndofs_total();

  // $S = A + m M$ on the full space.
  let mut system = CooMatrix::new(n_full, n_full);
  for (r, c, &v) in dirac.op.triplet_iter() {
    system.push(r, c, v);
  }
  if mass_term != 0.0 {
    for (r, c, &v) in dirac.mass_block.triplet_iter() {
      system.push(r, c, mass_term * v);
    }
  }
  let system = CsrMatrix::from(&system);

  // The block inclusion $E = plus.circle.big_k E_k$ of the relative DOFs into
  // the flat ambient layout.
  let n_relative: usize = (0..=relative.dim()).map(|k| relative.ndofs(k)).sum();
  let mut inclusion = CooMatrix::new(n_full, n_relative);
  let mut col_offset = 0;
  for k in 0..=relative.dim() {
    let e_k = relative.inclusion(k);
    for (r, c, &v) in e_k.triplet_iter() {
      inclusion.push(dirac.offsets[k] + r, col_offset + c, v);
    }
    col_offset += relative.ndofs(k);
  }
  let inclusion = CsrMatrix::from(&inclusion);

  let lift = dirac.flatten(boundary_values);
  let rhs = inclusion.transpose() * (dirac.flatten(load) - &system * &lift);
  let system_relative = inclusion.transpose() * &system * &inclusion;

  // At $m = 0$ border the system with the top-grade harmonic rather than
  // handing a singular matrix to LU. The mode is in the kernel exactly when the
  // essential part is the *whole* boundary, where $H^n (M, diff M) = RR$; on a
  // partial part -- the causal posing a hyperbolic problem needs -- that
  // relative cohomology vanishes, the operator is nonsingular, and bordering
  // with a non-kernel vector would over-constrain it. So the mode is tested,
  // not assumed.
  let harmonic = (mass_term == 0.0)
    .then(|| top_harmonic(relative))
    .flatten()
    .filter(|h| (&system_relative * h).norm() <= 1e-8 * h.norm());

  let interior = match harmonic {
    Some(h) => solve_bordered(&system_relative, &rhs, &h),
    None => FaerLu::new(system_relative).solve(&rhs),
  };
  dirac.unflatten(&(lift + inclusion * interior))
}

/// The relative harmonic field of the massless Hodge–Dirac operator, in the
/// flat relative DOF layout: top grade only, every lower grade zero.
///
/// $ h_n = M_n^(-1) z, quad z = "the fundamental class", $
///
/// with $z$ the coherent orientation read off the cells. This is the whole
/// kernel, not a member of it: a field supported in grade $n$ is annihilated by
/// $dif + delta$ exactly when $D_(n-1)^T M_n u_n = 0$, i.e. when $M_n u_n$ is a
/// relative $n$-cycle, and the relative $n$-cycles of a connected mesh are the
/// multiples of the fundamental class. Hence $dim ker = dim H^n (M, diff M) = 1$
/// --- the closed form of the Poincaré--Lefschetz statement, needing no
/// eigensolve.
///
/// `None` on a non-orientable mesh, where no fundamental class exists
/// (invariant 6: holding the orientation *is* the proof of orientability).
/// There the kernel is genuinely absent, $H^n (M, diff M) = 0$ over $RR$, and
/// the operator is invertible without deflation.
pub fn top_harmonic(relative: &RelativeWhitneyComplex) -> Option<Vector> {
  let dim = relative.dim();
  let orientation = relative.full().topology().orientation()?;

  let z = Vector::from_iterator(
    orientation.signs().len(),
    orientation.signs().iter().map(|s| s.as_f64()),
  );
  let h_n = FaerLu::new(CsrMatrix::from(&relative.mass(dim))).solve(&z);

  let ndofs: Vec<usize> = (0..=dim).map(|k| relative.ndofs(k)).collect();
  let mut h = Vector::zeros(ndofs.iter().sum());
  let offset: usize = ndofs[..dim].iter().sum();
  h.rows_mut(offset, ndofs[dim]).copy_from(&h_n);
  Some(h)
}

/// Solve the consistent singular system $S u = f$ with $ker S = "span"{h}$ by
/// bordering, returning the unique solution with $h^T u = 0$:
///
/// $ mat(S, h; h^T, 0) vec(u, c) = vec(f, 0). $
///
/// The bordered matrix is nonsingular precisely because $h$ spans the kernel,
/// so one LU replaces a pseudo-inverse. The multiplier $c$ is the residual
/// along $h$, zero when $f perp ker S$.
fn solve_bordered(system: &CsrMatrix, rhs: &Vector, h: &Vector) -> Vector {
  let n = rhs.len();
  let mut bordered = CooMatrix::new(n + 1, n + 1);
  for (r, c, &v) in system.triplet_iter() {
    bordered.push(r, c, v);
  }
  for (i, &v) in h.iter().enumerate() {
    if v != 0.0 {
      bordered.push(i, n, v);
      bordered.push(n, i, v);
    }
  }

  let mut bordered_rhs = Vector::zeros(n + 1);
  bordered_rhs.rows_mut(0, n).copy_from(rhs);

  FaerLu::new(CsrMatrix::from(&bordered))
    .solve(&bordered_rhs)
    .rows(0, n)
    .into_owned()
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::{
    linalg::faer::FaerCholesky, problems::elliptic::HodgeBlocks, whitney_complex::WhitneyComplex,
  };
  use simplicial::mesher::cartesian::CartesianGrid;

  use approx::assert_relative_eq;

  /// A deterministic full field: every grade populated with a reproducible
  /// pattern, enough to couple all rungs of the complex.
  fn seed_field(dirac: &HodgeDirac) -> MixedField {
    let grades = (0..=dirac.dim())
      .map(|k| {
        let n = dirac.offsets[k + 1] - dirac.offsets[k];
        Cochain::new(
          k,
          Vector::from_fn(n, |i, _| ((7 * i + 3 * k + 1) % 11) as f64 - 5.0),
        )
      })
      .collect();
    MixedField::new(grades)
  }

  /// The discrete codifferential is the adjoint of the exterior derivative:
  /// the assembled Hodge–Dirac operator is skew-symmetric, $A + A^T = 0$, to
  /// roundoff and at every dimension. This is integration by parts made
  /// structural --- the super-diagonal blocks are the negated transposes of the
  /// Poincaré--Lefschetz, discretely: the closed-form top-grade harmonic
  /// $h_n = M_n^(-1) z$ is annihilated by the massless self-adjoint
  /// Hodge--Dirac operator, in every dimension and on either signature. This is
  /// the statement that $ker(dif + delta) supset.eq H^n (M, diff M)$ is realized
  /// exactly on the nose by the fundamental class, with no eigensolve.
  #[test]
  fn top_harmonic_is_annihilated() {
    for dim in 1..=4 {
      for minkowski in [false, true] {
        let (topology, coords) = if minkowski {
          CartesianGrid::minkowski(dim, 2)
        } else {
          CartesianGrid::new_unit(dim, 2).triangulate()
        };
        let regge = coords.to_edge_lengths_sq(&topology);
        let whitney = WhitneyComplex::new(&topology, &regge);
        let relative = whitney.relative();

        let h = top_harmonic(&relative).expect("a box is orientable");
        let dirac = HodgeDirac::assemble_selfadjoint(&relative);
        let residual = (dirac.op() * &h).norm() / h.norm();

        assert!(
          residual < 1e-9,
          "dim {dim} minkowski {minkowski}: |A h|/|h| = {residual:.3e}"
        );
      }
    }
  }

  /// sub-diagonal ones by construction --- and it is what conserves energy.
  #[test]
  fn operator_is_skew_symmetric() {
    for dim in 1..=3 {
      let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      let metric = coords.to_edge_lengths_sq(&topology);
      let whitney = WhitneyComplex::new(&topology, &metric);
      let dirac = HodgeDirac::assemble(&whitney);

      let a = &dirac.op;
      let skew = a + &a.transpose();
      assert_relative_eq!(
        skew.values().iter().fold(0.0, |m: f64, &v| m.max(v.abs())),
        0.0
      );
    }
  }

  /// The defining Dirac law: $sans(D)^2 = -Delta$. The discrete Hodge–Dirac
  /// operator $M^(-1) A = dif - delta_h$ squared equals the negative discrete
  /// Hodge–Laplacian, grade by grade --- the grade-shifting-by-two terms
  /// cancelling by $dif compose dif = 0$. Checked against the independently
  /// assembled up/down Laplacian blocks of [`HodgeBlocks`], at every grade.
  #[test]
  fn dirac_squared_is_negative_hodge_laplacian() {
    for dim in 1..=3 {
      let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      let metric = coords.to_edge_lengths_sq(&topology);
      let whitney = WhitneyComplex::new(&topology, &metric);
      let dirac = HodgeDirac::assemble(&whitney);

      // Mass solves for $M^(-1)$, one factorization per grade.
      let chol: Vec<_> = (0..=dim)
        .map(|k| FaerCholesky::new(dirac.masses[k].clone()))
        .collect();
      // The strong Hodge–Dirac action $v = (dif - delta_h) u$, from $M v = A u$.
      let apply_dirac = |u: &MixedField| {
        let mv = &dirac.op * dirac.flatten(u);
        let grades = (0..=dim)
          .map(|k| {
            let (off, n) = (dirac.offsets[k], dirac.offsets[k + 1] - dirac.offsets[k]);
            Cochain::new(k, chol[k].solve(&mv.rows(off, n).into_owned()))
          })
          .collect();
        MixedField::new(grades)
      };

      let u = seed_field(&dirac);
      let d2u = apply_dirac(&apply_dirac(&u));

      #[allow(clippy::needless_range_loop)] // grade is the mathematical index
      for grade in 0..=dim {
        // The Hodge–Laplacian $Delta_h u|_k = M_k^(-1)(K^"up" + K^"dn") u_k$.
        let hb = HodgeBlocks::compute(&whitney, grade);
        let uk = u.grade(grade).coeffs();
        let up = hb.stiff() * uk;
        let dn = if hb.n_sigma > 0 {
          let s = FaerCholesky::new(hb.mass_sigma.clone()).solve(&(&hb.codif_dn() * uk));
          &hb.dif_sigma() * s
        } else {
          Vector::zeros(hb.n_u)
        };
        let lap = chol[grade].solve(&(up + dn));

        let lhs = d2u.grade(grade).coeffs();
        assert_relative_eq!(
          (lhs + &lap).norm(),
          0.0,
          epsilon = 1e-9 * lap.norm().max(1.0)
        );
      }
    }
  }

  /// The structure-preserving law, at every dimension: the total Hodge–Dirac
  /// energy $1/2 norm(u)_(L^2)^2 = 1/2 thin u^T M u$ is conserved to roundoff.
  /// Gauss–Legendre is symplectic and, on this linear skew system, conserves the
  /// quadratic invariant exactly --- across all grades of the coupled complex
  /// at once.
  #[test]
  fn energy_conserved_at_every_dimension() {
    for dim in 1..=3 {
      let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      let metric = coords.to_edge_lengths_sq(&topology);
      let whitney = WhitneyComplex::new(&topology, &metric);
      let dirac = HodgeDirac::assemble(&whitney);

      let initial = seed_field(&dirac);
      let times: Vec<f64> = (0..=100).map(|i| 0.05 * i as f64).collect();
      let solution = solve_dirac(&whitney, &times, initial);

      let energy0 = dirac.energy(&solution[0]);
      assert!(energy0 > 0.0);
      for state in &solution {
        let energy = dirac.energy(state);
        assert_relative_eq!(energy, energy0, epsilon = 1e-9 * energy0);
      }
    }
  }

  /// The explicit leapfrog is structure-preserving too. Built from the same
  /// Hodge–Dirac $M$ and skew $A$, 2-colored by grade parity, it conserves its
  /// staggered invariant to roundoff at every dimension --- within CFL. The same
  /// symplectic guarantee as Gauss–Legendre, for the cheap explicit scheme. This
  /// also exercises the coloring: [`Leapfrog::new`] asserts $A$ is
  /// block-antidiagonal under it, which holds iff $dif, delta$ never couple two
  /// grades of the same parity.
  #[test]
  fn leapfrog_conserves_staggered_energy_at_every_dimension() {
    for dim in 1..=3 {
      let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      let metric = coords.to_edge_lengths_sq(&topology);
      let whitney = WhitneyComplex::new(&topology, &metric);
      let dirac = HodgeDirac::assemble(&whitney);
      let color = dirac.grade_parity_coloring();

      // Within the CFL limit (wave speed c = 1 in vacuum).
      let dt = 0.1 * metric.mesh_width_min();
      let leapfrog = Leapfrog::new(&dirac.mass_block, &dirac.op, &color, dt);

      let mut y = dirac.flatten(&seed_field(&dirac));
      let e0 = leapfrog.conserved_energy(&y);
      assert!(e0 > 0.0);
      for _ in 0..200 {
        y = leapfrog.step(&y);
        assert_relative_eq!(leapfrog.conserved_energy(&y), e0, epsilon = 1e-9 * e0);
      }
    }
  }

  /// The canonical Hodge–Dirac operator is self-adjoint: $A = A^T$ to
  /// roundoff, on Riemannian and Lorentzian geometry alike -- the covariant
  /// counterpart of [`operator_is_skew_symmetric`], the one sign flipped.
  #[test]
  fn selfadjoint_operator_is_symmetric() {
    for dim in 1..=3 {
      let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      let (_, spacetime) = CartesianGrid::minkowski(dim, 2);
      let riemannian = coords.to_edge_lengths_sq(&topology);
      let lorentzian = spacetime.to_edge_lengths_sq(&topology);

      let check = |dirac: &HodgeDirac| {
        let a = &dirac.op;
        let sym = a - &a.transpose();
        assert_relative_eq!(
          sym.values().iter().fold(0.0, |m: f64, &v| m.max(v.abs())),
          0.0
        );
      };
      check(&HodgeDirac::assemble_selfadjoint(&WhitneyComplex::new(
        &topology,
        &riemannian,
      )));
      check(&HodgeDirac::assemble_selfadjoint(&WhitneyComplex::new(
        &topology,
        &lorentzian,
      )));
    }
  }

  /// The defining Dirac law in its covariant form, on a Lorentzian spacetime
  /// mesh: $sans(D)^2 = Delta$ -- the discrete $dif + delta$ squared equals
  /// the discrete Hodge–Laplacian, which on Minkowski geometry *is* the
  /// d'Alembertian, hyperbolic through the signature alone. Same
  /// grade-by-grade check as [`dirac_squared_is_negative_hodge_laplacian`],
  /// with the sign flipped and every solve LU, since the Lorentzian masses
  /// are symmetric indefinite, not s.p.d.
  #[test]
  fn selfadjoint_dirac_squares_to_hodge_laplacian_on_minkowski() {
    for dim in 1..=3 {
      let (topology, spacetime) = CartesianGrid::minkowski(dim, 2);
      let regge = spacetime.to_edge_lengths_sq(&topology);
      let whitney = WhitneyComplex::new(&topology, &regge);
      let dirac = HodgeDirac::assemble_selfadjoint(&whitney);

      let lu: Vec<_> = (0..=dim)
        .map(|k| crate::linalg::faer::FaerLu::new(dirac.masses[k].clone()))
        .collect();
      let apply_dirac = |u: &MixedField| {
        let mv = &dirac.op * dirac.flatten(u);
        let grades = (0..=dim)
          .map(|k| {
            let (off, n) = (dirac.offsets[k], dirac.offsets[k + 1] - dirac.offsets[k]);
            Cochain::new(k, lu[k].solve(&mv.rows(off, n).into_owned()))
          })
          .collect();
        MixedField::new(grades)
      };

      let u = seed_field(&dirac);
      let d2u = apply_dirac(&apply_dirac(&u));

      #[allow(clippy::needless_range_loop)] // grade is the mathematical index
      for grade in 0..=dim {
        let hb = HodgeBlocks::compute(&whitney, grade);
        let uk = u.grade(grade).coeffs();
        let up = hb.stiff() * uk;
        let dn = if hb.n_sigma > 0 {
          let s =
            crate::linalg::faer::FaerLu::new(hb.mass_sigma.clone()).solve(&(&hb.codif_dn() * uk));
          &hb.dif_sigma() * s
        } else {
          Vector::zeros(hb.n_u)
        };
        let lap = lu[grade].solve(&(up + dn));

        let lhs = d2u.grade(grade).coeffs();
        assert_relative_eq!(
          (lhs - &lap).norm(),
          0.0,
          epsilon = 1e-9 * lap.norm().max(1.0)
        );
      }
    }
  }

  /// [`solve_dirac_source`] reproduces a solution that lies in the Whitney
  /// space exactly: a constant mixed-grade form $u$ has $dif u = delta u = 0$,
  /// so $(sans(D) + m) u = m u$, and with load $m M u$ and essential data $u$
  /// the discrete solution is $u$ itself to solver precision -- on Riemannian
  /// and Lorentzian geometry alike.
  #[test]
  fn dirac_source_reproduces_constant_field() {
    use simplicial::{
      geometry::{coord::mesh::MeshCoords, metric::mesh::MeshLengthsSq},
      topology::complex::Complex,
    };

    fn run(topology: &Complex, coords: &MeshCoords, geometry: &MeshLengthsSq) {
      let dim = topology.dim();
      let whitney = WhitneyComplex::new(topology, geometry);
      let relative = whitney.relative();
      let dirac = HodgeDirac::assemble_selfadjoint(&whitney);

      // The de Rham coefficients of the constant form with every component
      // 1 on every grade: integrals of $sum_I dif x^I$ over the simplices.
      let exact = MixedField::new(
        (0..=dim)
          .map(|k| {
            let form = glatt::field::DiffFormClosure::new(
              move |_: &coorder::Coord| {
                exterior::MultiForm::new(
                  Vector::from_element(exterior::exterior_dim(dim, k), 1.0),
                  dim,
                  k,
                )
              },
              dim,
              k,
            );
            let field = derham::section::CoordFieldExt::pullback_on(&form, topology, coords);
            derham::project::derham_map(&field, topology, 2)
          })
          .collect(),
      );

      let mass_term = 1.0;
      let load = dirac.unflatten(&(&dirac.mass_block * dirac.flatten(&exact) * mass_term));
      let solution = solve_dirac_source(&relative, mass_term, &load, &exact);

      for k in 0..=dim {
        assert_relative_eq!(
          solution.grade(k).coeffs(),
          exact.grade(k).coeffs(),
          epsilon = 1e-9
        );
      }
    }

    for dim in 1..=3 {
      let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      let riemannian = coords.to_edge_lengths_sq(&topology);
      run(&topology, &coords, &riemannian);

      // On the Minkowski side the constant field is interpolated on the
      // causally generic (time-scaled) coordinates themselves.
      let (topology, spacetime) = CartesianGrid::minkowski(dim, 2);
      let euclidean_view = MeshCoords::new(spacetime.matrix().clone());
      run(
        &topology,
        &euclidean_view,
        &spacetime.to_edge_lengths_sq(&topology),
      );
    }
  }

  /// The explicit and implicit solvers integrate the *same* equation: over a
  /// short run at small $dif t$ they agree to the leapfrog's second order. This
  /// validates the [`solve_dirac_leapfrog`] wiring (restrict/extend, flatten,
  /// grade-parity coloring) against the trusted Gauss–Legendre [`solve_dirac`].
  #[test]
  fn leapfrog_agrees_with_gauss_legendre() {
    let (topology, coords) = CartesianGrid::new_unit(2, 2).triangulate();
    let metric = coords.to_edge_lengths_sq(&topology);
    let whitney = WhitneyComplex::new(&topology, &metric);
    let dirac = HodgeDirac::assemble(&whitney);

    let dt = 0.02 * metric.mesh_width_min();
    let times: Vec<f64> = (0..=50).map(|i| dt * i as f64).collect();
    let initial = seed_field(&dirac);

    let implicit = solve_dirac(&whitney, &times, initial.clone());
    let explicit = solve_dirac_leapfrog(&whitney, &times, initial);

    let last_implicit = dirac.flatten(implicit.last().unwrap());
    let last_explicit = dirac.flatten(explicit.last().unwrap());
    let rel_err = (&last_implicit - &last_explicit).norm() / last_implicit.norm();
    assert!(rel_err < 1e-2, "solvers disagree by {rel_err}");
  }
}
