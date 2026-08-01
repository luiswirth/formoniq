//! The Whitney finite element complex: the FEEC discretization of the
//! L^2 de Rham complex on a simplicial pseudo-Riemannian manifold.
//!
//! The geometry is the intrinsic Regge primitive, [`MeshLengthsSq`], of any
//! signature: a Riemannian manifold given by edge lengths and a Lorentzian
//! spacetime given by signed edge lengths discretize through the same code. An
//! embedded mesh or raw per-cell metrics reach it by converting to edge
//! lengths at the door. On an indefinite metric the
//! $L^2 Lambda^k$ "inner products" are the indefinite $L^2$ pairings and the
//! mass matrices are symmetric non-degenerate rather than s.p.d., which is
//! the honest structural difference, not a separate code path.

use crate::{
  galerkin::{BilinearForm, GalerkinMatrix},
  linalg::faer::FaerLu,
  operators::WhitneyPairing,
};
use regge::subcomplex::SubcomplexExt;

use {
  crate::linalg::quadratic_form_sparse,
  derham::{Chain, Cochain},
  multialgebra::ExteriorGrade,
  regge::lengths::mesh::MeshLengthsSq,
  simplicial::{
    Dim,
    linalg::{CooMatrix, CsrMatrix, Vector},
    topology::{complex::Complex, handle::KSimplexIdx, role::Facet, subcomplex::Subcomplex},
  },
};

use std::collections::HashSet;

/// The interface the mixed Hodge-Laplace solver consumes from a discrete
/// Hilbert complex: the $L^2 Lambda^k$ inner products, the exterior derivative
/// and its stiffness, the DOF counts, and the harmonic dimension.
///
/// Implemented by both the full [`WhitneyComplex`] (absolute / natural boundary
/// conditions, harmonic space $H^k (K)$) and its [`RelativeWhitneyComplex`]
/// (essential / homogeneous Dirichlet, harmonic space $H^k (K, diff K)$), so the
/// solver is one monomorphized piece of code over both, the choice of
/// boundary condition is just the choice of complex.
///
/// The [`Self::inclusion`] $E: cal(W)^"rel" Lambda^k arrow.hook cal(W) Lambda^k$
/// (the identity for the full complex) is what lets the solver take its source
/// and return its solution in the ambient $cal(W) Lambda^k$ regardless: it
/// restricts the right-hand side by $E^T$ and extends the solution by $E$.
///
/// The four pairings of the complex are provided, not required: they are one
/// [`WhitneyPairing`] each, put through [`Self::assemble`], and that one method
/// is where an implementor says which space it is. They carry the same four
/// names as the local forms and the same grade convention, the grade of the
/// inner product, so a block reads identically whether it is wanted on one
/// cell, on the mesh, or on the mesh with boundary conditions imposed.
pub trait HilbertComplex {
  fn dim(&self) -> Dim;
  fn ndofs(&self, grade: impl Into<ExteriorGrade>) -> usize;
  fn dif(&self, grade: impl Into<ExteriorGrade>) -> CsrMatrix;

  /// The Galerkin matrix of a bilinear form *on this complex*, which is the
  /// one thing distinguishing an implementor from any other.
  ///
  /// The form knows nothing of boundary conditions; a complex is the subspace
  /// it is restricted to, so this is the $i^* b i$ of the Galerkin
  /// discretization with $i$ this complex's [`inclusion`](Self::inclusion).
  fn assemble(&self, form: &impl BilinearForm) -> GalerkinMatrix;

  /// The dimension of the discrete harmonic space $cal(H)^k$: the Betti number
  /// of the complex by the discrete Hodge theorem ($b_k (K)$ for the full
  /// complex, $b_k (K, diff K)$ for the relative one), an exact topological
  /// invariant.
  fn harmonic_dim(&self, grade: impl Into<ExteriorGrade>) -> usize;

  /// Representative integral cocycles of a basis of the cohomology this
  /// complex's harmonic space realizes, expressed in *this* complex's DOFs:
  /// $H^k (K; ZZ)$ for the full complex, $H^k (K, diff K; ZZ)$ for the
  /// relative one. One per [`Self::harmonic_dim`].
  ///
  /// Metric-free, and closed exactly: the coefficients are integers and the
  /// incidence entries are $plus.minus 1$. This is the input the harmonic
  /// projection of [`crate::harmonic`] turns into a harmonic basis, so an
  /// implementor whose DOFs are not the simplices owes the de Rham map into
  /// its own space here.
  fn integral_cocycles(&self, grade: impl Into<ExteriorGrade>) -> Vec<Cochain<i64>>;

  /// Representative integral cycles of a basis of the homology Kronecker-dual
  /// to [`Self::integral_cocycles`], in the same DOF indexing: $H_k (K; ZZ)$
  /// for the full complex, $H_k (K, diff K; ZZ)$ for the relative one. One per
  /// [`Self::harmonic_dim`], the two ranks agreeing by universal coefficients.
  ///
  /// These are the holes a period measures. Pairing a cochain against them
  /// (`simplicial::topology::chain::pairing`) is what pins the harmonic basis
  /// to the individual holes in [`crate::harmonic`], and the pairing matrix of
  /// the two bases is nonsingular over $QQ$, which is what makes that pinning
  /// well posed.
  ///
  /// A period is an integral over a $k$-chain, so an implementor whose DOFs are
  /// not the simplices owes the de Rham map here as it does for the cocycles.
  fn integral_cycles(&self, grade: impl Into<ExteriorGrade>) -> Vec<Chain<i64>>;

  /// The inclusion $E: C^k arrow.hook cal(W) Lambda^k$ of this complex's DOFs
  /// into the ambient Whitney space, extending by zero on the constrained
  /// boundary. The identity on the full complex.
  fn inclusion(&self, grade: impl Into<ExteriorGrade>) -> CsrMatrix;

  /// A [`WhitneyPairing`] on this complex, with the degenerate grades answered
  /// by the correctly shaped empty matrix rather than by an empty assembly.
  ///
  /// Total in grade, and once here rather than in each of the four: a pairing
  /// whose test or trial space is trivial has no entries to assemble, which
  /// happens past either end of the complex and at the ends themselves, where a
  /// differentiated side sits one grade outside.
  fn pairing(&self, form: &WhitneyPairing) -> GalerkinMatrix {
    let (rows, cols) = (
      self.ndofs(form.test_grade()),
      self.ndofs(form.trial_grade()),
    );
    if rows == 0 || cols == 0 {
      return GalerkinMatrix::zeros(rows, cols);
    }
    self.assemble(form)
  }

  /// Galerkin matrix of the $L^2 Lambda^k$ inner product $(u, v)$,
  ///
  /// $M_k = [inner(lambda_tau, lambda_sigma)_(L^2 Lambda^k)]_(sigma tau)$.
  ///
  /// Symmetric positive definite on a Riemannian geometry, symmetric
  /// non-degenerate on any other.
  fn mass(&self, grade: impl Into<ExteriorGrade>) -> GalerkinMatrix {
    self.pairing(&WhitneyPairing::mass(self.dim(), grade))
  }

  /// Galerkin matrix of $(dif u, v)_(L^2 Lambda^k)$, that is $M_k D^(k-1)$,
  /// shape $"ndofs"(k) times "ndofs"(k-1)$: the exterior derivative on the
  /// trial side.
  ///
  /// The transpose of [`dif_test`](Self::dif_test) at the same grade, since the
  /// mass is symmetric on every signature. That adjointness is what makes the
  /// mixed saddle point symmetric.
  fn dif_trial(&self, grade: impl Into<ExteriorGrade>) -> GalerkinMatrix {
    self.pairing(&WhitneyPairing::dif_trial(self.dim(), grade))
  }

  /// Galerkin matrix of $(u, dif tau)_(L^2 Lambda^k)$, that is
  /// $(D^(k-1))^T M_k$, shape $"ndofs"(k-1) times "ndofs"(k)$: the exterior
  /// derivative on the test side.
  ///
  /// This is what a weak codifferential is, and it is not the codifferential:
  /// the mass inverse that would make it one is left to whoever solves with it
  /// ([`codif_cochain`](Self::codif_cochain)), which is exactly what keeps this
  /// sparse.
  fn dif_test(&self, grade: impl Into<ExteriorGrade>) -> GalerkinMatrix {
    self.pairing(&WhitneyPairing::dif_test(self.dim(), grade))
  }

  /// Galerkin matrix of $(dif u, dif v)_(L^2 Lambda^k)$, that is
  /// $(D^(k-1))^T M_k D^(k-1)$, shape $"ndofs"(k-1)^2$: the exterior derivative
  /// on both sides, hence the up-Laplacian $delta dif$ one grade below.
  ///
  /// Assembled from the element form directly, not as the global product of
  /// three assembled matrices. The exterior derivative of a Whitney form is the
  /// coboundary of the *reference* cell, a $plus.minus 1$ incidence, so the
  /// sandwich is already local to a cell. Routing it through the global
  /// matrices instead materializes the whole grade-$k$ mass, over a skeleton
  /// the answer never mentions, only to contract it away: for the vertex
  /// Laplacian in 3D that is the mass on the edges, six times the entries per
  /// cell of the four-by-four this produces.
  fn dif_both(&self, grade: impl Into<ExteriorGrade>) -> GalerkinMatrix {
    self.pairing(&WhitneyPairing::dif_both(self.dim(), grade))
  }

  /// Gram matrix of the full $H Lambda^k (dif)$ inner product,
  /// $M_k + D^T M_(k+1) D$: the $L^2$ mass plus the up-stiffness.
  ///
  /// SPD on a Riemannian geometry, and the diagonal block of the stable mixed
  /// Hodge-Laplace preconditioner on the space $Lambda^k$
  /// (Arnold-Falk-Winther): the norm the formulation is well-posed in. Sparse,
  /// since $dif$ is metric-free and no mass inverse enters.
  fn hdif_gram(&self, grade: impl Into<ExteriorGrade>) -> CsrMatrix {
    let grade = grade.into();
    &self.mass(grade) + &self.dif_both(grade + 1)
  }

  /// The discrete codifferential $delta: Lambda^k -> Lambda^(k-1)$, the
  /// $L^2$-adjoint of $dif$. $sigma = delta u$ is characterized weakly by
  /// $angle.l sigma, tau angle.r = angle.l u, dif tau angle.r$ for all $tau$,
  /// i.e. the mass solve $M_(k-1) sigma = (D^(k-1))^T M_k u$.
  ///
  /// Total in grade: $delta$ maps $C^k$ into $C^(k-1)$, and where that codomain
  /// is trivial, at grade $0$, where it is $Lambda^(-1) = 0$, or past either
  /// end of the complex, the only element is the zero cochain, returned
  /// directly rather than through a degenerate empty solve. This is the
  /// $Z$-graded degree at work: $delta$ of a $0$-form is the empty cochain of
  /// $Lambda^(-1)$, not a missing value. Unlike $dif$, $delta$ is not
  /// metric-free (invariant 5): it carries the mass inverse, realized here as a
  /// solve, well conditioned, since the mass is. Total over signature (the
  /// solve is an LU).
  fn codif_cochain(&self, u: &Cochain) -> Cochain {
    let grade = u.grade();
    let lower = grade - 1;
    if self.ndofs(lower) == 0 {
      return Cochain::new(lower, Vector::zeros(0));
    }
    let mass_lower = self.mass(lower);
    let coupling = self.dif_test(grade);
    let sigma = FaerLu::new(mass_lower).solve(&(coupling * u.coeffs()));
    Cochain::new(lower, sigma)
  }

  /// $dif u$, the exterior derivative of a cochain of this complex.
  ///
  /// Total in grade: at the top the codomain is trivial and the result is the
  /// empty cochain of $Lambda^(n+1) = 0$.
  fn dif_cochain(&self, u: &Cochain) -> Cochain {
    let grade = u.grade();
    Cochain::new(grade + 1, self.dif(grade) * u.coeffs())
  }

  /// $L^2 Lambda^k$ norm of a discrete differential form.
  fn norm_l2(&self, u: &Cochain) -> f64 {
    quadratic_form_sparse(&self.mass(u.grade()), u.coeffs()).sqrt()
  }

  /// $H Lambda^k (dif)$ seminorm: the $L^2$ norm of the exterior derivative.
  ///
  /// Total at the top grade with no guard: $dif u$ there is the empty cochain of
  /// $Lambda^(n+1) = 0$, and its $L^2$ norm against the total $0 times 0$ mass is
  /// $0$.
  fn seminorm_hdif(&self, u: &Cochain) -> f64 {
    self.norm_l2(&self.dif_cochain(u))
  }

  /// The full $H Lambda^k (dif)$ (graph) norm
  /// $norm(u)_(H Lambda(dif))^2 = norm(u)_(L^2)^2 + norm(dif u)_(L^2)^2$.
  ///
  /// The norm the mixed Hodge-Laplacian is well-posed in (Arnold-Falk-Winther),
  /// hence the one its stable block preconditioner is built from. Unlike the
  /// $H^*(delta)$ norm it is sparse: $dif$ is metric-free, so no mass inverse
  /// enters. Its Gram matrix is [`Self::hdif_gram`].
  fn norm_hdif(&self, u: &Cochain) -> f64 {
    self.norm_l2(u).hypot(self.seminorm_hdif(u))
  }

  /// $H^* Lambda^k (delta)$ seminorm: the $L^2$ norm of the codifferential,
  /// $norm(delta u)_(L^2 Lambda^(k-1))$.
  ///
  /// $0$ at grade $0$ ($delta$ maps into the trivial space, so $delta u$ is the
  /// empty cochain and its norm is $0$ with no guard). Unlike
  /// [`Self::seminorm_hdif`] it costs a mass solve, since $delta$ carries the
  /// mass inverse.
  fn seminorm_hcodif(&self, u: &Cochain) -> f64 {
    self.norm_l2(&self.codif_cochain(u))
  }

  /// The full $H^* Lambda^k (delta)$ norm
  /// $norm(u)^2 = norm(u)_(L^2)^2 + norm(delta u)_(L^2)^2$.
  fn norm_hcodif(&self, u: &Cochain) -> f64 {
    self.norm_l2(u).hypot(self.seminorm_hcodif(u))
  }

  /// The Hodge-Laplace energy seminorm
  /// $abs(u)^2 = norm(dif u)_(L^2)^2 + norm(delta u)_(L^2)^2 =
  /// angle.l Delta u, u angle.r$: the form the Hodge-Laplacian is coercive in
  /// (modulo harmonics). The norm convergence rates are naturally measured in.
  fn seminorm_energy(&self, u: &Cochain) -> f64 {
    self.seminorm_hdif(u).hypot(self.seminorm_hcodif(u))
  }

  /// The full $H Lambda^k$ (Hodge-Dirac graph) norm
  /// $norm(u)^2 = norm(u)_(L^2)^2 + norm(dif u)_(L^2)^2 + norm(delta u)_(L^2)^2$:
  /// the graph norm of $D = dif + delta$, the complete energy space of the de
  /// Rham complex, $H Lambda(dif) sect H^* Lambda(delta)$.
  fn norm_full(&self, u: &Cochain) -> f64 {
    self.norm_l2(u).hypot(self.seminorm_energy(u))
  }
}

/// The $L^2 Lambda^k$ pairing of two cochains of a discrete complex,
/// $angle.l u, v angle.r_(L^2) = u^top M_k v$.
///
/// The metric duality, where [`pairing`](derham::pairing) is the
/// metric-free one. A chain-cochain pairing needs nothing but the incidence;
/// this needs the mass matrix, hence a geometry, and the two must not be
/// conflated: the first is a statement about the complex, the second about the
/// manifold it discretizes.
///
/// Symmetric and positive definite on a Riemannian geometry, indefinite on a
/// Lorentzian one, exactly as the metric it is assembled from.
///
/// A free function, not a method: an inner product privileges neither argument.
///
/// # Panics
/// If the grades disagree or either does not match the complex.
pub fn l2_pairing(complex: &impl HilbertComplex, left: &Cochain, right: &Cochain) -> f64 {
  assert_eq!(
    left.grade(),
    right.grade(),
    "an L2 pairing is between cochains of one grade"
  );
  let mass = complex.mass(left.grade());
  assert_eq!(
    mass.ncols(),
    left.coeffs().len(),
    "an L2 pairing is over the complex's own degrees of freedom"
  );
  crate::linalg::bilinear_form_sparse(&mass, left.coeffs(), right.coeffs())
}

/// The discrete Hilbert complex of Whitney forms,
///
/// $cal(W) Lambda^0 -> cal(W) Lambda^1 -> dots.c -> cal(W) Lambda^n$
///
/// with the $L^2 Lambda^k$ inner products: the central object of FEEC.
/// The topology supplies the exterior derivative, the geometry the inner
/// products.
///
/// The geometry is the intrinsic Regge primitive [`MeshLengthsSq`], of any
/// signature: on a Lorentzian geometry the mass matrices carry the indefinite
/// $L^2$ pairing.
#[derive(Clone, Copy)]
pub struct WhitneyComplex<'a> {
  topology: &'a Complex,
  geometry: &'a MeshLengthsSq,
}

impl<'a> WhitneyComplex<'a> {
  pub fn new(topology: &'a Complex, geometry: &'a MeshLengthsSq) -> Self {
    Self { topology, geometry }
  }

  pub fn topology(&self) -> &'a Complex {
    self.topology
  }
  pub fn geometry(&self) -> &'a MeshLengthsSq {
    self.geometry
  }

  /// The relative complex of the pair $(K, diff K)$.
  pub fn relative(self) -> RelativeWhitneyComplex<'a> {
    RelativeWhitneyComplex::new(self)
  }
  /// The relative complex of the pair $(K, Gamma)$ for a boundary part
  /// $Gamma subset.eq diff K$: mixed boundary conditions constrain only the
  /// DOFs on $Gamma$.
  pub fn relative_to(self, constrained: &BoundaryWhitneyComplex) -> RelativeWhitneyComplex<'a> {
    RelativeWhitneyComplex::with_constrained(self, |grade| {
      if grade <= constrained.topology().dim() {
        constrained
          .boundary_complex()
          .parent_kidxs(grade)
          .iter()
          .copied()
          .collect()
      } else {
        HashSet::new()
      }
    })
  }
}

/// The trace geometry of the boundary is the restriction of the squared edge
/// lengths, a pure data restriction that is total on any signature, which is
/// the subsimplex-geometry generalization at work: the boundary facets are
/// subsimplices of the cells, and their induced metric is read off the shared
/// edge lengths directly. On an indefinite parent a null facet carries
/// degenerate induced data, the degeneracy surfaces where a facet metric is
/// actually built, which is the honest mathematical boundary of the concept.
impl<'a> WhitneyComplex<'a> {
  /// The Whitney complex of the boundary $diff K$ with the induced metric,
  /// together with the trace map. `None` on closed manifolds.
  pub fn boundary(&self) -> Option<BoundaryWhitneyComplex> {
    let facets = self.topology.boundary_facets();
    (!facets.is_empty()).then(|| self.boundary_part(facets))
  }

  /// The Whitney complex of a boundary part $Gamma subset.eq diff K$
  /// (a set of boundary facets): the carrier of one kind of mixed boundary
  /// condition.
  pub fn boundary_part(&self, facets: Vec<Facet>) -> BoundaryWhitneyComplex {
    let boundary = self.topology.facet_subcomplex(facets);
    let geometry = boundary.trace_lengths_sq(self.geometry);
    BoundaryWhitneyComplex { boundary, geometry }
  }
}

impl HilbertComplex for WhitneyComplex<'_> {
  fn dim(&self) -> Dim {
    self.topology.dim()
  }

  /// $dim cal(W) Lambda^k$: one DOF per $k$-simplex.
  ///
  /// Total in grade: $0$ outside $[0, n]$, where the space $Lambda^k$ is trivial
  /// and there are no $k$-simplices to carry a DOF.
  fn ndofs(&self, grade: impl Into<ExteriorGrade>) -> usize {
    let grade = grade.into();
    if grade.in_range(self.dim()) {
      self.topology.nsimplices(grade)
    } else {
      0
    }
  }

  /// The form assembled over the mesh, with no restriction: the full complex
  /// is the whole Whitney space, so its inclusion is the identity.
  fn assemble(&self, form: &impl BilinearForm) -> GalerkinMatrix {
    form.assemble(self.topology, self.geometry)
  }

  /// Exterior derivative $dif: cal(W) Lambda^k -> cal(W) Lambda^(k+1)$.
  ///
  /// Purely topological: the coboundary operator on cochains. Total in grade:
  /// the zero map of shape $"ndofs"(k+1) times "ndofs"(k)$ outside $[0, n]$,
  /// where one of $Lambda^k$, $Lambda^(k+1)$ is trivial (the interior top-grade
  /// case $k = n$ is the coboundary's own, already zero-columned codomain).
  fn dif(&self, grade: impl Into<ExteriorGrade>) -> CsrMatrix {
    let grade = grade.into();
    if !grade.in_range(self.dim()) {
      return CsrMatrix::zeros(self.ndofs(grade + 1), self.ndofs(grade));
    }
    CsrMatrix::from(&self.topology.coboundary_operator(grade))
  }

  /// The absolute harmonic space $H^k (K)$: the Betti number $b_k (K)$.
  /// Total in grade: $0$ outside $[0, n]$, where the complex is trivial.
  fn harmonic_dim(&self, grade: impl Into<ExteriorGrade>) -> usize {
    let grade = grade.into();
    if grade.in_range(self.dim()) {
      self.topology.betti_number(grade)
    } else {
      0
    }
  }
  /// The absolute cocycles, on the DOFs of the full complex, which are the
  /// $k$-simplices themselves.
  fn integral_cocycles(&self, grade: impl Into<ExteriorGrade>) -> Vec<Cochain<i64>> {
    let grade = grade.into();
    if grade.in_range(self.dim()) {
      self.topology.cohomology_generators(grade)
    } else {
      Vec::new()
    }
  }
  /// The absolute cycles, on the DOFs of the full complex, which are the
  /// $k$-simplices themselves.
  fn integral_cycles(&self, grade: impl Into<ExteriorGrade>) -> Vec<Chain<i64>> {
    let grade = grade.into();
    if grade.in_range(self.dim()) {
      self.topology.homology_generators(grade)
    } else {
      Vec::new()
    }
  }
  /// No boundary is constrained, so the inclusion is the identity.
  fn inclusion(&self, grade: impl Into<ExteriorGrade>) -> CsrMatrix {
    let grade = grade.into();
    let n = WhitneyComplex::ndofs(self, grade);
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
      coo.push(i, i, 1.0);
    }
    CsrMatrix::from(&coo)
  }
}

/// The Whitney complex of the boundary $diff K$ (the image of the trace map),
/// carrying the geometry induced from the parent mesh.
pub struct BoundaryWhitneyComplex {
  boundary: Subcomplex,
  geometry: MeshLengthsSq,
}

impl BoundaryWhitneyComplex {
  /// The Whitney complex of $diff K$ itself, treated as any other mesh.
  pub fn whitney_complex(&self) -> WhitneyComplex<'_> {
    WhitneyComplex::new(self.boundary.complex(), &self.geometry)
  }
  pub fn topology(&self) -> &Complex {
    self.boundary.complex()
  }
  pub fn geometry(&self) -> &MeshLengthsSq {
    &self.geometry
  }
  pub fn boundary_complex(&self) -> &Subcomplex {
    &self.boundary
  }
  /// Total in grade: $0$ outside $[0, dim diff K]$, where $diff K$ carries no
  /// simplices of that grade.
  pub fn ndofs(&self, grade: impl Into<ExteriorGrade>) -> usize {
    let grade = grade.into();
    let boundary = self.boundary.complex();
    if grade.in_range(boundary.dim()) {
      boundary.nsimplices(grade)
    } else {
      0
    }
  }

  /// The trace $"tr": C^k (K) -> C^k (diff K)$, a cochain map.
  pub fn trace(&self, grade: impl Into<ExteriorGrade>) -> CsrMatrix {
    let grade = grade.into();
    CsrMatrix::from(&self.boundary.trace_operator(grade))
  }
  /// Restrict a cochain on $K$ to the boundary.
  pub fn trace_cochain(&self, u: &Cochain) -> Cochain {
    Cochain::new(u.grade(), self.trace(u.grade()) * u.coeffs())
  }
  /// Extend a boundary cochain by zero onto the full mesh: $"tr"^T$,
  /// the canonical affine lift of essential boundary values.
  pub fn extend_cochain(&self, g: &Cochain) -> Cochain {
    Cochain::new(g.grade(), self.trace(g.grade()).transpose() * g.coeffs())
  }
}

/// The relative Whitney complex of the pair $(K, diff K)$: the subcomplex of
/// cochains with vanishing trace on the boundary, realizing essential
/// (homogeneous Dirichlet) conditions for every grade at once.
///
/// All operators are conjugates $E^T A E$ by the inclusion
/// $E: C^k (K, diff K) arrow.hook C^k (K)$. On a boundaryless mesh this
/// coincides with the full complex.
pub struct RelativeWhitneyComplex<'a> {
  full: WhitneyComplex<'a>,
  /// Per grade: sorted indices of the interior (non-boundary) simplices,
  /// which carry the DOFs of the relative complex.
  interior_simps: Vec<Vec<KSimplexIdx>>,
}

impl<'a> RelativeWhitneyComplex<'a> {
  /// Constrain the full boundary $diff K$.
  pub fn new(full: WhitneyComplex<'a>) -> Self {
    Self::with_constrained(full, |grade| {
      full
        .topology()
        .boundary_simplices(grade)
        .into_iter()
        .map(|idx| idx.kidx)
        .collect()
    })
  }
  /// Constrain the given simplices per grade: the mixed complex
  /// $C^k (K, Gamma)$ of cochains whose trace vanishes on a chosen part
  /// $Gamma subset.eq diff K$ only, the rest of the boundary carrying the
  /// natural condition.
  ///
  /// `constrained` must return the simplices of the closure of $Gamma$, a
  /// half-open part is not a subcomplex, and the conjugates $E^T A E$ would no
  /// longer restrict a cochain complex.
  ///
  /// The two extremes are the familiar ones: all of $diff K$ is
  /// [`Self::new`] (fully essential), the empty set the full
  /// [`WhitneyComplex`] (fully natural). The genuinely mixed choice is what a
  /// hyperbolic problem needs: on a spacetime mesh the Dirichlet part is the
  /// past face together with the timelike sides, the future face left free,
  /// because prescribing data on the whole boundary of a hyperbolic operator is
  /// the ill-posed Hadamard problem rather than a stricter one.
  pub fn with_constrained(
    full: WhitneyComplex<'a>,
    constrained: impl Fn(ExteriorGrade) -> HashSet<KSimplexIdx>,
  ) -> Self {
    let interior_simps = full
      .dim()
      .range_inclusive()
      .map(|grade| {
        let constrained = constrained(grade);
        (0..full.ndofs(grade))
          .filter(|kidx| !constrained.contains(kidx))
          .collect()
      })
      .collect();
    Self {
      full,
      interior_simps,
    }
  }

  pub fn full(&self) -> WhitneyComplex<'a> {
    self.full
  }

  /// Extension by zero of a relative cochain to the full mesh.
  pub fn extend_by_zero(&self, u: &Cochain) -> Cochain {
    Cochain::new(u.grade(), self.inclusion(u.grade()) * u.coeffs())
  }

  /// Restriction of a full cochain to the interior DOFs.
  pub fn restrict(&self, u: &Cochain) -> Cochain {
    Cochain::new(
      u.grade(),
      self.inclusion(u.grade()).transpose() * u.coeffs(),
    )
  }
}

impl HilbertComplex for RelativeWhitneyComplex<'_> {
  fn dim(&self) -> Dim {
    self.full.dim()
  }
  /// Total in grade: $0$ outside $[0, n]$, where the relative complex is trivial.
  fn ndofs(&self, grade: impl Into<ExteriorGrade>) -> usize {
    let grade = grade.into();
    if grade.in_range(self.dim()) {
      self.interior_simps[grade.index()].len()
    } else {
      0
    }
  }

  /// The form assembled on the full complex and restricted to the interior
  /// degrees of freedom, $E_"test"^T A E_"trial"$, which is the Galerkin
  /// discretization on the subspace read literally.
  ///
  /// The restriction is exact, not an approximation of the relative operator,
  /// and the subcomplex property is what makes it so. Writing $P = E E^T$ for
  /// the projection onto the interior DOFs, $dif$ of a boundary-vanishing
  /// cochain vanishes on the boundary, so $P D E = D E$, and a projection
  /// standing between a form's two factors collapses. Hence a differentiated
  /// side needs no relative mass of its own, and none is assembled.
  fn assemble(&self, form: &impl BilinearForm) -> GalerkinMatrix {
    let test = self.inclusion(form.test_grade());
    let trial = self.inclusion(form.trial_grade());
    test.transpose() * self.full.assemble(form) * trial
  }

  /// Exterior derivative on the relative complex: $E_(k+1)^T D E_k$.
  ///
  /// The boundary-vanishing cochains form a subcomplex, so this is a
  /// genuine restriction of the full exterior derivative.
  fn dif(&self, grade: impl Into<ExteriorGrade>) -> CsrMatrix {
    let grade = grade.into();
    self.inclusion(grade + 1).transpose() * self.full.dif(grade) * self.inclusion(grade)
  }

  /// The relative harmonic space $H^k (K, diff K)$: the relative Betti number.
  /// Total in grade: $0$ outside $[0, n]$, where the complex is trivial.
  fn harmonic_dim(&self, grade: impl Into<ExteriorGrade>) -> usize {
    let grade = grade.into();
    if grade.in_range(self.dim()) {
      self.full.topology().relative_betti_number(grade)
    } else {
      0
    }
  }
  /// The relative cocycles, restricted to the interior DOFs by $E^T$. They are
  /// supported in the interior already, so the restriction loses nothing.
  ///
  /// Like [`Self::harmonic_dim`], this reads the pair $(K, diff K)$ even where
  /// only a part $Gamma$ is constrained: the invariants of the genuinely mixed
  /// pair $(K, Gamma)$ are not what either function returns.
  fn integral_cocycles(&self, grade: impl Into<ExteriorGrade>) -> Vec<Cochain<i64>> {
    let grade = grade.into();
    if !grade.in_range(self.dim()) {
      return Vec::new();
    }
    let interior = &self.interior_simps[grade.index()];
    self
      .full
      .topology()
      .relative_cohomology_generators(grade)
      .into_iter()
      .map(|cocycle| {
        Cochain::new(
          grade,
          na::DVector::from_iterator(
            interior.len(),
            interior.iter().map(|&kidx| cocycle.coeffs()[kidx]),
          ),
        )
      })
      .collect()
  }
  /// The relative cycles, restricted to the interior DOFs by $E^T$, exactly as
  /// the relative cocycles are: they are supported in the interior already, so
  /// the restriction loses nothing.
  fn integral_cycles(&self, grade: impl Into<ExteriorGrade>) -> Vec<Chain<i64>> {
    let grade = grade.into();
    if !grade.in_range(self.dim()) {
      return Vec::new();
    }
    let interior = &self.interior_simps[grade.index()];
    self
      .full
      .topology()
      .relative_homology_generators(grade)
      .into_iter()
      .map(|cycle| {
        Chain::new(
          grade,
          na::DVector::from_iterator(
            interior.len(),
            interior.iter().map(|&kidx| cycle.coeffs()[kidx]),
          ),
        )
      })
      .collect()
  }
  /// The inclusion $E: C^k (K, diff K) arrow.hook C^k (K)$,
  /// extending interior cochains by zero onto the boundary.
  ///
  /// A cochain map: $D E_k = E_(k+1) dif_k$. Its transpose restricts
  /// cochains to the interior DOFs. Total in grade: the $0$-columned (or empty)
  /// matrix outside $[0, n]$, since both DOF counts vanish there.
  fn inclusion(&self, grade: impl Into<ExteriorGrade>) -> CsrMatrix {
    let grade = grade.into();
    let mut coo = CooMatrix::new(self.full.ndofs(grade), self.ndofs(grade));
    if grade.in_range(self.dim()) {
      for (relative, &full) in self.interior_simps[grade.index()].iter().enumerate() {
        coo.push(full, relative, 1.0);
      }
    }
    CsrMatrix::from(&coo)
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use derham::Cochain;
  use regge::mesher::cartesian::CartesianGrid;
  use simplicial::Dim;
  use simplicial::linalg::Vector;

  /// The stiffness is one matrix with two routes to it: the element-local
  /// sandwich that [`HilbertComplex::dif_both`] assembles, and the global
  /// product $D^T M_(k+1) D$ of three separately assembled matrices.
  ///
  /// They agree because the exterior derivative of a Whitney form is the
  /// coboundary of the reference cell, so the contraction commutes with the
  /// scatter. Swept over every dimension and grade, the extremes included:
  /// at the top grade both sides are the zero operator, which is the case a
  /// route that special-cased the empty codomain would get wrong.
  #[test]
  fn dif_both_local_and_global_routes_agree() {
    for dim in (1..=3).map(Dim::from) {
      let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      let lengths = coords.to_edge_lengths_sq(&topology);
      let whitney = WhitneyComplex::new(&topology, &lengths);

      for grade in dim.range_inclusive() {
        let local = whitney.dif_both(grade + 1);

        let dif = whitney.dif(grade);
        let mass = whitney.mass(grade + 1);
        let global = dif.transpose() * mass * dif;

        assert_eq!(local.nrows(), global.nrows());
        assert_eq!(local.ncols(), global.ncols());
        let residual = (&local - &global)
          .values()
          .iter()
          .fold(0.0f64, |acc, v| acc.max(v.abs()));
        let scale = local
          .values()
          .iter()
          .fold(0.0f64, |acc, v| acc.max(v.abs()));
        assert!(
          residual <= 1e-12 * scale.max(1.0),
          "dim {dim:?} grade {grade:?}: routes differ by {residual:e}"
        );
        // The law must be able to fail: at every grade below the top the
        // stiffness is a nonzero operator, so agreement is not two zeros
        // matching.
        if grade < dim {
          assert!(
            scale > 1e-6,
            "dim {dim:?} grade {grade:?}: stiffness vanished"
          );
        }
      }
    }
  }

  /// The weak codifferential has the same two routes as the stiffness, and
  /// they agree: the element-local pairing that [`HilbertComplex::dif_test`]
  /// assembles, and the global product $(D^(k-1))^T M_k$ of two assembled
  /// matrices. Swept over every dimension and grade, the degenerate grade $0$
  /// included, where the $sigma$ space is empty and both are the $0 times
  /// "ndofs"(0)$ matrix.
  #[test]
  fn codif_local_and_global_routes_agree() {
    for dim in (1..=3).map(Dim::from) {
      let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      let lengths = coords.to_edge_lengths_sq(&topology);
      let whitney = WhitneyComplex::new(&topology, &lengths);

      for grade in dim.range_inclusive() {
        let local = whitney.dif_test(grade);
        let global = whitney.dif(grade - 1).transpose() * &whitney.mass(grade);

        assert_eq!(local.nrows(), global.nrows());
        assert_eq!(local.ncols(), global.ncols());
        let residual = (&local - &global)
          .values()
          .iter()
          .fold(0.0f64, |acc, v| acc.max(v.abs()));
        let scale = local
          .values()
          .iter()
          .fold(0.0f64, |acc, v| acc.max(v.abs()));
        assert!(
          residual <= 1e-12 * scale.max(1.0),
          "dim {dim:?} grade {grade:?}: routes differ by {residual:e}"
        );
        // Above grade 0 the sigma space is nonempty and the coupling is a
        // nonzero matrix, so the law is not two empties agreeing.
        if grade > 0 {
          assert!(
            scale > 1e-6,
            "dim {dim:?} grade {grade:?}: coupling vanished"
          );
        }
      }
    }
  }

  /// The relative stiffness is the restriction of the full one, which is the
  /// subcomplex property in matrix form and is why the relative complex never
  /// has to assemble a mass one grade up.
  ///
  /// Checked against the definition it replaces, $D_"rel"^T M_"rel" D_"rel"$
  /// built from the relative operators, on a mesh with a genuine boundary so
  /// that the inclusion is not the identity and the law can fail.
  #[test]
  fn the_relative_stiffness_is_the_restriction_of_the_full_one() {
    for dim in (1..=3).map(Dim::from) {
      let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      let lengths = coords.to_edge_lengths_sq(&topology);
      let relative = WhitneyComplex::new(&topology, &lengths).relative();

      for grade in dim.range_inclusive() {
        // Below the top the boundary carries simplices of that grade and the
        // inclusion is a proper one, so the law is not two identities agreeing.
        // At the top there are none: the boundary subcomplex has dimension
        // $n - 1$, so no cell is constrained and the relative complex coincides
        // with the full one there.
        if grade < dim {
          assert!(relative.ndofs(grade) < topology.nsimplices(grade));
        }

        let restricted = relative.dif_both(grade + 1);

        let dif = relative.dif(grade);
        let mass = relative.mass(grade + 1);
        let by_definition = dif.transpose() * mass * dif;

        let residual = (&restricted - &by_definition)
          .values()
          .iter()
          .fold(0.0f64, |acc, v| acc.max(v.abs()));
        let scale = restricted
          .values()
          .iter()
          .fold(0.0f64, |acc, v| acc.max(v.abs()));
        assert!(
          residual <= 1e-12 * scale.max(1.0),
          "dim {dim:?} grade {grade:?}: differs by {residual:e}"
        );
      }
    }
  }

  /// The full $H Lambda(dif)$ norm is the Pythagorean sum of the $L^2$ norm and
  /// the $dif$ seminorm, and its Gram matrix [`HilbertComplex::hdif_gram`]
  /// realizes it as a quadratic form: two views of one inner product.
  #[test]
  fn hdif_norm_and_gram_agree() {
    for dim in (1..=3).map(Dim::from) {
      let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      let lengths = coords.to_edge_lengths_sq(&topology);
      let whitney = WhitneyComplex::new(&topology, &lengths);

      for grade in dim.range_inclusive() {
        let ndofs = topology.nsimplices(grade);
        let u = Cochain::new(
          grade,
          Vector::from_iterator(ndofs, (0..ndofs).map(|i| ((i % 5) as f64) - 2.0)),
        );

        let full = whitney.norm_hdif(&u);
        let pythag = (whitney.norm_l2(&u).powi(2) + whitney.seminorm_hdif(&u).powi(2)).sqrt();
        let gram = quadratic_form_sparse(&whitney.hdif_gram(grade), u.coeffs()).sqrt();

        assert!((full - pythag).abs() < 1e-12, "dim={dim} grade={grade}");
        assert!(
          (full - gram).abs() < 1e-10,
          "dim={dim} grade={grade}: {full} vs {gram}"
        );
        assert!(
          full >= whitney.seminorm_hdif(&u) - 1e-12,
          "full norm dominates seminorm"
        );
      }
    }
  }

  fn sample(grade: ExteriorGrade, topology: &Complex) -> Cochain {
    let ndofs = topology.nsimplices(grade);
    Cochain::new(
      grade,
      Vector::from_iterator(ndofs, (0..ndofs).map(|i| ((i * 3 % 7) as f64) - 3.0)),
    )
  }

  /// The defining law of the codifferential: it is the $L^2$-adjoint of $dif$,
  /// $angle.l delta u, tau angle.r_(k-1) = angle.l u, dif tau angle.r_k$ for
  /// every $tau in Lambda^(k-1)$. Swept over dimension and grade.
  #[test]
  fn codif_is_the_adjoint_of_dif() {
    use crate::linalg::bilinear_form_sparse;
    for dim in (1..=3).map(Dim::from) {
      let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      let lengths = coords.to_edge_lengths_sq(&topology);
      let whitney = WhitneyComplex::new(&topology, &lengths);

      for grade in Dim::ONE.range_to_inclusive(dim) {
        let u = sample(grade, &topology);
        let tau = sample(grade - 1, &topology);
        let sigma = whitney.codif_cochain(&u);

        let mass_lower = whitney.mass(grade - 1);
        let mass_k = whitney.mass(grade);
        let lhs = bilinear_form_sparse(&mass_lower, sigma.coeffs(), tau.coeffs());
        let rhs = bilinear_form_sparse(&mass_k, u.coeffs(), tau.dif(&topology).coeffs());

        assert!(
          (lhs - rhs).abs() < 1e-9,
          "dim={dim} grade={grade}: {lhs} vs {rhs}"
        );
      }
    }
  }

  /// $delta compose delta = 0$: the codifferential is nilpotent, dual to
  /// $dif compose dif = 0$. Needs grade $>= 2$ so both codifferentials land in a
  /// real space.
  #[test]
  fn codif_is_nilpotent() {
    for dim in (2..=3).map(Dim::from) {
      let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      let lengths = coords.to_edge_lengths_sq(&topology);
      let whitney = WhitneyComplex::new(&topology, &lengths);

      for grade in Dim::new(2).range_to_inclusive(dim) {
        let u = sample(grade, &topology);
        let ddu = whitney.codif_cochain(&whitney.codif_cochain(&u));
        assert!(whitney.norm_l2(&ddu) < 1e-9, "dim={dim} grade={grade}");
      }
    }
  }

  /// The energy and full Hodge-Dirac norms decompose as the Pythagorean sums
  /// they are defined to be, total over every grade including the degenerate
  /// $0$ and $n$ where a seminorm vanishes.
  #[test]
  fn delta_norms_are_total_and_pythagorean() {
    for dim in (1..=3).map(Dim::from) {
      let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      let lengths = coords.to_edge_lengths_sq(&topology);
      let whitney = WhitneyComplex::new(&topology, &lengths);

      for grade in dim.range_inclusive() {
        let u = sample(grade, &topology);
        let (l2, hd, hcd) = (
          whitney.norm_l2(&u),
          whitney.seminorm_hdif(&u),
          whitney.seminorm_hcodif(&u),
        );
        assert!((whitney.seminorm_energy(&u) - (hd * hd + hcd * hcd).sqrt()).abs() < 1e-12);
        assert!((whitney.norm_full(&u) - (l2 * l2 + hd * hd + hcd * hcd).sqrt()).abs() < 1e-12);
        if grade == 0 {
          assert_eq!(whitney.seminorm_hcodif(&u), 0.0, "delta = 0 at grade 0");
          // delta u is the empty cochain of the trivial space Lambda^(-1) = 0,
          // not a missing value.
          let du = whitney.codif_cochain(&u);
          assert_eq!(du.grade(), Dim::new(-1));
          assert_eq!(du.coeffs().len(), 0);
        }
      }
    }
  }

  /// The de Rham operators are total at the trivial ends: a degree past either
  /// end of the complex names the zero space $Lambda^k = 0$ ($k in.not [0, n]$),
  /// so every accessor returns the correctly-shaped empty object rather than
  /// panicking. This is the $Z$-graded degree cashed out, one step past each
  /// end runs the same code and returns the mathematically trivial answer.
  #[test]
  fn operators_are_total_at_the_trivial_ends() {
    for dim in (1..=3).map(Dim::from) {
      let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      let lengths = coords.to_edge_lengths_sq(&topology);
      let whitney = WhitneyComplex::new(&topology, &lengths);

      let ndofs0 = whitney.ndofs(Dim::ZERO);
      let ndofs_top = whitney.ndofs(dim);

      for ghost in [Dim::new(-1), dim + 1] {
        assert_eq!(whitney.ndofs(ghost), 0, "dim={dim} ghost={ghost}");
        assert_eq!(whitney.harmonic_dim(ghost), 0);

        let mass = whitney.mass(ghost);
        assert_eq!((mass.nrows(), mass.ncols()), (0, 0));

        let ldd = whitney.dif_both(ghost + 1);
        assert_eq!((ldd.nrows(), ldd.ncols()), (0, 0));

        let incl = HilbertComplex::inclusion(&whitney, ghost);
        assert_eq!((incl.nrows(), incl.ncols()), (0, 0));
      }

      // $dif$ at each end keeps one honest zero dimension: $dif^(-1): 0 ->
      // Lambda^0$ is $"ndofs"(0) times 0$, the top $dif^n: Lambda^n -> 0$ is
      // $0 times "ndofs"(n)$.
      let d_below = whitney.dif(Dim::new(-1));
      assert_eq!((d_below.nrows(), d_below.ncols()), (ndofs0, 0));
      let d_top = whitney.dif(dim);
      assert_eq!((d_top.nrows(), d_top.ncols()), (0, ndofs_top));

      // The top-grade stiffness is the honest $"ndofs"(n)^2$ zero operator, from
      // the general formula with no special case.
      let ldd_top = whitney.dif_both(dim + 1);
      assert_eq!((ldd_top.nrows(), ldd_top.ncols()), (ndofs_top, ndofs_top));

      assert!(ldd_top.values().iter().all(|&v| v == 0.0), "dim={dim}");
    }
  }

  /// The $L^2$ pairing is symmetric and positive definite on a Riemannian
  /// geometry, and it is not the chain-cochain pairing.
  ///
  /// The two dualities a discrete complex carries, kept apart. The metric-free
  /// one integrates a cochain over a chain and needs only the incidence; this
  /// one needs the mass matrix, hence a geometry. Asserting they disagree on a
  /// nontrivial input is what stops a later refactor from quietly routing one
  /// through the other.
  #[test]
  fn the_l2_pairing_is_the_metric_duality_and_the_other_is_not() {
    use approx::assert_relative_eq;
    use derham::pairing;
    use simplicial::topology::chain::Chain;

    for dim in 1..=3 {
      let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      let geometry = coords.to_edge_lengths_sq(&topology);
      let complex = WhitneyComplex::new(&topology, &geometry);

      for grade in 0..=dim {
        let ndofs = complex.ndofs(grade);
        let u = Cochain::new(grade, Vector::from_fn(ndofs, |i, _| ((i % 5) as f64) - 2.0));
        let v = Cochain::new(grade, Vector::from_fn(ndofs, |i, _| ((i % 7) as f64) - 3.0));

        assert_relative_eq!(
          l2_pairing(&complex, &u, &v),
          l2_pairing(&complex, &v, &u),
          epsilon = 1e-10
        );
        assert!(
          l2_pairing(&complex, &u, &u) > 0.0,
          "dim {dim} grade {grade}: the L2 pairing is not positive definite"
        );

        // The same cochain against the chain with those coefficients: no
        // geometry consulted, and a different number.
        let chain = Chain::from_vec(grade, v.coeffs().iter().map(|c| c.round() as i64).collect());
        let combinatorial = pairing(&u, &chain.extend_scalars(|&c| c as f64));
        assert!(
          (combinatorial - l2_pairing(&complex, &u, &v)).abs() > 1e-9,
          "dim {dim} grade {grade}: the two pairings coincide, so one is not what it claims"
        );
      }
    }
  }
}
