//! The Whitney finite element complex: the FEEC discretization of the
//! L^2 de Rham complex on a simplicial pseudo-Riemannian manifold.
//!
//! The geometry is the intrinsic Regge primitive, [`MeshLengthsSq`], of any
//! signature: a Riemannian manifold given by edge lengths and a Lorentzian
//! spacetime given by signed edge lengths discretize through the same code. An
//! embedded mesh or raw per-cell metrics reach it by converting to edge
//! lengths at the door. On an indefinite metric the
//! $L^2 Lambda^k$ "inner products" are the indefinite $L^2$ pairings and the
//! mass matrices are symmetric non-degenerate rather than s.p.d. -- which is
//! the honest structural difference, not a separate code path.

use crate::{
  assemble::{GalMat, assemble_galmat},
  linalg::faer::FaerLu,
  operators::HodgeMassElmat,
};

use {
  crate::linalg::quadratic_form_sparse,
  derham::cochain::Cochain,
  exterior::ExteriorGrade,
  simplicial::{
    Dim,
    geometry::metric::mesh::MeshLengthsSq,
    linalg::{CooMatrix, CsrMatrix},
    topology::{boundary::BoundaryComplex, complex::Complex, handle::KSimplexIdx, role::Facet},
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
/// solver is one monomorphized piece of code over both --- the choice of
/// boundary condition is just the choice of complex.
///
/// The [`Self::inclusion`] $E: cal(W)^"rel" Lambda^k arrow.hook cal(W) Lambda^k$
/// (the identity for the full complex) is what lets the solver take its source
/// and return its solution in the ambient $cal(W) Lambda^k$ regardless: it
/// restricts the right-hand side by $E^T$ and extends the solution by $E$.
pub trait HilbertComplex {
  fn dim(&self) -> Dim;
  fn ndofs(&self, grade: impl Into<ExteriorGrade>) -> usize;
  fn mass(&self, grade: impl Into<ExteriorGrade>) -> GalMat;
  fn dif(&self, grade: impl Into<ExteriorGrade>) -> CsrMatrix;
  fn codif_dif(&self, grade: impl Into<ExteriorGrade>) -> GalMat;

  /// The dimension of the discrete harmonic space $cal(H)^k$: the Betti number
  /// of the complex by the discrete Hodge theorem ($b_k (K)$ for the full
  /// complex, $b_k (K, diff K)$ for the relative one), an exact topological
  /// invariant.
  fn harmonic_dim(&self, grade: impl Into<ExteriorGrade>) -> usize;

  /// The inclusion $E: C^k arrow.hook cal(W) Lambda^k$ of this complex's DOFs
  /// into the ambient Whitney space, extending by zero on the constrained
  /// boundary. The identity on the full complex.
  fn inclusion(&self, grade: impl Into<ExteriorGrade>) -> CsrMatrix;

  /// Gram matrix of the full $H Lambda^k (dif)$ inner product,
  /// $M_k + D^T M_(k+1) D$: the $L^2$ mass plus the up-stiffness.
  ///
  /// SPD on a Riemannian geometry, and the diagonal block of the stable mixed
  /// Hodge-Laplace preconditioner on the space $Lambda^k$
  /// (Arnold-Falk-Winther): the norm the formulation is well-posed in. Sparse,
  /// since $dif$ is metric-free and no mass inverse enters.
  fn hdif_gram(&self, grade: impl Into<ExteriorGrade>) -> CsrMatrix {
    let grade = grade.into();
    &CsrMatrix::from(&self.mass(grade)) + &CsrMatrix::from(&self.codif_dif(grade))
  }

  /// The discrete codifferential $delta: Lambda^k -> Lambda^(k-1)$, the
  /// $L^2$-adjoint of $dif$. $sigma = delta u$ is characterized weakly by
  /// $angle.l sigma, tau angle.r = angle.l u, dif tau angle.r$ for all $tau$,
  /// i.e. the mass solve $M_(k-1) sigma = (D^(k-1))^T M_k u$.
  ///
  /// `None` at grade $0$: $delta$ maps $Lambda^0$ into the trivial space
  /// $Lambda^(-1) = 0$, so the only codomain element is the zero cochain, which
  /// `None` names without materializing an empty space. Unlike $dif$, $delta$ is
  /// *not* metric-free (invariant 5): it carries the mass inverse, realized here
  /// as a solve --- well conditioned, since the mass is. Total over signature
  /// (the solve is an LU).
  fn codif(&self, u: &Cochain) -> Option<Cochain> {
    let grade = u.grade();
    if grade == 0 {
      return None;
    }
    let mass_lower = CsrMatrix::from(&self.mass(grade - 1));
    let coupling = self.dif(grade - 1).transpose() * &CsrMatrix::from(&self.mass(grade));
    let sigma = FaerLu::new(mass_lower).solve(&(coupling * u.coeffs()));
    Some(Cochain::new(grade - 1, sigma))
  }
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

  pub fn dim(&self) -> Dim {
    self.topology.dim()
  }
  pub fn topology(&self) -> &'a Complex {
    self.topology
  }
  pub fn geometry(&self) -> &'a MeshLengthsSq {
    self.geometry
  }

  /// $dim cal(W) Lambda^k$: one DOF per $k$-simplex.
  pub fn ndofs(&self, grade: impl Into<ExteriorGrade>) -> usize {
    let grade = grade.into();
    self.topology.nsimplices(grade)
  }

  /// Galerkin mass matrix of the $L^2 Lambda^k$ inner product,
  ///
  /// $M = [inner(lambda_tau, lambda_sigma)_(L^2 Lambda^k)]_(sigma tau)$
  pub fn mass(&self, grade: impl Into<ExteriorGrade>) -> GalMat {
    let grade = grade.into();
    assemble_galmat(
      self.topology,
      self.geometry,
      HodgeMassElmat::new(self.dim(), grade),
    )
  }

  /// Exterior derivative $dif: cal(W) Lambda^k -> cal(W) Lambda^(k+1)$.
  ///
  /// Purely topological: the coboundary operator on cochains.
  pub fn dif(&self, grade: impl Into<ExteriorGrade>) -> CsrMatrix {
    let grade = grade.into();
    CsrMatrix::from(&self.topology.coboundary_operator(grade))
  }

  /// Galerkin matrix of the bilinear form $(dif u, dif v)_(L^2 Lambda^(k+1))$,
  ///
  /// the stiffness matrix $D^T M_(k+1) D$ of the up-part of the Hodge-Laplacian.
  pub fn codif_dif(&self, grade: impl Into<ExteriorGrade>) -> GalMat {
    let grade = grade.into();
    // At top grade $dif: Lambda^n -> Lambda^(n+1)$ maps into the zero space, so
    // $delta dif$ is the zero operator; there is no $(n+1)$-skeleton to assemble
    // a mass over. Return it explicitly, sized $"ndofs"^2$, keeping the operator
    // total at the degenerate top grade rather than indexing past the skeleton.
    if grade == self.dim() {
      return GalMat::new(self.ndofs(grade), self.ndofs(grade));
    }
    let dif = self.dif(grade);
    let mass = CsrMatrix::from(&self.mass(grade + 1));
    GalMat::from(&(dif.transpose() * mass * dif))
  }

  /// $L^2 Lambda^k$ norm of a discrete differential form.
  pub fn norm_l2(&self, u: &Cochain) -> f64 {
    let mass = CsrMatrix::from(&self.mass(u.grade()));
    quadratic_form_sparse(&mass, u.coeffs()).sqrt()
  }

  /// $H Lambda^k (dif)$ seminorm: the $L^2$ norm of the exterior derivative.
  pub fn seminorm_hdif(&self, u: &Cochain) -> f64 {
    // At top grade $dif$ maps into the zero space $Lambda^(n+1)$, so the
    // seminorm is $0$ and there is no $(n+1)$-skeleton to assemble a mass over
    // (cf. [`Self::codif_dif`]): total at the degenerate top grade rather than
    // indexing past the skeleton.
    if u.grade() == self.dim() {
      return 0.0;
    }
    self.norm_l2(&u.dif(self.topology))
  }

  /// The full $H Lambda^k (dif)$ (graph) norm
  /// $norm(u)_(H Lambda(dif))^2 = norm(u)_(L^2)^2 + norm(dif u)_(L^2)^2$.
  ///
  /// The norm the mixed Hodge-Laplacian is well-posed in (Arnold-Falk-Winther),
  /// hence the one its stable block preconditioner is built from. Unlike the
  /// $H^*(delta)$ norm it is sparse: $dif$ is metric-free, so no mass inverse
  /// enters. Its Gram matrix is [`Self::hdif_gram`].
  pub fn norm_hdif(&self, u: &Cochain) -> f64 {
    (self.norm_l2(u).powi(2) + self.seminorm_hdif(u).powi(2)).sqrt()
  }

  /// $H^* Lambda^k (delta)$ seminorm: the $L^2$ norm of the codifferential,
  /// $norm(delta u)_(L^2 Lambda^(k-1))$.
  ///
  /// $0$ at grade $0$ ($delta$ maps into the trivial space). Unlike
  /// [`Self::seminorm_hdif`] it costs a mass solve, since $delta$ carries the
  /// mass inverse.
  pub fn seminorm_hcodif(&self, u: &Cochain) -> f64 {
    match self.codif(u) {
      Some(sigma) => self.norm_l2(&sigma),
      None => 0.0,
    }
  }

  /// The full $H^* Lambda^k (delta)$ norm
  /// $norm(u)^2 = norm(u)_(L^2)^2 + norm(delta u)_(L^2)^2$.
  pub fn norm_hcodif(&self, u: &Cochain) -> f64 {
    (self.norm_l2(u).powi(2) + self.seminorm_hcodif(u).powi(2)).sqrt()
  }

  /// The Hodge-Laplace *energy* seminorm
  /// $abs(u)^2 = norm(dif u)_(L^2)^2 + norm(delta u)_(L^2)^2 =
  /// angle.l Delta u, u angle.r$: the form the Hodge-Laplacian is coercive in
  /// (modulo harmonics). The norm convergence rates are naturally measured in.
  pub fn seminorm_energy(&self, u: &Cochain) -> f64 {
    (self.seminorm_hdif(u).powi(2) + self.seminorm_hcodif(u).powi(2)).sqrt()
  }

  /// The full $H Lambda^k$ (Hodge-Dirac graph) norm
  /// $norm(u)^2 = norm(u)_(L^2)^2 + norm(dif u)_(L^2)^2 + norm(delta u)_(L^2)^2$:
  /// the graph norm of $D = dif + delta$, the complete energy space of the de
  /// Rham complex, $H Lambda(dif) sect H^* Lambda(delta)$.
  pub fn norm_full(&self, u: &Cochain) -> f64 {
    (self.norm_l2(u).powi(2) + self.seminorm_hdif(u).powi(2) + self.seminorm_hcodif(u).powi(2))
      .sqrt()
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
/// lengths, a pure data restriction that is total on any signature -- which is
/// the subsimplex-geometry generalization at work: the boundary facets are
/// subsimplices of the cells, and their induced metric is read off the shared
/// edge lengths directly. On an indefinite parent a *null* facet carries
/// degenerate induced data -- the degeneracy surfaces where a facet metric is
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
    WhitneyComplex::dim(self)
  }
  fn ndofs(&self, grade: impl Into<ExteriorGrade>) -> usize {
    let grade = grade.into();
    WhitneyComplex::ndofs(self, grade)
  }
  fn mass(&self, grade: impl Into<ExteriorGrade>) -> GalMat {
    let grade = grade.into();
    WhitneyComplex::mass(self, grade)
  }
  fn dif(&self, grade: impl Into<ExteriorGrade>) -> CsrMatrix {
    let grade = grade.into();
    WhitneyComplex::dif(self, grade)
  }
  fn codif_dif(&self, grade: impl Into<ExteriorGrade>) -> GalMat {
    let grade = grade.into();
    WhitneyComplex::codif_dif(self, grade)
  }
  /// The absolute harmonic space $H^k (K)$: the Betti number $b_k (K)$.
  fn harmonic_dim(&self, grade: impl Into<ExteriorGrade>) -> usize {
    let grade = grade.into();
    self.topology.betti_number(grade)
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
  boundary: BoundaryComplex,
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
  pub fn boundary_complex(&self) -> &BoundaryComplex {
    &self.boundary
  }
  pub fn ndofs(&self, grade: impl Into<ExteriorGrade>) -> usize {
    let grade = grade.into();
    self.boundary.complex().nsimplices(grade)
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
  /// Constrain the given simplices per grade: the *mixed* complex
  /// $C^k (K, Gamma)$ of cochains whose trace vanishes on a chosen part
  /// $Gamma subset.eq diff K$ only, the rest of the boundary carrying the
  /// natural condition.
  ///
  /// `constrained` must return the simplices of the *closure* of $Gamma$ --- a
  /// half-open part is not a subcomplex, and the conjugates $E^T A E$ would no
  /// longer restrict a cochain complex.
  ///
  /// The two extremes are the familiar ones: all of $diff K$ is
  /// [`Self::new`] (fully essential), the empty set the full
  /// [`WhitneyComplex`] (fully natural). The genuinely mixed choice is what a
  /// *hyperbolic* problem needs: on a spacetime mesh the Dirichlet part is the
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
  pub fn dim(&self) -> Dim {
    self.full.dim()
  }
  pub fn ndofs(&self, grade: impl Into<ExteriorGrade>) -> usize {
    let grade = grade.into();
    self.interior_simps[grade.index()].len()
  }

  /// The inclusion $E: C^k (K, diff K) arrow.hook C^k (K)$,
  /// extending interior cochains by zero onto the boundary.
  ///
  /// A cochain map: $D E_k = E_(k+1) dif_k$. Its transpose restricts
  /// cochains to the interior DOFs.
  pub fn inclusion(&self, grade: impl Into<ExteriorGrade>) -> CsrMatrix {
    let grade = grade.into();
    let mut coo = CooMatrix::new(self.full.ndofs(grade), self.ndofs(grade));
    for (relative, &full) in self.interior_simps[grade.index()].iter().enumerate() {
      coo.push(full, relative, 1.0);
    }
    CsrMatrix::from(&coo)
  }

  /// Galerkin mass matrix on the relative complex: $E^T M E$.
  pub fn mass(&self, grade: impl Into<ExteriorGrade>) -> GalMat {
    let grade = grade.into();
    let incl = self.inclusion(grade);
    let mass = CsrMatrix::from(&self.full.mass(grade));
    GalMat::from(&(incl.transpose() * mass * incl))
  }

  /// Exterior derivative on the relative complex: $E_(k+1)^T D E_k$.
  ///
  /// The boundary-vanishing cochains form a subcomplex, so this is a
  /// genuine restriction of the full exterior derivative.
  pub fn dif(&self, grade: impl Into<ExteriorGrade>) -> CsrMatrix {
    let grade = grade.into();
    self.inclusion(grade + 1).transpose() * self.full.dif(grade) * self.inclusion(grade)
  }

  /// Galerkin matrix of $(dif u, dif v)_(L^2 Lambda^(k+1))$ on the
  /// relative complex.
  pub fn codif_dif(&self, grade: impl Into<ExteriorGrade>) -> GalMat {
    let grade = grade.into();
    // At top grade $dif: Lambda^n -> Lambda^(n+1)$ maps into the zero space, so
    // $delta dif$ is the zero operator; there is no $(n+1)$-skeleton to assemble
    // a mass over. Return it explicitly, sized $"ndofs"^2$, keeping the operator
    // total at the degenerate top grade rather than indexing past the skeleton.
    if grade == self.dim() {
      return GalMat::new(self.ndofs(grade), self.ndofs(grade));
    }
    let dif = self.dif(grade);
    let mass = CsrMatrix::from(&self.mass(grade + 1));
    GalMat::from(&(dif.transpose() * mass * dif))
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
    RelativeWhitneyComplex::dim(self)
  }
  fn ndofs(&self, grade: impl Into<ExteriorGrade>) -> usize {
    let grade = grade.into();
    RelativeWhitneyComplex::ndofs(self, grade)
  }
  fn mass(&self, grade: impl Into<ExteriorGrade>) -> GalMat {
    let grade = grade.into();
    RelativeWhitneyComplex::mass(self, grade)
  }
  fn dif(&self, grade: impl Into<ExteriorGrade>) -> CsrMatrix {
    let grade = grade.into();
    RelativeWhitneyComplex::dif(self, grade)
  }
  fn codif_dif(&self, grade: impl Into<ExteriorGrade>) -> GalMat {
    let grade = grade.into();
    RelativeWhitneyComplex::codif_dif(self, grade)
  }
  /// The relative harmonic space $H^k (K, diff K)$: the relative Betti number.
  fn harmonic_dim(&self, grade: impl Into<ExteriorGrade>) -> usize {
    let grade = grade.into();
    self.full.topology().relative_betti_number(grade)
  }
  fn inclusion(&self, grade: impl Into<ExteriorGrade>) -> CsrMatrix {
    let grade = grade.into();
    RelativeWhitneyComplex::inclusion(self, grade)
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use derham::cochain::Cochain;
  use simplicial::Dim;
  use simplicial::linalg::Vector;
  use simplicial::mesher::cartesian::CartesianGrid;

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
        let sigma = whitney.codif(&u).expect("grade >= 1");

        let mass_lower = CsrMatrix::from(&whitney.mass(grade - 1));
        let mass_k = CsrMatrix::from(&whitney.mass(grade));
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
        let ddu = whitney.codif(&whitney.codif(&u).unwrap()).unwrap();
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
          assert!(whitney.codif(&u).is_none());
        }
      }
    }
  }
}
