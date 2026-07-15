//! The Whitney finite element complex: the FEEC discretization of the
//! L^2 de Rham complex on a simplicial Riemannian manifold.

use crate::{
  assemble::{assemble_galmat, GalMat},
  operators::HodgeMassElmat,
};

use {
  common::linalg::nalgebra::{quadratic_form_sparse, CooMatrix, CsrMatrix},
  ddf::cochain::Cochain,
  exterior::ExteriorGrade,
  manifold::{
    geometry::metric::mesh::MeshLengths,
    topology::{
      boundary::BoundaryComplex,
      complex::Complex,
      handle::{KSimplexIdx, SimplexIdx},
    },
    Dim,
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
  fn ndofs(&self, grade: ExteriorGrade) -> usize;
  fn mass(&self, grade: ExteriorGrade) -> GalMat;
  fn dif(&self, grade: ExteriorGrade) -> CsrMatrix;
  fn codif_dif(&self, grade: ExteriorGrade) -> GalMat;

  /// The dimension of the discrete harmonic space $cal(H)^k$: the Betti number
  /// of the complex by the discrete Hodge theorem ($b_k (K)$ for the full
  /// complex, $b_k (K, diff K)$ for the relative one), an exact topological
  /// invariant.
  fn harmonic_dim(&self, grade: ExteriorGrade) -> usize;

  /// The inclusion $E: C^k arrow.hook cal(W) Lambda^k$ of this complex's DOFs
  /// into the ambient Whitney space, extending by zero on the constrained
  /// boundary. The identity on the full complex.
  fn inclusion(&self, grade: ExteriorGrade) -> CsrMatrix;
}

/// The discrete Hilbert complex of Whitney forms,
///
/// $cal(W) Lambda^0 -> cal(W) Lambda^1 -> dots.c -> cal(W) Lambda^n$
///
/// with the $L^2 Lambda^k$ inner products: the central object of FEEC.
/// The topology supplies the exterior derivative, the geometry the inner
/// products.
#[derive(Copy, Clone)]
pub struct WhitneyComplex<'a> {
  topology: &'a Complex,
  geometry: &'a MeshLengths,
}

impl<'a> WhitneyComplex<'a> {
  pub fn new(topology: &'a Complex, geometry: &'a MeshLengths) -> Self {
    Self { topology, geometry }
  }

  pub fn dim(&self) -> Dim {
    self.topology.dim()
  }
  pub fn topology(&self) -> &'a Complex {
    self.topology
  }
  pub fn geometry(&self) -> &'a MeshLengths {
    self.geometry
  }

  /// $dim cal(W) Lambda^k$: one DOF per $k$-simplex.
  pub fn ndofs(&self, grade: ExteriorGrade) -> usize {
    self.topology.nsimplices(grade)
  }

  /// Galerkin mass matrix of the $L^2 Lambda^k$ inner product,
  ///
  /// $M = [inner(lambda_tau, lambda_sigma)_(L^2 Lambda^k)]_(sigma tau)$
  pub fn mass(&self, grade: ExteriorGrade) -> GalMat {
    assemble_galmat(
      self.topology,
      self.geometry,
      HodgeMassElmat::new(self.dim(), grade),
    )
  }

  /// Exterior derivative $dif: cal(W) Lambda^k -> cal(W) Lambda^(k+1)$.
  ///
  /// Purely topological: the coboundary operator on cochains.
  pub fn dif(&self, grade: ExteriorGrade) -> CsrMatrix {
    CsrMatrix::from(&self.topology.coboundary_operator(grade))
  }

  /// Galerkin matrix of the bilinear form $(dif u, dif v)_(L^2 Lambda^(k+1))$,
  ///
  /// the stiffness matrix $D^T M_(k+1) D$ of the up-part of the Hodge-Laplacian.
  pub fn codif_dif(&self, grade: ExteriorGrade) -> GalMat {
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
    self.norm_l2(&u.dif(self.topology))
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

  /// The Whitney complex of the boundary $diff K$ with the induced metric,
  /// together with the trace map. `None` on closed manifolds.
  pub fn boundary(&self) -> Option<BoundaryWhitneyComplex> {
    let facets = self.topology.boundary_facets();
    (!facets.is_empty()).then(|| self.boundary_part(facets))
  }

  /// The Whitney complex of a boundary part $Gamma subset.eq diff K$
  /// (a set of boundary facets): the carrier of one kind of mixed boundary
  /// condition.
  pub fn boundary_part(&self, facets: Vec<SimplexIdx>) -> BoundaryWhitneyComplex {
    let boundary = self.topology.facet_subcomplex(facets);
    let geometry = boundary.trace_lengths(self.geometry);
    BoundaryWhitneyComplex { boundary, geometry }
  }
}

impl HilbertComplex for WhitneyComplex<'_> {
  fn dim(&self) -> Dim {
    WhitneyComplex::dim(self)
  }
  fn ndofs(&self, grade: ExteriorGrade) -> usize {
    WhitneyComplex::ndofs(self, grade)
  }
  fn mass(&self, grade: ExteriorGrade) -> GalMat {
    WhitneyComplex::mass(self, grade)
  }
  fn dif(&self, grade: ExteriorGrade) -> CsrMatrix {
    WhitneyComplex::dif(self, grade)
  }
  fn codif_dif(&self, grade: ExteriorGrade) -> GalMat {
    WhitneyComplex::codif_dif(self, grade)
  }
  /// The absolute harmonic space $H^k (K)$: the Betti number $b_k (K)$.
  fn harmonic_dim(&self, grade: ExteriorGrade) -> usize {
    self.topology.betti_number(grade)
  }
  /// No boundary is constrained, so the inclusion is the identity.
  fn inclusion(&self, grade: ExteriorGrade) -> CsrMatrix {
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
  geometry: MeshLengths,
}

impl BoundaryWhitneyComplex {
  /// The Whitney complex of $diff K$ itself, treated as any other mesh.
  pub fn whitney_complex(&self) -> WhitneyComplex<'_> {
    WhitneyComplex::new(self.boundary.complex(), &self.geometry)
  }
  pub fn topology(&self) -> &Complex {
    self.boundary.complex()
  }
  pub fn geometry(&self) -> &MeshLengths {
    &self.geometry
  }
  pub fn boundary_complex(&self) -> &BoundaryComplex {
    &self.boundary
  }
  pub fn ndofs(&self, grade: ExteriorGrade) -> usize {
    self.boundary.complex().nsimplices(grade)
  }

  /// The trace $"tr": C^k (K) -> C^k (diff K)$, a cochain map.
  pub fn trace(&self, grade: ExteriorGrade) -> CsrMatrix {
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
  /// Constrain the given simplices per grade (the closure of the Dirichlet
  /// boundary part).
  fn with_constrained(
    full: WhitneyComplex<'a>,
    constrained: impl Fn(ExteriorGrade) -> HashSet<KSimplexIdx>,
  ) -> Self {
    let interior_simps = (0..=full.dim())
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
  pub fn ndofs(&self, grade: ExteriorGrade) -> usize {
    self.interior_simps[grade].len()
  }

  /// The inclusion $E: C^k (K, diff K) arrow.hook C^k (K)$,
  /// extending interior cochains by zero onto the boundary.
  ///
  /// A cochain map: $D E_k = E_(k+1) dif_k$. Its transpose restricts
  /// cochains to the interior DOFs.
  pub fn inclusion(&self, grade: ExteriorGrade) -> CsrMatrix {
    let mut coo = CooMatrix::new(self.full.ndofs(grade), self.ndofs(grade));
    for (relative, &full) in self.interior_simps[grade].iter().enumerate() {
      coo.push(full, relative, 1.0);
    }
    CsrMatrix::from(&coo)
  }

  /// Galerkin mass matrix on the relative complex: $E^T M E$.
  pub fn mass(&self, grade: ExteriorGrade) -> GalMat {
    let incl = self.inclusion(grade);
    let mass = CsrMatrix::from(&self.full.mass(grade));
    GalMat::from(&(incl.transpose() * mass * incl))
  }

  /// Exterior derivative on the relative complex: $E_(k+1)^T D E_k$.
  ///
  /// The boundary-vanishing cochains form a subcomplex, so this is a
  /// genuine restriction of the full exterior derivative.
  pub fn dif(&self, grade: ExteriorGrade) -> CsrMatrix {
    self.inclusion(grade + 1).transpose() * self.full.dif(grade) * self.inclusion(grade)
  }

  /// Galerkin matrix of $(dif u, dif v)_(L^2 Lambda^(k+1))$ on the
  /// relative complex.
  pub fn codif_dif(&self, grade: ExteriorGrade) -> GalMat {
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
  fn ndofs(&self, grade: ExteriorGrade) -> usize {
    RelativeWhitneyComplex::ndofs(self, grade)
  }
  fn mass(&self, grade: ExteriorGrade) -> GalMat {
    RelativeWhitneyComplex::mass(self, grade)
  }
  fn dif(&self, grade: ExteriorGrade) -> CsrMatrix {
    RelativeWhitneyComplex::dif(self, grade)
  }
  fn codif_dif(&self, grade: ExteriorGrade) -> GalMat {
    RelativeWhitneyComplex::codif_dif(self, grade)
  }
  /// The relative harmonic space $H^k (K, diff K)$: the relative Betti number.
  fn harmonic_dim(&self, grade: ExteriorGrade) -> usize {
    self.full.topology().relative_betti_number(grade)
  }
  fn inclusion(&self, grade: ExteriorGrade) -> CsrMatrix {
    RelativeWhitneyComplex::inclusion(self, grade)
  }
}
