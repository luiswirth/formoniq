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
    topology::{complex::Complex, handle::KSimplexIdx},
    Dim,
  },
};

use std::collections::HashSet;

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
}

/// The relative Whitney complex of the pair $(K, diff K)$:
/// Whitney forms with vanishing trace on the boundary.
///
/// The subcomplex of cochains vanishing on all boundary simplices. This
/// realizes essential (homogeneous Dirichlet) boundary conditions for every
/// grade at once, replacing ad-hoc DOF surgery: all operators are conjugates
/// $E^T A E$ by the inclusion $E: C^k (K, diff K) arrow.hook C^k (K)$.
///
/// On a boundaryless mesh this coincides with the full complex.
pub struct RelativeWhitneyComplex<'a> {
  full: WhitneyComplex<'a>,
  /// Per grade: sorted indices of the interior (non-boundary) simplices,
  /// which carry the DOFs of the relative complex.
  interior_simps: Vec<Vec<KSimplexIdx>>,
}

impl<'a> RelativeWhitneyComplex<'a> {
  pub fn new(full: WhitneyComplex<'a>) -> Self {
    let interior_simps = (0..=full.dim())
      .map(|grade| {
        let boundary: HashSet<KSimplexIdx> = full
          .topology()
          .boundary_simplices(grade)
          .into_iter()
          .map(|idx| idx.kidx)
          .collect();
        (0..full.ndofs(grade))
          .filter(|kidx| !boundary.contains(kidx))
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
