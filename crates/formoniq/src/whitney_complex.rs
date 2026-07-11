//! The Whitney finite element complex: the FEEC discretization of the
//! L^2 de Rham complex on a simplicial Riemannian manifold.

use crate::{
  assemble::{assemble_galmat, GalMat},
  operators::HodgeMassElmat,
};

use {
  common::linalg::nalgebra::{quadratic_form_sparse, CsrMatrix},
  ddf::cochain::Cochain,
  exterior::ExteriorGrade,
  manifold::{geometry::metric::mesh::MeshLengths, topology::complex::Complex, Dim},
};

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
}
