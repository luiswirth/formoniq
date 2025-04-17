use crate::{
  assemble::assemble_galmat,
  operators::{CodifDifElmat, HodgeMassElmat},
};

use {
  common::linalg::nalgebra::{quadratic_form_sparse, CsrMatrix},
  ddf::cochain::Cochain,
  manifold::{geometry::metric::mesh::MeshLengths, topology::complex::Complex},
};

pub fn l2_norm(fe: &Cochain, topology: &Complex, geometry: &MeshLengths) -> f64 {
  let mass = assemble_galmat(
    topology,
    geometry,
    HodgeMassElmat::new(topology.dim(), fe.dim()),
  );
  let mass = CsrMatrix::from(&mass);
  quadratic_form_sparse(&mass, fe.coeffs()).sqrt()
}

pub fn hdif_norm(fe: &Cochain, topology: &Complex, geometry: &MeshLengths) -> f64 {
  let codifdif = assemble_galmat(
    topology,
    geometry,
    CodifDifElmat::new(topology.dim(), fe.dim),
  );
  let codifdif = CsrMatrix::from(&codifdif);
  quadratic_form_sparse(&codifdif, fe.coeffs()).sqrt()
}
