use manifold::topology::handle::SimplexHandle;

use crate::{
  assemble::assemble_galmat,
  operators::{CodifDifElmat, HodgeMassElmat},
};

use {
  common::{
    gramian::Gramian,
    linalg::nalgebra::{quadratic_form_sparse, CsrMatrix},
  },
  ddf::{cochain::Cochain, whitney::form::WhitneyForm},
  exterior::{field::ExteriorField, term::multi_gramian},
  manifold::{
    geometry::{
      coord::{mesh::MeshCoords, quadrature::SimplexQuadRule, CoordRef},
      metric::mesh::MeshLengths,
    },
    topology::complex::Complex,
  },
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

pub fn fe_l2_error<E: ExteriorField>(
  fe_cochain: &Cochain,
  exact: &E,
  topology: &Complex,
  coords: &MeshCoords,
) -> f64 {
  let dim = topology.dim();
  let qr = SimplexQuadRule::order3(dim);
  let fe_whitney = WhitneyForm::new(fe_cochain.clone(), topology, coords);
  let inner = multi_gramian(&Gramian::standard(dim), fe_cochain.dim());
  let error_pointwise = |x: CoordRef, cell: SimplexHandle| {
    inner.norm_sq((exact.at_point(x) - fe_whitney.eval_known_cell(cell, x)).coeffs())
  };
  qr.integrate_mesh(&error_pointwise, topology, coords).sqrt()
}
