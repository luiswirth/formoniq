use manifold::topology::handle::SimplexRef;

use {
  common::gramian::Gramian,
  ddf::{cochain::Cochain, whitney::form::WhitneyForm},
  exterior::{field::ExteriorField, multi_gramian},
  manifold::{
    geometry::coord::{mesh::MeshCoords, quadrature::SimplexQuadRule, CoordRef},
    topology::complex::Complex,
  },
};

pub fn fe_l2_error<E: ExteriorField>(
  fe_cochain: &Cochain,
  exact: &E,
  topology: &Complex,
  coords: &MeshCoords,
) -> f64 {
  let dim = topology.dim();
  let qr = SimplexQuadRule::degree(dim, 3);
  let fe_whitney = WhitneyForm::new(fe_cochain.clone(), topology, coords);
  let inner = multi_gramian(&Gramian::standard(dim), fe_cochain.grade());
  let error_pointwise = |x: CoordRef, cell: SimplexRef| {
    inner.norm_sq((exact.at_point(x) - fe_whitney.eval_known_cell(cell, x)).coeffs())
  };
  qr.integrate_mesh(&error_pointwise, topology, coords).sqrt()
}
