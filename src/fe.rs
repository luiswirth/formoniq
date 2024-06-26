use crate::mesh::{EntityId, Mesh};

pub trait ElmatProvider {
  fn eval(&self, mesh: &Mesh, icell: EntityId) -> na::DMatrix<f64>;
}

impl<F> ElmatProvider for F
where
  F: Fn(&Mesh, EntityId) -> na::DMatrix<f64>,
{
  fn eval(&self, mesh: &Mesh, icell: EntityId) -> na::DMatrix<f64> {
    self(mesh, icell)
  }
}

/// The exact Element Matrix for the negative laplacian in linear lagrangian FE.
pub fn laplacian_neg_elmat(mesh: &Mesh, icell: EntityId) -> na::DMatrix<f64> {
  let cell_geo = mesh.coordinate_simplex(icell);
  let m = cell_geo.barycentric_functions_grad();
  cell_geo.vol() * m.transpose() * m
}

pub trait ElvecProvider {
  fn eval(&self, mesh: &Mesh, icell: EntityId) -> na::DVector<f64>;
}
impl<F> ElvecProvider for F
where
  F: Fn(&Mesh, EntityId) -> na::DVector<f64>,
{
  fn eval(&self, mesh: &Mesh, icell: EntityId) -> nalgebra::DVector<f64> {
    self(mesh, icell)
  }
}

// Element vector for scalar load function, computed using trapezoidal rule.
pub struct LoadElvec<F>
where
  F: Fn(na::DVectorView<f64>) -> f64,
{
  load_fn: F,
}
impl<F> ElvecProvider for LoadElvec<F>
where
  F: Fn(na::DVectorView<f64>) -> f64,
{
  fn eval(&self, mesh: &Mesh, icell: EntityId) -> na::DVector<f64> {
    let cell_geo = mesh.coordinate_simplex(icell);
    let nverts = cell_geo.nvertices();
    let verts = cell_geo.vertices();
    cell_geo.vol() / nverts as f64
      * na::DVector::from_iterator(nverts, verts.column_iter().map(|v| (self.load_fn)(v)))
  }
}
impl<F> LoadElvec<F>
where
  F: Fn(na::DVectorView<f64>) -> f64,
{
  pub fn new(load_fn: F) -> Self {
    Self { load_fn }
  }
}
