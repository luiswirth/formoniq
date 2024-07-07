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

/// The exact Element Matrix for the negative laplacian in linear lagrangian FE.
pub fn laplacian_neg_elmat(mesh: &Mesh, icell: EntityId) -> na::DMatrix<f64> {
  let cell_geo = mesh.coordinate_simplex(icell);
  let m = cell_geo.barycentric_functions_grad();
  cell_geo.vol() * m.transpose() * m
}

/// Approximated Element Matrix for mass bilinear form,
/// obtained through trapezoidal quadrature rule.
pub fn lumped_mass_elmat(mesh: &Mesh, icell: EntityId) -> na::DMatrix<f64> {
  let cell_geo = mesh.coordinate_simplex(icell);
  let n = cell_geo.nvertices();
  let v = cell_geo.vol() / n as f64;
  na::DMatrix::from_diagonal_element(n, n, v)
}

/// The Element Matrix for the linear advection bilinar form,
/// computed using upwind quadrature.
pub struct UpwindAdvectionElmat<V>
where
  V: Fn(na::DVectorView<f64>) -> na::DVector<f64>,
{
  advection_vel: V,
  vertex_masses: Vec<f64>,
}
impl<V> UpwindAdvectionElmat<V>
where
  V: Fn(na::DVectorView<f64>) -> na::DVector<f64>,
{
  pub fn new(advection_vel: V, mesh: &Mesh) -> Self {
    let mut vertex_masses = vec![0.0; mesh.nnodes()];

    let intrinsic_dim = mesh.dim_intrinsic();
    for (icell, cell) in mesh.cells().iter().enumerate() {
      let cell_geo = mesh.coordinate_simplex((intrinsic_dim, icell));
      let vol = cell_geo.vol();
      for &ivertex in cell.vertices() {
        vertex_masses[ivertex] += vol / cell.nvertices() as f64;
      }
    }

    Self {
      advection_vel,
      vertex_masses,
    }
  }
}
impl<V> ElmatProvider for UpwindAdvectionElmat<V>
where
  V: Fn(na::DVectorView<f64>) -> na::DVector<f64>,
{
  fn eval(&self, mesh: &Mesh, icell: EntityId) -> na::DMatrix<f64> {
    let cell = mesh.simplex_by_id(icell);
    let cell_geo = mesh.coordinate_simplex(icell);
    let bary_grads = cell_geo.barycentric_functions_grad();
    let normals = -bary_grads.clone();
    let nvertices = cell_geo.nvertices();

    assert!(nvertices == 3, "only works for triangles");

    let mut elmat = na::DMatrix::zeros(nvertices, nvertices);
    for ivertex in 0..nvertices {
      let advection_vel = (self.advection_vel)(cell_geo.vertices().column(ivertex));
      let normal0 = normals.column((ivertex + 2) % 3);
      let normal1 = normals.column((ivertex + 1) % 3);
      let dot0 = advection_vel.dot(&normal0);
      let dot1 = advection_vel.dot(&normal1);
      let is_upwind = dot0 >= 0.0 && dot1 >= 0.0;
      if is_upwind {
        let mass = self.vertex_masses[cell.vertices()[ivertex]];
        let mut row = mass * advection_vel.transpose() * &bary_grads;
        if dot0 == 0.0 && dot1 == 0.0 {
          row *= 0.5;
        }
        elmat.row_mut(ivertex).copy_from(&row);
      }
    }
    elmat
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

// NOTE: In general this should depend on the FE Space and not just the mesh.
pub fn l2_norm(fn_coeffs: na::DVector<f64>, mesh: &Mesh) -> f64 {
  let mut norm = 0.0;
  let d = mesh.dim_intrinsic();
  for (icell, cell) in mesh.dsimplicies(d).iter().enumerate() {
    let mut sum = 0.0;
    for &ivertex in cell.vertices() {
      sum += fn_coeffs[ivertex].powi(2);
    }
    let nvertices = cell.nvertices();
    let vol = mesh.coordinate_simplex((d, icell)).vol();
    norm += (vol / nvertices as f64) * sum;
  }
  norm.sqrt()
}
