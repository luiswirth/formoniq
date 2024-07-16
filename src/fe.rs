use crate::{
  assemble::DofCoeffMap,
  mesh::{CellId, SimplicialMesh},
  space::{DofId, FeSpace},
};

pub trait ElmatProvider {
  fn eval(&self, space: &FeSpace, icell: CellId) -> na::DMatrix<f64>;
}

impl<F> ElmatProvider for F
where
  F: Fn(&FeSpace, CellId) -> na::DMatrix<f64>,
{
  fn eval(&self, space: &FeSpace, icell: CellId) -> na::DMatrix<f64> {
    self(space, icell)
  }
}

pub trait ElvecProvider {
  fn eval(&self, space: &FeSpace, icell: CellId) -> na::DVector<f64>;
}
impl<F> ElvecProvider for F
where
  F: Fn(&FeSpace, CellId) -> na::DVector<f64>,
{
  fn eval(&self, space: &FeSpace, icell: CellId) -> nalgebra::DVector<f64> {
    self(space, icell)
  }
}

/// The exact Element Matrix for the negative laplacian in linear lagrangian FE.
pub fn laplacian_neg_elmat(space: &FeSpace, icell: CellId) -> na::DMatrix<f64> {
  let cell_geo = space.mesh().cell(icell).geometry_simplex();
  let m = cell_geo.barycentric_functions_grad();
  cell_geo.vol() * m.transpose() * m
}

/// Approximated Element Matrix for mass bilinear form,
/// obtained through trapezoidal quadrature rule.
pub fn lumped_mass_elmat(space: &FeSpace, icell: CellId) -> na::DMatrix<f64> {
  let cell_geo = space.mesh().cell(icell).geometry_simplex();
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
  pub fn new(advection_vel: V, mesh: &SimplicialMesh) -> Self {
    let mut vertex_masses = vec![0.0; mesh.nnodes()];

    for (icell, cell) in mesh.cells().iter().enumerate() {
      let cell_geo = mesh.cell(icell).geometry_simplex();
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
  fn eval(&self, space: &FeSpace, icell: CellId) -> na::DMatrix<f64> {
    let cell = space.mesh().cell(icell);
    let cell_geo = space.mesh().cell(icell).geometry_simplex();
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
  fn eval(&self, space: &FeSpace, icell: CellId) -> na::DVector<f64> {
    let cell_geo = space.mesh().cell(icell).geometry_simplex();
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
pub fn l2_norm(fn_coeffs: na::DVector<f64>, mesh: &SimplicialMesh) -> f64 {
  let mut norm = 0.0;
  for (icell, cell) in mesh.cells().iter().enumerate() {
    let mut sum = 0.0;
    for &ivertex in cell.vertices() {
      sum += fn_coeffs[ivertex].powi(2);
    }
    let nvertices = cell.nvertices();
    let vol = mesh.cell(icell).geometry_simplex().vol();
    norm += (vol / nvertices as f64) * sum;
  }
  norm.sqrt()
}

pub struct DirichletBcMap {
  dirichlet_coeffs: Vec<Option<f64>>,
}
impl DofCoeffMap for DirichletBcMap {
  fn eval(&self, idof: DofId) -> Option<f64> {
    self.dirichlet_coeffs[idof]
  }
}
impl DirichletBcMap {
  pub fn new<F>(space: &FeSpace, dirichlet_data: F) -> Self
  where
    F: Fn(na::DVectorView<f64>) -> f64,
  {
    let mut dirichlet_coeffs = vec![None; space.ndofs()];
    let boundary_dofs = space.mesh().boundary_nodes();
    for idof in boundary_dofs {
      let pos = space.mesh().node_coords().column(idof);
      let dof_value = dirichlet_data(pos);
      dirichlet_coeffs[idof] = Some(dof_value);
    }
    Self { dirichlet_coeffs }
  }
}

pub struct DirichletInflowBcMap {
  dirichlet_coeffs: Vec<Option<f64>>,
}
impl DofCoeffMap for DirichletInflowBcMap {
  fn eval(&self, idof: DofId) -> Option<f64> {
    self.dirichlet_coeffs[idof]
  }
}
#[allow(
  unused_variables,
  unreachable_code,
  unused_mut,
  clippy::diverging_sub_expression
)]
impl DirichletInflowBcMap {
  pub fn new<F, V>(space: &FeSpace, dirichlet_data: F, velocity: V) -> Self
  where
    F: Fn(na::DVectorView<f64>) -> f64,
    V: Fn(na::DVectorView<f64>) -> na::DVector<f64>,
  {
    let mut dirichlet_coeffs = vec![None; space.ndofs()];
    let boundary_dofs = space.mesh().boundary_nodes();
    for idof in boundary_dofs {
      let pos = space.mesh().node_coords().column(idof);

      let vel = velocity(pos);
      let normal0: na::DVector<f64> = todo!();
      let normal1: na::DVector<f64> = todo!();
      let dot0 = vel.dot(&normal0);
      let dot1 = vel.dot(&normal1);
      let is_inflow = dot0 <= 0.0 && dot1 <= 0.0;
      if is_inflow {
        let dof_value = dirichlet_data(pos);
        dirichlet_coeffs[idof] = Some(dof_value);
      }
    }
    Self { dirichlet_coeffs }
  }
}

#[cfg(test)]
mod test {
  use super::laplacian_neg_elmat;
  use crate::{geometry::GeometrySimplex, space::FeSpace};

  fn check_galmat_refd(d: usize, expected_elmat: na::DMatrixView<f64>) {
    let space = FeSpace::new(GeometrySimplex::new_ref(d).into_singleton_mesh());
    let computed_elmat = laplacian_neg_elmat(&space, 0);
    assert_eq!(computed_elmat, expected_elmat);

    let space = FeSpace::new(GeometrySimplex::new_ref_embedded(d, d + 1).into_singleton_mesh());
    let computed_elmat = laplacian_neg_elmat(&space, 0);
    assert!((computed_elmat - expected_elmat).norm_squared() < f64::EPSILON);
  }

  #[test]
  fn laplacian_galmat_ref1d() {
    #[rustfmt::skip]
    let expected_elmat = na::DMatrix::from_row_slice(2, 2, &[
      1.0, -1.0,
      -1.0, 1.0
    ]);
    check_galmat_refd(1, expected_elmat.as_view());
  }

  #[test]
  fn laplacian_galmat_ref2d() {
    #[rustfmt::skip]
    let expected_elmat = na::DMatrix::from_row_slice(3, 3, &[
      1.0, -0.5, -0.5,
      -0.5, 0.5, 0.0,
      -0.5, 0.0, 0.5
    ]);
    check_galmat_refd(2, expected_elmat.as_view());
  }

  #[test]
  fn laplacian_galmat_ref3d() {
    let a = 1. / 6.;
    #[rustfmt::skip]
    let expected_elmat = na::DMatrix::from_row_slice(4, 4, &[
      0.5, -a, -a, -a,
      -a, a, 0., 0.,
      -a, 0., a, 0.,
      -a, 0., 0., a
    ]);
    check_galmat_refd(3, expected_elmat.as_view());
  }
}
