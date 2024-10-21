use crate::{
  geometry::GeometrySimplex,
  mesh::{util::NodeData, CellIdx, SimplicialManifold},
  space::FeSpace,
};

pub trait ElmatProvider {
  fn eval(&self, space: &FeSpace, icell: CellIdx) -> na::DMatrix<f64>;
}

impl<F> ElmatProvider for F
where
  F: Fn(&FeSpace, CellIdx) -> na::DMatrix<f64>,
{
  fn eval(&self, space: &FeSpace, icell: CellIdx) -> na::DMatrix<f64> {
    self(space, icell)
  }
}

pub trait ElvecProvider {
  fn eval(&self, space: &FeSpace, icell: CellIdx) -> na::DVector<f64>;
}
impl<F> ElvecProvider for F
where
  F: Fn(&FeSpace, CellIdx) -> na::DVector<f64>,
{
  fn eval(&self, space: &FeSpace, icell: CellIdx) -> nalgebra::DVector<f64> {
    self(space, icell)
  }
}

pub fn laplacian_neg_elmat_geo(cell_geo: &GeometrySimplex) -> na::DMatrix<f64> {
  let dim = cell_geo.dim();
  let mut reference_gradbarys = na::DMatrix::zeros(dim, dim + 1);
  for i in 0..dim {
    reference_gradbarys[(i, 0)] = -1.0;
  }
  for i in 0..dim {
    reference_gradbarys[(i, i + 1)] = 1.0;
  }
  let metric = cell_geo.metric_tensor();
  let vol = cell_geo.vol();

  vol * reference_gradbarys.transpose() * metric.lu().solve(&reference_gradbarys).unwrap()
}

/// The exact Element Matrix for the negative laplacian in linear lagrangian FE.
pub fn laplacian_neg_elmat(space: &FeSpace, icell: CellIdx) -> na::DMatrix<f64> {
  let cell_geo = space.mesh().cells().get(icell).geometry_simplex();
  laplacian_neg_elmat_geo(&cell_geo)
}

// Element vector for scalar load function, computed using trapezoidal rule.
pub struct LoadElvec {
  dof_data: NodeData<f64>,
}
impl ElvecProvider for LoadElvec {
  fn eval(&self, space: &FeSpace, icell: CellIdx) -> na::DVector<f64> {
    let cell = space.mesh().cells().get(icell);
    let cell_geo = cell.geometry_simplex();
    let nverts = cell_geo.nvertices();

    cell_geo.vol() / nverts as f64
      * na::DVector::from_iterator(
        nverts,
        cell.vertices().iter().copied().map(|iv| self.dof_data[iv]),
      )
  }
}
impl LoadElvec {
  pub fn new(dof_data: NodeData<f64>) -> Self {
    Self { dof_data }
  }
}

// NOTE: In general this should depend on the FE Space and not just the mesh.
pub fn l2_norm(fn_coeffs: na::DVector<f64>, mesh: &SimplicialManifold) -> f64 {
  let mut norm: f64 = 0.0;
  for cell in mesh.cells().iter() {
    let mut sum = 0.0;
    for &ivertex in cell.vertices() {
      sum += fn_coeffs[ivertex].powi(2);
    }
    let nvertices = cell.nvertices();
    let cell_geo = cell.geometry_simplex();
    let vol = cell_geo.vol();
    norm += (vol / nvertices as f64) * sum;
  }
  norm.sqrt()
}

#[cfg(test)]
mod test {
  use crate::{fe, geometry::GeometrySimplex, space::FeSpace};
  use std::rc::Rc;

  fn check_elmat_refd(d: usize, expected_elmat: na::DMatrixView<f64>) {
    let ref_geo = GeometrySimplex::new_ref(d);

    let geo_elmat = fe::laplacian_neg_elmat_geo(&ref_geo);
    println!("{geo_elmat:.3}");
    assert!((geo_elmat - expected_elmat).norm() < 10.0 * f64::EPSILON);

    let space = FeSpace::new(Rc::new(ref_geo.into_singleton_mesh()));
    let mesh_elmat = fe::laplacian_neg_elmat(&space, 0);
    println!("{mesh_elmat:.3}");
    assert!((mesh_elmat - expected_elmat).norm() < 10.0 * f64::EPSILON);
  }

  #[test]
  fn laplacian_elmat_ref1d() {
    #[rustfmt::skip]
    let expected_elmat = na::DMatrix::from_row_slice(2, 2, &[
      1.0, -1.0,
      -1.0, 1.0
    ]);
    check_elmat_refd(1, expected_elmat.as_view());
  }

  #[test]
  fn laplacian_elmat_ref2d() {
    #[rustfmt::skip]
    let expected_elmat = na::DMatrix::from_row_slice(3, 3, &[
      1.0, -0.5, -0.5,
      -0.5, 0.5, 0.0,
      -0.5, 0.0, 0.5
    ]);
    check_elmat_refd(2, expected_elmat.as_view());
  }

  #[test]
  fn laplacian_elmat_ref3d() {
    let a = 1. / 6.;
    #[rustfmt::skip]
    let expected_elmat = na::DMatrix::from_row_slice(4, 4, &[
      0.5, -a, -a, -a,
      -a, a, 0., 0.,
      -a, 0., a, 0.,
      -a, 0., 0., a
    ]);
    check_elmat_refd(3, expected_elmat.as_view());
  }
}
