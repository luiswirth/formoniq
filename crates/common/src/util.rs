use std::path::Path;

pub trait CumsumExt {
  fn cumsum(self) -> impl Iterator<Item = usize>;
}
impl<I: IntoIterator<Item = usize>> CumsumExt for I {
  fn cumsum(self) -> impl Iterator<Item = usize> {
    self.into_iter().scan(0, |acc, x| {
      *acc += x;
      Some(*acc)
    })
  }
}

pub trait IterAllEqExt<T> {
  fn all_eq(self) -> Option<T>;
}
impl<T: PartialEq, I: IntoIterator<Item = T>> IterAllEqExt<T> for I {
  fn all_eq(self) -> Option<T> {
    let mut iter = self.into_iter();
    let first = iter.next()?;
    iter.all(|elem| elem == first).then_some(first)
  }
}

pub fn algebraic_convergence_rate(next: f64, prev: f64) -> f64 {
  let quot: f64 = next / prev;
  -quot.log2()
}

pub fn indicies_to_flags(indicies: &[usize], len: usize) -> Vec<bool> {
  let mut flags = vec![false; len];
  indicies.iter().for_each(|&i| flags[i] = true);
  flags
}

pub fn flags_to_indicies(flags: &[bool]) -> Vec<usize> {
  flags
    .iter()
    .enumerate()
    .filter_map(|(i, &flag)| flag.then_some(i))
    .collect()
}

pub fn sparse_to_dense_data<T>(sparse: Vec<(usize, T)>, len: usize) -> Vec<Option<T>> {
  let mut dense = Vec::from_iter((0..len).map(|_| None));
  sparse.into_iter().for_each(|(i, t)| dense[i] = Some(t));
  dense
}

pub fn dense_to_sparse_data<T>(dense: Vec<Option<T>>) -> Vec<(usize, T)> {
  dense
    .into_iter()
    .enumerate()
    .filter_map(|(i, o)| o.map(|v| (i, v)))
    .collect()
}

pub trait VecInto<S> {
  fn vec_into(self) -> Vec<S>;
}

impl<T, S> VecInto<S> for Vec<T>
where
  T: Into<S>,
{
  fn vec_into(self) -> Vec<S> {
    self.into_iter().map(|v| v.into()).collect()
  }
}

pub fn save_vector(mu: &na::DVector<f64>, path: impl AsRef<Path>) -> std::io::Result<()> {
  use std::io::Write;
  let mut file = std::fs::File::create(path).unwrap();
  for v in mu.iter() {
    writeln!(file, "{v}")?;
  }
  Ok(())
}

pub fn faervec2navec(faer: &faer::Mat<f64>) -> na::DVector<f64> {
  assert!(faer.ncols() == 1);
  na::DVector::from_iterator(faer.nrows(), faer.row_iter().map(|r| r[0]))
}

pub fn navec2faervec(na: &na::DVector<f64>) -> faer::Mat<f64> {
  let mut faer = faer::Mat::zeros(na.nrows(), 1);
  for (i, &v) in na.iter().enumerate() {
    faer[(i, 0)] = v;
  }
  faer
}

type SparseMatrixFaer = faer::sparse::SparseRowMat<usize, f64>;

pub fn nalgebra2faer(m: nas::CsrMatrix<f64>) -> SparseMatrixFaer {
  let nrows = m.nrows();
  let ncols = m.ncols();
  let (col_ptrs, row_indices, values) = m.disassemble();

  let symbolic =
    faer::sparse::SymbolicSparseRowMat::new_checked(nrows, ncols, col_ptrs, None, row_indices);
  faer::sparse::SparseRowMat::new(symbolic, values)
}

pub fn faer2nalgebra(m: SparseMatrixFaer) -> nas::CscMatrix<f64> {
  let (symbolic, values) = m.into_parts();
  let (nrows, ncols, col_ptrs, _, row_indices) = symbolic.into_parts();
  nas::CscMatrix::try_from_csc_data(nrows, ncols, col_ptrs, row_indices, values).unwrap()
}

pub struct FaerLu {
  raw: faer::sparse::linalg::solvers::Lu<usize, f64>,
}
impl FaerLu {
  pub fn new(a: nas::CsrMatrix<f64>) -> Self {
    let raw = nalgebra2faer(a).sp_lu().unwrap();
    Self { raw }
  }

  pub fn solve(&self, b: &na::DVector<f64>) -> na::DVector<f64> {
    use faer::solvers::SpSolver as _;

    let b = faer::col::from_slice(b.as_slice());
    na::DVector::from_vec(self.raw.solve(b).as_slice().to_vec())
  }
}

pub struct FaerCholesky {
  raw: faer::sparse::linalg::solvers::Cholesky<usize, f64>,
}
impl FaerCholesky {
  pub fn new(a: nas::CsrMatrix<f64>) -> Self {
    let raw = nalgebra2faer(a).sp_cholesky(faer::Side::Upper).unwrap();
    Self { raw }
  }

  pub fn solve(&self, b: &na::DVector<f64>) -> na::DVector<f64> {
    use faer::solvers::SpSolver as _;

    let b = faer::col::from_slice(b.as_slice());
    na::DVector::from_vec(self.raw.solve(b).as_slice().to_vec())
  }
}

pub fn phase_to_rgb(phase: f64) -> [f32; 3] {
  let hue = (phase / std::f64::consts::PI + 1.0) / 2.0;
  hsv_to_rgb(hue as f32, 1.0, 1.0)
}

pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
  let i = (h * 6.0).floor() as i32;
  let f = h * 6.0 - i as f32;
  let p = v * (1.0 - s);
  let q = v * (1.0 - f * s);
  let t = v * (1.0 - (1.0 - f) * s);
  match i % 6 {
    0 => [v, t, p],
    1 => [q, v, p],
    2 => [p, v, t],
    3 => [p, q, v],
    4 => [t, p, v],
    5 => [v, p, q],
    _ => [0.0, 0.0, 0.0],
  }
}
