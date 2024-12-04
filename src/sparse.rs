use faer::solvers::SpSolver;

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

#[derive(Default)]
pub struct SparseMatrix {
  nrows: usize,
  ncols: usize,
  triplets: Vec<(usize, usize, f64)>,
}

impl SparseMatrix {
  pub fn zeros(nrows: usize, ncols: usize) -> Self {
    Self::new(nrows, ncols, Vec::new())
  }
  pub fn new(nrows: usize, ncols: usize, triplets: Vec<(usize, usize, f64)>) -> Self {
    Self {
      nrows,
      ncols,
      triplets,
    }
  }

  pub fn nrows(&self) -> usize {
    self.nrows
  }
  pub fn ncols(&self) -> usize {
    self.ncols
  }
  pub fn triplets(&self) -> &[(usize, usize, f64)] {
    &self.triplets
  }

  pub fn into_parts(self) -> (usize, usize, Vec<(usize, usize, f64)>) {
    (self.nrows, self.ncols, self.triplets)
  }

  pub fn push(&mut self, r: usize, c: usize, v: f64) {
    if v != 0.0 {
      self.triplets.push((r, c, v));
    }
  }

  pub fn set_zero<F>(&mut self, predicate: F)
  where
    F: Fn(usize, usize) -> bool,
  {
    let mut i = 0;
    while i < self.triplets.len() {
      let triplet = self.triplets[i];
      let r = triplet.0;
      let c = triplet.1;
      if predicate(r, c) {
        self.triplets.swap_remove(i);
      } else {
        i += 1;
      }
    }
  }

  pub fn to_nalgebra_coo(&self) -> nas::CooMatrix<f64> {
    let rows = self.triplets.iter().map(|t| t.0).collect();
    let cols = self.triplets.iter().map(|t| t.1).collect();
    let vals = self.triplets.iter().map(|t| t.2).collect();
    nas::CooMatrix::try_from_triplets(self.nrows, self.ncols, rows, cols, vals).unwrap()
  }

  pub fn to_nalgebra_csc(&self) -> nas::CscMatrix<f64> {
    (&self.to_nalgebra_coo()).into()
  }

  pub fn to_nalgebra_dense(&self) -> na::DMatrix<f64> {
    (&self.to_nalgebra_coo()).into()
  }

  pub fn to_faer_csc(&self) -> faer::sparse::SparseColMat<usize, f64> {
    faer::sparse::SparseColMat::try_new_from_triplets(self.nrows, self.ncols, &self.triplets)
      .unwrap()
  }

  /// Returns `None` if matrix is not diagonal.
  pub fn try_into_diagonal(self) -> Option<na::DVector<f64>> {
    let mut diagonal = na::DVector::zeros(self.nrows.max(self.ncols));
    for (r, c, v) in self.triplets {
      if r == c {
        diagonal[r] += v;
      } else {
        println!("not diag ({r},{c})");
        return None;
      }
    }
    Some(diagonal)
  }

  pub fn mul_left_by_diagonal(&self, diagonal: &na::DVector<f64>) -> Self {
    let triplets = self
      .triplets
      .iter()
      .map(|&(r, c, mut v)| {
        v *= diagonal[r];
        (r, c, v)
      })
      .collect();
    Self::new(self.nrows, self.ncols, triplets)
  }
}
