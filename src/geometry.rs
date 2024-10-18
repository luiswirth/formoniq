use std::{collections::HashMap, f64::consts::SQRT_2, rc::Rc};

use num_integer::binomial;

use crate::{
  combinatorics::factorial,
  mesh::{EdgeBetweenVertices, SimplicialManifold},
  orientation::Orientation,
  Dim,
};

fn nedges(dim: Dim) -> usize {
  let nvertices = dim + 1;
  binomial(nvertices, 2)
}

#[derive(Debug, Clone)]
pub struct GeometrySimplex {
  /// Lengths of all edges in simplex
  /// Order of edges is lexicographical in vertex tuples.
  /// E.g. For a 3-simplex the edges are sorted as (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
  dim: Dim,
  edge_lengths: Vec<f64>,
}
impl GeometrySimplex {
  pub fn new(dim: Dim, edge_lengths: Vec<f64>) -> Self {
    assert!(edge_lengths.len() == nedges(dim));
    Self { dim, edge_lengths }
  }

  pub fn edge_lengths(&self) -> &[f64] {
    &self.edge_lengths
  }

  /// Constructs a reference simplex in `dim` dimensions.
  pub fn new_ref(dim: Dim) -> Self {
    let nedges = nedges(dim);
    let mut edge_lengths = vec![0.0; nedges];
    for l in edge_lengths.iter_mut().take(dim) {
      *l = 1.0;
    }
    for l in edge_lengths.iter_mut().take(nedges).skip(dim) {
      *l = SQRT_2;
    }
    Self { dim, edge_lengths }
  }

  pub fn dim(&self) -> Dim {
    self.dim
  }
  pub fn nvertices(&self) -> usize {
    self.dim + 1
  }
  pub fn nedges(&self) -> usize {
    self.edge_lengths.len()
  }

  /// The determinate (signed volume) of the simplex.
  pub fn det(&self) -> f64 {
    ref_vol(self.dim) * self.metric_tensor().determinant().sqrt()
  }

  /// The (unsigned) volume of the simplex.
  pub fn vol(&self) -> f64 {
    self.det().abs()
  }

  /// The orientation of the simplex.
  pub fn orientation(&self) -> Orientation {
    match self.det().is_sign_positive() {
      true => Orientation::Pos,
      false => Orientation::Neg,
    }
  }

  /// The diameter of the simplex.
  /// This is the maximum distance of two points inside the simplex.
  pub fn diameter(&self) -> f64 {
    self
      .edge_lengths
      .iter()
      .copied()
      .max_by(|a, b| a.partial_cmp(b).unwrap())
      .unwrap()
  }

  /// Returns the result of the Regge metric on the
  /// edge vectors (tangent vectors) i and j.
  /// This is the entry $G_(i j)$ of the metric tensor $G$.
  pub fn metric(&self, mut ei: usize, mut ej: usize) -> f64 {
    if ei == ej {
      self.edge_lengths[ei].powi(2)
    } else {
      if ei > ej {
        std::mem::swap(&mut ei, &mut ej);
      }

      let l0i = self.edge_lengths[ei];
      let l0j = self.edge_lengths[ej];

      // TODO: improve index computation
      let vi = ei + 1;
      let vj = ej + 1;
      let mut eij = 0;
      for i in 0..vi {
        eij += self.nvertices() - i - 1;
      }
      eij += (vj - vi) - 1;
      let lij = self.edge_lengths[eij];

      0.5 * (l0i.powi(2) + l0j.powi(2) - lij.powi(2))
    }
  }

  pub fn metric_tensor(&self) -> na::DMatrix<f64> {
    let mut mat = na::DMatrix::zeros(self.dim, self.dim);
    for i in 0..self.dim {
      for j in i..self.dim {
        let v = self.metric(i, j);
        mat[(i, j)] = v;
        mat[(j, i)] = v;
      }
    }
    mat
  }

  /// The shape regularity measure of the simplex.
  pub fn shape_reguarity_measure(&self) -> f64 {
    self.diameter().powi(self.dim() as i32) / self.vol()
  }

  pub fn into_singleton_mesh(&self) -> Rc<SimplicialManifold> {
    let mut edge_lengths = HashMap::new();
    let mut idx = 0;
    for i in 0..self.nvertices() {
      for j in (i + 1)..self.nvertices() {
        edge_lengths.insert(EdgeBetweenVertices::new(i, j), self.edge_lengths[idx]);
        idx += 1;
      }
    }
    SimplicialManifold::from_cells(
      self.nvertices(),
      vec![(0..self.nvertices()).collect()],
      edge_lengths,
    )
  }
}

pub fn ref_vol(dim: Dim) -> f64 {
  (factorial(dim) as f64).recip()
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn reference_transform() {
    let refsimp = GeometrySimplex::new_ref(3);
    let simp = GeometrySimplex::new(3, vec![1.0, 1.0, 1.0, SQRT_2, SQRT_2, SQRT_2]);
    assert_eq!(refsimp.edge_lengths, simp.edge_lengths);
  }

  #[test]
  fn ref_vol_test() {
    for d in 0..=8 {
      let simp = GeometrySimplex::new_ref(d);
      assert_eq!(simp.det(), ref_vol(d));
    }
  }
}
