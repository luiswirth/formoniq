use crate::{
  combinatorics::{
    factorial, nsubedges, nsubsimplicies, rank_of_combination, CanonicalVertplex, OrderedVertplex,
    Orientation, OrientedVertplex,
  },
  mesh::{raw::RawSimplicialManifold, KSimplexIdx, SimplicialManifold},
  Dim, VertexIdx,
};

use std::{collections::HashMap, f64::consts::SQRT_2};

pub type Length = f64;

#[derive(Debug, Clone, PartialEq)]
pub struct StandaloneCell {
  faces: Vec<Vec<KSimplexIdx>>,
  orientation: Orientation,
  edge_lengths: Vec<f64>,
}
impl StandaloneCell {
  pub fn new(
    faces: Vec<Vec<KSimplexIdx>>,
    orientation: Orientation,
    edge_lengths: Vec<f64>,
  ) -> Self {
    Self {
      faces,
      orientation,
      edge_lengths,
    }
  }

  /// Constructs a reference cell in `dim` dimensions.
  pub fn new_ref(dim: Dim) -> Self {
    let faces = (0..=dim)
      .map(|sub_dim| {
        let num_ksubs = nsubsimplicies(dim, sub_dim);
        (0..num_ksubs).collect()
      })
      .collect();

    let orientation = Orientation::default();

    let nedges = nsubedges(dim);
    let mut edge_lengths = vec![0.0; nedges];
    for l in edge_lengths.iter_mut().take(dim) {
      *l = 1.0;
    }
    for l in edge_lengths.iter_mut().take(nedges).skip(dim) {
      *l = SQRT_2;
    }

    Self {
      faces,
      orientation,
      edge_lengths,
    }
  }

  pub fn dim(&self) -> Dim {
    self.faces.len() - 1
  }
  pub fn nvertices(&self) -> usize {
    self.faces[0].len()
  }
  pub fn vertices(&self) -> &[VertexIdx] {
    &self.faces[0]
  }

  pub fn edge_lengths(&self) -> &[f64] {
    &self.edge_lengths
  }
  pub fn orientation(&self) -> Orientation {
    self.orientation
  }

  pub fn faces(&self) -> &[Vec<KSimplexIdx>] {
    &self.faces
  }

  /// The (unsigned) volume of this cell.
  pub fn vol(&self) -> f64 {
    ref_vol(self.dim()) * self.metric_tensor().determinant().sqrt()
  }

  /// The determinate (signed volume) of this cell.
  pub fn det(&self) -> f64 {
    self.orientation.as_f64() * self.vol()
  }

  /// The diameter of this cell.
  /// This is the maximum distance of two points inside the cell.
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

      let vi = ei + 1;
      let vj = ej + 1;
      let eij = rank_of_combination(&[vi, vj], self.nvertices());
      let lij = self.edge_lengths[eij];

      0.5 * (l0i.powi(2) + l0j.powi(2) - lij.powi(2))
    }
  }

  pub fn metric_tensor(&self) -> na::DMatrix<f64> {
    let mut mat = na::DMatrix::zeros(self.dim(), self.dim());
    for i in 0..self.dim() {
      for j in i..self.dim() {
        let v = self.metric(i, j);
        mat[(i, j)] = v;
        mat[(j, i)] = v;
      }
    }
    mat
  }

  /// The shape regularity measure of this cell.
  pub fn shape_reguarity_measure(&self) -> f64 {
    self.diameter().powi(self.dim() as i32) / self.vol()
  }

  pub fn to_singleton_mesh(&self) -> SimplicialManifold {
    let vertices = OrderedVertplex::new((0..self.nvertices()).collect());

    let mut edge_lengths = HashMap::new();

    let mut idx = 0;
    for i in 0..self.nvertices() {
      for j in (i + 1)..self.nvertices() {
        edge_lengths.insert(CanonicalVertplex::edge(i, j), self.edge_lengths[idx]);
        idx += 1;
      }
    }

    SimplicialManifold::new(RawSimplicialManifold::new(
      self.nvertices(),
      vec![OrientedVertplex::new(vertices, self.orientation)],
      edge_lengths,
    ))
  }
}

pub fn ref_vol(dim: Dim) -> f64 {
  (factorial(dim) as f64).recip()
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn refcell_vol() {
    for d in 0..=8 {
      let simp = StandaloneCell::new_ref(d);
      assert_eq!(simp.det(), ref_vol(d));
    }
  }
}
