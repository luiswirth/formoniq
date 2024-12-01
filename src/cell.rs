use crate::{
  combinatorics::{factorial, nsubedges, rank_of_combination, CanonicalVertplex, Orientation},
  mesh::{raw::RawSimplicialManifold, KSimplexIdx, SimplicialManifold},
  Dim, Rank, VertexIdx,
};

use std::{collections::HashMap, f64::consts::SQRT_2, sync::LazyLock};

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

  pub fn dim(&self) -> Dim {
    self.faces.len() - 1
  }
  pub fn nvertices(&self) -> usize {
    self.faces[0].len()
  }
  pub fn vertices(&self) -> &[VertexIdx] {
    &self.faces[0]
  }
  pub fn orientation(&self) -> Orientation {
    self.orientation
  }
  pub fn faces(&self) -> &[Vec<KSimplexIdx>] {
    &self.faces
  }

  pub fn edge_lengths(&self) -> &[f64] {
    &self.edge_lengths
  }

  /// The volume of this cell.
  pub fn vol(&self) -> f64 {
    ref_vol(self.dim()) * self.metric_tensor().determinant().sqrt()
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
}

// TODO: find better solution
pub static REFCELLS: LazyLock<Vec<ReferenceCell>> =
  LazyLock::new(|| (0..=4).map(ReferenceCell::new).collect());

pub struct ReferenceCell {
  faces: Vec<Vec<CanonicalVertplex>>,
  edge_lengths: Vec<f64>,
}
impl ReferenceCell {
  /// Constructs a reference cell in `dim` dimensions.
  pub fn new(dim: Dim) -> Self {
    let nvertices = dim + 1;
    let cell = CanonicalVertplex::new((0..nvertices).collect());

    let faces = (0..=dim).map(|face_dim| cell.subs(face_dim)).collect();

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
      edge_lengths,
    }
  }

  pub fn dim(&self) -> Dim {
    self.nvertices() - 1
  }
  pub fn nvertices(&self) -> usize {
    self.num_kfaces(0)
  }
  pub fn num_kfaces(&self, k: Dim) -> usize {
    self.faces[k].len()
  }

  pub fn as_vertplex(&self) -> &CanonicalVertplex {
    &self.faces[self.dim()][0]
  }

  /// $diff^k: Delta_k -> Delta_(k-1)$
  pub fn kboundary_operator(&self, k: Rank) -> na::DMatrix<f64> {
    let sups = &self.faces[k];
    let subs = &self.faces[k - 1];
    let mut mat = na::DMatrix::zeros(subs.len(), sups.len());
    for (isup, sup) in sups.iter().enumerate() {
      let sup_subs = sup.boundary();
      for (sub, orientation) in sup_subs {
        let isub = subs.iter().position(|other| sub == *other).unwrap();
        mat[(isub, isup)] = orientation.as_f64();
      }
    }
    mat
  }

  pub fn to_standalone_cell(&self) -> StandaloneCell {
    let faces = self
      .faces
      .iter()
      .map(|fs| (0..fs.len()).collect())
      .collect();
    let orientation = Orientation::Pos;
    let edge_lengths = self.edge_lengths.clone();
    StandaloneCell::new(faces, orientation, edge_lengths)
  }

  pub fn to_singleton_mesh(&self) -> SimplicialManifold {
    let nnodes = self.nvertices();
    let cells = vec![self.as_vertplex().clone().into_oriented()];

    let mut edge_lengths = HashMap::new();
    let mut idx = 0;
    for i in 0..self.nvertices() {
      for j in (i + 1)..self.nvertices() {
        edge_lengths.insert(CanonicalVertplex::edge(i, j), self.edge_lengths[idx]);
        idx += 1;
      }
    }

    RawSimplicialManifold::new(nnodes, cells, edge_lengths).build()
  }
}

pub fn ref_vol(dim: Dim) -> f64 {
  (factorial(dim) as f64).recip()
}