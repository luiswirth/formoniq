use super::{
  complex::{KSimplexIdx, VertexIdx},
  geometry::{EdgeLengths, RiemannianManifold, RiemannianMetric},
};
use crate::{
  combo::{binomial, combinators::GradedIndexSubsets, factorial, variants::*, IndexSet, Sign},
  mesh::RiemannianComplex,
  Dim,
};

use std::{f64::consts::SQRT_2, sync::LazyLock};

pub type Vertplex<B, O, S> = IndexSet<B, O, S>;

pub type SortedVertplex = Vertplex<Unspecified, Sorted, Unsigned>;
pub type OrderedVertplex = Vertplex<Unspecified, Ordered, Unsigned>;
pub type OrientedVertplex = Vertplex<Unspecified, Ordered, Signed>;

pub type RefVertplex = Vertplex<Local, Sorted, Unsigned>;

pub fn nsubsimplicies(dim: Dim, dim_sub: Dim) -> usize {
  let nverts = dim + 1;
  let nverts_sub = dim_sub + 1;
  binomial(nverts, nverts_sub)
}

pub fn subvertplicies(dim: Dim) -> Vec<Vec<RefVertplex>> {
  GradedIndexSubsets::canonical(dim + 1)
    .skip(1)
    .map(|subs| subs.collect())
    .collect()
}

pub trait SimplexExt {
  fn dim(&self) -> Dim;
  fn nvertices(&self) -> usize;
}
impl<B: Base, O: Order, S: Signedness> SimplexExt for Vertplex<B, O, S> {
  fn dim(&self) -> Dim {
    self.len() - 1
  }
  fn nvertices(&self) -> Dim {
    self.len()
  }
}

#[derive(Debug, Clone)]
pub struct LocalComplex {
  subs: Vec<Vec<KSimplexIdx>>,
  orientation: Sign,
  edge_lengths: Vec<f64>,
  metric: RiemannianMetric,
}
impl LocalComplex {
  pub fn new(subs: Vec<Vec<KSimplexIdx>>, orientation: Sign, edge_lengths: Vec<f64>) -> Self {
    let dim = subs.len() - 1;
    let metric = RiemannianMetric::regge(dim, &edge_lengths);
    Self {
      subs,
      orientation,
      edge_lengths,
      metric,
    }
  }

  pub fn dim(&self) -> Dim {
    self.subs.len() - 1
  }
  pub fn nvertices(&self) -> usize {
    self.vertices().len()
  }
  pub fn vertices(&self) -> &[VertexIdx] {
    &self.subs[0]
  }
  pub fn orientation(&self) -> Sign {
    self.orientation
  }
  pub fn faces(&self) -> &[Vec<KSimplexIdx>] {
    &self.subs
  }

  pub fn edge_lengths(&self) -> &[f64] {
    &self.edge_lengths
  }

  pub fn metric(&self) -> &RiemannianMetric {
    &self.metric
  }

  /// The volume of this cell.
  pub fn vol(&self) -> f64 {
    ref_vol(self.dim()) * self.metric().det_sqrt()
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

  /// The shape regularity measure of this cell.
  pub fn shape_reguarity_measure(&self) -> f64 {
    self.diameter().powi(self.dim() as i32) / self.vol()
  }
}

// TODO: find better solution
pub static REFCELLS: LazyLock<Vec<ReferenceCell>> =
  LazyLock::new(|| (0..=4).map(ReferenceCell::new).collect());

pub struct ReferenceCell {
  faces: Vec<Vec<RefVertplex>>,
  edge_lengths: Vec<f64>,
}
impl ReferenceCell {
  /// Constructs a reference cell in `dim` dimensions.
  pub fn new(dim: Dim) -> Self {
    let faces = subvertplicies(dim);

    let nedges = nsubsimplicies(dim, 1);
    let edge_lengths: Vec<f64> = (0..dim)
      .map(|_| 1.0)
      .chain((dim..nedges).map(|_| SQRT_2))
      .collect();
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

  pub fn as_vertplex(&self) -> &RefVertplex {
    &self.faces[self.dim()][0]
  }

  /// $diff^k: Delta_k -> Delta_(k-1)$
  pub fn kboundary_operator(&self, k: Dim) -> na::DMatrix<f64> {
    let sups = &self.faces[k];
    let subs = &self.faces[k - 1];
    let mut mat = na::DMatrix::zeros(subs.len(), sups.len());
    for (isup, sup) in sups.iter().enumerate() {
      let sup_subs = sup.boundary();
      for sub in sup_subs {
        let isub = subs
          .iter()
          .position(|other| sub.clone().forget_sign() == *other)
          .unwrap();
        mat[(isub, isup)] = sub.sign().as_f64();
      }
    }
    mat
  }

  pub fn to_cell_complex(&self) -> LocalComplex {
    let faces = self
      .faces
      .iter()
      .map(|fs| (0..fs.len()).collect())
      .collect();
    let orientation = Sign::Pos;
    let edge_lengths = self.edge_lengths.clone();
    LocalComplex::new(faces, orientation, edge_lengths)
  }

  pub fn to_singleton_mesh(&self) -> RiemannianComplex {
    let facets = vec![self.as_vertplex().clone().forget_base().into_oriented()];
    let edge_lengths = EdgeLengths::new(self.edge_lengths.clone().into());

    RiemannianManifold {
      facets,
      edge_lengths,
    }
    .into_complex()
  }
}

pub fn ref_vol(dim: Dim) -> f64 {
  (factorial(dim) as f64).recip()
}
