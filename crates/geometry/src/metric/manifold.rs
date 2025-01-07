//! Simplicial Manifold Datastructure for working with Topology and Geometry.
//!
//! - Container for mesh entities (Simplicies).
//! - Global numbering for unique identification of the entities.
//! - Entity Iteration
//! - Topological Information (Incidence)
//! - Geometrical information (Lengths, Volumes)

pub mod local;

use crate::metric::EdgeLengths;

use index_algebra::factorial;
use local::LocalMetricComplex;
use topology::{
  complex::{handle::FacetHandle, ManifoldComplex},
  simplex::nsubsimplicies,
  skeleton::ManifoldSkeleton,
  Dim,
};

use std::f64::consts::SQRT_2;

pub struct MetricSkeleton {
  pub skeleton: ManifoldSkeleton,
  pub lengths: EdgeLengths,
}
impl MetricSkeleton {
  pub fn into_complex(self) -> MetricComplex {
    let Self { skeleton, lengths } = self;
    let topology = ManifoldComplex::from_facet_skeleton(skeleton);
    MetricComplex::new(topology, lengths)
  }
}

/// A simplicial manifold with both topological and geometric information.
#[derive(Debug)]
pub struct MetricComplex {
  topology: ManifoldComplex,
  lengths: EdgeLengths,
}

impl MetricComplex {
  pub fn new(topology: ManifoldComplex, lengths: EdgeLengths) -> Self {
    Self { topology, lengths }
  }

  pub fn reference(dim: Dim) -> Self {
    let topology = ManifoldComplex::reference(dim);

    let nedges = nsubsimplicies(dim, 1);
    let edge_lengths: Vec<f64> = (0..dim)
      .map(|_| 1.0)
      .chain((dim..nedges).map(|_| SQRT_2))
      .collect();
    let edge_lengths = EdgeLengths::new(edge_lengths.into());

    Self::new(topology, edge_lengths)
  }

  pub fn topology(&self) -> &ManifoldComplex {
    &self.topology
  }
  pub fn edge_lengths(&self) -> &EdgeLengths {
    &self.lengths
  }

  // TODO: check for valid changes
  pub fn edge_lengths_mut(&mut self) -> &mut EdgeLengths {
    &mut self.lengths
  }

  pub fn dim(&self) -> Dim {
    self.topology.dim()
  }

  pub fn local_complexes(&self) -> Vec<LocalMetricComplex> {
    self
      .topology
      .facets()
      .iter()
      .map(|facet| self.local_complex(facet))
      .collect()
  }

  /// The mesh width $h_max$, equal to the largest diameter of all cells.
  pub fn mesh_width_max(&self) -> f64 {
    self
      .lengths
      .iter()
      .max_by(|a, b| a.partial_cmp(b).unwrap())
      .copied()
      .unwrap()
  }

  /// By convexity the smallest length of a line inside a simplex is the length
  /// one of the edges.
  pub fn mesh_width_min(&self) -> f64 {
    self
      .lengths
      .iter()
      .min_by(|a, b| a.partial_cmp(b).unwrap())
      .copied()
      .unwrap()
  }

  pub fn local_complex(&self, facet: FacetHandle) -> LocalMetricComplex {
    LocalMetricComplex::new(
      facet.to_local_complex(),
      self
        .edge_lengths()
        .restriction(facet.edges().map(|e| e.kidx())),
    )
  }

  /// The shape regularity measure $rho$ of the whole mesh, which is the largest
  /// shape regularity measure over all cells.
  pub fn shape_regularity_measure(&self) -> f64 {
    self
      .topology
      .facets()
      .iter()
      .map(|facet| self.local_complex(facet).shape_reguarity_measure())
      .max_by(|a, b| a.partial_cmp(b).unwrap())
      .unwrap()
  }
}

pub fn ref_vol(dim: Dim) -> f64 {
  (factorial(dim) as f64).recip()
}
