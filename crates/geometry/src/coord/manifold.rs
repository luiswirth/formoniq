pub mod cartesian;
pub mod dim3;
pub mod gmsh;

use super::{Coord, CoordRef, VertexCoords};
use crate::metric::manifold::{ref_vol, MetricComplex};

use common::linalg::DMatrixExt;
use index_algebra::sign::Sign;
use itertools::Itertools;
use topology::{
  complex::{dim::DimInfoProvider, handle::SimplexHandle, ManifoldComplex},
  simplex::{Simplex, SimplexExt},
  skeleton::ManifoldSkeleton,
  Dim,
};

#[derive(Debug, Clone)]
pub struct CoordSkeleton {
  skeleton: ManifoldSkeleton,
  coords: VertexCoords,
}
impl CoordSkeleton {
  pub fn new(skeleton: ManifoldSkeleton, coords: VertexCoords) -> Self {
    Self { skeleton, coords }
  }
  pub fn dim_embedded(&self) -> Dim {
    self.coords.dim()
  }
  pub fn dim_intrinsic(&self) -> Dim {
    self.skeleton.dim()
  }

  pub fn skeleton(&self) -> &ManifoldSkeleton {
    &self.skeleton
  }
  pub fn coords(&self) -> &VertexCoords {
    &self.coords
  }
  pub fn coords_mut(&mut self) -> &mut VertexCoords {
    &mut self.coords
  }
  pub fn into_parts(self) -> (ManifoldSkeleton, VertexCoords) {
    (self.skeleton, self.coords)
  }

  pub fn embed_euclidean(mut self, dim: Dim) -> CoordSkeleton {
    self.coords = self.coords.embed_euclidean(dim);
    self
  }

  pub fn into_coord_complex(self) -> CoordComplex {
    let Self { skeleton, coords } = self;
    let complex = ManifoldComplex::from_facet_skeleton(skeleton);
    CoordComplex::new(complex, coords)
  }

  pub fn into_metric_complex(self) -> (MetricComplex, VertexCoords) {
    self.into_coord_complex().into_metric_complex()
  }
}

#[derive(Debug, Clone)]
pub struct CoordComplex {
  topology: ManifoldComplex,
  coords: VertexCoords,
}
impl CoordComplex {
  pub fn new(topology: ManifoldComplex, coords: VertexCoords) -> Self {
    Self { topology, coords }
  }

  pub fn reference(dim: Dim) -> Self {
    let topology = ManifoldComplex::reference(dim);

    let coords = topology
      .vertices()
      .iter()
      .map(|v| v.kidx())
      .map(|v| {
        let mut vec = na::DVector::zeros(dim);
        if v > 0 {
          vec[v - 1] = 1.0;
        }
        vec
      })
      .collect_vec();
    let coords = na::DMatrix::from_columns(&coords);
    let coords = VertexCoords::new(coords);

    Self::new(topology, coords)
  }

  pub fn topology(&self) -> &ManifoldComplex {
    &self.topology
  }
  pub fn coords(&self) -> &VertexCoords {
    &self.coords
  }

  pub fn dim_intrinsic(&self) -> Dim {
    self.topology.dim()
  }
  pub fn dim_embedded(&self) -> Dim {
    self.coords.dim()
  }

  pub fn into_metric_complex(self) -> (MetricComplex, VertexCoords) {
    let Self { topology, coords } = self;
    let edges = topology.edges();
    let edges = edges
      .iter()
      .map(|e| e.simplex_set().clone().try_into().unwrap());
    let edge_lengths = coords.to_edge_lengths(edges);
    (MetricComplex::new(topology, edge_lengths), coords)
  }
}

#[derive(Debug, Clone)]
pub struct CoordSimplex {
  pub vertices: VertexCoords,
}
impl CoordSimplex {
  pub fn new(vertices: impl Into<VertexCoords>) -> Self {
    let vertices = vertices.into();
    Self { vertices }
  }

  pub fn standard(ndim: Dim) -> Self {
    let nvertices = ndim + 1;
    let mut vertices = na::DMatrix::<f64>::zeros(ndim, nvertices);
    for i in 0..ndim {
      vertices[(i, i + 1)] = 1.0;
    }
    Self::new(vertices)
  }

  pub fn from_simplex<O>(simp: &Simplex<O>, coords: &VertexCoords) -> CoordSimplex
  where
    O: index_algebra::variants::SetOrder,
  {
    let mut vert_coords = na::DMatrix::zeros(coords.dim(), simp.len());
    for (i, v) in simp.iter().enumerate() {
      vert_coords.set_column(i, &coords.coord(v));
    }
    CoordSimplex::new(vert_coords)
  }

  pub fn edges(&self) -> impl Iterator<Item = CoordSimplex> + use<'_> {
    Simplex::increasing(self.nvertices())
      .subsimps(1)
      .map(|edge| Self::from_simplex(&edge, &self.vertices))
  }
}
impl CoordSimplex {
  pub fn nvertices(&self) -> usize {
    self.vertices.nvertices()
  }
  pub fn dim_intrinsic(&self) -> Dim {
    self.nvertices() - 1
  }
  pub fn dim_embedded(&self) -> Dim {
    self.vertices.dim()
  }
  pub fn is_euclidean(&self) -> bool {
    self.dim_intrinsic() == self.dim_embedded()
  }

  pub fn spanning_vectors(&self) -> na::DMatrix<f64> {
    let mut mat = na::DMatrix::zeros(self.dim_embedded(), self.dim_intrinsic());
    let v0 = self.vertices.coord(0);
    for (i, vi) in self.vertices.coord_iter().skip(1).enumerate() {
      let v0i = vi - &v0;
      mat.set_column(i, &v0i);
    }
    mat
  }

  /// Affine transform from reference simplex to this simplex.
  pub fn affine_transform(&self) -> AffineCoordTransform {
    let linear = self.spanning_vectors();
    let translation = self.vertices.coord(0);
    AffineCoordTransform::new(linear, translation)
  }

  pub fn barycenter(&self) -> Coord {
    let mut barycenter = na::DVector::zeros(self.dim_embedded());
    self.vertices.coord_iter().for_each(|v| barycenter += v);
    barycenter /= self.nvertices() as f64;
    barycenter
  }

  pub fn det(&self) -> f64 {
    let det = if self.is_euclidean() {
      self.spanning_vectors().determinant()
    } else {
      self.spanning_vectors().gram_det_sqrt()
    };
    ref_vol(self.dim_intrinsic()) * det
  }
  pub fn vol(&self) -> f64 {
    self.det().abs()
  }
  pub fn orientation(&self) -> Sign {
    Sign::from_f64(self.det())
  }

  pub fn global_to_local_coords(&self, global: CoordRef) -> na::DVector<f64> {
    self.affine_transform().try_apply_inverse(global).unwrap()
  }

  pub fn global_to_bary_coords(&self, global: CoordRef) -> na::DVector<f64> {
    let local = self.global_to_local_coords(global);
    let bary0 = 1.0 - local.sum();
    local.insert_row(0, bary0)
  }

  pub fn is_coord_inside(&self, global: CoordRef) -> bool {
    let bary = self.global_to_bary_coords(global);
    bary.iter().all(|&b| (0.0..=1.0).contains(&b))
  }
}

pub trait SimplexHandleExt {
  fn coord(&self, coords: &VertexCoords) -> CoordSimplex;
}
impl<'c, D: DimInfoProvider> SimplexHandleExt for SimplexHandle<'c, D> {
  fn coord(&self, coords: &VertexCoords) -> CoordSimplex {
    CoordSimplex::from_simplex(self.simplex_set(), coords)
  }
}

pub struct AffineCoordTransform {
  pub linear: na::DMatrix<f64>,
  pub translation: na::DVector<f64>,
}
impl AffineCoordTransform {
  pub fn new(linear: na::DMatrix<f64>, translation: na::DVector<f64>) -> Self {
    Self {
      linear,
      translation,
    }
  }
  pub fn apply(&self, coord: CoordRef) -> Coord {
    &self.linear * coord + &self.translation
  }
  pub fn try_apply_inverse(&self, coord: CoordRef) -> Option<Coord> {
    self.linear.clone().qr().solve(&(coord - &self.translation))
  }

  pub fn try_inverse(&self) -> Option<Self> {
    todo!()
  }
}
