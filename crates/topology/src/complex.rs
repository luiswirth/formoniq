pub mod attribute;
pub mod dim;
pub mod handle;
pub mod local;

use attribute::KSimplexCollection;
use dim::{ConstCodim, ConstDim};
use handle::{FaceCodim, FacetCodim, KSimplexIdx, VertexDim};
use local::LocalComplex;

use crate::{
  simplex::{graded_subsimplicies, SortedSimplex},
  skeleton::ManifoldSkeleton,
  Dim,
};

use common::sparse::SparseMatrix;

use indexmap::IndexMap;
use itertools::Itertools;
use std::sync::LazyLock;

/// A simplicial manifold complex.
#[derive(Debug, Clone)]
pub struct ManifoldComplex {
  skeletons: Vec<ComplexSkeleton>,
}

pub type ComplexSkeleton = IndexMap<SortedSimplex, SimplexData>;

#[derive(Default, Debug, Clone)]
pub struct SimplexData {
  pub cofacets: Vec<KSimplexIdx>,
}

impl ManifoldComplex {
  pub fn new(skeletons: Vec<ComplexSkeleton>) -> Self {
    Self { skeletons }
  }

  pub fn reference(dim: Dim) -> Self {
    let data = SimplexData { cofacets: vec![0] };
    let skeletons = graded_subsimplicies(dim)
      .map(|simps| simps.map(|simp| (simp, data.clone())).collect())
      .collect();
    Self::new(skeletons)
  }

  pub fn dim(&self) -> Dim {
    self.skeletons.len() - 1
  }

  pub fn local_complexes(&self) -> Vec<LocalComplex> {
    self
      .facets()
      .iter()
      .map(|facet| facet.to_local_complex())
      .collect()
  }

  pub fn has_boundary(&self) -> bool {
    !self.boundary_faces().is_empty()
  }

  /// For a d-mesh computes the boundary, which consists of faces ((d-1)-subs).
  ///
  /// The boundary faces are characterized by the fact that they
  /// only have 1 facet as super entity.
  pub fn boundary_faces(&self) -> KSimplexCollection<FaceCodim> {
    self
      .faces()
      .iter()
      .filter(|f| f.anti_boundary().len() == 1)
      .collect()
  }

  pub fn boundary_facets(&self) -> KSimplexCollection<FacetCodim> {
    let facets = self
      .boundary_faces()
      .handle_iter(self)
      // the boundary has only one parent facet by definition
      .map(|face| face.anti_boundary().kidxs()[0])
      .unique()
      .collect();
    KSimplexCollection::new(facets, ConstCodim)
  }

  /// The vertices that lie on the boundary of the mesh.
  /// No particular order of vertices.
  pub fn boundary_vertices(&self) -> KSimplexCollection<VertexDim> {
    let vertices = self
      .boundary_faces()
      .handle_iter(self)
      .flat_map(|face| face.simplex_set().vertices.clone())
      .unique()
      .collect();
    KSimplexCollection::new(vertices, ConstDim)
  }

  /// $diff^k: Delta_k -> Delta_(k-1)$
  pub fn boundary_operator(&self, dim: Dim) -> SparseMatrix {
    let sups = &self.skeleton(dim);
    let subs = &self.skeleton(dim - 1);
    let mut mat = SparseMatrix::zeros(subs.len(), sups.len());
    for (isup, sup) in sups.iter().enumerate() {
      let sup_boundary = sup.simplex_set().boundary();
      for sub in sup_boundary {
        let sign = sub.sign.as_f64();
        let isub = subs.get_by_simplex(&sub.into_simplex()).kidx();
        mat.push(isub, isup, sign);
      }
    }
    mat
  }
}

impl ManifoldComplex {
  pub fn from_facet_skeleton(facets: ManifoldSkeleton) -> Self {
    let dim = facets.dim();
    let facets = facets.into_simplicies();

    let mut skeletons = vec![ComplexSkeleton::new(); dim + 1];
    for (ifacet, facet) in facets.iter().enumerate() {
      for (dim_sub, subs) in skeletons.iter_mut().enumerate() {
        for sub in facet.subsimps(dim_sub) {
          let sub = subs.entry(sub.clone()).or_insert(SimplexData::default());
          sub.cofacets.push(ifacet);
        }
      }
    }

    // TODO: verify correct!
    // sort vertices
    skeletons[0].sort_by(|v0, _, v1, _| v0.vertices[0].cmp(&v1.vertices[0]));

    // Topology checks.
    let faces = &skeletons[dim - 1];
    for (_, SimplexData { cofacets }) in faces {
      let nparents = cofacets.len();
      let is_manifold = nparents == 2 || nparents == 1;
      assert!(is_manifold, "Topology must be manifold.");
    }

    Self { skeletons }
  }
}

pub const DIM_PRECOMPUTED: usize = 3;

pub static REFERENCE_COMPLEXES: LazyLock<Vec<ManifoldComplex>> = LazyLock::new(|| {
  (0..=DIM_PRECOMPUTED)
    .map(ManifoldComplex::reference)
    .collect()
});

pub static LOCAL_BOUNDARY_OPERATORS: LazyLock<Vec<Vec<na::DMatrix<f64>>>> = LazyLock::new(|| {
  REFERENCE_COMPLEXES
    .iter()
    .enumerate()
    .map(|(dim, complex)| {
      (1..=dim)
        .map(|sub_dim| complex.boundary_operator(sub_dim).to_nalgebra_dense())
        .collect()
    })
    .collect()
});

#[cfg(test)]
mod test {
  use super::ManifoldComplex;
  use crate::simplex::{nsubsimplicies, Simplex};

  #[test]
  fn incidence() {
    let dim = 3;
    let complex = ManifoldComplex::reference(dim);
    let facet = complex.facets().iter().next().unwrap();

    // print
    for dim_sub in 0..=dim {
      let skeleton = complex.skeleton(dim_sub);
      for simp in skeleton.iter() {
        let simp_vertices = simp.simplex_set();
        print!("{simp_vertices:?},");
      }
      println!();
    }

    let facet_simplex = Simplex::standard(dim);
    for dim_sub in 0..=dim {
      let subs: Vec<_> = facet.subsimps(dim_sub).collect();
      assert_eq!(subs.len(), nsubsimplicies(dim, dim_sub));
      let subs_vertices: Vec<_> = facet_simplex.subsimps(dim_sub).collect();
      assert_eq!(
        subs
          .iter()
          .map(|sub| sub.simplex_set().clone())
          .collect::<Vec<_>>(),
        subs_vertices
      );

      for (isub, sub) in subs.iter().enumerate() {
        let sub_vertices = &subs_vertices[isub];
        for dim_sup in dim_sub..dim {
          let sups: Vec<_> = sub.sups(dim_sup);
          let sups_vertices = sups
            .iter()
            .map(|sub| sub.simplex_set().clone())
            .collect::<Vec<_>>();
          sups_vertices
            .iter()
            .all(|sup| sub_vertices.is_subsimp_of(sup) && sup.is_subsimp_of(&facet_simplex));
        }
      }
    }
  }
}
