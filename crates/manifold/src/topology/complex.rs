pub mod attribute;
pub mod handle;

use attribute::KSimplexCollection;
use handle::SimplexIdx;

use super::{
  simplex::{graded_subsimplicies, SortedSimplex},
  skeleton::Skeleton,
};
use crate::Dim;

use common::sparse::SparseMatrix;

use indexmap::IndexMap;
use itertools::Itertools;
use std::sync::LazyLock;

/// A simplicial manifold complex.
#[derive(Debug, Clone)]
pub struct Complex {
  skeletons: Vec<ComplexSkeleton>,
}
pub type ComplexSkeleton = IndexMap<SortedSimplex, SimplexData>;

#[derive(Default, Debug, Clone)]
pub struct SimplexData {
  pub cocells: Vec<SimplexIdx>,
}

impl Complex {
  pub fn new(skeletons: Vec<ComplexSkeleton>) -> Self {
    Self { skeletons }
  }

  pub fn standard(dim: Dim) -> Self {
    let data = SimplexData {
      cocells: vec![SimplexIdx::new(dim, 0)],
    };
    let skeletons = graded_subsimplicies(dim)
      .map(|simps| simps.map(|simp| (simp, data.clone())).collect())
      .collect();
    Self::new(skeletons)
  }

  pub fn dim(&self) -> Dim {
    self.skeletons.len() - 1
  }

  pub fn has_boundary(&self) -> bool {
    !self.boundary_facets().is_empty()
  }

  /// For a d-mesh computes the boundary, which consists of facets ((d-1)-subs).
  ///
  /// The boundary facets are characterized by the fact that they
  /// only have 1 cell as super entity.
  pub fn boundary_facets(&self) -> KSimplexCollection {
    self
      .facets()
      .handle_iter()
      .filter(|f| f.anti_boundary().len() == 1)
      .collect()
  }

  pub fn boundary_cells(&self) -> KSimplexCollection {
    let cells = self
      .boundary_facets()
      .handle_iter(self)
      // the boundary has only one parent cell by definition
      .map(|facet| facet.anti_boundary().kidxs()[0])
      .unique()
      .collect();
    KSimplexCollection::new(cells, self.dim())
  }

  /// The vertices that lie on the boundary of the mesh.
  /// No particular order of vertices.
  pub fn boundary_vertices(&self) -> KSimplexCollection {
    let vertices = self
      .boundary_facets()
      .handle_iter(self)
      .flat_map(|facet| facet.simplex_set().vertices.clone())
      .unique()
      .collect();
    KSimplexCollection::new(vertices, 0)
  }

  /// $diff^k: Delta_k -> Delta_(k-1)$
  pub fn boundary_operator(&self, dim: Dim) -> SparseMatrix {
    let sups = &self.skeleton(dim);

    if dim == 0 {
      return SparseMatrix::zeros(0, sups.len());
    }

    let subs = &self.skeleton(dim - 1);
    let mut mat = SparseMatrix::zeros(subs.len(), sups.len());
    for (isup, sup) in sups.handle_iter().enumerate() {
      let sup_boundary = sup.simplex_set().boundary();
      for sub in sup_boundary {
        let sign = sub.sign.as_f64();
        let isub = subs.get_by_simplex(&sub.into_simplex()).kidx();
        mat.push(isub, isup, sign);
      }
    }
    mat
  }

  /// Dimension of the k-th homology group.
  ///
  /// k-th Betti number.
  /// Number of k-dimensional holes in the manifold.
  /// Computed using simplicial homology.
  pub fn homology_dim(&self, dim: Dim) -> usize {
    // TODO: use sparse matrix!
    let boundary_this = self.boundary_operator(dim).to_nalgebra_dense();
    let boundary_plus = self.boundary_operator(dim + 1).to_nalgebra_dense();

    const RANK_TOL: f64 = 1e-12;

    let dim_image = |op: &na::DMatrix<f64>| -> usize { op.rank(RANK_TOL) };
    let dim_kernel = |op: &na::DMatrix<f64>| -> usize { op.ncols() - dim_image(op) };

    let dim_cycles = dim_kernel(&boundary_this);
    let dim_boundaries = dim_image(&boundary_plus);

    dim_cycles - dim_boundaries
  }
}

impl Complex {
  pub fn from_cell_skeleton(cells: Skeleton) -> Self {
    let dim = cells.dim();
    let cells = cells.into_simplicies();

    let mut skeletons = vec![ComplexSkeleton::new(); dim + 1];
    for (icell, cell) in cells.iter().enumerate() {
      for (dim_sub, subs) in skeletons.iter_mut().enumerate() {
        for sub in cell.subsimps(dim_sub) {
          let sub = subs.entry(sub.clone()).or_insert(SimplexData::default());
          sub.cocells.push(SimplexIdx::new(dim, icell));
        }
      }
    }

    // TODO: verify correct!
    // sort vertices
    skeletons[0].sort_by(|v0, _, v1, _| v0.vertices[0].cmp(&v1.vertices[0]));

    // Topology checks.
    let facets = &skeletons[dim - 1];
    for (_, SimplexData { cocells }) in facets {
      let nparents = cocells.len();
      let is_manifold = nparents == 2 || nparents == 1;
      assert!(is_manifold, "Topology must be manifold.");
    }

    Self { skeletons }
  }
}

pub const DIM_PRECOMPUTED: usize = 4;

pub static REFERENCE_COMPLEXES: LazyLock<Vec<Complex>> =
  LazyLock::new(|| (0..=DIM_PRECOMPUTED).map(Complex::standard).collect());

pub static LOCAL_BOUNDARY_OPERATORS: LazyLock<Vec<Vec<na::DMatrix<f64>>>> = LazyLock::new(|| {
  REFERENCE_COMPLEXES
    .iter()
    .enumerate()
    .map(|(dim, complex)| {
      (0..=dim)
        .map(|sub_dim| complex.boundary_operator(sub_dim).to_nalgebra_dense())
        .collect()
    })
    .collect()
});

#[cfg(test)]
mod test {
  use crate::topology::simplex::{nsubsimplicies, Simplex};

  use super::*;

  #[test]
  fn incidence() {
    let dim = 3;
    let complex = Complex::standard(dim);
    let cell = complex.cells().handle_iter().next().unwrap();

    // print
    for dim_sub in 0..=dim {
      let skeleton = complex.skeleton(dim_sub);
      for simp in skeleton.handle_iter() {
        let simp_vertices = simp.simplex_set();
        print!("{simp_vertices:?},");
      }
      println!();
    }

    let cell_simplex = Simplex::standard(dim);
    for dim_sub in 0..=dim {
      let subs: Vec<_> = cell.subsimps(dim_sub).collect();
      assert_eq!(subs.len(), nsubsimplicies(dim, dim_sub));
      let subs_vertices: Vec<_> = cell_simplex.subsimps(dim_sub).collect();
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
            .all(|sup| sub_vertices.is_subsimp_of(sup) && sup.is_subsimp_of(&cell_simplex));
        }
      }
    }
  }
}
