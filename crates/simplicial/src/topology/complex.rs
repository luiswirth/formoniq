use super::{
  handle::{KSimplexIdx, SimplexIdx, SkeletonRef},
  skeleton::Skeleton,
};
use crate::Dim;

use formoniq_linalg::nalgebra::{CooMatrix, CooMatrixExt};

use itertools::Itertools;

use std::{io, path::Path, sync::OnceLock};

/// A simplicial manifold complex.
///
/// Its skeletons are derived from the cells and stored in canonical colex
/// order. The only incidence stored is vertex → cells; all other up-incidence
/// (a simplex's cells, cofaces, neighbors) is the intersection of the
/// vertex-cell lists over the simplex's vertices, and all down-incidence is
/// pure combinatorics on the vertex set.
///
/// The boundary operators $diff_k$ -- which double as the oriented incidence
/// backbone -- are computed lazily on first use and cached.
#[derive(Default, Debug, Clone)]
pub struct Complex {
  skeletons: Vec<Skeleton>,
  /// Per vertex, by kidx: the sorted list of cells containing it.
  vertex_cells: Vec<Vec<KSimplexIdx>>,
  /// Cached boundary operators $diff_k: C_k -> C_(k-1)$, indexed by $k$ in
  /// `0..=dim + 1`.
  boundary_operators: Vec<OnceLock<CooMatrix>>,
}

impl Complex {
  pub fn skeletons(&self) -> impl Iterator<Item = SkeletonRef<'_>> {
    (0..=self.dim()).map(|d| SkeletonRef::new(self, d))
  }
  pub fn skeleton(&self, dim: Dim) -> SkeletonRef<'_> {
    SkeletonRef::new(self, dim)
  }
  pub fn skeleton_raw(&self, dim: Dim) -> &Skeleton {
    &self.skeletons[dim]
  }
  /// The cells containing the given vertex, by kidx (sorted).
  pub fn vertex_cells(&self, vertex: KSimplexIdx) -> &[KSimplexIdx] {
    &self.vertex_cells[vertex]
  }
  pub fn nsimplices(&self, dim: Dim) -> usize {
    self.skeleton(dim).len()
  }
  pub fn vertices(&self) -> SkeletonRef<'_> {
    self.skeleton(0)
  }
  pub fn edges(&self) -> SkeletonRef<'_> {
    self.skeleton(1)
  }
  pub fn facets(&self) -> SkeletonRef<'_> {
    self.skeleton(self.dim() - 1)
  }
  pub fn cells(&self) -> SkeletonRef<'_> {
    self.skeleton(self.dim())
  }
}

/// Serialization is just the top-dimensional [`Skeleton`] (its cells): every
/// other skeleton, and every cached operator, is [`Complex::from_cells`]'s
/// job to rederive, not data to store.
impl Complex {
  pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
    self.skeleton_raw(self.dim()).save(path)
  }
  pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
    Ok(Self::from_cells(Skeleton::load(path)?))
  }
}

impl Complex {
  pub fn standard(dim: Dim) -> Self {
    Self::from_cells(Skeleton::standard(dim))
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
  pub fn boundary_facets(&self) -> Vec<SimplexIdx> {
    // A 0-dimensional complex is closed.
    if self.dim() == 0 {
      return Vec::new();
    }
    self
      .facets()
      .handle_iter()
      .filter(|f| f.cells().count() == 1)
      .map(|f| f.idx())
      .collect()
  }

  pub fn boundary_cells(&self) -> Vec<SimplexIdx> {
    self
      .boundary_facets()
      .into_iter()
      // the boundary has only one parent cell by definition
      .map(|facet| {
        facet
          .handle(self)
          .cells()
          .next()
          .expect("Boundary facets have exactly one cell.")
          .idx()
      })
      .unique()
      .collect()
  }

  /// The dim-simplices that lie on the boundary of the mesh:
  /// the subsimplices of the boundary facets.
  ///
  /// These span the boundary subcomplex; their complement spans the
  /// relative cochain complex of the pair $(K, diff K)$.
  pub fn boundary_simplices(&self, dim: Dim) -> Vec<SimplexIdx> {
    self
      .boundary_facets()
      .into_iter()
      .flat_map(|facet| {
        facet
          .handle(self)
          .faces(dim)
          .map(|sub| sub.idx())
          .collect::<Vec<_>>()
      })
      .unique()
      .sorted_by_key(|idx| idx.kidx)
      .collect()
  }

  /// The vertices that lie on the boundary of the mesh.
  pub fn boundary_vertices(&self) -> Vec<usize> {
    self
      .boundary_simplices(0)
      .into_iter()
      .map(|idx| idx.kidx)
      .collect()
  }

  /// $diff_k: Delta_k -> Delta_(k-1)$, cached after first use.
  ///
  /// The chain complex extends by zero: outside $0 <= k <= n$ the operator
  /// maps to/from the zero space.
  pub fn boundary_operator(&self, dim: Dim) -> &CooMatrix {
    self.boundary_operators[dim].get_or_init(|| self.compute_boundary_operator(dim))
  }

  fn compute_boundary_operator(&self, dim: Dim) -> CooMatrix {
    if dim == self.dim() + 1 {
      return CooMatrix::zeros(self.nsimplices(self.dim()), 0);
    }
    let sups = &self.skeleton(dim);

    if dim == 0 {
      return CooMatrix::zeros(0, sups.len());
    }

    let subs = &self.skeleton(dim - 1);
    let mut mat = CooMatrix::zeros(subs.len(), sups.len());
    for (isup, sup) in sups.handle_iter().enumerate() {
      for (sign, sub) in sup.boundary() {
        mat.push(sub.kidx(), isup, sign.as_f64());
      }
    }
    mat
  }

  /// $dif^k: Delta^k -> Delta^(k+1)$
  ///
  /// The coboundary operator, which is the discrete exterior derivative
  /// on cochains. It is the transpose of the boundary operator.
  ///
  /// The Betti numbers of the complex are in [`homology`](super::homology).
  pub fn coboundary_operator(&self, dim: Dim) -> CooMatrix {
    self.boundary_operator(dim + 1).clone().transpose()
  }
}

impl Complex {
  pub fn from_cells(cells: Skeleton) -> Self {
    let dim = cells.dim();

    // Vertices must be contiguous and fully used: labels 0..nvertices, each
    // appearing in some cell. External imports should close gaps first (see
    // `geometry::coord::mesh::close_vertex_gaps`).
    let mut vertex_used = vec![false; cells.nvertices()];
    for cell in cells.iter() {
      for v in cell.iter() {
        vertex_used[v] = true;
      }
    }
    assert!(
      vertex_used.iter().all(|&used| used),
      "Mesh vertices must be contiguous and fully used (no dangling vertices)."
    );

    // Every skeleton, derived and canonically colex-ordered: the deduplicated
    // d-subsimplices of all cells. `Skeleton::new` sorts and dedups.
    let skeletons: Vec<Skeleton> = (0..=dim)
      .map(|d| Skeleton::new(cells.iter().flat_map(|cell| cell.subsimps(d)).collect()))
      .collect();

    // Vertex -> cells incidence, built from the final (colex) cell order, so
    // each list is sorted.
    let mut vertex_cells = vec![Vec::new(); skeletons[0].len()];
    for (icell, cell) in skeletons[dim].iter().enumerate() {
      for v in cell.iter() {
        vertex_cells[v].push(icell);
      }
    }

    // Manifold check: every facet is shared by one or two cells.
    if dim >= 1 {
      let facets = &skeletons[dim - 1];
      let mut nparents = vec![0usize; facets.len()];
      for cell in skeletons[dim].iter() {
        for facet in cell.subsimps(dim - 1) {
          nparents[facets.kidx_by_simplex(&facet)] += 1;
        }
      }
      assert!(
        nparents.iter().all(|&n| n == 1 || n == 2),
        "Topology must be manifold."
      );
    }

    Self {
      skeletons,
      vertex_cells,
      boundary_operators: (0..dim + 2).map(|_| OnceLock::new()).collect(),
    }
  }
}

#[cfg(test)]
mod test {
  use crate::topology::simplex::{nsubsimplices, standard_boundary_operator, Simplex};

  use super::*;
  use formoniq_linalg::nalgebra::Matrix;

  /// Round-tripping through CBOR reproduces the same topology: the boundary
  /// (an $O(n^3)$ derived quantity, not stored data) matches, so the save/load
  /// pair is exercised through `Complex::from_cells`, not just through serde.
  #[test]
  fn save_load_roundtrip() {
    use crate::gen::cartesian::CartesianMeshInfo;

    let (topology, _) = CartesianMeshInfo::new_unit(3, 2).compute_coord_complex();

    let path = std::env::temp_dir().join(format!("formoniq_test_{}.cbor", std::process::id()));
    topology.save(&path).unwrap();
    let loaded = Complex::load(&path).unwrap();
    std::fs::remove_file(&path).unwrap();

    assert_eq!(loaded.dim(), topology.dim());
    for dim in 0..=topology.dim() {
      assert_eq!(loaded.nsimplices(dim), topology.nsimplices(dim));
    }
    assert_eq!(loaded.betti_numbers(), topology.betti_numbers());
  }

  /// Every skeleton is in canonical colexicographic order, and the vertices
  /// are contiguous and fully used. This is the ordering contract the file
  /// formats and cochain indexing rely on.
  #[test]
  fn skeletons_are_colex_ordered_and_vertices_contiguous() {
    use crate::gen::cartesian::CartesianMeshInfo;

    for dim in 1..=3 {
      let (topology, _) = CartesianMeshInfo::new_unit(dim, 3).compute_coord_complex();

      // Vertices are exactly 0..nvertices, each labelled by its own kidx.
      let vertices = topology.skeleton(0);
      for (kidx, vertex) in vertices.iter().enumerate() {
        assert_eq!(vertex.vertices, vec![kidx]);
      }

      // Each skeleton is strictly increasing in colex order.
      for k in 0..=dim {
        let skeleton = topology.skeleton(k);
        let simplices: Vec<_> = skeleton.iter().collect();
        assert!(
          simplices.windows(2).all(|w| w[0] < w[1]),
          "skeleton {k} is not colex-ordered"
        );
      }
    }
  }

  /// $dif compose dif = 0$: the defining law of a cochain complex.
  #[test]
  fn coboundary_squares_to_zero() {
    use crate::gen::cartesian::CartesianMeshInfo;
    use formoniq_linalg::nalgebra::CsrMatrix;

    for dim in 1..=3 {
      let (topology, _) = CartesianMeshInfo::new_unit(dim, 2).compute_coord_complex();
      for k in 0..dim.saturating_sub(1) {
        let dif_k = CsrMatrix::from(&topology.coboundary_operator(k));
        let dif_kk = CsrMatrix::from(&topology.coboundary_operator(k + 1));
        let dif_dif = dif_kk * dif_k;
        assert!(dif_dif.values().iter().all(|&v| v == 0.0));
      }
    }
  }

  #[test]
  fn boundary_simplices_facets_are_boundary_facets() {
    for dim in 1..=3 {
      let (topology, _) =
        crate::gen::cartesian::CartesianMeshInfo::new_unit(dim, 2).compute_coord_complex();
      assert_eq!(topology.boundary_simplices(dim - 1), {
        let mut facets = topology.boundary_facets();
        facets.sort_by_key(|idx| idx.kidx);
        facets
      });
    }
  }

  #[test]
  fn standard_boundary_operator_agrees_with_complex() {
    for dim in 1..=4 {
      let complex = Complex::standard(dim);
      for k in 0..=dim {
        let combinatorial = standard_boundary_operator(dim, k);
        let from_complex = Matrix::from(complex.boundary_operator(k));
        assert_eq!(combinatorial, from_complex);
      }
    }
  }

  #[test]
  fn incidence() {
    let dim = 3;
    let complex = Complex::standard(dim);
    let cell = complex.cells().handle_iter().next().unwrap();

    let cell_simplex = Simplex::standard(dim);
    for dim_sub in 0..=dim {
      let subs: Vec<_> = cell.faces(dim_sub).collect();
      assert_eq!(subs.len(), nsubsimplices(dim, dim_sub));
      let subs_vertices: Vec<_> = cell_simplex.subsimps(dim_sub).collect();
      assert_eq!(
        subs
          .iter()
          .map(|sub| sub.simplex().clone())
          .collect::<Vec<_>>(),
        subs_vertices
      );

      for (isub, sub) in subs.iter().enumerate() {
        let sub_vertices = &subs_vertices[isub];
        for dim_sup in dim_sub..dim {
          for sup in sub.cofaces(dim_sup) {
            assert!(
              sub_vertices.is_subsimplex_of(sup.simplex())
                && sup.simplex().is_subsimplex_of(&cell_simplex)
            );
          }
        }
      }
    }
  }

  /// Navigation on a triangulation: facets, neighbors, star and link behave
  /// as their topological definitions demand.
  #[test]
  fn ref_navigation() {
    use crate::gen::cartesian::CartesianMeshInfo;

    let (topology, _) = CartesianMeshInfo::new_unit(2, 3).compute_coord_complex();

    for cell in topology.cells().handle_iter() {
      // A triangle has 3 facets (edges) and at most 3 neighbors across them.
      assert_eq!(cell.facets().count(), 3);
      assert!(cell.neighbors().count() <= 3);
      // Neighbors share a facet and are distinct cells.
      for nb in cell.neighbors() {
        assert_eq!(nb.dim(), 2);
        assert_ne!(nb.idx(), cell.idx());
        let shared = cell
          .facets()
          .filter(|f| nb.facets().any(|g| g.idx() == f.idx()));
        assert_eq!(shared.count(), 1);
      }
      // The star of a top cell is just itself.
      assert_eq!(cell.star().count(), 1);
    }

    // Boundary edges have a single cell; interior edges two.
    assert!(topology.edges().handle_iter().any(|e| e.is_boundary()));
    for edge in topology.edges().handle_iter() {
      assert_eq!(edge.cells().count(), if edge.is_boundary() { 1 } else { 2 });
    }

    // The link of a vertex never touches the vertex itself; its star does.
    for vertex in topology.vertices().handle_iter() {
      assert!(vertex.star().any(|s| s.idx() == vertex.idx()));
      for linked in vertex.link() {
        assert!(!linked.simplex().contains(vertex.kidx()));
      }
    }
  }
}
