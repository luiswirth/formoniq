use super::{
  handle::{SimplexIdx, SkeletonRef},
  simplex::Simplex,
  skeleton::Skeleton,
};
use crate::Dim;

use common::linalg::nalgebra::{CooMatrix, CooMatrixExt, Matrix};

use itertools::Itertools;

/// A simplicial manifold complex.
#[derive(Default, Debug, Clone)]
pub struct Complex {
  skeletons: Vec<ComplexSkeleton>,
}

/// A skeleton inside of a complex.
#[derive(Default, Debug, Clone)]
pub struct ComplexSkeleton {
  skeleton: Skeleton,
  complex_data: SkeletonComplexData,
}
impl ComplexSkeleton {
  pub fn skeleton(&self) -> &Skeleton {
    &self.skeleton
  }
  pub fn complex_data(&self) -> &[SimplexComplexData] {
    &self.complex_data
  }
}

pub type SkeletonComplexData = Vec<SimplexComplexData>;

#[derive(Default, Debug, Clone)]
pub struct SimplexComplexData {
  pub cocells: Vec<SimplexIdx>,
}

impl Complex {
  pub fn skeletons(&self) -> impl Iterator<Item = SkeletonRef<'_>> {
    (0..=self.dim()).map(|d| SkeletonRef::new(self, d))
  }
  pub fn skeleton(&self, dim: Dim) -> SkeletonRef<'_> {
    SkeletonRef::new(self, dim)
  }
  pub fn complex_skeleton(&self, dim: Dim) -> &ComplexSkeleton {
    &self.skeletons[dim]
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

  /// $diff^k: Delta_k -> Delta_(k-1)$
  ///
  /// The chain complex extends by zero: outside $0 <= k <= n$ the operator
  /// maps to/from the zero space.
  pub fn boundary_operator(&self, grade: Dim) -> CooMatrix {
    if grade == self.dim() + 1 {
      return CooMatrix::zeros(self.nsimplices(self.dim()), 0);
    }
    let sups = &self.skeleton(grade);

    if grade == 0 {
      return CooMatrix::zeros(0, sups.len());
    }

    let subs = &self.skeleton(grade - 1);
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
  pub fn coboundary_operator(&self, grade: Dim) -> CooMatrix {
    self.boundary_operator(grade + 1).transpose()
  }

  /// Dimension of the k-th homology group.
  ///
  /// k-th Betti number.
  /// Number of k-dimensional holes in the manifold.
  /// Computed using simplicial homology.
  pub fn homology_dim(&self, dim: Dim) -> usize {
    // TODO: use sparse matrix!
    let boundary_this = Matrix::from(&self.boundary_operator(dim));
    let boundary_plus = Matrix::from(&self.boundary_operator(dim + 1));

    const RANK_TOL: f64 = 1e-12;

    let dim_image = |op: &Matrix| -> usize {
      if op.is_empty() {
        0
      } else {
        op.rank(RANK_TOL)
      }
    };
    let dim_kernel = |op: &Matrix| -> usize { op.ncols() - dim_image(op) };

    let dim_cycles = dim_kernel(&boundary_this);
    let dim_boundaries = dim_image(&boundary_plus);

    dim_cycles - dim_boundaries
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

    let mut skeletons = vec![ComplexSkeleton::default(); dim + 1];
    skeletons[0] = ComplexSkeleton {
      skeleton: Skeleton::new((0..cells.nvertices()).map(Simplex::single).collect()),
      complex_data: (0..cells.nvertices())
        .map(|_| SimplexComplexData::default())
        .collect(),
    };

    for (icell, cell) in cells.iter().enumerate() {
      for (
        dim_skeleton,
        ComplexSkeleton {
          skeleton,
          complex_data: mesh_data,
        },
      ) in skeletons.iter_mut().enumerate()
      {
        for sub in cell.subsimps(dim_skeleton) {
          let (sub_idx, is_new) = skeleton.insert(sub);
          let sub_data = if is_new {
            mesh_data.push(SimplexComplexData::default());
            mesh_data.last_mut().unwrap()
          } else {
            &mut mesh_data[sub_idx]
          };
          sub_data.cocells.push(SimplexIdx::new(dim, icell));
        }
      }
    }

    // Topology checks.
    if dim >= 1 {
      let facet_data = skeletons[dim - 1].complex_data();
      for SimplexComplexData { cocells } in facet_data {
        let nparents = cocells.len();
        let is_manifold = nparents == 2 || nparents == 1;
        assert!(is_manifold, "Topology must be manifold.");
      }
    }

    Self { skeletons }.into_colex_ordered()
  }

  /// Re-key every skeleton into canonical colexicographic order. The cell
  /// indices referenced by `cocells` are remapped to the new cell order, so
  /// all incidence stays consistent. Idempotent.
  fn into_colex_ordered(mut self) -> Self {
    let dim = self.dim();

    // Permutation sorting the top (cell) skeleton, and its inverse
    // old-cell-index -> new-cell-index.
    let cell_perm = colex_permutation(self.skeletons[dim].skeleton());
    let mut cell_new_of_old = vec![0usize; cell_perm.len()];
    for (new, &old) in cell_perm.iter().enumerate() {
      cell_new_of_old[old] = new;
    }

    // Reorder each skeleton and its parallel complex data by colex.
    for cs in &mut self.skeletons {
      let perm = colex_permutation(&cs.skeleton);
      let old: Vec<Simplex> = cs.skeleton.iter().cloned().collect();
      let nvertices = cs.skeleton.nvertices();
      let simplices = perm.iter().map(|&o| old[o].clone()).collect();
      cs.complex_data = perm.iter().map(|&o| cs.complex_data[o].clone()).collect();
      cs.skeleton = Skeleton::from_ordered(simplices, nvertices);
    }

    // Remap the cell indices stored in every cocell to the new cell order.
    for cs in &mut self.skeletons {
      for data in &mut cs.complex_data {
        for cocell in &mut data.cocells {
          cocell.kidx = cell_new_of_old[cocell.kidx];
        }
      }
    }

    self
  }
}

/// The permutation `new_pos -> old_kidx` that orders a skeleton's simplices
/// colexicographically.
fn colex_permutation(skeleton: &Skeleton) -> Vec<usize> {
  let mut perm: Vec<usize> = (0..skeleton.len()).collect();
  perm.sort_by(|&a, &b| skeleton.simplex_by_kidx(a).cmp(skeleton.simplex_by_kidx(b)));
  perm
}

#[cfg(test)]
mod test {
  use crate::topology::simplex::{nsubsimplices, standard_boundary_operator, Simplex};

  use super::*;

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
    use common::linalg::nalgebra::CsrMatrix;

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
        let from_complex = Matrix::from(&complex.boundary_operator(k));
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
