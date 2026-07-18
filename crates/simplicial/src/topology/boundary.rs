//! The boundary $diff K$ of a complex as a complex in its own right,
//! together with the inclusion $diff K arrow.hook K$.
//!
//! This is the third object of the short exact sequence of the pair,
//!
//! $0 -> C^k (K, diff K) -> C^k (K) -->^"tr" C^k (diff K) -> 0$
//!
//! whose kernel is the relative complex. The trace (restriction of cochains
//! to boundary simplices) is a cochain map: $"tr" compose dif = dif compose "tr"$.

use super::{
  complex::Complex,
  handle::{KSimplexIdx, SimplexIdx},
  simplex::Simplex,
  skeleton::Skeleton,
};
use crate::{
  geometry::{coord::mesh::MeshCoords, metric::mesh::MeshLengths},
  topology::VertexIdx,
  Dim,
};

use formoniq_linalg::nalgebra::{CooMatrix, Matrix};

/// A codimension-1 subcomplex of $K$ as a complex in its own right, with
/// its own (monotone) vertex numbering and the simplex-wise inclusion into
/// the parent complex.
///
/// For the full boundary $diff K$ this is a closed $(n-1)$-manifold; a
/// subset $Gamma$ of the boundary facets gives the boundary part of mixed
/// boundary conditions.
pub struct BoundaryComplex {
  complex: Complex,
  /// Per grade: boundary k-simplex index -> parent k-simplex index.
  /// The inclusion is monotone, so no signs appear.
  parent_kidxs: Vec<Vec<KSimplexIdx>>,
  /// Per grade: number of k-simplices of the parent complex.
  parent_nsimplices: Vec<usize>,
}

impl Complex {
  /// The boundary $diff K$ as a first-class complex.
  /// `None` if the manifold is closed.
  pub fn boundary_complex(&self) -> Option<BoundaryComplex> {
    let facets = self.boundary_facets();
    if facets.is_empty() {
      return None;
    }
    Some(self.facet_subcomplex(facets))
  }

  /// The subcomplex spanned by the given facets: for a subset
  /// $Gamma subset.eq diff K$ of the boundary facets this is the boundary
  /// part carrying mixed (Dirichlet/Neumann/Robin) boundary conditions.
  pub fn facet_subcomplex(&self, facets: Vec<SimplexIdx>) -> BoundaryComplex {
    assert!(!facets.is_empty(), "Facet subcomplex must not be empty.");
    assert!(facets.iter().all(|f| f.dim == self.dim() - 1));

    // Monotone vertex renumbering: sorted parent vertices -> 0..m.
    let mut parent_vertices: Vec<VertexIdx> = facets
      .iter()
      .flat_map(|facet| facet.handle(self).simplex().vertices.clone())
      .collect();
    parent_vertices.sort_unstable();
    parent_vertices.dedup();
    let to_local = |parent: VertexIdx| -> VertexIdx {
      parent_vertices
        .binary_search(&parent)
        .expect("Vertex lies on the subcomplex.")
    };

    let cells: Vec<Simplex> = facets
      .into_iter()
      .map(|facet| {
        let facet = facet.handle(self);
        Simplex::new(facet.simplex().iter().map(to_local).collect())
      })
      .collect();
    let complex = Complex::from_cells(Skeleton::new(cells));

    // Indexed by the full parent grade range $0..=n$, not just the boundary's
    // own $0..=n-1$: at grade $n$ the $(n-1)$-dimensional boundary carries no
    // simplices, so the trace is the zero map into an empty codomain. Storing
    // that empty row explicitly keeps `trace_operator` total at top grade
    // rather than indexing one past the boundary dimension.
    let parent_kidxs = (0..=self.dim())
      .map(|grade| {
        if grade > complex.dim() {
          return Vec::new();
        }
        complex
          .skeleton(grade)
          .iter()
          .map(|boundary_simp| {
            let parent_simp =
              Simplex::new(boundary_simp.iter().map(|v| parent_vertices[v]).collect());
            self.skeleton(grade).kidx_by_simplex(&parent_simp)
          })
          .collect()
      })
      .collect();
    let parent_nsimplices = (0..=self.dim()).map(|k| self.nsimplices(k)).collect();

    BoundaryComplex {
      complex,
      parent_kidxs,
      parent_nsimplices,
    }
  }
}

impl BoundaryComplex {
  pub fn complex(&self) -> &Complex {
    &self.complex
  }
  pub fn dim(&self) -> Dim {
    self.complex.dim()
  }
  /// The parent indices of the boundary k-simplices.
  pub fn parent_kidxs(&self, grade: Dim) -> &[KSimplexIdx] {
    &self.parent_kidxs[grade]
  }
  pub fn parent_idx(&self, boundary_idx: SimplexIdx) -> SimplexIdx {
    SimplexIdx::new(
      boundary_idx.dim,
      self.parent_kidxs[boundary_idx.dim][boundary_idx.kidx],
    )
  }

  /// The trace $"tr": C^k (K) -> C^k (diff K)$: restriction of cochains to
  /// the boundary simplices. A cochain map, $"tr" compose dif = dif compose "tr"$,
  /// and the cokernel projection of the relative inclusion.
  pub fn trace_operator(&self, grade: Dim) -> CooMatrix {
    let mut trace = CooMatrix::new(
      self.parent_kidxs[grade].len(),
      self.parent_nsimplices[grade],
    );
    for (boundary_kidx, &parent_kidx) in self.parent_kidxs[grade].iter().enumerate() {
      trace.push(boundary_kidx, parent_kidx, 1.0);
    }
    trace
  }

  /// The induced geometry: parent edge lengths restricted to the boundary.
  pub fn trace_lengths(&self, parent: &MeshLengths) -> MeshLengths {
    // A 0-dimensional boundary (of a 1d mesh) has no edges.
    let lengths: Vec<f64> = if self.dim() == 0 {
      Vec::new()
    } else {
      self
        .parent_kidxs(1)
        .iter()
        .map(|&iedge| parent[iedge])
        .collect()
    };
    MeshLengths::new_unchecked(lengths.into())
  }

  /// The vertex coordinates restricted to the boundary.
  pub fn trace_coords(&self, parent: &MeshCoords) -> MeshCoords {
    let columns: Vec<_> = self
      .parent_kidxs(0)
      .iter()
      .map(|&ivertex| parent.matrix().column(ivertex))
      .collect();
    MeshCoords::new(Matrix::from_columns(&columns))
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::gen::cartesian::CartesianMeshInfo;

  use formoniq_linalg::nalgebra::CsrMatrix;

  /// The boundary of the n-cube is a closed manifold with the homology of
  /// the (n-1)-sphere.
  #[test]
  fn boundary_of_cube_is_sphere() {
    for dim in 1..=3 {
      let (topology, _) = CartesianMeshInfo::new_unit(dim, 2).compute_coord_complex();
      let boundary = topology.boundary_complex().unwrap();
      assert!(!boundary.complex().has_boundary());
      for k in 0..dim {
        // S^(n-1) betti numbers; the 0-sphere is two points.
        let expected = if dim == 1 {
          2
        } else {
          usize::from(k == 0 || k == dim - 1)
        };
        assert_eq!(
          boundary.complex().betti_number(k),
          expected,
          "dim={dim} k={k}"
        );
      }
    }
  }

  /// The trace is a cochain map: $"tr" compose dif = dif compose "tr"$.
  #[test]
  fn trace_is_cochain_map() {
    for dim in 2..=3 {
      let (topology, _) = CartesianMeshInfo::new_unit(dim, 2).compute_coord_complex();
      let boundary = topology.boundary_complex().unwrap();
      for k in 0..(dim - 1) {
        let trace_k = CsrMatrix::from(&boundary.trace_operator(k));
        let trace_kk = CsrMatrix::from(&boundary.trace_operator(k + 1));
        let dif_parent = CsrMatrix::from(&topology.coboundary_operator(k));
        let dif_boundary = CsrMatrix::from(&boundary.complex().coboundary_operator(k));

        let tr_dif = Matrix::from(&CooMatrix::from(&(trace_kk * dif_parent)));
        let dif_tr = Matrix::from(&CooMatrix::from(&(dif_boundary * trace_k)));
        assert_eq!(tr_dif, dif_tr);
      }
    }
  }

  /// Exactness of 0 -> C(K, dK) -> C(K) -> C(dK) -> 0 at the level of
  /// dimensions: interior and boundary DOFs partition the full complex.
  #[test]
  fn trace_dimensions_are_exact() {
    for dim in 1..=3 {
      let (topology, _) = CartesianMeshInfo::new_unit(dim, 2).compute_coord_complex();
      let boundary = topology.boundary_complex().unwrap();
      for k in 0..dim {
        let nboundary = boundary.complex().nsimplices(k);
        let ninterior = topology.nsimplices(k) - topology.boundary_simplices(k).len();
        assert_eq!(nboundary + ninterior, topology.nsimplices(k));
        assert_eq!(nboundary, topology.boundary_simplices(k).len());
      }
    }
  }
}
