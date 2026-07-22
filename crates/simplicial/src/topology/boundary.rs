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
  role::Facet,
  simplex::Simplex,
  skeleton::Skeleton,
};
use crate::{
  Dim,
  geometry::{coord::mesh::MeshCoords, metric::mesh::MeshLengthsSq},
  topology::VertexIdx,
};

use crate::linalg::{CooMatrix, Matrix};

/// A codimension-1 subcomplex of $K$ as a complex in its own right, with
/// its own (monotone) vertex numbering and the simplex-wise inclusion into
/// the parent complex.
///
/// For the full boundary $diff K$ this is a closed $(n-1)$-manifold; a
/// subset $Gamma$ of the boundary facets gives the boundary part of mixed
/// boundary conditions.
#[derive(Debug, Clone)]
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
  ///
  /// The [`Facet`] witness carries the codimension-1 precondition; what it
  /// cannot carry is *which* complex it proves it for, hence the ownership
  /// check.
  pub fn facet_subcomplex(&self, facets: Vec<Facet>) -> BoundaryComplex {
    assert!(!facets.is_empty(), "Facet subcomplex must not be empty.");
    assert!(
      facets.iter().all(|f| f.belongs_to(self)),
      "Facets must belong to this complex."
    );

    // Monotone vertex renumbering: sorted parent vertices -> 0..m.
    let mut parent_vertices: Vec<VertexIdx> = facets
      .iter()
      .flat_map(|facet| facet.simplex().vertices.clone())
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
      .map(|facet| Simplex::new(facet.simplex().iter().map(to_local).collect()))
      .collect();
    let complex = Complex::from_cells(Skeleton::new(cells));

    // Indexed by the full parent grade range $0..=n$, not just the boundary's
    // own $0..=n-1$: at grade $n$ the $(n-1)$-dimensional boundary carries no
    // simplices, so the trace is the zero map into an empty codomain. Storing
    // that empty row explicitly keeps `trace_operator` total at top grade
    // rather than indexing one past the boundary dimension.
    let parent_kidxs = self
      .dim()
      .range_inclusive()
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
    let parent_nsimplices = self
      .dim()
      .range_inclusive()
      .map(|k| self.nsimplices(k))
      .collect();

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
    &self.parent_kidxs[grade.index()]
  }
  pub fn parent_idx(&self, boundary_idx: SimplexIdx) -> SimplexIdx {
    SimplexIdx::new(
      boundary_idx.dim,
      self.parent_kidxs[boundary_idx.dim.index()][boundary_idx.kidx],
    )
  }

  /// The trace $"tr": C^k (K) -> C^k (diff K)$: restriction of cochains to
  /// the boundary simplices. A cochain map, $"tr" compose dif = dif compose "tr"$,
  /// and the cokernel projection of the relative inclusion.
  pub fn trace_operator(&self, grade: Dim) -> CooMatrix {
    let mut trace = CooMatrix::new(
      self.parent_kidxs[grade.index()].len(),
      self.parent_nsimplices[grade.index()],
    );
    for (boundary_kidx, &parent_kidx) in self.parent_kidxs[grade.index()].iter().enumerate() {
      trace.push(boundary_kidx, parent_kidx, 1.0);
    }
    trace
  }

  /// The induced geometry: parent squared edge lengths restricted to the
  /// boundary. A pure data restriction, total on any signature; on an
  /// indefinite parent a null facet carries degenerate data, which surfaces
  /// where a facet metric is actually built from it, not here.
  pub fn trace_lengths_sq(&self, parent: &MeshLengthsSq) -> MeshLengthsSq {
    // A 0-dimensional boundary (of a 1d mesh) has no edges.
    let lengths_sq: Vec<f64> = if self.dim() == 0 {
      Vec::new()
    } else {
      self
        .parent_kidxs(Dim::ONE)
        .iter()
        .map(|&iedge| parent[iedge])
        .collect()
    };
    MeshLengthsSq::new_unchecked(lengths_sq.into())
  }

  /// The vertex coordinates restricted to the boundary.
  pub fn trace_coords(&self, parent: &MeshCoords) -> MeshCoords {
    let columns: Vec<_> = self
      .parent_kidxs(Dim::ZERO)
      .iter()
      .map(|&ivertex| parent.matrix().column(ivertex))
      .collect();
    MeshCoords::with_ambient(Matrix::from_columns(&columns), parent.ambient().clone())
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::Dim;
  use crate::mesher::cartesian::CartesianGrid;

  use crate::linalg::CsrMatrix;

  /// The boundary of the n-cube is a closed manifold with the homology of
  /// the (n-1)-sphere.
  #[test]
  fn boundary_of_cube_is_sphere() {
    for dim in (1..=3usize).map(Dim::from) {
      let (topology, _) = CartesianGrid::new_unit(dim, 2).triangulate();
      let boundary = topology.boundary_complex().unwrap();
      assert!(!boundary.complex().has_boundary());
      for k in dim.range() {
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
    for dim in (2..=3usize).map(Dim::from) {
      let (topology, _) = CartesianGrid::new_unit(dim, 2).triangulate();
      let boundary = topology.boundary_complex().unwrap();
      for k in (dim - 1).range() {
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
    for dim in (1..=3usize).map(Dim::from) {
      let (topology, _) = CartesianGrid::new_unit(dim, 2).triangulate();
      let boundary = topology.boundary_complex().unwrap();
      for k in dim.range() {
        let nboundary = boundary.complex().nsimplices(k);
        let ninterior = topology.nsimplices(k) - topology.boundary_simplices(k).len();
        assert_eq!(nboundary + ninterior, topology.nsimplices(k));
        assert_eq!(nboundary, topology.boundary_simplices(k).len());
      }
    }
  }
}
