//! A codimension-1 subcomplex $L subset.eq K$ as a complex in its own right,
//! together with the inclusion $L arrow.hook K$.
//!
//! Three of them are the same construction, which is why there is one type: the
//! boundary $partial K$, a part $Gamma subset.eq partial K$ carrying its own boundary
//! condition, and the link of a vertex. Each is spanned by a set of facets, each
//! renumbers its vertices monotonically, and each keeps the map back.
//!
//! For the boundary this is the third object of the short exact sequence of the
//! pair,
//!
//! $0 -> C^k (K, partial K) -> C^k (K) -->^"tr" C^k (partial K) -> 0$,
//!
//! whose kernel is the relative complex. The trace (restriction of cochains
//! to the subcomplex's simplices) is a cochain map: $"tr" compose dif = dif
//! compose "tr"$.

use super::{
  chain::{Cochain, Coefficient},
  complex::Complex,
  handle::{KSimplexIdx, SimplexIdx},
  role::Facet,
  simplex::Simplex,
  skeleton::Skeleton,
};
use crate::{Dim, topology::VertexIdx};

use crate::linalg::{CooMatrix, Selection, Vector};

/// A codimension-1 subcomplex of $K$ as a complex in its own right, with
/// its own (monotone) vertex numbering and the simplex-wise inclusion into
/// the parent complex.
///
/// For the full boundary $partial K$ this is a closed $(n-1)$-manifold; a subset
/// $Gamma$ of the boundary facets gives the boundary part of mixed boundary
/// conditions; and the facets opposite a vertex give its link
/// ([`Complex::vertex_link`]).
#[derive(Debug, Clone)]
pub struct Subcomplex {
  complex: Complex,
  /// Per grade: which of the parent's $k$-simplices are the subcomplex's,
  /// hence the inclusion $C_k (L) arrow.hook C_k (K)$ and the trace back down.
  /// Both are needed, and a [`Selection`] is both.
  inclusion: Vec<Selection>,
}

impl Complex {
  /// The boundary $partial K$ as a first-class complex.
  /// `None` if the manifold is closed.
  pub fn boundary_complex(&self) -> Option<Subcomplex> {
    let facets = self.boundary_facets();
    if facets.is_empty() {
      return None;
    }
    Some(self.facet_subcomplex(facets))
  }

  /// The subcomplex spanned by the given facets: for a subset
  /// $Gamma subset.eq partial K$ of the boundary facets this is the boundary
  /// part carrying mixed (Dirichlet/Neumann/Robin) boundary conditions.
  ///
  /// The [`Facet`] witness carries the codimension-1 precondition; what it
  /// cannot carry is which complex it proves it for, hence the ownership
  /// check.
  pub fn facet_subcomplex(&self, facets: Vec<Facet>) -> Subcomplex {
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

    // Indexed by the full parent grade range $0..=n$, not just the
    // subcomplex's own $0..=n-1$: at grade $n$ it carries no simplices, so the
    // inclusion is the empty selection of the parent's $n$-simplices, which is
    // what keeps every grade-keyed accessor total at the top rather than
    // indexing one past the subcomplex's dimension.
    let inclusion = self
      .dim()
      .range_inclusive()
      .map(|grade| {
        let parent = self.skeleton(grade);
        Selection::new(
          parent.len(),
          complex
            .skeleton(grade)
            .iter()
            .map(|sub_simp| {
              let parent_simp = Simplex::new(sub_simp.iter().map(|v| parent_vertices[v]).collect());
              parent.kidx_by_simplex(&parent_simp)
            })
            .collect(),
        )
      })
      .collect();

    Subcomplex { complex, inclusion }
  }
}

impl Subcomplex {
  pub fn complex(&self) -> &Complex {
    &self.complex
  }
  pub fn dim(&self) -> Dim {
    self.complex.dim()
  }
  /// The inclusion $L arrow.hook K$ at one grade: which of the parent's
  /// $k$-simplices are the subcomplex's.
  ///
  /// Its [`complement`](Selection::complement) is the relative chain group
  /// $C_k (K, L)$, so this one datum carries the whole short exact sequence of
  /// the pair.
  pub fn inclusion(&self, grade: impl Into<Dim>) -> &Selection {
    &self.inclusion[grade.into().index()]
  }
  /// The parent indices of the subcomplex's k-simplices.
  pub fn parent_kidxs(&self, grade: impl Into<Dim>) -> &[KSimplexIdx] {
    self.inclusion(grade).indices()
  }
  pub fn parent_idx(&self, sub_idx: SimplexIdx) -> SimplexIdx {
    SimplexIdx::new(sub_idx.dim, self.parent_kidxs(sub_idx.dim)[sub_idx.kidx])
  }

  /// The trace $"tr": C^k (K) -> C^k (partial K)$: restriction of cochains to
  /// the boundary simplices. A cochain map, $"tr" compose dif = dif compose "tr"$,
  /// and the cokernel projection of the relative inclusion.
  pub fn trace_operator(&self, grade: impl Into<Dim>) -> CooMatrix {
    self.inclusion(grade).restriction()
  }

  /// The trace $"tr" c$ of a cochain, applied without materializing
  /// [`trace_operator`](Self::trace_operator): gathering the parent
  /// coefficients at [`parent_kidxs`](Self::parent_kidxs) is that matrix's
  /// definition, and a select needs no sparse product.
  ///
  /// Metric-free, being a selection of simplices, and total over grade: the
  /// subcomplex carries no simplices at the parent's top grade, so
  /// $C^n (L) = 0$ and the trace of an $n$-cochain is the empty cochain.
  pub fn trace<R: Coefficient>(&self, cochain: &Cochain<R>) -> Cochain<R> {
    let grade = cochain.grade();
    let parent_kidxs = self.parent_kidxs(grade);
    Cochain::new(
      grade,
      Vector::from_iterator(
        parent_kidxs.len(),
        parent_kidxs
          .iter()
          .map(|&kidx| cochain.coeffs()[kidx].clone()),
      ),
    )
  }
}
