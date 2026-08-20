//! Whether a complex is a manifold, as far as that is decidable.
//!
//! A simplicial complex is a *piecewise-linear manifold* of dimension $n$ when
//! the link of every vertex is a PL sphere $S^(n-1)$, or a PL ball where the
//! vertex lies on the boundary. That condition cannot be checked: recognizing
//! whether a complex is a PL sphere is undecidable for $S^n$ with $n >= 5$
//! (Novikov), and open for $n = 4$. No amount of care in the implementation
//! changes that, so what a library can offer is the ladder of conditions that
//! *are* computable, each strictly weaker than the one above it:
//!
//! 1. **pure**: every simplex is a face of a cell. Structural here rather than
//!    checked, since [`Complex::from_cells`] derives every skeleton as the
//!    subsimplices of the cells.
//! 2. **pseudomanifold**: every facet lies in one or two cells, hence the
//!    two-sidedness that makes [`Facet::adjacent_cells`](super::role::Facet)
//!    a pair and the boundary well defined. Checked eagerly by
//!    [`Complex::from_cells`], since the rest of the crate reads incidence
//!    through that pair.
//! 3. **homology manifold**: the link of every vertex has the homology of
//!    $S^(n-1)$, or is acyclic where the vertex is on the boundary.
//!    [`Complex::is_homology_manifold`], and the strongest rung there is.
//!
//! The gap at the top is real, not a formality: the double suspension of a
//! homology 3-sphere is a homology manifold, and is homeomorphic to $S^5$, yet
//! carries no PL manifold structure at the suspension points. So a complex
//! passing rung 3 need not be a PL manifold, and the code says only what it
//! checked.
//!
//! None of this is a metric statement. A manifold condition is a fact about
//! incidence alone.

use super::{
  complex::Complex,
  handle::{KSimplexIdx, SimplexRef},
  role::Facet,
  subcomplex::Subcomplex,
};
use crate::Dim;

impl Complex {
  /// Whether every facet lies in one or two cells: the pseudomanifold
  /// condition, and what makes a facet's cells a pair and the boundary well
  /// defined.
  ///
  /// Always true of a complex built by [`Complex::from_cells`], which checks
  /// it; the predicate is here for one built by
  /// [`Complex::from_cells_unchecked`].
  pub fn is_pseudomanifold(&self) -> bool {
    let Some(facets) = self.role_skeleton::<super::role::roles::Facet>() else {
      // A 0-complex has no facets, and no condition to violate.
      return true;
    };
    facets
      .handle_iter()
      .all(|facet| (1..=2).contains(&facet.get().cells().count()))
  }

  /// The link of a vertex as a complex in its own right: the face opposite
  /// that vertex in each cell containing it, and their faces.
  ///
  /// Those opposite faces are facets of this complex, so the link is a
  /// [`Subcomplex`] like the boundary is, carrying the same inclusion back into
  /// the parent. For a vertex of a triangulated surface it is the polygon
  /// around it.
  ///
  /// `None` on a 0-complex, where a vertex is its own cell and its link is the
  /// empty complex $S^(-1)$.
  pub fn vertex_link(&self, vertex: KSimplexIdx) -> Option<Subcomplex> {
    let facets: Vec<Facet> = self
      .vertices()
      .handle_by_kidx(vertex)
      .cells()
      .filter_map(|cell| {
        cell
          .get()
          .facets()
          .find(|facet| !facet.contains(vertex))
          .map(SimplexRef::role)
      })
      .collect();
    (!facets.is_empty()).then(|| self.facet_subcomplex(facets))
  }

  /// Whether every vertex link has the homology of a sphere $S^(n-1)$, or is
  /// acyclic as it may be on the boundary: the strongest rung of the manifold
  /// condition that is computable at all (see the [module docs](self)).
  ///
  /// Exact, being read off the integer homology of each link, and metric-free.
  /// Costs a homology computation per vertex, so it is a check to run, never an
  /// invariant to maintain.
  pub fn is_homology_manifold(&self) -> bool {
    let dim = self.dim();
    self.vertices().handle_iter().all(|vertex| {
      match self.vertex_link(vertex.kidx()) {
        // The link in a 0-complex is empty, which is $S^(-1)$.
        None => true,
        Some(link) => {
          let betti = link.complex().betti_numbers();
          betti == sphere_betti(dim - 1) || betti == ball_betti(dim - 1)
        }
      }
    })
  }
}

/// The Betti numbers of $S^d$: one class in grade $0$ and one in grade $d$,
/// which coincide at $d = 0$, where $S^0$ is two points.
fn sphere_betti(dim: Dim) -> Vec<usize> {
  let mut betti = vec![0; (dim + 1).index()];
  betti[0] = 1;
  betti[dim.index()] += 1;
  betti
}

/// The Betti numbers of a ball of dimension $d$: contractible, so one class in
/// grade $0$ and nothing above.
fn ball_betti(dim: Dim) -> Vec<usize> {
  let mut betti = vec![0; (dim + 1).index()];
  betti[0] = 1;
  betti
}
