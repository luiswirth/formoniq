//! The render surface: the 2-manifold a field is actually seen on.
//!
//! The bake reduces an $n$-manifold to the render primitive $min(n, 2)$, and
//! for a solid ($n >= 3$) that primitive is the boundary $partial M$, itself a
//! genuine closed $(n-1)$-manifold, carrying its own `Complex`, its own
//! coherent orientation and its own metric, not a bag of faces. This type is
//! that reduction named once, so every mark reads the object it is drawn on
//! rather than the object it was solved on.
//!
//! Below $n = 3$ the reduction is the identity: the mesh is already its own
//! render surface. That is why there is no dimension dispatch here beyond the
//! single construction, a caller asks the surface for its complex and gets
//! either the parent or the boundary, and cannot tell which.
//!
//! A field reaches the surface by its trace. $i^*: C^k (M) -> C^k (partial M)$
//! ([`Subcomplex::trace_operator`]) is a cochain map, so $i^* dif = dif i^*$
//! and the traced coefficients are a genuine Whitney form on $partial M$, not a
//! resample, not a nodal recovery. This is what makes drawing it honest.
//!
//! The trace is total in grade, but it is zero at the top. $partial M$ has no
//! $n$-simplices, so $C^n (partial M) = 0$ and an $n$-form's trace vanishes
//! identically. That is the correct answer to the wrong question: the top-grade
//! density of a solid is a volume quantity, and reading it on the boundary is
//! a sampling of the interior, never a trace. [`Surface::traces`] is the
//! predicate that separates the two, and a mark that needs the volume must say
//! so rather than trace to zero and draw nothing.

use regge::subcomplex::SubcomplexExt;
use std::borrow::Cow;

use derham::Cochain;
use multialgebra::ExteriorGrade;
use regge::coord::mesh::MeshCoords;
use simplicial::topology::{complex::Complex, handle::KSimplexIdx, subcomplex::Subcomplex};

/// The 2-manifold (or lower) a scene's marks are drawn on, together with the
/// map back to the parent's vertex numbering.
///
/// Holds only what the reduction adds: the parent it reduces is passed back
/// in at each access, so the identity case stores nothing and copies nothing.
#[derive(Debug, Clone)]
pub struct Surface {
  /// `None` exactly when the mesh is already its own render surface: either
  /// $n <= 2$, or a closed solid, which has no boundary to draw at all.
  boundary: Option<Subcomplex>,
  /// The boundary's own vertex coordinates, restricted from the parent's.
  coords: Option<MeshCoords>,
}

impl Surface {
  /// The render surface of a mesh: the mesh itself for $n <= 2$, its boundary
  /// for a solid.
  ///
  /// A closed solid reduces to the identity as well, and that is deliberate
  /// rather than a fallback: it has no boundary, so there is no surface, and
  /// the honest response is to leave the parent in place and let the marks
  /// find nothing of dimension $<= 2$ to draw. Panicking on a manifold with no
  /// boundary would make closedness an error, which it is not.
  pub fn of(topology: &Complex, coords: &MeshCoords) -> Self {
    let boundary = (topology.dim() > 2)
      .then(|| topology.boundary_complex())
      .flatten();
    let surface_coords = boundary.as_ref().map(|b| b.trace_coords(coords));
    Self {
      boundary,
      coords: surface_coords,
    }
  }

  /// The surface's own complex. A proper manifold in its own right, whichever
  /// branch the reduction took.
  pub fn complex<'a>(&'a self, parent: &'a Complex) -> &'a Complex {
    self.boundary.as_ref().map_or(parent, Subcomplex::complex)
  }

  /// The surface's own vertex coordinates.
  pub fn coords<'a>(&'a self, parent: &'a MeshCoords) -> &'a MeshCoords {
    self.coords.as_ref().unwrap_or(parent)
  }

  /// The surface's intrinsic dimension, the $n$ every grade reduction must
  /// be taken against, since a mark is chosen for the manifold it is drawn on.
  ///
  /// The distinction is not pedantic: a $2$-form on a solid has reduced grade
  /// $min(2, 1) = 1$ in the volume (a line field) but $min(2, 0) = 0$ on the
  /// boundary (a density), because $2$ is the boundary's top grade. Reading
  /// the parent's $n$ here would draw arrows for a flux that has no direction
  /// on the surface it is shown on.
  pub fn dim(&self, parent: &Complex) -> simplicial::Dim {
    self.complex(parent).dim()
  }

  /// Whether a grade-`k` field has a nonzero trace on the surface.
  ///
  /// False above the surface's own top grade, where $C^k (partial M) = 0$. A
  /// caller that gets `false` is holding a volume quantity and must reach for
  /// a volume mark, not trace it to zero.
  pub fn traces(&self, parent: &Complex, grade: ExteriorGrade) -> bool {
    grade <= self.dim(parent)
  }

  /// The trace $i^* c$ of a cochain onto the surface: a genuine grade-$k$
  /// cochain on $partial M$, borrowed unchanged where the reduction is the
  /// identity.
  ///
  /// [`Subcomplex::trace`] is the restriction itself; what this adds is the
  /// identity case, where the surface is the parent and nothing is copied.
  ///
  /// Returns `None` when the grade does not trace (see [`Self::traces`]).
  pub fn trace<'a>(&self, parent: &Complex, cochain: &'a Cochain) -> Option<Cow<'a, Cochain>> {
    if !self.traces(parent, cochain.grade()) {
      return None;
    }
    match self.boundary.as_ref() {
      None => Some(Cow::Borrowed(cochain)),
      Some(boundary) => Some(Cow::Owned(boundary.trace(cochain))),
    }
  }

  /// The surface's vertices in the parent's numbering, or `None` where the
  /// surface is the parent and the map is the identity.
  ///
  /// The one place the reduction leaks, and it leaks for a concrete reason:
  /// the baked vertex table is the parent's, so a datum computed on the
  /// surface has to be scattered back into it.
  pub fn vertex_to_parent(&self) -> Option<&[KSimplexIdx]> {
    self
      .boundary
      .as_ref()
      .map(|b| b.parent_kidxs(simplicial::Dim::ZERO))
  }
}
