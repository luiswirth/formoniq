//! The render surface: the 2-manifold a field is actually seen on.
//!
//! The bake reduces an $n$-manifold to the render primitive $min(n, 2)$, and
//! for a solid ($n >= 3$) that primitive is the boundary $diff M$ -- itself a
//! genuine closed $(n-1)$-manifold, carrying its own `Complex`, its own
//! coherent orientation and its own metric, not a bag of faces. This type is
//! that reduction named once, so every mark reads the object it is drawn on
//! rather than the object it was solved on.
//!
//! Below $n = 3$ the reduction is the *identity*: the mesh is already its own
//! render surface. That is why there is no dimension dispatch here beyond the
//! single construction -- a caller asks the surface for its complex and gets
//! either the parent or the boundary, and cannot tell which.
//!
//! **A field reaches the surface by its trace.** $i^*: C^k (M) -> C^k (diff M)$
//! ([`BoundaryComplex::trace_operator`]) is a cochain map, so $i^* dif = dif i^*$
//! and the traced coefficients are a genuine Whitney form on $diff M$ -- not a
//! resample, not a nodal recovery. This is what makes drawing it honest.
//!
//! **The trace is total in grade, but it is zero at the top.** $diff M$ has no
//! $n$-simplices, so $C^n (diff M) = 0$ and an $n$-form's trace vanishes
//! identically. That is the correct answer to the wrong question: the top-grade
//! density of a solid is a *volume* quantity, and reading it on the boundary is
//! a sampling of the interior, never a trace. [`Surface::traces`] is the
//! predicate that separates the two, and a mark that needs the volume must say
//! so rather than trace to zero and draw nothing.

use std::borrow::Cow;

use derham::cochain::Cochain;
use exterior::ExteriorGrade;
use simplicial::{
  geometry::coord::mesh::MeshCoords,
  linalg::Vector,
  topology::{boundary::BoundaryComplex, complex::Complex, handle::KSimplexIdx},
};

/// The 2-manifold (or lower) a scene's marks are drawn on, together with the
/// map back to the parent's vertex numbering.
///
/// Holds only what the reduction *adds*: the parent it reduces is passed back
/// in at each access, so the identity case stores nothing and copies nothing.
#[derive(Debug, Clone)]
pub(crate) struct Surface {
  /// `None` exactly when the mesh is already its own render surface: either
  /// $n <= 2$, or a closed solid, which has no boundary to draw at all.
  boundary: Option<BoundaryComplex>,
  /// The boundary's own vertex coordinates, restricted from the parent's.
  coords: Option<MeshCoords>,
}

impl Surface {
  /// The render surface of a mesh: the mesh itself for $n <= 2$, its boundary
  /// for a solid.
  ///
  /// A *closed* solid reduces to the identity as well, and that is deliberate
  /// rather than a fallback: it has no boundary, so there is no surface, and
  /// the honest response is to leave the parent in place and let the marks
  /// find nothing of dimension $<= 2$ to draw. Panicking on a manifold with no
  /// boundary would make closedness an error, which it is not.
  pub(crate) fn of(topology: &Complex, coords: &MeshCoords) -> Self {
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
  pub(crate) fn complex<'a>(&'a self, parent: &'a Complex) -> &'a Complex {
    self
      .boundary
      .as_ref()
      .map_or(parent, BoundaryComplex::complex)
  }

  /// The surface's own vertex coordinates.
  pub(crate) fn coords<'a>(&'a self, parent: &'a MeshCoords) -> &'a MeshCoords {
    self.coords.as_ref().unwrap_or(parent)
  }

  /// The surface's intrinsic dimension -- the $n$ every *grade* reduction must
  /// be taken against, since a mark is chosen for the manifold it is drawn on.
  ///
  /// The distinction is not pedantic: a $2$-form on a solid has reduced grade
  /// $min(2, 1) = 1$ in the volume (a line field) but $min(2, 0) = 0$ on the
  /// boundary (a density), because $2$ is the boundary's *top* grade. Reading
  /// the parent's $n$ here would draw arrows for a flux that has no direction
  /// on the surface it is shown on.
  pub(crate) fn dim(&self, parent: &Complex) -> simplicial::Dim {
    self.complex(parent).dim()
  }

  /// Whether a grade-`k` field has a nonzero trace on the surface.
  ///
  /// False exactly at the parent's top grade, where $C^k (diff M) = 0$. A
  /// caller that gets `false` is holding a volume quantity and must reach for
  /// a volume mark, not trace it to zero.
  pub(crate) fn traces(&self, parent: &Complex, grade: ExteriorGrade) -> bool {
    grade <= self.dim(parent)
  }

  /// The trace $i^* c$ of a cochain onto the surface: a genuine grade-$k$
  /// cochain on $diff M$, borrowed unchanged where the reduction is the
  /// identity.
  ///
  /// This *is* [`BoundaryComplex::trace_operator`] -- gathering the parent
  /// coefficients at `parent_kidxs` is that matrix's definition, applied
  /// without materializing it, since a permutation-and-select needs no sparse
  /// product.
  ///
  /// Returns `None` when the grade does not trace (see [`Self::traces`]).
  pub(crate) fn trace<'a>(
    &self,
    parent: &Complex,
    cochain: &'a Cochain,
  ) -> Option<Cow<'a, Cochain>> {
    let grade = cochain.grade();
    if !self.traces(parent, grade) {
      return None;
    }
    let Some(boundary) = self.boundary.as_ref() else {
      return Some(Cow::Borrowed(cochain));
    };
    let coeffs = Vector::from_iterator(
      boundary.parent_kidxs(grade).len(),
      boundary
        .parent_kidxs(grade)
        .iter()
        .map(|&parent_kidx| cochain.coeffs()[parent_kidx]),
    );
    Some(Cow::Owned(Cochain::new(grade, coeffs)))
  }

  /// The surface's vertices in the parent's numbering, or `None` where the
  /// surface *is* the parent and the map is the identity.
  ///
  /// The one place the reduction leaks, and it leaks for a concrete reason:
  /// the baked vertex table is the parent's, so a datum computed on the
  /// surface has to be scattered back into it.
  pub(crate) fn vertex_to_parent(&self) -> Option<&[KSimplexIdx]> {
    self.boundary.as_ref().map(|b| b.parent_kidxs(0))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use simplicial::gen::{cartesian::CartesianGrid, sphere::mesh_sphere_surface};

  /// The reduction is the identity at and below the render primitive's own
  /// dimension: a surface is its own render surface, and nothing is copied or
  /// traced. This is the base case invariant "total on the degenerate boundary"
  /// asks for -- the same code, returning the trivial answer.
  #[test]
  fn a_surface_is_its_own_render_surface() {
    let (topology, coords) = mesh_sphere_surface(2);
    let surface = Surface::of(&topology, &coords);

    assert!(std::ptr::eq(surface.complex(&topology), &topology));
    assert!(std::ptr::eq(surface.coords(&coords), &coords));
    assert_eq!(surface.dim(&topology), topology.dim());
    assert!(surface.vertex_to_parent().is_none());

    let cochain = Cochain::constant(1.0, topology.skeleton_raw(1));
    let traced = surface.trace(&topology, &cochain).expect("a 1-form traces");
    assert!(
      matches!(traced, Cow::Borrowed(_)),
      "the identity must borrow"
    );
  }

  /// A solid reduces to its boundary, and the boundary is a *proper manifold
  /// one dimension down*: it has its own complex of the right dimension, its
  /// own vertices, and strictly fewer of them than the solid.
  #[test]
  fn a_solid_reduces_to_its_boundary_manifold() {
    let (topology, coords) = CartesianGrid::new_unit(3, 2).triangulate();
    let surface = Surface::of(&topology, &coords);

    assert_eq!(topology.dim(), 3);
    assert_eq!(surface.dim(&topology), 2, "the boundary is a 2-manifold");

    let to_parent = surface.vertex_to_parent().expect("a cube has a boundary");
    assert_eq!(surface.coords(&coords).nvertices(), to_parent.len());
    assert!(
      to_parent.len() < topology.skeleton_raw(0).len(),
      "a cube has interior vertices the boundary does not"
    );
  }

  /// The trace is the *restriction* of coefficients: each boundary simplex
  /// carries exactly its parent's value. Stated on a field that distinguishes
  /// every simplex, so an index permutation cannot pass.
  #[test]
  fn the_trace_restricts_each_coefficient_to_its_parent() {
    let (topology, coords) = CartesianGrid::new_unit(3, 2).triangulate();
    let surface = Surface::of(&topology, &coords);

    for grade in 0..=surface.dim(&topology) {
      let cochain = Cochain::from_function(|s| s.kidx() as f64, grade, &topology);
      let traced = surface.trace(&topology, &cochain).expect("grade traces");

      let to_parent = surface
        .boundary
        .as_ref()
        .expect("a cube has a boundary")
        .parent_kidxs(grade);
      assert_eq!(traced.len(), to_parent.len());
      for (boundary_kidx, &parent_kidx) in to_parent.iter().enumerate() {
        assert_eq!(traced.coeffs()[boundary_kidx], parent_kidx as f64);
      }
    }
  }

  /// The top grade does not trace: $C^n (diff M) = 0$, so an $n$-form has no
  /// surface representative at all. The predicate must say so rather than hand
  /// back a zero cochain that a mark would draw as a vanishing field.
  #[test]
  fn the_top_grade_has_no_trace() {
    let (topology, coords) = CartesianGrid::new_unit(3, 2).triangulate();
    let surface = Surface::of(&topology, &coords);

    for grade in 0..topology.dim() {
      assert!(surface.traces(&topology, grade), "grade {grade} traces");
    }
    assert!(!surface.traces(&topology, topology.dim()));

    let volume = Cochain::constant(1.0, topology.skeleton_raw(3));
    assert!(surface.trace(&topology, &volume).is_none());
  }
}
