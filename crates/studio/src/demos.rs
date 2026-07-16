//! Scene constructors: builds the [`crate::scene::Scene`] for a given
//! `View` -- the mesh's eigenmodes, the Whitney basis galleries, and the
//! worked examples on the triforce teaching mesh.

use crate::gallery::{Mesh, View};
use crate::scene::Scene;
use crate::ui::panel::Selection;

// Chosen so both grades close on a complete degeneracy shell: grade 0 fills
// $l = 0..=3$ ($sum (2l+1) = 16$) and grade 1 fills $l = 1, 2$
// ($6 + 10 = 16$), so the orbital pyramid the UI lays these out in has no
// half-built final row.
const SPHERE_MODES: usize = 16;

/// Builds the scene for `view`. For a mesh grade this runs that grade's dense
/// eigensolve against the shared `mesh` -- the expensive path the caller runs
/// on a background thread and memoizes; the Whitney basis is cheap and ignores
/// the mesh.
pub(crate) fn build_view(view: View, mesh: &Mesh) -> Scene {
  match view {
    View::MeshGrade(grade) => {
      let (topology, coords) = mesh;
      Scene::mesh_grade(topology, coords, grade, SPHERE_MODES)
    }
    View::WhitneyBasis => Scene::whitney_basis(2),
    View::WhitneyBasisMesh => {
      let (topology, coords) = crate::mesh3d::triforce();
      Scene::whitney_basis_mesh(topology, coords)
    }
    View::WhitneyExamplesMesh => {
      let (topology, coords) = crate::mesh3d::triforce();
      Scene::whitney_examples(topology, coords)
    }
  }
}

/// The field a freshly shown scene opens on: its first mode. A sphere grade
/// carries only one render mark, so exactly one of the two lists is nonempty; a
/// scene with neither (never produced here) falls back harmlessly to the first
/// scalar slot.
pub(crate) fn default_selection(scene: &Scene) -> Selection {
  if !scene.fields.is_empty() {
    Selection::Scalar(0)
  } else if !scene.line_fields.is_empty() {
    Selection::Line(0)
  } else {
    Selection::Scalar(0)
  }
}
