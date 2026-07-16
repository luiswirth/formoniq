//! Scene constructors: builds the [`crate::scene::Scene`] for a given
//! `View` -- the mesh's eigenmodes, the Whitney basis galleries, and the
//! worked examples on the triforce teaching mesh.

use common::linalg::nalgebra::Matrix;
use manifold::{
  geometry::coord::mesh::MeshCoords,
  topology::{complex::Complex, simplex::Simplex, skeleton::Skeleton},
};

use crate::gallery::{Mesh, View};
use crate::scene::Scene;
use crate::ui::Selection;

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
      let (topology, coords) = triforce();
      Scene::whitney_basis_mesh(topology, coords)
    }
    View::WhitneyExamplesMesh => {
      let (topology, coords) = triforce();
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

/// The "triforce" teaching mesh: a central equilateral triangle with one
/// congruent triangle mirrored outward across each of its three edges, four
/// cells in all. The same layout as `plot/in/triforce`, the thesis figures'
/// worked example of a global shape function -- reused here rather than
/// invented anew, so [`Scene::whitney_basis_mesh`] on it reproduces those
/// figures. Every interior edge is shared by exactly two cells and the one
/// interior vertex by three, enough to show a global shape function's
/// multi-cell support without the mesh itself being interesting.
///
/// Flat, and embedded in $RR^3$'s $z = 0$ plane: the viewer's one ambient space
/// is $RR^3$, and a planar mesh embeds as itself there.
pub fn triforce() -> (Complex, MeshCoords) {
  let sqrt3_2 = 3f64.sqrt() / 2.0;
  #[rustfmt::skip]
  let positions: [[f64; 2]; 6] = [
    [ 0.0, 0.0],
    [ 1.0, 0.0],
    [ 0.5, sqrt3_2],
    [-0.5, sqrt3_2],
    [ 1.5, sqrt3_2],
    [ 0.5, -sqrt3_2],
  ];
  let cells: [[usize; 3]; 4] = [[0, 1, 2], [0, 2, 3], [1, 4, 2], [0, 1, 5]];

  let columns: Vec<_> = positions.iter().map(|p| na::dvector![p[0], p[1]]).collect();
  let coords = MeshCoords::from(Matrix::from_columns(&columns)).embed_euclidean(3);
  let simplices = cells
    .into_iter()
    .map(|c| Simplex::from_word(c.to_vec()).1)
    .collect();
  (Complex::from_cells(Skeleton::new(simplices)), coords)
}
