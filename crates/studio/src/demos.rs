//! Scene data shared across studies: the triforce teaching mesh and its worked
//! example cochains, plus the default opening field of a freshly shown scene.

use common::linalg::nalgebra::Matrix;
use manifold::{
  geometry::coord::mesh::MeshCoords,
  topology::{complex::Complex, simplex::Simplex, skeleton::Skeleton},
};

use crate::gallery::{CochainSpec, NamedCochain};
use crate::scene::Scene;
use crate::ui::Selection;

/// The field a freshly shown scene opens on: its first mode. A single mesh
/// grade carries only one render mark, so exactly one of the two lists is
/// nonempty; a scene with neither (never produced here) falls back harmlessly
/// to the first scalar slot.
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
/// invented anew, so the Whitney-basis study on it reproduces those figures.
/// Every interior edge is shared by exactly two cells and the one interior
/// vertex by three, enough to show a global shape function's multi-cell support
/// without the mesh itself being interesting.
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

/// The constant / pure-curl / pure-divergence worked grade-1 fields on the
/// triforce mesh, as data: each a [`CochainSpec::ByEdges`] whose coefficients
/// reproduce `plot/in/triforce`'s `constant`/`rot`/`div` cochains. Addressed by
/// vertex pair rather than by the exporter's file order, since a mesh's own
/// edge indexing need not agree with it -- resolving against the mesh is
/// [`CochainSpec`]'s own, at build time.
pub fn triforce_examples() -> Vec<NamedCochain> {
  // (v0, v1, constant, curl, div), v0 < v1 matching the canonical (positively
  // oriented) edge orientation both `plot/in/triforce` and the triforce
  // topology agree on.
  #[rustfmt::skip]
  let edges: [(usize, usize, f64, f64, f64); 9] = [
    (0, 1,  1.0,  1.0, 0.0),
    (0, 2,  0.5, -1.0, 0.0),
    (1, 2, -0.5,  1.0, 0.0),
    (0, 3, -0.5, -1.0, 0.5),
    (2, 3, -1.0,  1.0, 0.5),
    (1, 4,  0.5,  1.0, 0.5),
    (2, 4,  1.0, -1.0, 0.5),
    (0, 5,  0.5,  1.0, 0.5),
    (1, 5, -0.5, -1.0, 0.5),
  ];
  let by = |pick: fn(&(usize, usize, f64, f64, f64)) -> f64| {
    CochainSpec::ByEdges(edges.iter().map(|e| (e.0, e.1, pick(e))).collect())
  };
  vec![
    NamedCochain {
      name: "constant field".to_string(),
      spec: by(|e| e.2),
    },
    NamedCochain {
      name: "pure curl".to_string(),
      spec: by(|e| e.3),
    },
    NamedCochain {
      name: "pure div".to_string(),
      spec: by(|e| e.4),
    },
  ]
}
