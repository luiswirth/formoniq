//! Scene data shared across studies: the worked example cochains of the
//! triforce teaching mesh, plus the default opening field of a freshly shown
//! scene.
//!
//! The mesh itself is [`realize::demos::triforce`], re-exported here: it is a
//! mesh and nothing more, so it needs no scene, and the reductions below the
//! viewer test against the same one the studies are shown on.

pub use realize::demos::triforce;

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

/// The constant / pure-curl / pure-divergence worked grade-1 fields on the
/// triforce mesh, as data: each a [`CochainSpec::ByEdges`] whose coefficients
/// reproduce the thesis figures' `constant`/`rot`/`div` cochains. Addressed by
/// vertex pair rather than by the exporter's file order, since a mesh's own
/// edge indexing need not agree with it -- resolving against the mesh is
/// [`CochainSpec`]'s own, at build time.
pub fn triforce_examples() -> Vec<NamedCochain> {
  // (v0, v1, constant, curl, div), v0 < v1 matching the canonical (positively
  // oriented) edge orientation the triforce topology agrees on.
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
