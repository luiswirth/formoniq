//! Writing a deforming mesh as a `.mdd` vertex-animation cache, the point
//! cache Blender and Maya read alongside a static mesh.
//!
//! The format is a table of positions: the topology is fixed and lives in the
//! `.obj` beside it (see [`obj`](super::obj)), and every frame is that same
//! vertex table at a different place. So a solution evolving in time leaves as
//! two files, one saying what the mesh is and one saying where it goes, which
//! is the same split as the bake's own, static half and field half, carried
//! into interchange.
//!
//! Big-endian throughout, `f32` positions and times, as the format fixes.

use std::io::{self, Write};
use std::path::Path;

use regge::coord::mesh::MeshCoords;

use crate::bake::to_vec3;

/// Writes the frames as an MDD point cache at `path`, `times` giving each
/// frame's instant in seconds.
///
/// Every frame is the same mesh's vertex table, in the mesh's own vertex order,
/// which is the order the OBJ writer emits and hence what makes the two files
/// one animated object. A frame's coordinates are read in $RR^3$ like every
/// other extrinsic quantity, a lower-dimensional embedding sitting in the zero
/// planes of the axes it does not use.
pub fn write<'a>(
  path: impl AsRef<Path>,
  frames: impl IntoIterator<Item = &'a MeshCoords>,
  times: impl IntoIterator<Item = f64>,
) -> io::Result<()> {
  let frames: Vec<Vec<[f32; 3]>> = frames
    .into_iter()
    .map(|coords| {
      coords
        .coord_iter()
        .map(|coord| {
          let p = to_vec3(&coord.view().into_owned());
          [p.x as f32, p.y as f32, p.z as f32]
        })
        .collect()
    })
    .collect();
  let times: Vec<f32> = times.into_iter().map(|t| t as f32).collect();

  let mut writer = io::BufWriter::new(std::fs::File::create(path)?);
  writer.write_all(&(frames.len() as u32).to_be_bytes())?;
  writer.write_all(&(frames.first().map_or(0, Vec::len) as u32).to_be_bytes())?;
  for time in times {
    writer.write_all(&time.to_be_bytes())?;
  }
  for positions in &frames {
    for position in positions {
      for component in position {
        writer.write_all(&component.to_be_bytes())?;
      }
    }
  }
  writer.flush()
}
