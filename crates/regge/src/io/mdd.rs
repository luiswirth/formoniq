//! Writing a deforming mesh as a `.mdd` vertex-animation cache, the point
//! cache Blender and Maya read alongside a static mesh.
//!
//! The format is a table of positions: the topology is fixed and lives in the
//! `.obj` beside it (see [`obj`](super::obj)), and every frame is that same
//! vertex table at a different place. So a manifold deforming in time leaves
//! as two files, one saying what the mesh is and one saying where it goes.
//!
//! Big-endian throughout, `f32` positions and times, as the format fixes.

use std::io::{self, Write};
use std::path::Path;

use crate::coord::mesh::MeshCoords;

/// Writes the frames as an MDD point cache at `path`, `times` giving each
/// frame's instant in seconds.
///
/// Every frame is the same mesh's vertex table, in the mesh's own vertex order,
/// which is the order an OBJ beside it lists them in and hence what makes the
/// two files one animated object. A frame's coordinates are padded out to
/// $RR^3$, a lower-dimensional embedding sitting in the zero planes of the
/// axes it does not use.
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
          // Padded out to the format's 3-tuple: an embedding of lower
          // dimension sits in the zero planes of the axes it does not use.
          let mut position = [0.0f32; 3];
          for (slot, value) in position.iter_mut().zip(coord.iter()) {
            *slot = *value as f32;
          }
          position
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
