//! Enumerates the shipped meshes in `assets/meshes` into a table the gallery
//! reads, so the picker is a function of what is in that directory rather than
//! a list kept in step with it by hand: dropping a mesh in is the whole of
//! adding it.
//!
//! Generated at build time, not scanned at run time, on purpose. The assets are
//! embedded with `include_bytes!` so they travel inside the binary and need no
//! filesystem when the viewer runs -- which is what lets the same code serve the
//! web build, where there is no filesystem to scan at all.

use std::{env, fs, path::Path};

/// The extensions that name a mesh, and the reader each one is parsed with.
/// A file with any other extension (`SOURCES.md`) is not a mesh and is skipped.
const FORMATS: [(&str, &str); 2] = [("obj", "Format::Obj"), ("msh", "Format::Gmsh")];

fn main() {
  let manifest = env::var("CARGO_MANIFEST_DIR").expect("cargo sets the manifest dir");
  let assets = Path::new(&manifest).join("assets/meshes");

  // A mesh added, removed or renamed changes the table, so the directory
  // itself is the dependency -- not just this script.
  println!("cargo:rerun-if-changed={}", assets.display());

  let mut meshes: Vec<(String, &str, String)> = Vec::new();
  if let Ok(entries) = fs::read_dir(&assets) {
    for entry in entries.flatten() {
      let path = entry.path();
      let Some(extension) = path.extension().and_then(|e| e.to_str()) else {
        continue;
      };
      let Some((_, format)) = FORMATS.iter().find(|(e, _)| *e == extension) else {
        continue;
      };
      let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
        continue;
      };
      meshes.push((stem.to_string(), format, path.display().to_string()));
    }
  }
  // Sorted by name so the picker's order is stable across filesystems, which
  // do not agree on directory order.
  meshes.sort_by(|a, b| a.0.cmp(&b.0));

  let entries: String = meshes
    .iter()
    .map(|(name, format, path)| {
      format!(
        "  Builtin {{ name: {name:?}, format: {format}, bytes: include_bytes!({path:?}) }},\n"
      )
    })
    .collect();
  let table = format!("pub(crate) static BUILTINS: &[Builtin] = &[\n{entries}];\n");

  let out = Path::new(&env::var("OUT_DIR").expect("cargo sets OUT_DIR")).join("builtin_meshes.rs");
  fs::write(&out, table).expect("the mesh table is writable");
}
