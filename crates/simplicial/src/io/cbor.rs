//! CBOR (de)serialization to and from a file: the one on-disk encoding shared
//! by every serializable type in the workspace.
//!
//! Self-describing (field-tagged, not positional like `bincode`) and
//! schema-evolvable, which matters while the types it serializes are still
//! being reshaped. Generic and type-agnostic on its own -- it knows nothing of
//! meshes or cochains -- but persistence for both the mesh types here and the
//! cochains in `derham` (which already depends on `simplicial`) is exactly two
//! call sites, not a third crate's worth of concept.

use serde::{de::DeserializeOwned, Serialize};
use std::{fs::File, io, path::Path};

pub fn save_cbor<T: Serialize>(value: &T, path: impl AsRef<Path>) -> io::Result<()> {
  let file = File::create(path)?;
  ciborium::into_writer(value, file).map_err(io::Error::other)
}

pub fn load_cbor<T: DeserializeOwned>(path: impl AsRef<Path>) -> io::Result<T> {
  let file = File::open(path)?;
  ciborium::from_reader(file).map_err(io::Error::other)
}
