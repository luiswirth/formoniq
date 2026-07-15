//! CBOR (de)serialization to and from a file: the one on-disk encoding shared
//! by every serializable type in the workspace.
//!
//! Self-describing (field-tagged, not positional like `bincode`) and
//! schema-evolvable, which matters while the types it serializes are still
//! being reshaped. Generic and type-agnostic on purpose -- it knows nothing of
//! meshes, geometry or cochains, so it lives here rather than in any crate
//! that does; each type's own (de)serialization support lives next to its
//! definition and calls through to this.

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
