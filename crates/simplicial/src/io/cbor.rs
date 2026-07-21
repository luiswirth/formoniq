//! CBOR (de)serialization to and from a file: the one on-disk encoding shared
//! by every serializable type in the workspace.
//!
//! Self-describing (field-tagged, not positional like `bincode`) and
//! schema-evolvable, which matters while the types it serializes are still
//! being reshaped. Generic and type-agnostic on its own -- it knows nothing of
//! the types it stores -- so it lives at the lowest crate that needs on-disk
//! persistence and is reused, unchanged, by every serializable type above it:
//! one small helper, not a crate's worth of concept per consumer.

use serde::{Serialize, de::DeserializeOwned};
use std::{fs::File, io, path::Path};

pub fn save_cbor<T: Serialize>(value: &T, path: impl AsRef<Path>) -> io::Result<()> {
  let file = File::create(path)?;
  ciborium::into_writer(value, file).map_err(io::Error::other)
}

pub fn load_cbor<T: DeserializeOwned>(path: impl AsRef<Path>) -> io::Result<T> {
  let file = File::open(path)?;
  ciborium::from_reader(file).map_err(io::Error::other)
}
