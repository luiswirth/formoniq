//! Persistence for the combinatorial types.
//!
//! Only the format-agnostic CBOR helpers live here. Reading a mesh means
//! reading coordinates with it, so the mesh formats are `regge`'s.
#[cfg(feature = "serde")]
pub mod cbor;
