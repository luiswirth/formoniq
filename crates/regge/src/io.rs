//! Mesh formats: a topology and the coordinates that come with it.
//!
//! A file names its vertices by position, so reading or writing one needs an
//! embedding and lives on the extrinsic side of invariant 2. A deforming mesh
//! splits along the same seam: the topology stays in one file and the moving
//! vertex table in another, which is what [`mdd`] writes.
pub mod gmsh;
pub mod mdd;
pub mod obj;
