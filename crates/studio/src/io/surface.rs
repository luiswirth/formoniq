//! A triangle surface embedded in $RR^3$: the thin intermediate the `.obj` and
//! `.mdd` paths read and write through.
//!
//! Not a seam -- `bake.rs` is the one place the viewer commits to a rasterizer's
//! primitives. This is an interchange format's own shape, which happens to want
//! the same two things a graphics API does and the core keeps out: an explicit
//! winding, and an explicit embedding, fixed at 3.

use simplicial::linalg::VectorView;
use simplicial::{
  geometry::coord::mesh::MeshCoords,
  topology::{complex::Complex, simplex::Simplex, skeleton::Skeleton},
};

use crate::bake::orient_triangles;

pub type TriangleTopology = Vec<[usize; 3]>;

#[derive(Debug, Clone)]
pub struct TriangleSurface3D {
  triangles: TriangleTopology,
  coords: MeshCoords,
}
impl TriangleSurface3D {
  pub fn new(triangles: TriangleTopology, coords: impl Into<MeshCoords>) -> Self {
    let coords = coords.into();
    Self { triangles, coords }
  }
  pub fn triangles(&self) -> &[[usize; 3]] {
    &self.triangles
  }
  pub fn vertex_coords(&self) -> &MeshCoords {
    &self.coords
  }
  pub fn vertex_coords_mut(&mut self) -> &mut MeshCoords {
    &mut self.coords
  }
  pub fn into_parts(self) -> (TriangleTopology, MeshCoords) {
    (self.triangles, self.coords)
  }
}

impl TriangleSurface3D {
  pub fn from_coord_skeleton(skeleton: Skeleton, coords: MeshCoords) -> Self {
    assert!(skeleton.dim() == 2, "Topology is not 2D.");
    assert!(coords.dim() <= 3, "Skeleton is not embeddable in 3D.");
    let coords = coords.embed_euclidean(3);

    let triangles: Vec<[u32; 3]> = skeleton
      .into_index_set()
      .into_iter()
      .map(|simp| {
        let vertices: [usize; 3] = simp.try_into().unwrap();
        vertices.map(|v| v as u32)
      })
      .collect();
    let triangles = orient_triangles(&triangles);
    let triangles = triangles
      .into_iter()
      .map(|t| t.map(|v| v as usize))
      .collect();

    Self::new(triangles, coords)
  }

  pub fn into_coord_skeleton(self) -> (Skeleton, MeshCoords) {
    let simps = self
      .triangles
      .into_iter()
      .map(|tria| Simplex::from_word(tria.to_vec()).1)
      .collect();
    let skeleton = Skeleton::new(simps);
    let coords = self.coords;
    (skeleton, coords)
  }

  pub fn into_coord_complex(self) -> (Complex, MeshCoords) {
    let (skeleton, coords) = self.into_coord_skeleton();
    let complex = Complex::from_cells(skeleton);
    (complex, coords)
  }

  /// Displaces each vertex along its own normal.
  ///
  /// The normals are [`crate::bake::vertex_normals`]' unnormalized ones: the
  /// average of a vertex's incident triangles' unit normals falls short of unit
  /// length at a crease, and this intentionally does not renormalize.
  pub fn displace_normal<'a>(&mut self, displacements: impl Into<VectorView<'a>>) {
    let displacements = displacements.into();
    let positions: Vec<na::Vector3<f64>> = self
      .coords
      .coord_iter()
      .map(|c| na::Vector3::new(c[0], c[1], c[2]))
      .collect();
    let triangles: Vec<[u32; 3]> = self.triangles.iter().map(|t| t.map(|v| v as u32)).collect();
    let vertex_normals = crate::bake::vertex_normals(&triangles, &positions);
    for ((mut v, n), &d) in self
      .coords
      .coord_iter_mut()
      .zip(vertex_normals)
      .zip(displacements)
    {
      v += d * n;
    }
  }
}
