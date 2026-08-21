//! Laws for [`realize::deposit::DepositLayout`]: only the 2-dimensional
//! manifold carries a deposit atlas, its blocks are disjoint and stay in
//! bounds, and a corner's texel coordinate is the block formula evaluated
//! at that corner, the affine map the fill's rasterization extends exactly.

use realize::{
  bake::BakedMesh,
  deposit::{ATLAS_SIZE, DepositLayout, GUTTER, MAX_RESOLUTION, MIN_RESOLUTION},
};
use regge::coord::mesh::{MeshCoords, unit_coord_complex};
use simplicial::topology::complex::Complex;

fn layout_of(dim: usize) -> (Complex, MeshCoords, DepositLayout) {
  let (topology, coords) = unit_coord_complex(dim);
  let coords = coords.embed_euclidean(3);
  let layout = DepositLayout::new(&topology, &coords);
  (topology, coords, layout)
}

/// Only the 2-dimensional manifold carries an atlas; every other dimension
/// gets the empty layout from the same code, not an error.
#[test]
fn atlas_exists_exactly_at_dimension_two() {
  for dim in 0..=3 {
    let (_, _, layout) = layout_of(dim);
    assert_eq!(!layout.blocks.is_empty(), dim == 2, "dim {dim}");
  }
}

/// Every block lies inside the atlas, respects the resolution bounds, and no
/// two blocks overlap, a texel belongs to at most one cell.
#[test]
fn blocks_are_disjoint_and_in_bounds() {
  let triforce = regge::mesher::teaching::triforce();
  let layout = DepositLayout::new(&triforce.0, &triforce.1);
  assert!(!layout.blocks.is_empty());
  for (i, a) in layout.blocks.iter().enumerate() {
    assert!((MIN_RESOLUTION..=MAX_RESOLUTION).contains(&a.resolution));
    assert!(a.origin[0] + a.resolution <= ATLAS_SIZE);
    assert!(a.origin[1] + a.resolution <= ATLAS_SIZE);
    for b in &layout.blocks[i + 1..] {
      let disjoint_x = a.origin[0] + a.resolution + GUTTER <= b.origin[0]
        || b.origin[0] + b.resolution + GUTTER <= a.origin[0];
      let disjoint_y = a.origin[1] + a.resolution + GUTTER <= b.origin[1]
        || b.origin[1] + b.resolution + GUTTER <= a.origin[1];
      assert!(disjoint_x || disjoint_y, "blocks overlap");
    }
  }
}

/// Each corner's texel coordinate is the block formula at that corner's
/// barycentric indicator: the affine map the fill's interpolation extends.
#[test]
fn corner_uvs_are_the_block_formula_at_the_corners() {
  let (topology, coords) = regge::mesher::teaching::triforce();
  let layout = DepositLayout::new(&topology, &coords);
  let baked = BakedMesh::new(&topology, &coords);
  let triangles = baked.fill_triangles();
  let uvs = layout.corner_uvs(&baked.cell_corners);
  assert_eq!(uvs.len(), 3 * triangles.len());

  let cells = topology.skeleton_raw(2);
  let atlas = ATLAS_SIZE as f32;
  for ((i, triangle), block) in triangles.iter().enumerate().zip(&layout.blocks) {
    for (corner, &vertex) in triangle.iter().enumerate() {
      let local = cells
        .iter()
        .nth(i)
        .unwrap()
        .vertices
        .iter()
        .position(|&v| v == vertex as usize)
        .unwrap();
      let uv = uvs[3 * i + corner];
      let (o, r) = (block.origin, block.resolution as f32);
      let expected = match local {
        0 => [o[0] as f32 + r, o[1] as f32],
        1 => [o[0] as f32, o[1] as f32 + r],
        _ => [o[0] as f32, o[1] as f32],
      };
      assert_eq!(uv, [expected[0] / atlas, expected[1] / atlas]);
    }
  }
}
