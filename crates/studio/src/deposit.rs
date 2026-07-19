//! The deposit atlas: where a particle population's trails live.
//!
//! The trail a moving speck leaves is *state*, and state belongs to the
//! manifold, not to the screen -- a screen-space accumulation would bake the
//! camera into it, and every orbit or export would smear history that was never
//! the field's. So the accumulation texture is laid out in the atlas's own
//! charts: each 2-cell owns a square block, and a texel of that block *is* a
//! point of the cell's barycentric lattice, $lambda = (i, j, R - i - j) \/ R$
//! (the `ref_lattice` of `simplicial::atlas`, stored as texels so that splatting
//! and filtered sampling are hardware). There is no UV unwrapping anywhere:
//! the map $("cell", lambda) -> "texel"$ is the one formula
//! $O_c + R_c (lambda_0, lambda_1)$, and because it is *affine*, the
//! rasterizer's own interpolation of the three corner texel coordinates
//! reproduces it exactly at every fragment of the fill.
//!
//! Texel density is allocated by metric area -- $R_c prop sqrt("area"_c)$ --
//! for the same reason the advection's seeding is: the resolution of the trail
//! is a property of the manifold, not of its triangulation. A sliver and a
//! large cell get the density their area earns, quantized and clamped, then
//! shelf-packed.
//!
//! Only an intrinsic dimension of exactly 2 carries an atlas: a curve's cells
//! have no area to rasterize into, and a solid's rendered surface is its
//! boundary, whose faces are not charts. Both give the empty layout, which
//! every consumer treats as "no deposit" by arithmetic rather than by branch.

use simplicial::{geometry::coord::mesh::MeshCoords, topology::complex::Complex};

/// The atlas texture's side, in texels. One fixed square texture: the budget
/// the per-cell resolutions are allocated out of. The trail is sampled from
/// this atlas by the fill and magnified onto whatever screen area the cell
/// covers, so it is this budget -- not the particle count or the splat -- that
/// sets how sharp a filament reads when a cell fills the view.
///
/// The bound is the *fade*: the whole atlas is blitted once per advection step,
/// and a slow display owes the [`crate::app`] step cap's worth of them every
/// frame, so the per-frame fade bandwidth grows as the square of this side. At
/// 2048 the ping-pong pair is ~16 MB of R16Float and the fade stays cheap on a
/// weak integrated GPU; the bicubic reconstruction in the fill recovers most of
/// the sharpness a larger atlas would buy, at a per-fragment cost the display
/// bounds rather than the atlas. A budget scaled to the adapter, rather than one
/// fixed side, is the way to spend more where the hardware affords it.
pub const ATLAS_SIZE: u32 = 2048;

/// Bounds on one cell's lattice resolution. The floor keeps a sliver from
/// degenerating below the splat kernel's own footprint; the cap keeps one huge
/// cell (a coarse mesh's) from spending the whole budget on itself.
const MIN_RESOLUTION: u32 = 4;
const MAX_RESOLUTION: u32 = 512;

/// Empty texels around each block, so a bilinear read at a cell's edge touches
/// only the cell's own block -- the neighbouring block in the *atlas* belongs
/// to an unrelated cell of the *manifold*.
const GUTTER: u32 = 1;

/// One cell's block: the texel origin of its lattice, and the resolution $R_c$.
/// The cell's triangle occupies the texels $(x, y)$ with
/// $x + y <= R_c$ relative to the origin.
#[derive(Clone, Copy, Debug)]
pub struct Block {
  pub origin: [u32; 2],
  pub resolution: u32,
}

/// The atlas layout: one [`Block`] per cell, in cell (colex) order -- the same
/// order the advection bake indexes cells by, which is what lets a particle's
/// `cell` index the block table directly.
pub struct DepositLayout {
  pub blocks: Vec<Block>,
}

impl DepositLayout {
  /// The layout of `topology` embedded by `coords`; empty for an intrinsic
  /// dimension other than 2 (see the module doc for why that is the honest
  /// answer, not an exclusion).
  pub fn new(topology: &Complex, coords: &MeshCoords) -> Self {
    if topology.dim() != 2 {
      return Self { blocks: Vec::new() };
    }
    let weights: Vec<f64> = topology
      .cells()
      .handle_iter()
      .map(|cell| coords.cell_metric(cell).det_sqrt().max(0.0))
      .collect();
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
      return Self { blocks: Vec::new() };
    }

    // Uniform texel density: each cell's square holds ~R^2/2 useful texels, so
    // aim R_c at the cell's share of the budget and let the packing loop below
    // shrink uniformly if the shelf overflows. The initial fill factor only
    // sets where that loop starts.
    let budget = 0.5 * f64::from(ATLAS_SIZE) * f64::from(ATLAS_SIZE);
    let mut scale = 1.0;
    for _ in 0..32 {
      let resolutions: Vec<u32> = weights
        .iter()
        .map(|w| {
          let texels = 2.0 * w / total * budget * scale * scale;
          (texels.sqrt() as u32).clamp(MIN_RESOLUTION, MAX_RESOLUTION)
        })
        .collect();
      if let Some(blocks) = shelf_pack(&resolutions) {
        return Self { blocks };
      }
      scale *= 0.9;
    }
    // Even MIN_RESOLUTION for every cell does not fit: the mesh has more cells
    // than the atlas has blocks for. No deposit is the honest answer.
    Self { blocks: Vec::new() }
  }

  /// The texels the triangles actually cover: what calibrates the splat
  /// energy's average against the particle count.
  pub fn used_texels(&self) -> u64 {
    self
      .blocks
      .iter()
      .map(|b| u64::from(b.resolution) * u64::from(b.resolution) / 2)
      .sum()
  }

  /// The atlas texel coordinate of each triangle corner, three per triangle in
  /// the triangles' own order, *normalized* to $[0, 1]$ texture coordinates.
  ///
  /// `triangles` must be the bake's cell triangles in cell order (each a
  /// permutation of its cell's vertices -- the winding pass may have swapped
  /// two). A corner's coordinate is the block formula at the corner's own
  /// barycentric indicator: vertex $0 -> O + (R, 0)$, $1 -> O + (0, R)$,
  /// $2 -> O$, where the local index is the vertex's position in the cell's
  /// colex-sorted tuple. Affine interpolation of these three values *is* the
  /// map $O + R (lambda_0, lambda_1)$ at every interior point, which is why no
  /// per-fragment lookup exists.
  ///
  /// Zeros (matching length) for the empty layout, so the vertex stream always
  /// exists and "no deposit" stays a material value rather than a pipeline.
  pub fn corner_uvs(&self, topology: &Complex, triangles: &[[u32; 3]]) -> Vec<[f32; 2]> {
    if self.blocks.is_empty() {
      return vec![[0.0; 2]; 3 * triangles.len()];
    }
    assert_eq!(self.blocks.len(), triangles.len());
    let cells = topology.skeleton_raw(topology.dim());
    let atlas = ATLAS_SIZE as f32;
    triangles
      .iter()
      .zip(cells.iter())
      .zip(&self.blocks)
      .flat_map(|((triangle, cell), block)| {
        triangle.map(|vertex| {
          let local = cell
            .vertices
            .iter()
            .position(|&v| v == vertex as usize)
            .expect("a bake triangle is a permutation of its cell");
          let origin = [block.origin[0] as f32, block.origin[1] as f32];
          let r = block.resolution as f32;
          let corner = match local {
            0 => [origin[0] + r, origin[1]],
            1 => [origin[0], origin[1] + r],
            _ => origin,
          };
          [corner[0] / atlas, corner[1] / atlas]
        })
      })
      .collect()
  }
}

/// Next-fit decreasing-height shelf packing of one square per cell (side
/// $R_c + 2 dot "GUTTER"$) into the atlas. `None` when the shelves overflow
/// the atlas height, which is the caller's signal to shrink and retry.
///
/// One square per cell rather than two triangles sharing one: the pairing
/// halves the waste but puts a foreign cell across the diagonal, exactly where
/// the gutter cannot protect a bilinear read. Half an atlas is cheaper than a
/// seam.
fn shelf_pack(resolutions: &[u32]) -> Option<Vec<Block>> {
  let mut order: Vec<usize> = (0..resolutions.len()).collect();
  order.sort_by_key(|&i| std::cmp::Reverse(resolutions[i]));

  let mut blocks = vec![
    Block {
      origin: [0; 2],
      resolution: 0
    };
    resolutions.len()
  ];
  let (mut x, mut y, mut shelf) = (0u32, 0u32, 0u32);
  for &i in &order {
    let side = resolutions[i] + 2 * GUTTER;
    if x + side > ATLAS_SIZE {
      y += shelf;
      x = 0;
      shelf = 0;
    }
    shelf = shelf.max(side);
    if y + side > ATLAS_SIZE {
      return None;
    }
    blocks[i] = Block {
      origin: [x + GUTTER, y + GUTTER],
      resolution: resolutions[i],
    };
    x += side;
  }
  Some(blocks)
}

#[cfg(test)]
mod tests {
  use super::*;
  use simplicial::geometry::coord::mesh::standard_coord_complex;

  fn layout_of(dim: usize) -> (Complex, MeshCoords, DepositLayout) {
    let (topology, coords) = standard_coord_complex(dim);
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
  /// two blocks overlap -- a texel belongs to at most one cell.
  #[test]
  fn blocks_are_disjoint_and_in_bounds() {
    let triforce = crate::demos::triforce();
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
    let (topology, coords) = crate::demos::triforce();
    let layout = DepositLayout::new(&topology, &coords);
    let baked = crate::bake::BakedMesh::new(&topology, &coords);
    let crate::bake::PrimBatch::Triangles(triangles) = &baked.cells else {
      panic!("a 2-manifold bakes to triangles");
    };
    let uvs = layout.corner_uvs(&topology, triangles);
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
}
