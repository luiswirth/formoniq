//! The draw list: baked geometry uploaded to the GPU, and the items a frame is
//! made of.
//!
//! A batch owns buffers and nothing else; an item is a batch plus the material
//! it is drawn with. A surface, its wireframe overlay and a line field's arrow
//! glyphs are three items over three batch kinds, and several manifolds in one
//! scene are simply more items -- the renderer never learns how many there will
//! be.
//!
//! Each batch mirrors `bake.rs`'s static/per-field split: positions and index
//! streams are uploaded once per mesh, and only `write_attributes` runs when the
//! field on display changes.

use bytemuck::Pod;
use wgpu::util::DeviceExt;

use super::particles::ParticleBatch;
use super::uniform::{GlyphMaterial, SegmentMaterial, SurfaceMaterial};
use crate::bake::{BakedMesh, BakedVertex, GlyphVertex, PrimBatch, SegmentVertex};

/// A `VERTEX` buffer holding `data`, never empty: a zero-length
/// `create_buffer_init` is rejected, and an empty batch (a field with no
/// glyphs, a curve with no wireframe overlay) still needs valid buffers to
/// bind. The batch's own count stays the true one, so the pad is never drawn.
fn vertex_buffer<T: Pod + Default>(device: &wgpu::Device, label: &str, data: &[T]) -> wgpu::Buffer {
  let pad = [T::default()];
  let contents = if data.is_empty() { &pad[..] } else { data };
  device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    label: Some(label),
    contents: bytemuck::cast_slice(contents),
    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
  })
}

const fn attribute(
  offset: u64,
  shader_location: u32,
  format: wgpu::VertexFormat,
) -> wgpu::VertexAttribute {
  wgpu::VertexAttribute {
    offset,
    shader_location,
    format,
  }
}

/// A `BakedVertex`'s three static attributes, at `loc..loc + 3`.
const fn baked_attributes(loc: u32) -> [wgpu::VertexAttribute; 3] {
  [
    attribute(0, loc, wgpu::VertexFormat::Float32x3),
    attribute(12, loc + 1, wgpu::VertexFormat::Float32x3),
    attribute(24, loc + 2, wgpu::VertexFormat::Float32),
  ]
}

/// A `SegmentVertex`'s four: a `BakedVertex`'s, then the opacity trailing it.
const fn segment_attributes(loc: u32) -> [wgpu::VertexAttribute; 4] {
  let [p, n, m] = baked_attributes(loc);
  [p, n, m, attribute(28, loc + 3, wgpu::VertexFormat::Float32)]
}

/// A `GlyphVertex`'s five: world position, the arrow-frame coordinate, the
/// arrow length, the opacity, then the cell-barycentric clip coordinate.
const fn glyph_attributes() -> [wgpu::VertexAttribute; 5] {
  [
    attribute(0, 0, wgpu::VertexFormat::Float32x3),
    attribute(12, 1, wgpu::VertexFormat::Float32x2),
    attribute(20, 2, wgpu::VertexFormat::Float32),
    attribute(24, 3, wgpu::VertexFormat::Float32),
    attribute(28, 4, wgpu::VertexFormat::Float32x4),
  ]
}

const fn value_attribute(loc: u32) -> [wgpu::VertexAttribute; 1] {
  [attribute(0, loc, wgpu::VertexFormat::Float32)]
}

const SURFACE_STATIC: [wgpu::VertexAttribute; 3] = baked_attributes(0);
const SURFACE_VALUE: [wgpu::VertexAttribute; 1] = value_attribute(3);
const SURFACE_DEPOSIT_UV: [wgpu::VertexAttribute; 1] =
  [attribute(0, 4, wgpu::VertexFormat::Float32x2)];
const SEGMENT_A: [wgpu::VertexAttribute; 4] = segment_attributes(0);
const SEGMENT_B: [wgpu::VertexAttribute; 4] = segment_attributes(4);
const SEGMENT_VALUE_A: [wgpu::VertexAttribute; 1] = value_attribute(8);
const SEGMENT_VALUE_B: [wgpu::VertexAttribute; 1] = value_attribute(9);
const GLYPH: [wgpu::VertexAttribute; 5] = glyph_attributes();

/// A filled triangle surface: per-corner vertex streams, three corners per
/// triangle, unshared.
///
/// Unshared, unlike the earlier indexed table, because the deposit-atlas texel
/// coordinate is per corner *per triangle*: two triangles sharing a mesh
/// vertex map it into two different atlas blocks, so a shared vertex has no
/// one coordinate to carry. The gather is the same move [`SegmentBatch`]
/// already makes for its endpoints, and the per-field values are re-gathered
/// through the retained triangle list exactly as its `write_attributes` does.
pub struct SurfaceBatch {
  corners: wgpu::Buffer,
  attributes: wgpu::Buffer,
  deposit_uvs: wgpu::Buffer,
  /// The triangle list over the *mesh* vertex table, retained so a field
  /// change re-gathers its per-vertex values into corner order.
  triangles: Vec<[u32; 3]>,
  ncorners: u32,
}

impl SurfaceBatch {
  /// The batch of a baked mesh whose cells are triangles; `None` for a bake that
  /// produced no fill (a curve, a point cloud), whose marks are its segments
  /// instead. `deposit_uvs` is the per-corner atlas texel coordinate stream
  /// (normalized), three per triangle -- zeros for a mesh with no atlas. The
  /// attribute stream starts at zero; a field fills it through
  /// [`Self::write_attributes`].
  pub fn new(device: &wgpu::Device, baked: &BakedMesh, deposit_uvs: &[[f32; 2]]) -> Option<Self> {
    let PrimBatch::Triangles(triangles) = &baked.cells else {
      return None;
    };
    assert_eq!(deposit_uvs.len(), 3 * triangles.len());
    let corners: Vec<BakedVertex> = triangles
      .iter()
      .flatten()
      .map(|&i| baked.positions[i as usize])
      .collect();
    Some(Self {
      corners: vertex_buffer(device, "Surface Corners", &corners),
      attributes: vertex_buffer(device, "Surface Attributes", &vec![0.0f32; corners.len()]),
      deposit_uvs: vertex_buffer(device, "Surface Deposit UVs", deposit_uvs),
      ncorners: corners.len() as u32,
      triangles: triangles.clone(),
    })
  }

  /// Rebinds the surface to a different field of the same mesh: `values` is
  /// per *mesh* vertex, gathered here into corner order. One buffer write, no
  /// rebake.
  pub fn write_attributes(&self, queue: &wgpu::Queue, values: &[f32]) {
    if values.is_empty() {
      return;
    }
    let gathered: Vec<f32> = self
      .triangles
      .iter()
      .flatten()
      .map(|&i| values[i as usize])
      .collect();
    if !gathered.is_empty() {
      queue.write_buffer(&self.attributes, 0, bytemuck::cast_slice(&gathered));
    }
  }

  /// The static stream at locations 0..=2, the per-field value at 3, the
  /// deposit texel coordinate at 4.
  pub fn layouts<'a>() -> [wgpu::VertexBufferLayout<'a>; 3] {
    [
      wgpu::VertexBufferLayout {
        array_stride: size_of::<BakedVertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &SURFACE_STATIC,
      },
      wgpu::VertexBufferLayout {
        array_stride: size_of::<f32>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &SURFACE_VALUE,
      },
      wgpu::VertexBufferLayout {
        array_stride: size_of::<[f32; 2]>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &SURFACE_DEPOSIT_UV,
      },
    ]
  }

  pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
    pass.set_vertex_buffer(0, self.corners.slice(..));
    pass.set_vertex_buffer(1, self.attributes.slice(..));
    pass.set_vertex_buffer(2, self.deposit_uvs.slice(..));
    pass.draw(0..self.ncorners, 0..1);
  }
}

/// A segment mark: the wireframe overlay, a line field's ribbons, or a
/// 1-manifold's own cells. One batch kind for all three -- the same instanced
/// billboard quad over a different index stream.
///
/// A segment's two endpoints are expanded into two parallel per-instance streams
/// (`a[i]` and `b[i]` are segment `i`'s two ends), because a `step_mode: Vertex`
/// buffer only ever exposes the current vertex to the shader, never a neighbor,
/// while both ends have to reach one invocation at once to compute the segment's
/// screen-facing perpendicular. The index stream is retained so a field change
/// re-gathers the endpoint values without the caller re-expanding them.
pub struct SegmentBatch {
  endpoints: [wgpu::Buffer; 2],
  values: [wgpu::Buffer; 2],
  segments: Vec<[u32; 2]>,
  nsegments: u32,
}

impl SegmentBatch {
  /// `vertices` and `values` are one table, indexed by `segments`. An empty
  /// `segments` is a valid batch that draws nothing.
  pub fn new(
    device: &wgpu::Device,
    vertices: &[SegmentVertex],
    values: &[f32],
    segments: &[[u32; 2]],
  ) -> Self {
    assert_eq!(vertices.len(), values.len());
    let endpoints_of = |end: usize| -> Vec<SegmentVertex> {
      segments.iter().map(|s| vertices[s[end] as usize]).collect()
    };
    Self {
      endpoints: [
        vertex_buffer(device, "Segment Endpoint A", &endpoints_of(0)),
        vertex_buffer(device, "Segment Endpoint B", &endpoints_of(1)),
      ],
      values: [
        vertex_buffer(device, "Segment Value A", &gather(values, segments, 0)),
        vertex_buffer(device, "Segment Value B", &gather(values, segments, 1)),
      ],
      nsegments: segments.len() as u32,
      segments: segments.to_vec(),
    }
  }

  /// Rebinds the mark to a different field over the same segments: the endpoint
  /// positions, normals and opacities are the mesh's, and stay.
  pub fn write_attributes(&self, queue: &wgpu::Queue, values: &[f32]) {
    for (end, buffer) in self.values.iter().enumerate() {
      let gathered = gather(values, &self.segments, end);
      if !gathered.is_empty() {
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(&gathered));
      }
    }
  }

  /// Endpoint A's static stream at locations 0..=3 and B's at 4..=7, their
  /// per-field values at 8 and 9.
  pub fn layouts<'a>() -> [wgpu::VertexBufferLayout<'a>; 4] {
    const fn endpoint(
      attributes: &'static [wgpu::VertexAttribute],
    ) -> wgpu::VertexBufferLayout<'static> {
      wgpu::VertexBufferLayout {
        array_stride: size_of::<SegmentVertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes,
      }
    }
    const fn value(
      attributes: &'static [wgpu::VertexAttribute],
    ) -> wgpu::VertexBufferLayout<'static> {
      wgpu::VertexBufferLayout {
        array_stride: size_of::<f32>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes,
      }
    }
    [
      endpoint(&SEGMENT_A),
      endpoint(&SEGMENT_B),
      value(&SEGMENT_VALUE_A),
      value(&SEGMENT_VALUE_B),
    ]
  }

  /// Draws the quads, if there are any: an empty batch draws nothing rather than
  /// being a case the caller has to exclude.
  pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
    if self.nsegments == 0 {
      return;
    }
    pass.set_vertex_buffer(0, self.endpoints[0].slice(..));
    pass.set_vertex_buffer(1, self.endpoints[1].slice(..));
    pass.set_vertex_buffer(2, self.values[0].slice(..));
    pass.set_vertex_buffer(3, self.values[1].slice(..));
    pass.draw(0..6, 0..self.nsegments);
  }
}

fn gather(values: &[f32], segments: &[[u32; 2]], end: usize) -> Vec<f32> {
  segments.iter().map(|s| values[s[end] as usize]).collect()
}

/// A glyph mark: the arrows of a line field, each a flat quad (two triangles,
/// six corners) lying in its surface cell. One vertex buffer, drawn unindexed --
/// the corners are already expanded, since a quad's diagonal is not shared the
/// way a mesh edge is, and the arrow-frame coordinate differs per corner anyway.
pub struct GlyphBatch {
  corners: wgpu::Buffer,
  ncorners: u32,
}

impl GlyphBatch {
  /// `vertices` is the flat corner stream, six per glyph. An empty stream is a
  /// valid batch that draws nothing.
  pub fn new(device: &wgpu::Device, vertices: &[GlyphVertex]) -> Self {
    Self {
      corners: vertex_buffer(device, "Glyph Corners", vertices),
      ncorners: vertices.len() as u32,
    }
  }

  pub fn layouts<'a>() -> [wgpu::VertexBufferLayout<'a>; 1] {
    [wgpu::VertexBufferLayout {
      array_stride: size_of::<GlyphVertex>() as wgpu::BufferAddress,
      step_mode: wgpu::VertexStepMode::Vertex,
      attributes: &GLYPH,
    }]
  }

  pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
    if self.ncorners == 0 {
      return;
    }
    pass.set_vertex_buffer(0, self.corners.slice(..));
    pass.draw(0..self.ncorners, 0..1);
  }
}

/// One thing a frame draws: a batch, and the material it is drawn with.
pub enum RenderItem<'a> {
  Surface(&'a SurfaceBatch, SurfaceMaterial),
  Segments(&'a SegmentBatch, SegmentMaterial),
  Glyphs(&'a GlyphBatch, GlyphMaterial),
}

/// Everything one frame draws, in submission order. The order is the caller's
/// statement of what lies over what: the surface writes depth and the segment
/// marks over it only test against it, so they blend in the order given.
///
/// The advected population and the atlas it trails into sit *beside* the items,
/// not among them: both are field state the frame advances before it draws, not
/// marks it draws. The population is stepped by the compute pass and read only
/// through the deposit it splats into -- it is never itself on screen.
#[derive(Default)]
pub struct DrawList<'a> {
  pub items: Vec<RenderItem<'a>>,
  /// The population the frame steps, when it has one. Written by the GPU, read
  /// by the deposit's splat -- so it is stepped, never drawn.
  pub particles: Option<&'a ParticleBatch>,
  /// The deposit atlas the particles trail into and the fill reads, when the
  /// frame has both. Beside the items for the same reason: it is stepped with
  /// the advection and read by the fill, so it belongs to no single mark.
  pub deposit: Option<&'a super::deposit::DepositBatch>,
}
