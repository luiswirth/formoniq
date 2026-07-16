//! The draw list: baked geometry uploaded to the GPU, and the items a frame is
//! made of.
//!
//! A batch owns buffers and nothing else; an item is a batch plus the material
//! it is drawn with. A surface, its wireframe overlay and its traced streamline
//! ribbons are three items over two batch kinds, and several manifolds in one
//! scene are simply more items -- the renderer never learns how many there will
//! be.
//!
//! Each batch mirrors `bake.rs`'s static/per-field split: positions and index
//! streams are uploaded once per mesh, and only `write_attributes` runs when the
//! field on display changes.

use bytemuck::Pod;
use wgpu::util::DeviceExt;

use super::uniform::{SegmentMaterial, SurfaceMaterial};
use crate::bake::{BakedMesh, BakedVertex, PrimBatch, SegmentVertex};

/// A `VERTEX` buffer holding `data`, never empty: a zero-length
/// `create_buffer_init` is rejected, and an empty batch (a field with no
/// streamlines, a curve with no wireframe overlay) still needs valid buffers to
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

const fn value_attribute(loc: u32) -> [wgpu::VertexAttribute; 1] {
  [attribute(0, loc, wgpu::VertexFormat::Float32)]
}

const SURFACE_STATIC: [wgpu::VertexAttribute; 3] = baked_attributes(0);
const SURFACE_VALUE: [wgpu::VertexAttribute; 1] = value_attribute(3);
const SEGMENT_A: [wgpu::VertexAttribute; 4] = segment_attributes(0);
const SEGMENT_B: [wgpu::VertexAttribute; 4] = segment_attributes(4);
const SEGMENT_VALUE_A: [wgpu::VertexAttribute; 1] = value_attribute(8);
const SEGMENT_VALUE_B: [wgpu::VertexAttribute; 1] = value_attribute(9);

/// A filled triangle surface: the shared vertex table, the per-field attribute
/// stream over it, and one index stream of wound triangles.
pub struct SurfaceBatch {
  positions: wgpu::Buffer,
  attributes: wgpu::Buffer,
  indices: wgpu::Buffer,
  nindices: u32,
}

impl SurfaceBatch {
  /// The batch of a baked mesh whose cells are triangles; `None` for a bake that
  /// produced no fill (a curve, a point cloud), whose marks are its segments
  /// instead. The attribute stream starts at zero; a field fills it through
  /// [`Self::write_attributes`].
  pub fn new(device: &wgpu::Device, baked: &BakedMesh) -> Option<Self> {
    let PrimBatch::Triangles(triangles) = &baked.cells else {
      return None;
    };
    let indices: Vec<u32> = triangles.iter().flatten().copied().collect();
    Some(Self {
      positions: vertex_buffer(device, "Surface Positions", &baked.positions),
      attributes: vertex_buffer(
        device,
        "Surface Attributes",
        &vec![0.0f32; baked.positions.len()],
      ),
      indices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Surface Indices"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
      }),
      nindices: indices.len() as u32,
    })
  }

  /// Rebinds the surface to a different field of the same mesh: one buffer
  /// write, no rebake.
  pub fn write_attributes(&self, queue: &wgpu::Queue, attributes: &[f32]) {
    if !attributes.is_empty() {
      queue.write_buffer(&self.attributes, 0, bytemuck::cast_slice(attributes));
    }
  }

  /// The static stream at locations 0..=2, the per-field value at 3.
  pub fn layouts<'a>() -> [wgpu::VertexBufferLayout<'a>; 2] {
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
    ]
  }

  pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
    pass.set_vertex_buffer(0, self.positions.slice(..));
    pass.set_vertex_buffer(1, self.attributes.slice(..));
    pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
    pass.draw_indexed(0..self.nindices, 0, 0..1);
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

/// One thing a frame draws: a batch, and the material it is drawn with.
pub enum RenderItem<'a> {
  Surface(&'a SurfaceBatch, SurfaceMaterial),
  Segments(&'a SegmentBatch, SegmentMaterial),
}

/// Everything one frame draws, in submission order. The order is the caller's
/// statement of what lies over what: the surface writes depth and the segment
/// marks over it only test against it, so they blend in the order given.
#[derive(Default)]
pub struct DrawList<'a> {
  pub items: Vec<RenderItem<'a>>,
}
