use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::streamline::Streamlines;

/// Number of samples over which a curve's ends ramp from invisible to full
/// opacity, so a streamline fades in rather than starting with a hard cap.
const TAPER_SAMPLES: f64 = 4.0;

/// One end of a streamline segment, as a per-instance vertex. A segment (a
/// consecutive pair of samples) is drawn as a camera-facing quad by
/// `streamline.wgsl`, exactly as the wireframe draws a mesh edge -- both ends
/// have to reach one shader invocation, so they arrive as two parallel
/// per-instance buffers (`endpoint_a[i]`/`endpoint_b[i]` are segment `i`).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct StreamEndpoint {
  pub position: [f32; 3],
  /// End taper in $[0, 1]$: 0 at the very ends of a curve, ramping to 1 over a
  /// fixed number of samples, multiplied into the ribbon's opacity.
  pub taper: f32,
}

impl StreamEndpoint {
  pub fn desc_a<'a>() -> wgpu::VertexBufferLayout<'a> {
    wgpu::VertexBufferLayout {
      array_stride: std::mem::size_of::<StreamEndpoint>() as wgpu::BufferAddress,
      step_mode: wgpu::VertexStepMode::Instance,
      attributes: &[
        wgpu::VertexAttribute {
          offset: 0,
          shader_location: 0,
          format: wgpu::VertexFormat::Float32x3,
        },
        wgpu::VertexAttribute {
          offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
          shader_location: 1,
          format: wgpu::VertexFormat::Float32,
        },
      ],
    }
  }

  pub fn desc_b<'a>() -> wgpu::VertexBufferLayout<'a> {
    wgpu::VertexBufferLayout {
      array_stride: std::mem::size_of::<StreamEndpoint>() as wgpu::BufferAddress,
      step_mode: wgpu::VertexStepMode::Instance,
      attributes: &[
        wgpu::VertexAttribute {
          offset: 0,
          shader_location: 2,
          format: wgpu::VertexFormat::Float32x3,
        },
        wgpu::VertexAttribute {
          offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
          shader_location: 3,
          format: wgpu::VertexFormat::Float32,
        },
      ],
    }
  }
}

/// The traced streamlines as GPU segment instances: the two endpoint buffers the
/// ribbon pipeline draws (6 vertices per segment, non-indexed). Empty is a valid
/// state (a field with no streamlines), and `num_segments == 0` skips the draw.
pub struct StreamlineBuffer {
  pub endpoint_a: wgpu::Buffer,
  pub endpoint_b: wgpu::Buffer,
  pub num_segments: u32,
}

impl StreamlineBuffer {
  pub fn new(device: &wgpu::Device, streamlines: &Streamlines) -> Self {
    let mut endpoint_a: Vec<StreamEndpoint> = Vec::new();
    let mut endpoint_b: Vec<StreamEndpoint> = Vec::new();

    for line in &streamlines.lines {
      let len = line.len();
      let endpoints: Vec<StreamEndpoint> = line
        .iter()
        .enumerate()
        .map(|(i, sp)| {
          let from_end = i.min(len - 1 - i) as f64;
          StreamEndpoint {
            position: [sp.pos.x as f32, sp.pos.y as f32, sp.pos.z as f32],
            taper: (from_end / TAPER_SAMPLES).min(1.0) as f32,
          }
        })
        .collect();
      for pair in endpoints.windows(2) {
        endpoint_a.push(pair[0]);
        endpoint_b.push(pair[1]);
      }
    }

    // A zero-length `create_buffer_init` is rejected; a field with no
    // streamlines still needs valid (unused) buffers. `num_segments` stays the
    // true count, so the pad is never drawn.
    let num_segments = endpoint_a.len() as u32;
    if endpoint_a.is_empty() {
      endpoint_a.push(StreamEndpoint::zeroed());
      endpoint_b.push(StreamEndpoint::zeroed());
    }

    let endpoint_a_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Streamline Endpoint A Buffer"),
      contents: bytemuck::cast_slice(&endpoint_a),
      usage: wgpu::BufferUsages::VERTEX,
    });
    let endpoint_b_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Streamline Endpoint B Buffer"),
      contents: bytemuck::cast_slice(&endpoint_b),
      usage: wgpu::BufferUsages::VERTEX,
    });

    Self {
      endpoint_a: endpoint_a_buffer,
      endpoint_b: endpoint_b_buffer,
      num_segments,
    }
  }
}
