//! The streamline ribbon pass: the traced integral curves of a line field as
//! billboard quads, one instance per segment. The same technique as the
//! wireframe, at a different width and with a tapered, wave-faded ink instead
//! of a solid edge. See `streamline.wgsl`.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::{
  color_target, compilation_options, depth_stencil, primitive, shader_module, uniform::Uniforms,
};
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

pub struct StreamlinePass {
  pipeline: wgpu::RenderPipeline,
}

impl StreamlinePass {
  pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, uniforms: &Uniforms) -> Self {
    let shader = shader_module(device, "Streamline Shader", include_str!("streamline.wgsl"));
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some("Streamline Pipeline Layout"),
      bind_group_layouts: &[
        Some(uniforms.camera.layout()),
        Some(uniforms.wave.layout()),
        Some(uniforms.streamline_width.layout()),
      ],
      immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
      label: Some("Streamline Pipeline"),
      layout: Some(&layout),
      vertex: wgpu::VertexState {
        module: &shader,
        entry_point: Some("vs_main"),
        compilation_options: compilation_options(),
        buffers: &[StreamEndpoint::desc_a(), StreamEndpoint::desc_b()],
      },
      fragment: Some(wgpu::FragmentState {
        module: &shader,
        entry_point: Some("fs_main"),
        compilation_options: compilation_options(),
        targets: &[color_target(format, wgpu::BlendState::ALPHA_BLENDING)],
      }),
      primitive: primitive(),
      // The ribbons are an alpha-blended overlay on the surface: they test
      // against it (so a curve on the far side stays hidden) but must not write
      // depth. Writing it would let a ribbon, biased toward the camera and drawn
      // first, occlude the wireframe edge it lies along -- and depth-writing
      // translucent geometry is wrong regardless of what is drawn after.
      depth_stencil: Some(depth_stencil(false)),
      multisample: wgpu::MultisampleState::default(),
      multiview_mask: None,
      cache: None,
    });
    Self { pipeline }
  }

  /// Draws the ribbons, if there are any: an empty trace (a field with no
  /// streamlines) draws nothing rather than being a case the caller has to
  /// exclude.
  pub fn draw(
    &self,
    pass: &mut wgpu::RenderPass<'_>,
    uniforms: &Uniforms,
    streamlines: &StreamlineBuffer,
  ) {
    if streamlines.num_segments == 0 {
      return;
    }
    pass.set_pipeline(&self.pipeline);
    pass.set_bind_group(0, uniforms.camera.bind_group(), &[]);
    pass.set_bind_group(1, uniforms.wave.bind_group(), &[]);
    pass.set_bind_group(2, uniforms.streamline_width.bind_group(), &[]);
    pass.set_vertex_buffer(0, streamlines.endpoint_a.slice(..));
    pass.set_vertex_buffer(1, streamlines.endpoint_b.slice(..));
    pass.draw(0..6, 0..streamlines.num_segments);
  }
}
