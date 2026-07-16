//! The triangle fill pass: the surface itself, colormapped by the per-vertex
//! value and normal-displaced as a standing wave. See `fill.wgsl`.

use super::{
  color_target, compilation_options, depth_stencil,
  mesh::{MeshBuffer, Vertex},
  primitive, shader_module,
  uniform::Uniforms,
};

pub struct FillPass {
  pipeline: wgpu::RenderPipeline,
}

impl FillPass {
  pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, uniforms: &Uniforms) -> Self {
    let shader = shader_module(device, "Fill Shader", include_str!("fill.wgsl"));
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some("Fill Pipeline Layout"),
      bind_group_layouts: &[
        Some(uniforms.camera.layout()),
        Some(uniforms.wave.layout()),
        Some(uniforms.bounds.layout()),
      ],
      immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
      label: Some("Fill Pipeline"),
      layout: Some(&layout),
      vertex: wgpu::VertexState {
        module: &shader,
        entry_point: Some("vs_main"),
        compilation_options: compilation_options(),
        buffers: &[Vertex::desc()],
      },
      fragment: Some(wgpu::FragmentState {
        module: &shader,
        entry_point: Some("fs_main"),
        compilation_options: compilation_options(),
        targets: &[color_target(format, wgpu::BlendState::REPLACE)],
      }),
      primitive: primitive(),
      depth_stencil: Some(depth_stencil(true)),
      multisample: wgpu::MultisampleState::default(),
      multiview_mask: None,
      cache: None,
    });
    Self { pipeline }
  }

  pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, uniforms: &Uniforms, mesh: &MeshBuffer) {
    pass.set_pipeline(&self.pipeline);
    pass.set_bind_group(0, uniforms.camera.bind_group(), &[]);
    pass.set_bind_group(1, uniforms.wave.bind_group(), &[]);
    pass.set_bind_group(2, uniforms.bounds.bind_group(), &[]);
    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
    pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
    pass.draw_indexed(0..mesh.num_indices, 0, 0..1);
  }
}
