//! The wireframe pass: the mesh's 1-skeleton as billboard quads of constant
//! world-space thickness. See `wireframe.wgsl`.

use super::{
  color_target, compilation_options, depth_stencil,
  mesh::{MeshBuffer, Vertex},
  primitive, shader_module,
  uniform::Uniforms,
};

pub struct WireframePass {
  pipeline: wgpu::RenderPipeline,
}

impl WireframePass {
  pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, uniforms: &Uniforms) -> Self {
    let shader = shader_module(device, "Wireframe Shader", include_str!("wireframe.wgsl"));
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some("Wireframe Pipeline Layout"),
      bind_group_layouts: &[
        Some(uniforms.camera.layout()),
        Some(uniforms.wave.layout()),
        Some(uniforms.wireframe_width.layout()),
      ],
      immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
      label: Some("Wireframe Pipeline"),
      layout: Some(&layout),
      vertex: wgpu::VertexState {
        module: &shader,
        entry_point: Some("vs_main"),
        compilation_options: compilation_options(),
        buffers: &[Vertex::desc_endpoint_a(), Vertex::desc_endpoint_b()],
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
    pass.set_bind_group(2, uniforms.wireframe_width.bind_group(), &[]);
    pass.set_vertex_buffer(0, mesh.wireframe_a_buffer.slice(..));
    pass.set_vertex_buffer(1, mesh.wireframe_b_buffer.slice(..));
    pass.draw(0..6, 0..mesh.num_wireframe_edges);
  }
}
