//! The triangle fill pass: the surface itself, colormapped by the per-vertex
//! value and normal-displaced as a standing wave. See `fill.wgsl`.

use super::{
  color_target, compilation_options, depth_stencil,
  item::SurfaceBatch,
  primitive, shader_module, ssaa_constants,
  uniform::{FrameUniform, SurfaceMaterial, UniformBinding, UniformPool},
};

pub struct FillPass {
  pipeline: wgpu::RenderPipeline,
}

impl FillPass {
  pub fn new(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    frame: &UniformBinding<FrameUniform>,
    materials: &UniformPool<SurfaceMaterial>,
    ssaa: u32,
  ) -> Self {
    let constants = ssaa_constants(ssaa);
    let shader = shader_module(device, "Fill Shader", include_str!("fill.wgsl"));
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some("Fill Pipeline Layout"),
      bind_group_layouts: &[Some(frame.layout()), Some(materials.layout())],
      immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
      label: Some("Fill Pipeline"),
      layout: Some(&layout),
      vertex: wgpu::VertexState {
        module: &shader,
        entry_point: Some("vs_main"),
        compilation_options: compilation_options(&constants),
        buffers: &SurfaceBatch::layouts(),
      },
      fragment: Some(wgpu::FragmentState {
        module: &shader,
        entry_point: Some("fs_main"),
        compilation_options: compilation_options(&constants),
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

  pub fn draw(
    &self,
    pass: &mut wgpu::RenderPass<'_>,
    frame: &wgpu::BindGroup,
    material: &wgpu::BindGroup,
    batch: &SurfaceBatch,
  ) {
    pass.set_pipeline(&self.pipeline);
    pass.set_bind_group(0, frame, &[]);
    pass.set_bind_group(1, material, &[]);
    batch.draw(pass);
  }
}
