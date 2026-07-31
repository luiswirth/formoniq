//! The triangle fill pass: the surface itself, colormapped by the per-vertex
//! value and normal-displaced as a standing wave. See `fill.wgsl`.

use super::{
  MarkPipeline,
  deposit::deposit_read_layout,
  depth_stencil,
  item::SurfaceBatch,
  shader_module,
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
  ) -> Self {
    let shader = shader_module(device, "Fill Shader", include_str!("fill.wgsl"));
    let deposit = deposit_read_layout(device);
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some("Fill Pipeline Layout"),
      bind_group_layouts: &[
        Some(frame.layout()),
        Some(materials.layout()),
        Some(&deposit),
      ],
      immediate_size: 0,
    });
    let pipeline = MarkPipeline {
      label: "Fill Pipeline",
      shader: &shader,
      layout: &layout,
      buffers: &SurfaceBatch::layouts(),
      format,
      blend: wgpu::BlendState::REPLACE,
      depth: depth_stencil(true),
    }
    .build(device);
    Self { pipeline }
  }

  /// `deposit` is the atlas read binding, a batch's current texture, or the
  /// renderer's 1x1 zero dummy for a frame without trails, which the material's
  /// floor-1/gain-0 makes the identity.
  pub fn draw(
    &self,
    pass: &mut wgpu::RenderPass<'_>,
    frame: &wgpu::BindGroup,
    material: &wgpu::BindGroup,
    deposit: &wgpu::BindGroup,
    batch: &SurfaceBatch,
  ) {
    pass.set_pipeline(&self.pipeline);
    pass.set_bind_group(0, frame, &[]);
    pass.set_bind_group(1, material, &[]);
    pass.set_bind_group(2, deposit, &[]);
    batch.draw(pass);
  }
}
