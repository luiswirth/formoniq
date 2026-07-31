//! The point pass: the 0-skeleton, drawn as instanced billboard circles of
//! constant world-space radius. The 0-dimensional sibling of the segment pass,
//! sharing its `SegmentMaterial` and its two inks, see `points.wgsl`.
//!
//! Alpha-blended and depth-testing without writing, exactly as the segments are:
//! the discs are translucent and sit over the surface that already wrote depth.

use super::{
  MarkPipeline, SURFACE_MARK_DEPTH_BIAS, depth_stencil_biased,
  item::PointBatch,
  shader_module,
  uniform::{FrameUniform, SegmentMaterial, UniformPool},
};

pub struct PointPass {
  pipeline: wgpu::RenderPipeline,
}

impl PointPass {
  pub fn new(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    frame: &super::uniform::UniformBinding<FrameUniform>,
    materials: &UniformPool<SegmentMaterial>,
  ) -> Self {
    let shader = shader_module(device, "Point Shader", include_str!("points.wgsl"));
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some("Point Pipeline Layout"),
      bind_group_layouts: &[Some(frame.layout()), Some(materials.layout())],
      immediate_size: 0,
    });
    let pipeline = MarkPipeline {
      label: "Point Pipeline",
      shader: &shader,
      layout: &layout,
      buffers: &PointBatch::layouts(),
      format,
      blend: wgpu::BlendState::ALPHA_BLENDING,
      depth: depth_stencil_biased(true, SURFACE_MARK_DEPTH_BIAS),
    }
    .build(device);
    Self { pipeline }
  }

  pub fn draw(
    &self,
    pass: &mut wgpu::RenderPass<'_>,
    frame: &wgpu::BindGroup,
    material: &wgpu::BindGroup,
    batch: &PointBatch,
  ) {
    pass.set_pipeline(&self.pipeline);
    pass.set_bind_group(0, frame, &[]);
    pass.set_bind_group(1, material, &[]);
    batch.draw(pass);
  }
}
