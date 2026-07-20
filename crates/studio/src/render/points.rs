//! The point pass: the 0-skeleton, drawn as instanced billboard circles of
//! constant world-space radius. The 0-dimensional sibling of the segment pass,
//! sharing its `SegmentMaterial` and its two inks -- see `points.wgsl`.
//!
//! Alpha-blended and depth-testing without writing, exactly as the segments are:
//! the discs are translucent and sit over the surface that already wrote depth.

use super::{
  color_target, depth_stencil_biased,
  item::PointBatch,
  primitive, shader_module,
  uniform::{FrameUniform, SegmentMaterial, UniformPool},
  MASK_FORMAT, SURFACE_MARK_DEPTH_BIAS,
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
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
      label: Some("Point Pipeline"),
      layout: Some(&layout),
      vertex: wgpu::VertexState {
        module: &shader,
        entry_point: Some("vs_main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        buffers: &PointBatch::layouts(),
      },
      fragment: Some(wgpu::FragmentState {
        module: &shader,
        entry_point: Some("fs_main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        targets: &[
          color_target(format, wgpu::BlendState::ALPHA_BLENDING),
          color_target(MASK_FORMAT, wgpu::BlendState::REPLACE),
        ],
      }),
      primitive: primitive(),
      depth_stencil: Some(depth_stencil_biased(true, SURFACE_MARK_DEPTH_BIAS)),
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
    batch: &PointBatch,
  ) {
    pass.set_pipeline(&self.pipeline);
    pass.set_bind_group(0, frame, &[]);
    pass.set_bind_group(1, material, &[]);
    batch.draw(pass);
  }
}
