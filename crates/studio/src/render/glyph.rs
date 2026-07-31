//! The glyph pass: a line field's arrow glyphs, drawn as flat quads lying in
//! their surface cells. See `glyph.wgsl`.
//!
//! Not billboarded, unlike the segment pass: an arrow has a plane, its cell's,
//! so the quad is baked into it once and never turned toward the camera. That is
//! also what lets the arrow clip itself to the cell, each corner's barycentric
//! coordinate is known in the plane, which a section needs, since it has a
//! value only on the open cell it was sampled in.
//!
//! Alpha-blended and depth-testing but not depth-writing, exactly as the segment
//! marks: the arrows are translucent and lie over the fill they are coplanar
//! with. The tie is broken in depth by the pipeline's bias, never by moving
//! the quad, see the depth-bias state the pass is built with.

use super::{
  MarkPipeline, SURFACE_MARK_DEPTH_BIAS, depth_stencil_biased,
  item::GlyphBatch,
  shader_module,
  uniform::{FrameUniform, GlyphMaterial, UniformBinding, UniformPool},
};

pub struct GlyphPass {
  pipeline: wgpu::RenderPipeline,
}

impl GlyphPass {
  pub fn new(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    frame: &UniformBinding<FrameUniform>,
    materials: &UniformPool<GlyphMaterial>,
  ) -> Self {
    let shader = shader_module(device, "Glyph Shader", include_str!("glyph.wgsl"));
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some("Glyph Pipeline Layout"),
      bind_group_layouts: &[Some(frame.layout()), Some(materials.layout())],
      immediate_size: 0,
    });
    let pipeline = MarkPipeline {
      label: "Glyph Pipeline",
      shader: &shader,
      layout: &layout,
      buffers: &GlyphBatch::layouts(),
      format,
      blend: wgpu::BlendState::ALPHA_BLENDING,
      depth: depth_stencil_biased(false, SURFACE_MARK_DEPTH_BIAS),
    }
    .build(device);
    Self { pipeline }
  }

  pub fn draw(
    &self,
    pass: &mut wgpu::RenderPass<'_>,
    frame: &wgpu::BindGroup,
    material: &wgpu::BindGroup,
    batch: &GlyphBatch,
  ) {
    pass.set_pipeline(&self.pipeline);
    pass.set_bind_group(0, frame, &[]);
    pass.set_bind_group(1, material, &[]);
    batch.draw(pass);
  }
}
