//! The glyph pass: a line field's arrow glyphs, drawn as flat quads lying in
//! their surface cells. See `glyph.wgsl`.
//!
//! Not billboarded, unlike the segment pass: an arrow has a plane, its cell's,
//! so the quad is baked into it once and never turned toward the camera. That is
//! also what lets the arrow clip itself to the cell -- each corner's barycentric
//! coordinate is known in the plane -- which a section needs, since it has a
//! value only on the open cell it was sampled in.
//!
//! Alpha-blended and depth-testing but not depth-writing, exactly as the segment
//! marks: the arrows are translucent and lie over the fill they are coplanar
//! with. The tie is broken in *depth* by the pipeline's bias, never by moving
//! the quad -- see the depth-bias state the pass is built with.

use super::{
  color_target, depth_stencil_biased,
  item::GlyphBatch,
  primitive, shader_module,
  uniform::{FrameUniform, GlyphMaterial, UniformBinding, UniformPool},
  MASK_FORMAT, SURFACE_MARK_DEPTH_BIAS,
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
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
      label: Some("Glyph Pipeline"),
      layout: Some(&layout),
      vertex: wgpu::VertexState {
        module: &shader,
        entry_point: Some("vs_main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        buffers: &GlyphBatch::layouts(),
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
      depth_stencil: Some(depth_stencil_biased(false, SURFACE_MARK_DEPTH_BIAS)),
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
    batch: &GlyphBatch,
  ) {
    pass.set_pipeline(&self.pipeline);
    pass.set_bind_group(0, frame, &[]);
    pass.set_bind_group(1, material, &[]);
    batch.draw(pass);
  }
}
