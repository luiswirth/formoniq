//! The segment pass: every 1-dimensional mark, drawn as instanced billboard
//! quads of constant world-space thickness. See `segments.wgsl`.
//!
//! One pipeline serves the wireframe overlay, a line field's traced streamline
//! ribbons and a 1-manifold's own cells. They were three descriptions of one
//! technique; what differed -- ink, width, end taper, whether the mark rides the
//! standing-wave displacement -- is data now: the material, and the endpoint's
//! own normal (zero where a mark does not sit on a displaced surface, which
//! makes the displacement the identity on it).
//!
//! The pass is alpha-blended and does not write depth. The ribbons must not:
//! they are translucent, and a ribbon biased toward the camera and drawn first
//! would otherwise occlude the wireframe edge it lies along. The wireframe need
//! not: nothing in the scene is drawn after it, so its depth would never be
//! read.

use super::{
  color_target, compilation_options, depth_stencil,
  item::SegmentBatch,
  primitive, shader_module,
  uniform::{FrameUniform, SegmentMaterial, UniformPool},
};

pub struct SegmentPass {
  pipeline: wgpu::RenderPipeline,
}

impl SegmentPass {
  pub fn new(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    frame: &super::uniform::UniformBinding<FrameUniform>,
    materials: &UniformPool<SegmentMaterial>,
  ) -> Self {
    let shader = shader_module(device, "Segment Shader", include_str!("segments.wgsl"));
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some("Segment Pipeline Layout"),
      bind_group_layouts: &[Some(frame.layout()), Some(materials.layout())],
      immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
      label: Some("Segment Pipeline"),
      layout: Some(&layout),
      vertex: wgpu::VertexState {
        module: &shader,
        entry_point: Some("vs_main"),
        compilation_options: compilation_options(),
        buffers: &SegmentBatch::layouts(),
      },
      fragment: Some(wgpu::FragmentState {
        module: &shader,
        entry_point: Some("fs_main"),
        compilation_options: compilation_options(),
        targets: &[color_target(format, wgpu::BlendState::ALPHA_BLENDING)],
      }),
      primitive: primitive(),
      depth_stencil: Some(depth_stencil(false)),
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
    batch: &SegmentBatch,
  ) {
    pass.set_pipeline(&self.pipeline);
    pass.set_bind_group(0, frame, &[]);
    pass.set_bind_group(1, material, &[]);
    batch.draw(pass);
  }
}
