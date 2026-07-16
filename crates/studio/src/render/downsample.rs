//! The supersampling resolve: a box filter from the offscreen scene color
//! target into the caller's own view. The one place the scene's pixels are
//! antialiased, and the only pass that writes the render target directly. See
//! `downsample.wgsl`.

use super::{color_target, compilation_options, primitive, shader_module, ssaa_constants};

/// The scene color texture as the downsample's one binding. Rebuilt with the
/// texture on resize.
pub struct SceneColorBinding {
  pub(super) bind_group: wgpu::BindGroup,
}

pub struct DownsamplePass {
  pipeline: wgpu::RenderPipeline,
  layout: wgpu::BindGroupLayout,
}

impl DownsamplePass {
  pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, ssaa: u32) -> Self {
    let constants = ssaa_constants(ssaa);
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
      label: Some("scene_color_bind_group_layout"),
      entries: &[wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
          sample_type: wgpu::TextureSampleType::Float { filterable: false },
          view_dimension: wgpu::TextureViewDimension::D2,
          multisampled: false,
        },
        count: None,
      }],
    });
    let shader = shader_module(device, "Downsample Shader", include_str!("downsample.wgsl"));
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some("Downsample Pipeline Layout"),
      bind_group_layouts: &[Some(&layout)],
      immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
      label: Some("Downsample Pipeline"),
      layout: Some(&pipeline_layout),
      vertex: wgpu::VertexState {
        module: &shader,
        entry_point: Some("vs_main"),
        compilation_options: compilation_options(&constants),
        buffers: &[],
      },
      fragment: Some(wgpu::FragmentState {
        module: &shader,
        entry_point: Some("fs_main"),
        compilation_options: compilation_options(&constants),
        targets: &[color_target(format, wgpu::BlendState::REPLACE)],
      }),
      primitive: primitive(),
      depth_stencil: None,
      multisample: wgpu::MultisampleState::default(),
      multiview_mask: None,
      cache: None,
    });
    Self { pipeline, layout }
  }

  /// The binding of a freshly (re)created scene color view.
  pub fn bind(&self, device: &wgpu::Device, scene_color: &wgpu::TextureView) -> SceneColorBinding {
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: Some("scene_color_bind_group"),
      layout: &self.layout,
      entries: &[wgpu::BindGroupEntry {
        binding: 0,
        resource: wgpu::BindingResource::TextureView(scene_color),
      }],
    });
    SceneColorBinding { bind_group }
  }

  pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, scene_color: &SceneColorBinding) {
    pass.set_pipeline(&self.pipeline);
    pass.set_bind_group(0, &scene_color.bind_group, &[]);
    pass.draw(0..3, 0..1);
  }
}
