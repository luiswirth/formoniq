//! The resolve: the supersampling box filter, the bloom composite and the tone
//! map, from the offscreen HDR scene target into the caller's own view. The only
//! pass that writes the render target directly. See `downsample.wgsl`.
//!
//! The three are one pass because they are one operation -- the crossing from
//! linear, unbounded radiance to a bounded sRGB display. Filtering and adding
//! light both have to happen on the radiance side of that crossing; the tone map
//! is the crossing itself.

use super::{
  color_target, compilation_options, primitive, shader_module, ssaa_constants,
  uniform::{PostUniform, UniformBinding},
};

/// The scene color texture and the bloom chain's glow, as the resolve's binding.
/// Rebuilt with the textures on resize.
pub struct SceneColorBinding {
  pub(super) bind_group: wgpu::BindGroup,
}

pub struct DownsamplePass {
  pipeline: wgpu::RenderPipeline,
  layout: wgpu::BindGroupLayout,
  sampler: wgpu::Sampler,
}

impl DownsamplePass {
  pub fn new(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    post: &UniformBinding<PostUniform>,
    ssaa: u32,
  ) -> Self {
    let constants = ssaa_constants(ssaa);
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
      label: Some("scene_color_bind_group_layout"),
      entries: &[
        wgpu::BindGroupLayoutEntry {
          binding: 0,
          visibility: wgpu::ShaderStages::FRAGMENT,
          // Unfiltered: the box filter loads the subsamples it is averaging, and
          // averaging them is the whole job -- a bilinear tap would be the
          // hardware guessing at the same mean with the wrong weights.
          ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
          },
          count: None,
        },
        wgpu::BindGroupLayoutEntry {
          binding: 1,
          visibility: wgpu::ShaderStages::FRAGMENT,
          // Filtered: the glow is half the scene's resolution and below, so it
          // is stretched back up and wants bilinear rather than blocks.
          ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
          },
          count: None,
        },
        wgpu::BindGroupLayoutEntry {
          binding: 2,
          visibility: wgpu::ShaderStages::FRAGMENT,
          ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
          count: None,
        },
      ],
    });
    let shader = shader_module(device, "Downsample Shader", include_str!("downsample.wgsl"));
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some("Downsample Pipeline Layout"),
      bind_group_layouts: &[Some(&layout), Some(post.layout())],
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
    Self {
      pipeline,
      layout,
      sampler: device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Glow Sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
      }),
    }
  }

  /// The binding of freshly (re)created scene color and glow views.
  pub fn bind(
    &self,
    device: &wgpu::Device,
    scene_color: &wgpu::TextureView,
    glow: &wgpu::TextureView,
  ) -> SceneColorBinding {
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: Some("scene_color_bind_group"),
      layout: &self.layout,
      entries: &[
        wgpu::BindGroupEntry {
          binding: 0,
          resource: wgpu::BindingResource::TextureView(scene_color),
        },
        wgpu::BindGroupEntry {
          binding: 1,
          resource: wgpu::BindingResource::TextureView(glow),
        },
        wgpu::BindGroupEntry {
          binding: 2,
          resource: wgpu::BindingResource::Sampler(&self.sampler),
        },
      ],
    });
    SceneColorBinding { bind_group }
  }

  pub fn draw(
    &self,
    pass: &mut wgpu::RenderPass<'_>,
    scene_color: &SceneColorBinding,
    post: &wgpu::BindGroup,
  ) {
    pass.set_pipeline(&self.pipeline);
    pass.set_bind_group(0, &scene_color.bind_group, &[]);
    pass.set_bind_group(1, post, &[]);
    pass.draw(0..3, 0..1);
  }
}
