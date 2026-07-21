//! The bloom chain: the light that spills out of whatever the display range
//! cannot hold. See `bloom.wgsl`.
//!
//! Unconditional, like the resolve and unlike the LIC's old G-buffer: the chain
//! is allocated with the targets and runs every frame whatever the draw list
//! holds. A scene with nothing above the threshold blooms nothing, by arithmetic
//! rather than by a branch, so the frame graph stays one linear sequence.
//!
//! The chain halves down and adds back up. Each level is a wider blur over a
//! quarter of the pixels, and their sum is a Gaussian far wider than any single
//! pass would afford -- which is what a glow is: not a halo of fixed radius, but
//! light falling off over the whole frame.

use super::{SCENE_FORMAT, color_target, primitive, shader_module};

/// How many halvings the chain takes below the scene's own resolution.
///
/// The widest level is what sets the glow's reach, so this is the *radius* knob
/// in disguise: each level doubles it. Five puts the broadest halo at 1/32 of
/// the frame, which reads as light in the air rather than an outline around a
/// speck. Levels stop early rather than degenerate when a dimension runs out.
const LEVELS: usize = 5;

/// The pipelines and the sampler; the textures live in a [`BloomChain`], which is
/// reallocated on resize.
pub struct BloomPass {
  prefilter: wgpu::RenderPipeline,
  downsample: wgpu::RenderPipeline,
  upsample: wgpu::RenderPipeline,
  layout: wgpu::BindGroupLayout,
  sampler: wgpu::Sampler,
}

/// One level of the chain: what it is drawn into, and what it is read from.
struct Level {
  view: wgpu::TextureView,
  bind_group: wgpu::BindGroup,
}

/// The chain's textures, sized from the scene's, plus the scene's own binding.
pub struct BloomChain {
  scene: wgpu::BindGroup,
  levels: Vec<Level>,
}

impl BloomChain {
  /// The blurred glow, for the resolve to composite: the top of the chain, after
  /// every level below it has been added back in.
  pub fn glow(&self) -> &wgpu::TextureView {
    &self.levels[0].view
  }
}

impl BloomPass {
  pub fn new(device: &wgpu::Device) -> Self {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
      label: Some("Bloom Bind Group Layout"),
      entries: &[
        wgpu::BindGroupLayoutEntry {
          binding: 0,
          visibility: wgpu::ShaderStages::FRAGMENT,
          ty: wgpu::BindingType::Texture {
            // Filterable, unlike the resolve's binding of the same texture: the
            // chain leans on bilinear taps to make four fetches average sixteen
            // texels, where the resolve wants the texels themselves.
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
          },
          count: None,
        },
        wgpu::BindGroupLayoutEntry {
          binding: 1,
          visibility: wgpu::ShaderStages::FRAGMENT,
          ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
          count: None,
        },
      ],
    });
    let shader = shader_module(device, "Bloom Shader", include_str!("bloom.wgsl"));
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some("Bloom Pipeline Layout"),
      bind_group_layouts: &[Some(&layout)],
      immediate_size: 0,
    });
    let stage = |label: &str, entry: &str, blend: wgpu::BlendState| {
      device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
          module: &shader,
          entry_point: Some("vs_main"),
          compilation_options: wgpu::PipelineCompilationOptions::default(),
          buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
          module: &shader,
          entry_point: Some(entry),
          compilation_options: wgpu::PipelineCompilationOptions::default(),
          targets: &[color_target(SCENE_FORMAT, blend)],
        }),
        primitive: primitive(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
      })
    };
    Self {
      prefilter: stage("Bloom Prefilter", "fs_prefilter", wgpu::BlendState::REPLACE),
      downsample: stage(
        "Bloom Downsample",
        "fs_downsample",
        wgpu::BlendState::REPLACE,
      ),
      // Additive, which is what makes the upward pass a *sum* of blurs rather
      // than the widest one alone.
      upsample: stage(
        "Bloom Upsample",
        "fs_upsample",
        wgpu::BlendState {
          color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
          },
          alpha: wgpu::BlendComponent::REPLACE,
        },
      ),
      layout,
      sampler: device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Bloom Sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
      }),
    }
  }

  fn bind(&self, device: &wgpu::Device, view: &wgpu::TextureView) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: Some("Bloom Bind Group"),
      layout: &self.layout,
      entries: &[
        wgpu::BindGroupEntry {
          binding: 0,
          resource: wgpu::BindingResource::TextureView(view),
        },
        wgpu::BindGroupEntry {
          binding: 1,
          resource: wgpu::BindingResource::Sampler(&self.sampler),
        },
      ],
    })
  }

  /// The chain for a scene target of `size` (the supersampled resolution).
  pub fn chain(
    &self,
    device: &wgpu::Device,
    scene_view: &wgpu::TextureView,
    size: (u32, u32),
  ) -> BloomChain {
    let mut levels = Vec::with_capacity(LEVELS);
    let (mut width, mut height) = size;
    for level in 0..LEVELS {
      width = width.div_ceil(2);
      height = height.div_ceil(2);
      // A chain that has run out of pixels stops. The degenerate case is not
      // excluded by the caller: a tiny target simply blooms over fewer levels.
      if (width < 2 || height < 2) && level > 0 {
        break;
      }
      let view = device
        .create_texture(&wgpu::TextureDescriptor {
          label: Some("Bloom Level"),
          size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
          },
          mip_level_count: 1,
          sample_count: 1,
          dimension: wgpu::TextureDimension::D2,
          format: SCENE_FORMAT,
          usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
          view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default());
      levels.push(Level {
        bind_group: self.bind(device, &view),
        view,
      });
    }
    BloomChain {
      scene: self.bind(device, scene_view),
      levels,
    }
  }

  /// Records the whole chain: threshold the scene into the top level, halve to
  /// the bottom, then add back up. Afterwards [`BloomChain::glow`] holds the sum.
  pub fn render(&self, encoder: &mut wgpu::CommandEncoder, chain: &BloomChain) {
    let mut stage = |label: &str,
                     pipeline: &wgpu::RenderPipeline,
                     source: &wgpu::BindGroup,
                     target: &wgpu::TextureView,
                     load: wgpu::LoadOp<wgpu::Color>| {
      let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
          view: target,
          resolve_target: None,
          ops: wgpu::Operations {
            load,
            store: wgpu::StoreOp::Store,
          },
          depth_slice: None,
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
      });
      pass.set_pipeline(pipeline);
      pass.set_bind_group(0, source, &[]);
      pass.draw(0..3, 0..1);
    };

    let clear = wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT);
    stage(
      "Bloom Prefilter Pass",
      &self.prefilter,
      &chain.scene,
      &chain.levels[0].view,
      clear,
    );
    for i in 1..chain.levels.len() {
      stage(
        "Bloom Downsample Pass",
        &self.downsample,
        &chain.levels[i - 1].bind_group,
        &chain.levels[i].view,
        clear,
      );
    }
    // Upward, loading rather than clearing: the additive blend is adding this
    // level's blur to the one already there.
    for i in (1..chain.levels.len()).rev() {
      stage(
        "Bloom Upsample Pass",
        &self.upsample,
        &chain.levels[i].bind_group,
        &chain.levels[i - 1].view,
        wgpu::LoadOp::Load,
      );
    }
  }
}
