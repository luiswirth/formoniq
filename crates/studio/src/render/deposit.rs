//! The deposit atlas on the GPU: the ping-pong texture pair holding the
//! trails, and the fade + splat pipelines that step it. See `deposit.wgsl`,
//! and `realize::deposit` for the layout.
//!
//! The state-on-the-manifold invariant is enforced structurally here: the
//! atlas is the *only* texture in the renderer that survives a frame, and it
//! is indexed by (cell, barycentric lattice), never by screen position. The
//! screen-side passes read it exactly as they read any other field datum.
//!
//! Like a [`super::particles::ParticleBatch`], a `DepositBatch` is field
//! geometry written by the GPU: built when the field changes, stepped by the
//! frame, read by the fill. The window and an exporter share it through the
//! display, so a still export shows exactly the trails on screen.

use std::cell::Cell as StdCell;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::{color_target, shader_module, uniform::UniformBinding};
use realize::deposit::{ATLAS_SIZE, DepositLayout};

/// The WGSL `DepositParams` of `preamble.wgsl`, byte for byte.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct DepositParams {
  pub atlas_size: f32,
  pub radius: f32,
  pub energy: f32,
  pub decay: f32,
  pub depth: u32,
  pub _pad0: u32,
  pub _pad1: u32,
  pub _pad2: u32,
}

/// The atlas texel format: float because deposits accumulate without bound
/// (the whole point -- the overflow is what blooms), 16 bits because the
/// range matters and the precision does not, single channel because a deposit
/// is a density. Renderable, blendable and filterable in core WebGPU.
pub const DEPOSIT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;

/// The layout the fill (and the fade) read the atlas through: the texture and
/// a bilinear sampler. A free function for the same reason the advection's is:
/// the pipelines are built from it once and every batch binds through it, and
/// wgpu compares layouts structurally.
pub fn deposit_read_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
  device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
    label: Some("Deposit Read Layout"),
    entries: &[
      wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
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
  })
}

fn splat_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
  let storage = |binding: u32| wgpu::BindGroupLayoutEntry {
    binding,
    visibility: wgpu::ShaderStages::VERTEX,
    ty: wgpu::BindingType::Buffer {
      ty: wgpu::BufferBindingType::Storage { read_only: true },
      has_dynamic_offset: false,
      min_binding_size: None,
    },
    count: None,
  };
  device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
    label: Some("Deposit Splat Layout"),
    entries: &[storage(0), storage(1), storage(2)],
  })
}

/// A read bind group over `view`: what the fill binds, and what the fade
/// pipeline reads the previous step through. Also how the renderer builds its
/// 1x1 zero dummy for frames with no deposit.
pub fn read_bind_group(device: &wgpu::Device, view: &wgpu::TextureView) -> wgpu::BindGroup {
  let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
    label: Some("Deposit Sampler"),
    address_mode_u: wgpu::AddressMode::ClampToEdge,
    address_mode_v: wgpu::AddressMode::ClampToEdge,
    mag_filter: wgpu::FilterMode::Linear,
    min_filter: wgpu::FilterMode::Linear,
    ..Default::default()
  });
  device.create_bind_group(&wgpu::BindGroupDescriptor {
    label: Some("Deposit Read Bind Group"),
    layout: &deposit_read_layout(device),
    entries: &[
      wgpu::BindGroupEntry {
        binding: 0,
        resource: wgpu::BindingResource::TextureView(view),
      },
      wgpu::BindGroupEntry {
        binding: 1,
        resource: wgpu::BindingResource::Sampler(&sampler),
      },
    ],
  })
}

/// A 1x1 zero atlas: the read binding of a frame with no deposit. Zero deposit
/// through the material's `floor + gain * D` is the identity, so "none" is a
/// value, never a second fill pipeline.
pub fn dummy_read_bind_group(device: &wgpu::Device) -> wgpu::BindGroup {
  let texture = device.create_texture(&wgpu::TextureDescriptor {
    label: Some("Deposit Dummy Texture"),
    size: wgpu::Extent3d {
      width: 1,
      height: 1,
      depth_or_array_layers: 1,
    },
    mip_level_count: 1,
    sample_count: 1,
    dimension: wgpu::TextureDimension::D2,
    format: DEPOSIT_FORMAT,
    usage: wgpu::TextureUsages::TEXTURE_BINDING,
    view_formats: &[],
  });
  read_bind_group(
    device,
    &texture.create_view(&wgpu::TextureViewDescriptor::default()),
  )
}

/// One field's trail state: the two atlas textures, and every binding the
/// stepping pair and the fill need. `current` is which texture holds the
/// latest step -- interior mutability because stepping happens during an
/// immutably borrowed frame, exactly like the particle buffer the GPU writes.
pub struct DepositBatch {
  views: [wgpu::TextureView; 2],
  read_groups: [wgpu::BindGroup; 2],
  splat_group: wgpu::BindGroup,
  params: UniformBinding<DepositParams>,
  count: u32,
  current: StdCell<usize>,
}

impl DepositBatch {
  /// The atlas of `layout`, splatted by the particles of `particle_buffer`
  /// fading by `decay` per step and inking `energy` per texel of path.
  /// `depth` is the advection's own dyadic depth -- the splat reads the
  /// population's whole-step flow level to derive each particle's
  /// displacement, exactly as the head speck's motion blur does. `None` for
  /// the empty layout: a manifold with no atlas has no trails, and draws none.
  pub fn new(
    device: &wgpu::Device,
    layout: &DepositLayout,
    population: &super::particles::ParticleBatch,
    depth: u32,
    energy: f32,
    decay: f32,
  ) -> Option<Self> {
    let count = population.count();
    if layout.blocks.is_empty() || count == 0 {
      return None;
    }
    let texture = |label: &str| {
      device
        .create_texture(&wgpu::TextureDescriptor {
          label: Some(label),
          size: wgpu::Extent3d {
            width: ATLAS_SIZE,
            height: ATLAS_SIZE,
            depth_or_array_layers: 1,
          },
          mip_level_count: 1,
          sample_count: 1,
          dimension: wgpu::TextureDimension::D2,
          format: DEPOSIT_FORMAT,
          usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
          view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
    };
    let views = [texture("Deposit Atlas A"), texture("Deposit Atlas B")];
    let read_groups = [
      read_bind_group(device, &views[0]),
      read_bind_group(device, &views[1]),
    ];

    let blocks: Vec<[u32; 4]> = layout
      .blocks
      .iter()
      .map(|b| [b.origin[0], b.origin[1], b.resolution, 0])
      .collect();
    let blocks = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Deposit Blocks"),
      contents: bytemuck::cast_slice(&blocks),
      usage: wgpu::BufferUsages::STORAGE,
    });
    let splat_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: Some("Deposit Splat Bind Group"),
      layout: &splat_layout(device),
      entries: &[
        wgpu::BindGroupEntry {
          binding: 0,
          resource: population.particle_buffer().as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding: 1,
          resource: blocks.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding: 2,
          resource: population.flow_buffer().as_entire_binding(),
        },
      ],
    });

    let params = UniformBinding::new(
      device,
      "Deposit Params",
      wgpu::ShaderStages::VERTEX_FRAGMENT,
      DepositParams {
        atlas_size: ATLAS_SIZE as f32,
        radius: SPLAT_RADIUS_TEXELS,
        energy,
        decay,
        depth,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
      },
    );

    Some(Self {
      views,
      read_groups,
      splat_group,
      params,
      count,
      current: StdCell::new(0),
    })
  }

  /// The binding of the latest-written atlas: what the fill samples.
  pub fn read_bind_group(&self) -> &wgpu::BindGroup {
    &self.read_groups[self.current.get()]
  }
}

/// The splat footprint's radius in texels. Texel density is uniform per metric
/// area by construction, so this is a world-space radius in disguise -- and
/// small: the trail's continuity comes from consecutive splats overlapping,
/// not from a wide stamp.
const SPLAT_RADIUS_TEXELS: f32 = 1.5;

/// The Gaussian falloff of one splat, in units of the radius. Mirrors
/// `SPLAT_FALLOFF` in `deposit.wgsl`.
const SPLAT_FALLOFF: f64 = 3.5;

/// The texels one splat deposits in total at unit energy,
/// $integral e^(-k r^2 \/ r_s^2) dif A = pi r_s^2 \/ k$: what calibrates the
/// display's gain so the equilibrium trail brightness is a function of the
/// particle count and the atlas budget rather than a hand-tuned number.
pub fn splat_footprint_integral() -> f64 {
  std::f64::consts::PI * f64::from(SPLAT_RADIUS_TEXELS) * f64::from(SPLAT_RADIUS_TEXELS)
    / SPLAT_FALLOFF
}

/// The fade and splat pipelines, built once, stepping any batch.
pub struct DepositPass {
  fade: wgpu::RenderPipeline,
  splat: wgpu::RenderPipeline,
}

impl DepositPass {
  pub fn new(device: &wgpu::Device) -> Self {
    let shader = shader_module(device, "Deposit Shader", include_str!("deposit.wgsl"));
    let params_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
      label: Some("Deposit Params Layout"),
      entries: &[wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
          ty: wgpu::BufferBindingType::Uniform,
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      }],
    });
    let read = deposit_read_layout(device);
    let splat = splat_layout(device);
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some("Deposit Pipeline Layout"),
      bind_group_layouts: &[Some(&params_layout), Some(&read), Some(&splat)],
      immediate_size: 0,
    });

    let pipeline = |label: &str, vs: &str, fs: &str, blend: wgpu::BlendState| {
      device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
          module: &shader,
          entry_point: Some(vs),
          compilation_options: wgpu::PipelineCompilationOptions::default(),
          buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
          module: &shader,
          entry_point: Some(fs),
          compilation_options: wgpu::PipelineCompilationOptions::default(),
          targets: &[color_target(DEPOSIT_FORMAT, blend)],
        }),
        primitive: super::primitive(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
      })
    };

    Self {
      fade: pipeline(
        "Deposit Fade",
        "vs_fade",
        "fs_fade",
        wgpu::BlendState::REPLACE,
      ),
      // Additive: overlapping footprints accumulate into density, exactly as
      // the head specks do in radiance.
      splat: pipeline(
        "Deposit Splat",
        "vs_splat",
        "fs_splat",
        wgpu::BlendState {
          color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
          },
          alpha: wgpu::BlendComponent::OVER,
        },
      ),
    }
  }

  /// One step of the trail: fade the previous atlas into the other texture,
  /// splat this step's particle positions on top, flip which is current.
  ///
  /// Recorded once per advection step, after that step's dispatch, so the
  /// trail is a pure function of the step count -- a frame owing several steps
  /// records several of these, and a window and an exporter that reach the
  /// same count show the same trail.
  pub fn record(&self, encoder: &mut wgpu::CommandEncoder, batch: &DepositBatch) {
    let source = batch.current.get();
    let target = 1 - source;
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
      label: Some("Deposit Pass"),
      color_attachments: &[Some(wgpu::RenderPassColorAttachment {
        view: &batch.views[target],
        resolve_target: None,
        ops: wgpu::Operations {
          load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
          store: wgpu::StoreOp::Store,
        },
        depth_slice: None,
      })],
      depth_stencil_attachment: None,
      timestamp_writes: None,
      occlusion_query_set: None,
      multiview_mask: None,
    });
    pass.set_bind_group(0, batch.params.bind_group(), &[]);
    pass.set_bind_group(1, &batch.read_groups[source], &[]);
    pass.set_bind_group(2, &batch.splat_group, &[]);
    pass.set_pipeline(&self.fade);
    pass.draw(0..3, 0..1);
    pass.set_pipeline(&self.splat);
    pass.draw(0..6, 0..batch.count);
    drop(pass);
    batch.current.set(target);
  }
}
