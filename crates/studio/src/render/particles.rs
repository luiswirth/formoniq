//! The particle mark: the batch that holds an advection's state on the GPU, and
//! the pipeline that draws it. See `particles.wgsl`.
//!
//! The batch is the field's own geometry, exactly as a [`super::item::SegmentBatch`]
//! is: built when the field changes, held across frames. What makes it unlike
//! the others is that it is *written by the GPU* -- the advection's compute pass
//! owns its contents, and nothing reads them back. A particle's position is
//! never stored anywhere, in any buffer: the draw derives it from the weights
//! and the cell, at the last possible moment.

use bytemuck::Pod;
use wgpu::util::DeviceExt;

use super::{
  advect::{compute_bind_group_layout, AdvectParams, Cell, Particle},
  color_target, compilation_options, depth_stencil, primitive, shader_module, ssaa_constants,
  uniform::{FrameUniform, ParticleMaterial, UniformBinding, UniformPool},
  MASK_FORMAT,
};
use crate::advect::{AdvectBake, Seed};

/// How long a particle lives, in frame steps, and how widely those lives are
/// spread.
///
/// The spread is not cosmetic: with one shared lifetime the whole population
/// expires together and the field visibly breathes. The floor is long enough to
/// cross a good part of the manifold, so a particle traces a real curve rather
/// than blinking in place.
const LIFE_MIN: u32 = 240;
const LIFE_SPREAD: u32 = 240;

/// An advection's state on the GPU: the particles, and the baked field they
/// flow along.
pub struct ParticleBatch {
  compute_bind_group: wgpu::BindGroup,
  draw_bind_group: wgpu::BindGroup,
  /// The particle storage buffer itself, kept beyond its bind groups: the
  /// deposit's splat pass binds the same population through its own layout.
  particles: wgpu::Buffer,
  /// The baked flow levels, kept for the same reason: the splat derives each
  /// particle's displacement from the whole-step flow, as the draw pass does.
  flows: wgpu::Buffer,
  count: u32,
}

impl ParticleBatch {
  /// A batch of `count` particles flowing the baked field.
  ///
  /// The initial population *is* the seeds, drawn by the same rule a respawn
  /// draws by, with the same jittered lives. The first generation is therefore
  /// not a special case: the field does not open on a uniform cohort that dies
  /// at once.
  pub fn new(device: &wgpu::Device, bake: &AdvectBake, count: u32) -> Option<Self> {
    if bake.seeds.is_empty() || count == 0 {
      // A field that vanishes everywhere seeds nowhere. The empty population is
      // the honest answer, and it draws nothing rather than being a case the
      // caller has to exclude.
      return None;
    }
    let seeds: Vec<Particle> = bake.seeds.iter().map(particle_from_seed).collect();
    let initial: Vec<Particle> = (0..count)
      .map(|id| {
        let draw = hash_u32(id ^ hash_u32(0));
        let mut particle = seeds[draw as usize % seeds.len()];
        particle.epoch = 1;
        particle.life = LIFE_MIN + hash_u32(draw) % LIFE_SPREAD;
        particle
      })
      .collect();

    let particles = storage_buffer(device, "Advect Particles", &initial);
    let cells: Vec<Cell> = bake
      .neighbours
      .iter()
      .map(|&neighbour| Cell { neighbour })
      .collect();
    let cells = storage_buffer(device, "Advect Cells", &cells);
    let flows = storage_buffer(device, "Advect Flows", &bake.flows);
    let transitions = storage_buffer(device, "Advect Transitions", &bake.transitions);
    let positions = storage_buffer(device, "Advect Cell Positions", &bake.cell_positions);
    let seeds = storage_buffer(device, "Advect Seeds", &seeds);
    let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Advect Params"),
      contents: bytemuck::bytes_of(&AdvectParams {
        particle_count: count,
        seed_count: bake.seeds.len() as u32,
        life_min: LIFE_MIN,
        life_spread: LIFE_SPREAD,
        depth: bake.depth,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
      }),
      usage: wgpu::BufferUsages::UNIFORM,
    });

    let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: Some("Advect Bind Group"),
      layout: &compute_bind_group_layout(device),
      entries: &[
        binding(0, &particles),
        binding(1, &cells),
        binding(2, &flows),
        binding(3, &transitions),
        binding(4, &seeds),
        binding(5, &params),
      ],
    });
    let draw_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: Some("Particle Draw Bind Group"),
      layout: &draw_bind_group_layout(device),
      entries: &[
        binding(0, &particles),
        binding(1, &positions),
        binding(2, &flows),
        binding(3, &params),
      ],
    });

    Some(Self {
      compute_bind_group,
      draw_bind_group,
      particles,
      flows,
      count,
    })
  }

  /// The particle storage buffer, for the deposit's splat bind group.
  pub(crate) fn particle_buffer(&self) -> &wgpu::Buffer {
    &self.particles
  }

  /// The baked flow-level buffer, for the same.
  pub(crate) fn flow_buffer(&self) -> &wgpu::Buffer {
    &self.flows
  }

  pub(super) fn compute_bind_group(&self) -> &wgpu::BindGroup {
    &self.compute_bind_group
  }

  pub fn count(&self) -> u32 {
    self.count
  }

  fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
    pass.set_bind_group(2, &self.draw_bind_group, &[]);
    pass.draw(0..6, 0..self.count);
  }
}

/// What `particles.wgsl` declares at group 2: the particles, the cells' ambient
/// positions, and the flow the speck's own velocity is read off.
///
/// The draw shares the advection's `flows` and `params` rather than being handed
/// a velocity the compute pass wrote. That is the cheaper *and* the truer
/// arrangement: a stored velocity would be one step stale by the time it was
/// drawn, and would cost a buffer; the flow matrix is the field itself, so
/// re-applying it is exact and free.
fn draw_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
    label: Some("Particle Draw Bind Group Layout"),
    entries: &[
      storage(0),
      storage(1),
      storage(2),
      wgpu::BindGroupLayoutEntry {
        binding: 3,
        visibility: wgpu::ShaderStages::VERTEX,
        ty: wgpu::BindingType::Buffer {
          ty: wgpu::BufferBindingType::Uniform,
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      },
    ],
  })
}

/// The pipeline that draws a [`ParticleBatch`]: one instanced billboard quad per
/// particle, additively blended.
pub struct ParticlePass {
  pipeline: wgpu::RenderPipeline,
}

impl ParticlePass {
  pub fn new(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    frame: &UniformBinding<FrameUniform>,
    materials: &UniformPool<ParticleMaterial>,
    ssaa: u32,
  ) -> Self {
    let constants = ssaa_constants(ssaa);
    let shader = shader_module(device, "Particle Shader", include_str!("particles.wgsl"));
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some("Particle Pipeline Layout"),
      bind_group_layouts: &[
        Some(frame.layout()),
        Some(materials.layout()),
        Some(&draw_bind_group_layout(device)),
      ],
      immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
      label: Some("Particle Pipeline"),
      layout: Some(&layout),
      vertex: wgpu::VertexState {
        module: &shader,
        entry_point: Some("vs_main"),
        compilation_options: compilation_options(&constants),
        // No vertex streams: the quad's corners come from the vertex index and
        // the particle from the instance index, both read out of storage.
        buffers: &[],
      },
      fragment: Some(wgpu::FragmentState {
        module: &shader,
        entry_point: Some("fs_main"),
        compilation_options: compilation_options(&constants),
        // Additive, so overlapping specks accumulate into density instead of
        // occluding each other -- which is also what makes the population
        // order-independent, and hence sortable not at all.
        targets: &[
          color_target(
            format,
            wgpu::BlendState {
              color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
              },
              alpha: wgpu::BlendComponent::OVER,
            },
          ),
          // `Max`, unlike the color target: the mask is a coverage fraction,
          // not an accumulating radiance, so a texel a hundred specks pile onto
          // should read as the coverage of whichever covers it most, not their
          // sum saturating on the first handful.
          color_target(
            MASK_FORMAT,
            wgpu::BlendState {
              color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Max,
              },
              alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Max,
              },
            },
          ),
        ],
      }),
      primitive: primitive(),
      // Tests depth so the far side of a surface stays hidden; writes none,
      // because a translucent speck must not occlude the one behind it.
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
    batch: &ParticleBatch,
  ) {
    pass.set_pipeline(&self.pipeline);
    pass.set_bind_group(0, frame, &[]);
    pass.set_bind_group(1, material, &[]);
    batch.draw(pass);
  }
}

fn particle_from_seed(seed: &Seed) -> Particle {
  Particle {
    lambda: seed.bary,
    cell: seed.cell,
    life: 0,
    epoch: 0,
    _pad: 0,
  }
}

fn storage_buffer<T: Pod>(device: &wgpu::Device, label: &str, contents: &[T]) -> wgpu::Buffer {
  device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    label: Some(label),
    contents: bytemuck::cast_slice(contents),
    usage: wgpu::BufferUsages::STORAGE,
  })
}

fn binding(index: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
  wgpu::BindGroupEntry {
    binding: index,
    resource: buffer.as_entire_binding(),
  }
}

/// The `lowbias32` of `advect.wgsl` and of the bake's seeding, so the first
/// generation is drawn by the rule every later one is.
fn hash_u32(value: u32) -> u32 {
  let mut h = value;
  h ^= h >> 16;
  h = h.wrapping_mul(0x7feb_352d);
  h ^= h >> 15;
  h = h.wrapping_mul(0x846c_a68b);
  h ^= h >> 16;
  h
}
