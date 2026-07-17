//! The advection pass: the compute pipeline that flows every particle along the
//! field, and the mirrors of the types it reads. See `advect.wgsl`.
//!
//! **The step count is the caller's, and that is the whole of this pass's
//! relationship with time.** A frame's step is a fixed quantity of field time,
//! because the bake exponentiated one; what varies is how many of them a frame
//! is worth. So [`AdvectPass::dispatch`] takes a number, an interactive loop
//! derives it from an accumulated wall clock, and an exporter derives it from
//! the instant it means to render. Neither can drift from the other, because
//! neither owns a clock this pass can see.
//!
//! That is the stateful reading of the renderer's own contract. Time is still
//! an argument, but a simulation cannot be *evaluated at* an instant the way a
//! standing wave can -- it can only be *stepped to* one. So the argument is a
//! count of steps rather than seconds, and the pass's deterministic seeding is
//! what makes a given count mean the same picture to either caller.

use bytemuck::{Pod, Zeroable};

use super::particles::ParticleBatch;

/// A particle: the WGSL `Particle` of `preamble.wgsl`, byte for byte.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Particle {
  pub lambda: [f32; 4],
  pub cell: u32,
  pub life: u32,
  pub epoch: u32,
  pub _pad: u32,
}

/// One cell's neighbours: the WGSL `Cell`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Cell {
  pub neighbour: [u32; 4],
}

/// The WGSL `AdvectParams`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct AdvectParams {
  pub particle_count: u32,
  pub seed_count: u32,
  pub life_min: u32,
  pub life_spread: u32,
  pub depth: u32,
  pub _pad0: u32,
  pub _pad1: u32,
  pub _pad2: u32,
}

/// Must match `@workgroup_size` in `advect.wgsl`.
pub(super) const WORKGROUP_SIZE: u32 = 64;

/// The bindings `advect.wgsl` declares, as a layout.
///
/// A free function because both sides need it and neither owns the other: the
/// pipeline is built from it once, and every [`ParticleBatch`] builds its bind
/// group from it. wgpu compares layouts structurally, so the two calls agree.
pub(super) fn compute_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
  let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
    binding,
    visibility: wgpu::ShaderStages::COMPUTE,
    ty: wgpu::BindingType::Buffer {
      ty: wgpu::BufferBindingType::Storage { read_only },
      has_dynamic_offset: false,
      min_binding_size: None,
    },
    count: None,
  };
  device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
    label: Some("Advect Bind Group Layout"),
    entries: &[
      storage(0, false),
      storage(1, true),
      storage(2, true),
      storage(3, true),
      storage(4, true),
      wgpu::BindGroupLayoutEntry {
        binding: 5,
        visibility: wgpu::ShaderStages::COMPUTE,
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

/// The compute pipeline alone: built once, run over any batch.
pub struct AdvectPass {
  pipeline: wgpu::ComputePipeline,
}

impl AdvectPass {
  pub fn new(device: &wgpu::Device) -> Self {
    let shader = super::shader_module(device, "Advect Shader", include_str!("advect.wgsl"));
    let layout = compute_bind_group_layout(device);
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some("Advect Pipeline Layout"),
      bind_group_layouts: &[Some(&layout)],
      immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
      label: Some("Advect Pipeline"),
      layout: Some(&pipeline_layout),
      module: &shader,
      entry_point: Some("advect"),
      compilation_options: wgpu::PipelineCompilationOptions::default(),
      cache: None,
    });
    Self { pipeline }
  }

  /// Advance `batch` by `steps` frame steps.
  ///
  /// Each step is its own dispatch, because it reads the state the one before
  /// it wrote and a workgroup barrier does not order across the whole
  /// population. They are cheap: one mat-vec per particle per dyadic level.
  pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, batch: &ParticleBatch, steps: u32) {
    let workgroups = batch.count().div_ceil(WORKGROUP_SIZE);
    if workgroups == 0 {
      return;
    }
    for _ in 0..steps {
      let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("Advect Pass"),
        timestamp_writes: None,
      });
      pass.set_pipeline(&self.pipeline);
      pass.set_bind_group(0, batch.compute_bind_group(), &[]);
      pass.dispatch_workgroups(workgroups, 1, 1);
    }
  }
}
