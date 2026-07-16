//! The renderer's uniforms: the binding helper, the values themselves, and the
//! bundle every pass reads them from.
//!
//! Every uniform in this crate is the same shape -- a `Pod` struct at
//! `@group(_) @binding(0)` -- so the buffer/layout/bind-group triple is written
//! once here rather than at each pipeline. Each struct mirrors a WGSL struct of
//! the same name in `preamble.wgsl`, field for field.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::camera::CameraUniform;

/// A `T`-valued uniform bound at `@group(_) @binding(0)`: the buffer that backs
/// it, the layout a pipeline declares it with, and the bind group a pass sets.
///
/// The explicit `_pad` fields on the values below are why `T` may be taken at
/// face value: WGSL rounds a uniform struct up to a 16-byte multiple and
/// `#[repr(C)]` does not, so the padding is written out rather than left to the
/// two languages to agree on by luck.
pub struct UniformBinding<T: Pod> {
  buffer: wgpu::Buffer,
  layout: wgpu::BindGroupLayout,
  bind_group: wgpu::BindGroup,
  _value: std::marker::PhantomData<T>,
}

impl<T: Pod> UniformBinding<T> {
  /// A uniform holding `value`, visible to `visibility`.
  pub fn new(device: &wgpu::Device, label: &str, visibility: wgpu::ShaderStages, value: T) -> Self {
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some(label),
      contents: bytemuck::bytes_of(&value),
      usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
      label: Some(label),
      entries: &[wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility,
        ty: wgpu::BindingType::Buffer {
          ty: wgpu::BufferBindingType::Uniform,
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      }],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: Some(label),
      layout: &layout,
      entries: &[wgpu::BindGroupEntry {
        binding: 0,
        resource: buffer.as_entire_binding(),
      }],
    });
    Self {
      buffer,
      layout,
      bind_group,
      _value: std::marker::PhantomData,
    }
  }

  pub fn write(&self, queue: &wgpu::Queue, value: T) {
    queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&value));
  }

  pub fn layout(&self) -> &wgpu::BindGroupLayout {
    &self.layout
  }

  pub fn bind_group(&self) -> &wgpu::BindGroup {
    &self.bind_group
  }
}

/// Per-frame standing-wave state: $u(t) = "amplitude" dot "value" dot cos(omega t)$,
/// displacing each vertex along its own normal and pulsing its color by the
/// same factor.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct WaveUniform {
  pub time: f32,
  pub amplitude: f32,
  pub omega: f32,
  pub _pad: f32,
}

/// Colormap normalization range for the field currently on display.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct BoundsUniform {
  pub min_val: f32,
  pub max_val: f32,
  pub _pad1: f32,
  pub _pad2: f32,
}

/// A segment pass's world-space half-width, in the same units the mesh's own
/// coordinates are in. One type, two bindings: the wireframe and the streamline
/// ribbons are the same billboard-quad technique at two different widths.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct SegmentWidth {
  pub half_width_world: f32,
  pub _pad0: f32,
  pub _pad1: f32,
  pub _pad2: f32,
}

/// Every uniform the frame graph binds, in one place: created once, rewritten
/// per frame, and handed to each pass both at pipeline creation (for the
/// layouts) and at draw time (for the bind groups).
pub struct Uniforms {
  pub camera: UniformBinding<CameraUniform>,
  pub wave: UniformBinding<WaveUniform>,
  pub bounds: UniformBinding<BoundsUniform>,
  pub wireframe_width: UniformBinding<SegmentWidth>,
  pub streamline_width: UniformBinding<SegmentWidth>,
}

impl Uniforms {
  pub fn new(device: &wgpu::Device) -> Self {
    use wgpu::ShaderStages as Stages;
    Self {
      camera: UniformBinding::new(device, "camera", Stages::VERTEX, CameraUniform::new()),
      // Also visible in the fragment stage: the fill pulses its colormap by the
      // same $cos(sqrt(lambda) t)$ that displaces it, and the ribbons fade by
      // it.
      wave: UniformBinding::new(
        device,
        "wave",
        Stages::VERTEX_FRAGMENT,
        WaveUniform::default(),
      ),
      bounds: UniformBinding::new(device, "bounds", Stages::FRAGMENT, BoundsUniform::default()),
      wireframe_width: UniformBinding::new(
        device,
        "wireframe width",
        Stages::VERTEX,
        SegmentWidth::default(),
      ),
      streamline_width: UniformBinding::new(
        device,
        "streamline width",
        Stages::VERTEX,
        SegmentWidth::default(),
      ),
    }
  }
}
