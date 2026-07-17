//! The renderer's uniforms: the binding helper, the values themselves, and the
//! per-item pools.
//!
//! Every uniform in this crate is the same shape -- a `Pod` struct at
//! `@group(_) @binding(0)` -- so the buffer/layout/bind-group triple is written
//! once here rather than at each pipeline. Each struct mirrors a WGSL struct of
//! the same name in `preamble.wgsl`, field for field.
//!
//! The split is frame versus item: [`FrameUniform`] is where and when the scene
//! is seen from, and every pipeline binds it at group 0; a *material* is what
//! one [`super::item::RenderItem`] is drawn like, at group 1, one binding per
//! item out of a [`UniformPool`].

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::camera::Camera;

/// A `T`-valued uniform bound at `@group(_) @binding(0)`: the buffer that backs
/// it, the layout a pipeline declares it with, and the bind group a pass sets.
///
/// The explicit `_pad` fields on the values below are why `T` may be taken at
/// face value: WGSL rounds a uniform struct up to a 16-byte multiple and
/// `#[repr(C)]` does not, so the padding is written out rather than left to the
/// two languages to agree on by luck.
///
/// Padding is written in *scalars*, never a vector: WGSL aligns a `vec3<f32>`
/// to 16 bytes where Rust aligns `[f32; 3]` to 4, so a `vec3` tail does not pad
/// a struct closed -- it opens a fresh slot and pads it wider than its mirror.
/// The rule that makes the mirroring real is that the two declarations agree on
/// *bytes*, which reading them side by side does not establish and
/// `render::tests::uniform_layouts_match_wgsl` does.
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

/// One `UniformBinding<T>` per item of a draw list, grown on demand and reused
/// across frames.
///
/// A material belongs to an item and a bind group belongs to a buffer, so a
/// frame drawing $m$ items of one kind needs $m$ bindings. The pool is what lets
/// that count be a property of the scene -- several manifolds, several overlays
/// -- rather than a fixed set the renderer declares up front.
pub struct UniformPool<T: Pod + Default> {
  label: &'static str,
  visibility: wgpu::ShaderStages,
  bindings: Vec<UniformBinding<T>>,
}

impl<T: Pod + Default> UniformPool<T> {
  pub fn new(device: &wgpu::Device, label: &'static str, visibility: wgpu::ShaderStages) -> Self {
    // One binding up front: the layout a pipeline is built with is a property
    // of a binding here, and pipelines exist before any item does.
    let bindings = vec![UniformBinding::new(device, label, visibility, T::default())];
    Self {
      label,
      visibility,
      bindings,
    }
  }

  /// The layout every binding of this pool shares.
  pub fn layout(&self) -> &wgpu::BindGroupLayout {
    self.bindings[0].layout()
  }

  /// Writes `value` into the `index`th binding, allocating up to it if needed.
  pub fn write(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, index: usize, value: T) {
    while self.bindings.len() <= index {
      self.bindings.push(UniformBinding::new(
        device,
        self.label,
        self.visibility,
        T::default(),
      ));
    }
    self.bindings[index].write(queue, value);
  }

  pub fn bind_group(&self, index: usize) -> &wgpu::BindGroup {
    self.bindings[index].bind_group()
  }
}

/// Where and when the scene is seen from: the one uniform every pipeline binds,
/// at group 0.
///
/// Time is here, not in a material, because it is the frame's -- the windowed
/// loop passes wall-clock seconds, an exporter passes $t_k = k \/ "fps"$ -- while
/// the frequency $omega$ it is multiplied by belongs to the field on display,
/// and so lives in that item's material.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct FrameUniform {
  view_proj: [[f32; 4]; 4],
  /// World-space eye position, `w` unused: the billboard construction needs it
  /// directly, not only the matrix it feeds into.
  eye: [f32; 4],
  time: f32,
  _pad: [f32; 3],
}

impl Default for FrameUniform {
  fn default() -> Self {
    Self {
      view_proj: nalgebra::Matrix4::identity().into(),
      eye: [0.0; 4],
      time: 0.0,
      _pad: [0.0; 3],
    }
  }
}

impl FrameUniform {
  pub fn new(camera: &Camera, time: f32) -> Self {
    let eye = camera.eye();
    Self {
      view_proj: camera.build_view_projection_matrix().into(),
      eye: [eye.x, eye.y, eye.z, 1.0],
      time,
      _pad: [0.0; 3],
    }
  }
}

/// How a filled surface is drawn: the colormap range it normalizes against and
/// the standing wave that displaces and pulses it.
///
/// A field with no eigenvalue passes `wave_omega = 0` and `wave_amplitude = 0`,
/// so $cos(0) = 1$ and the same code draws it static.
///
/// Diverging vs. sequential is a property of the *field* -- a signed quantity
/// centered at zero, or an unsigned magnitude -- not a global shader choice, so
/// it travels as material data alongside the range it is read against. `1.0`
/// for diverging (`min_val`/`max_val` symmetric about zero), `0.0` for
/// sequential. A scalar `f32`, never a `bool`: `Pod` requires it, and WGSL has
/// no bool uniform either.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct SurfaceMaterial {
  pub min_val: f32,
  pub max_val: f32,
  pub wave_amplitude: f32,
  pub wave_omega: f32,
  pub diverging: f32,
}

/// How the scene's radiance reaches the display: the WGSL `Post`.
///
/// Not a material -- it is a property of no mark, and of the frame as a whole --
/// so it arrives through [`super::FrameView`] beside `time`, and both callers
/// state it. Held by the renderer as a value the window and an exporter cannot
/// silently disagree on.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct PostUniform {
  pub exposure: f32,
  /// Zero disables bloom by arithmetic, so the chain can be skipped without the
  /// image depending on whether it was.
  pub bloom_intensity: f32,
  /// 1.0 for the filmic curve, 0.0 for a hard clamp.
  pub tonemap: f32,
  pub _pad0: f32,
}

/// How a particle mark is drawn: the ink and radius of one advected speck.
///
/// No wave: the standing-wave clock displaces a surface and fades a ribbon, but
/// a particle already carries the field's own motion. Riding a second
/// oscillation on top of the advection would be two clocks on one mark.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct ParticleMaterial {
  /// The speck's ink: `rgb` plus the opacity at its own center, before the
  /// Gaussian falloff.
  pub color: [f32; 4],
  /// The speck's radius in world space, on the same object-intrinsic scale
  /// every other mark's width uses.
  pub radius_world: f32,
  /// The ambient distance a particle at the field's peak magnitude covers in one
  /// step: the normalization that turns a speck's own speed into a fraction of
  /// the field's range, so the tint means the same thing whatever the cochain's
  /// units.
  pub speed_scale: f32,
  /// How many radii a peak-speed speck stretches along its motion.
  pub stretch: f32,
  pub _pad0: f32,
}

/// How a segment mark is drawn. One material serves the wireframe, a line
/// field's ribbons and a 1-manifold's own cells -- one technique at different
/// ink and width, not three passes.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct SegmentMaterial {
  /// The mark's ink: `rgb` plus the base opacity every fragment starts at.
  pub color: [f32; 4],
  /// Half-width in world space, in the same units the mesh's own coordinates
  /// are in -- not a pixel count, so a line reads the same thickness whether the
  /// mesh fills the screen or sits in a corner of it.
  pub half_width_world: f32,
  /// Opacity at the standing wave's node, relative to the crest: an eigenmode's
  /// ribbons fade where the field vanishes and the curves are meaningless. A
  /// mark that does not fade passes 1.
  pub fade_floor: f32,
  pub wave_amplitude: f32,
  pub wave_omega: f32,
}
