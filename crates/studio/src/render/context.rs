//! The GPU device the renderer draws with, without any claim about where the
//! result lands: a window's swapchain and an offscreen export target both hand
//! over the same pair.

/// A `wgpu` device and its queue. The renderer takes one of these plus a target
/// format; nothing about a surface, a window or a clock enters here.
pub struct GpuContext {
  pub device: wgpu::Device,
  pub queue: wgpu::Queue,
}
