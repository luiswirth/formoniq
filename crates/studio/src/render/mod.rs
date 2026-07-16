//! The renderer: pipelines, transient targets and the frame graph, recorded
//! into any caller-supplied `TextureView`.
//!
//! Nothing here knows about a window, a surface or a clock -- the target
//! format, the target size and the time are all inputs -- so the interactive
//! viewer and a headless export drive one frame graph and cannot drift.

pub mod camera;
pub mod context;
pub mod downsample;
pub mod fill;
pub mod item;
pub mod renderer;
pub mod segments;
pub mod uniform;

pub use context::GpuContext;
pub use renderer::{FrameView, Renderer};

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Supersampling factor applied per axis (so 4x the pixel count) to the scene
/// pass before a box filter downsamples it back to the target. MSAA was
/// rejected: a single supersampled offscreen target antialiases the fill and
/// the segment marks uniformly, at the cost of the extra fill rate. Reaches the
/// WGSL side as the preamble's `SSAA_SCALE` override, so the box filter and the
/// target allocation are one number rather than two that must be kept in sync.
const SSAA_SCALE: u32 = 2;

/// The WGSL source of one pass: the shared preamble (types, pure functions, the
/// `SSAA_SCALE` override) followed by the body. Concatenated because WGSL has
/// no `#include`; the shader test validates this same concatenation, so a
/// preamble/body mismatch fails `cargo test` rather than pipeline creation.
fn shader_source(body: &str) -> String {
  format!("{PREAMBLE}\n{body}")
}

/// The preamble alone: every shader's prefix, and a WGSL module in its own
/// right, which is what lets the uniform structs it declares be laid out and
/// checked against their Rust mirrors without a pass around them.
const PREAMBLE: &str = include_str!("preamble.wgsl");

/// The pipeline-constant assignment every pipeline here is built with: the WGSL
/// `SSAA_SCALE` override, from the Rust constant.
fn compilation_options<'a>() -> wgpu::PipelineCompilationOptions<'a> {
  wgpu::PipelineCompilationOptions {
    constants: &[("SSAA_SCALE", SSAA_SCALE as f64)],
    ..Default::default()
  }
}

fn shader_module(device: &wgpu::Device, label: &str, body: &str) -> wgpu::ShaderModule {
  device.create_shader_module(wgpu::ShaderModuleDescriptor {
    label: Some(label),
    source: wgpu::ShaderSource::Wgsl(shader_source(body).into()),
  })
}

/// The depth state the scene passes share: the fill writes it; the
/// alpha-blended segment marks only test against it.
fn depth_stencil(write: bool) -> wgpu::DepthStencilState {
  wgpu::DepthStencilState {
    format: DEPTH_FORMAT,
    depth_write_enabled: Some(write),
    depth_compare: Some(wgpu::CompareFunction::Less),
    stencil: wgpu::StencilState::default(),
    bias: wgpu::DepthBiasState::default(),
  }
}

/// The triangle-list primitive state shared by every pipeline here: nothing is
/// culled, since a surface is viewed from both sides.
fn primitive() -> wgpu::PrimitiveState {
  wgpu::PrimitiveState {
    topology: wgpu::PrimitiveTopology::TriangleList,
    front_face: wgpu::FrontFace::Ccw,
    cull_mode: None,
    ..Default::default()
  }
}

fn color_target(
  format: wgpu::TextureFormat,
  blend: wgpu::BlendState,
) -> Option<wgpu::ColorTargetState> {
  Some(wgpu::ColorTargetState {
    format,
    blend: Some(blend),
    write_mask: wgpu::ColorWrites::ALL,
  })
}

#[cfg(test)]
mod tests {
  /// Every WGSL source in this module parses and validates against naga's own
  /// frontend -- the one wgpu itself uses to build a pipeline -- so a broken
  /// shader fails `cargo test` instead of the pipeline-creation panic at
  /// startup a syntax or type error would otherwise cause. Validated as the
  /// preamble/body concatenation the pipelines are actually built from, never
  /// the body alone.
  #[test]
  fn shaders_parse_and_validate() {
    let bodies: &[(&str, &str)] = &[
      ("fill.wgsl", include_str!("fill.wgsl")),
      ("segments.wgsl", include_str!("segments.wgsl")),
      ("downsample.wgsl", include_str!("downsample.wgsl")),
    ];
    for (name, body) in bodies {
      let source = super::shader_source(body);
      let module = naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("{name} failed to parse: {e}"));
      naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
      )
      .validate(&module)
      .unwrap_or_else(|e| panic!("{name} failed to validate: {e}"));
    }
  }

  /// Every uniform's WGSL struct has the size its `#[repr(C)]` Rust mirror
  /// does, as naga lays it out -- the same computation wgpu validates a bind
  /// group against.
  ///
  /// The two languages do not share an alignment rule (a WGSL vector is
  /// 16-aligned; a Rust array is aligned as its element), so "mirrors it field
  /// for field" is a claim about bytes that reading the two declarations
  /// side by side does not check. A mismatch is otherwise invisible until a
  /// draw call fails validation at runtime, which is exactly the error this
  /// test exists to turn into a compile-time-adjacent one.
  #[test]
  fn uniform_layouts_match_wgsl() {
    use naga::proc::Layouter;

    let module = naga::front::wgsl::parse_str(super::PREAMBLE).expect("preamble failed to parse");
    naga::valid::Validator::new(
      naga::valid::ValidationFlags::all(),
      naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("preamble failed to validate");

    let mut layouter = Layouter::default();
    layouter
      .update(module.to_ctx())
      .expect("preamble failed to lay out");

    let expected: &[(&str, usize)] = &[
      ("Frame", size_of::<super::uniform::FrameUniform>()),
      (
        "SurfaceMaterial",
        size_of::<super::uniform::SurfaceMaterial>(),
      ),
      (
        "SegmentMaterial",
        size_of::<super::uniform::SegmentMaterial>(),
      ),
    ];

    for (name, rust_size) in expected {
      let (handle, _) = module
        .types
        .iter()
        .find(|(_, ty)| ty.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("preamble declares no struct `{name}`"));
      let wgsl_size = layouter[handle].size as usize;
      assert_eq!(
        wgsl_size, *rust_size,
        "`{name}`: WGSL lays it out at {wgsl_size} bytes, Rust at {rust_size}"
      );
    }
  }
}
