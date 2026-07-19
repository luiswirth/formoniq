//! The renderer: pipelines, transient targets and the frame graph, recorded
//! into any caller-supplied `TextureView`.
//!
//! Nothing here knows about a window, a surface or a clock -- the target
//! format, the target size and the time are all inputs -- so the interactive
//! viewer and a headless export drive one frame graph and cannot drift.

pub mod advect;
pub mod bloom;
pub mod camera;
pub mod context;
pub mod deposit;
pub mod downsample;
pub mod fill;
pub mod glyph;
pub mod item;
pub mod particles;
pub mod renderer;
pub mod segments;
pub mod uniform;

pub use context::GpuContext;
pub use renderer::{FrameView, Renderer};

/// Float depth, which reversed-Z requires: the precision argument for reversing
/// is an argument about where floats are dense, and a unorm target, being
/// uniform, gains nothing from it.
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// The far plane, since depth is reversed. Exactly representable, so an
/// unoccluded fragment at any depth strictly beats the cleared value.
const DEPTH_CLEAR: f32 = 0.0;

/// The format every scene pass draws into, and the bloom chain with it.
///
/// Float, and therefore *unbounded*, which is the whole of the point: a dense
/// trail lifts the surface's radiance above 1, so a filament where the flow
/// bunches carries many times a still region's light. An 8-bit target saturates
/// at 1.0 and throws that away -- a bright filament and a faint one come out the
/// same white -- so the overflow that is meant to be the image is destroyed at
/// the blend, before any pass could shade it. Here it survives to the resolve,
/// which is the only place that has to fit it into a display.
///
/// The range is what matters, not the precision: a 16-bit *unorm* target would
/// give 256 times the levels and clip at 1.0 just the same.
pub const SCENE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The format of the unbounded-coverage mask the scene pass writes alongside
/// [`SCENE_FORMAT`]: how much of a texel's radiance came from a mark that can
/// actually exceed 1, as opposed to one that is clamped to `[0, 1]` by
/// construction (a colormapped fill, a wireframe or glyph ribbon).
///
/// Only a deposit-lifted fill overflows -- see the invariant in `bloom.wgsl` --
/// so this is the material-specific fact the resolve needs to decide, per pixel,
/// whether the tone curve or a plain clamp is the correct crossing. A single
/// scalar is enough: coverage, not radiance, so `R8Unorm` rather than a float
/// format.
pub const MASK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

/// The supersampling factor per axis a [`Renderer`] uses when its caller names
/// none: what an interactive window can afford at frame rate. An export is not
/// bound by that budget and passes its own.
///
/// Supersampling is applied per axis (so the square of this in pixel count) to
/// the scene pass, before a box filter downsamples it back to the target. MSAA
/// was rejected: a single supersampled offscreen target antialiases the fill
/// and the segment marks uniformly, at the cost of the extra fill rate.
pub const DEFAULT_SSAA_SCALE: u32 = 2;

/// The supersampling factor for a display of device-pixel ratio `dpr`.
///
/// Supersampling models the observation -- it antialiases what the eye resolves
/// -- and a dense display already oversamples the layout by its own pixel ratio.
/// The quantity that reads on screen is therefore the *total* samples per layout
/// pixel per axis, `dpr * ssaa`; past [`DEFAULT_SSAA_SCALE`] of them the extra
/// draws detail the display cannot show while the scene fill grows as its
/// square. So hold that product near the target the desktop uses rather than the
/// multiplier: a `dpr = 1` monitor keeps the full factor, a `dpr = 2` or `3`
/// phone drops toward the floor of `1`. This is where a phone's *always-on*
/// scene cost lives -- a `dpr = 3` panel at the fixed factor renders every scene
/// pass at nine times the pixels the screen resolves.
///
/// The scene target alone rides this; the UI still composites at full device
/// resolution (a later pass onto the surface), so its text stays crisp.
pub fn ssaa_for_dpr(dpr: f32) -> u32 {
  #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
  let scaled = (f64::from(DEFAULT_SSAA_SCALE) / f64::from(dpr.max(1.0))).round() as u32;
  scaled.clamp(1, DEFAULT_SSAA_SCALE)
}

/// The WGSL source of one pass: the shared preamble (types, pure functions)
/// followed by the body. Concatenated because WGSL has no `#include`; the shader
/// test validates this same concatenation, so a preamble/body mismatch fails
/// `cargo test` rather than pipeline creation.
fn shader_source(body: &str) -> String {
  format!("{PREAMBLE}\n{body}")
}

/// The preamble alone: every shader's prefix, and a WGSL module in its own
/// right, which is what lets the uniform structs it declares be laid out and
/// checked against their Rust mirrors without a pass around them.
const PREAMBLE: &str = include_str!("preamble.wgsl");

/// The `SSAA_SCALE` declaration prepended to the one shader that reads it (the
/// downsample resolve, whose box filter divides by it). Baked as a plain `const`
/// from the same `ssaa` that sizes the targets, so the filter's divisor and the
/// target allocation are one number -- NOT a pipeline-overridable `override`,
/// which WebGPU on WebKit fails to specialize ("Vertex library failed
/// creation"). The factor is fixed for a pipeline's life anyway, so nothing is
/// lost by baking it at shader-build instead of pipeline-build.
fn ssaa_prelude(ssaa: u32) -> String {
  format!("const SSAA_SCALE: i32 = {ssaa};\n")
}

fn shader_module(device: &wgpu::Device, label: &str, body: &str) -> wgpu::ShaderModule {
  device.create_shader_module(wgpu::ShaderModuleDescriptor {
    label: Some(label),
    source: wgpu::ShaderSource::Wgsl(shader_source(body).into()),
  })
}

/// The depth state the scene passes share: the fill writes it; the
/// alpha-blended segment marks only test against it.
///
/// `Greater`, and the pass clears to [`DEPTH_CLEAR`], because depth is reversed:
/// nearer is *larger*. See [`OPENGL_TO_REVERSED_WGPU_MATRIX`] for why. The three
/// -- the projection's sign, the comparison, the clear -- are one decision, and
/// any two of them agreeing without the third is a scene drawn inside out.
///
/// [`OPENGL_TO_REVERSED_WGPU_MATRIX`]: camera::OPENGL_TO_REVERSED_WGPU_MATRIX
fn depth_stencil(write: bool) -> wgpu::DepthStencilState {
  wgpu::DepthStencilState {
    format: DEPTH_FORMAT,
    depth_write_enabled: Some(write),
    depth_compare: Some(wgpu::CompareFunction::Greater),
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
      ("glyph.wgsl", include_str!("glyph.wgsl")),
      ("downsample.wgsl", include_str!("downsample.wgsl")),
      ("advect.wgsl", include_str!("advect.wgsl")),
      ("bloom.wgsl", include_str!("bloom.wgsl")),
      ("deposit.wgsl", include_str!("deposit.wgsl")),
    ];
    for (name, body) in bodies {
      // The downsample body reads `SSAA_SCALE`, which the pipeline bakes in as a
      // `const` (see `ssaa_prelude`); validate the same composed source.
      let composed = if *name == "downsample.wgsl" {
        format!("{}{body}", super::ssaa_prelude(super::DEFAULT_SSAA_SCALE))
      } else {
        (*body).to_string()
      };
      let source = super::shader_source(&composed);
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
      ("GlyphMaterial", size_of::<super::uniform::GlyphMaterial>()),
      ("Post", size_of::<super::uniform::PostUniform>()),
      ("Particle", size_of::<super::advect::Particle>()),
      ("Cell", size_of::<super::advect::Cell>()),
      ("AdvectParams", size_of::<super::advect::AdvectParams>()),
      ("DepositParams", size_of::<super::deposit::DepositParams>()),
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
