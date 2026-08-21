//! Laws for [`studio::render`]'s shaders: every WGSL source parses and
//! validates against naga's own frontend, the advect dispatch's workgroup
//! count matches the shader's own `@workgroup_size`, and every uniform's
//! WGSL struct has the size its `#[repr(C)]` Rust mirror does.

use formoniq_studio::render::{
  DEFAULT_SSAA_SCALE, PREAMBLE, advect, deposit, shader_source, ssaa_prelude, uniform, volume,
};
use std::mem::size_of;

macro_rules! shader_src {
  ($name:literal) => {
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/render/", $name))
  };
}
/// Every WGSL source in this module parses and validates against naga's own
/// frontend, the one wgpu itself uses to build a pipeline, so a broken
/// shader fails `cargo test` instead of the pipeline-creation panic at
/// startup a syntax or type error would otherwise cause. Validated as the
/// preamble/body concatenation the pipelines are actually built from, never
/// the body alone.
#[test]
fn shaders_parse_and_validate() {
  let bodies: &[(&str, &str)] = &[
    ("fill.wgsl", shader_src!("fill.wgsl")),
    ("segments.wgsl", shader_src!("segments.wgsl")),
    ("points.wgsl", shader_src!("points.wgsl")),
    ("glyph.wgsl", shader_src!("glyph.wgsl")),
    ("downsample.wgsl", shader_src!("downsample.wgsl")),
    ("advect.wgsl", shader_src!("advect.wgsl")),
    ("bloom.wgsl", shader_src!("bloom.wgsl")),
    ("deposit.wgsl", shader_src!("deposit.wgsl")),
    ("volume.wgsl", shader_src!("volume.wgsl")),
  ];
  for (name, body) in bodies {
    // The downsample body reads `SSAA_SCALE`, which the pipeline bakes in as a
    // `const` (see `ssaa_prelude`); validate the same composed source.
    let composed = if *name == "downsample.wgsl" {
      format!("{}{body}", ssaa_prelude(DEFAULT_SSAA_SCALE))
    } else {
      (*body).to_string()
    };
    let source = shader_source(&composed);
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

/// The dispatch's workgroup count is derived from
/// [`advect::WORKGROUP_SIZE`], so it is the shader's own `@workgroup_size`
/// that decides whether the population is covered. A larger declared size
/// leaves the tail of the population unstepped, which reads as a patch of
/// particles frozen in the flow rather than as a failure, so the two are
/// checked against each other rather than against a written number.
#[test]
fn advect_workgroup_size_matches_dispatch() {
  let source = shader_source(shader_src!("advect.wgsl"));
  let module = naga::front::wgsl::parse_str(&source).expect("advect failed to parse");
  let entry = module
    .entry_points
    .iter()
    .find(|e| e.name == "advect")
    .expect("advect has no `advect` entry point");
  assert_eq!(entry.workgroup_size, [advect::WORKGROUP_SIZE, 1, 1]);
}

/// Every uniform's WGSL struct has the size its `#[repr(C)]` Rust mirror
/// does, as naga lays it out, the same computation wgpu validates a bind
/// group against.
///
/// The two languages do not share an alignment rule (a WGSL vector is
/// 16-aligned. A Rust array is aligned as its element), so mirrors it field"
/// for field" is a claim about bytes that reading the two declarations
/// side by side does not check. A mismatch is otherwise invisible until a
/// draw call fails validation at runtime, which is exactly the error this
/// test exists to turn into a compile-time-adjacent one.
#[test]
fn uniform_layouts_match_wgsl() {
  use naga::proc::Layouter;

  let module = naga::front::wgsl::parse_str(PREAMBLE).expect("preamble failed to parse");
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
    ("Frame", size_of::<uniform::FrameUniform>()),
    ("SurfaceMaterial", size_of::<uniform::SurfaceMaterial>()),
    ("SegmentMaterial", size_of::<uniform::SegmentMaterial>()),
    ("GlyphMaterial", size_of::<uniform::GlyphMaterial>()),
    ("Post", size_of::<uniform::PostUniform>()),
    ("Particle", size_of::<advect::Particle>()),
    ("Cell", size_of::<advect::Cell>()),
    ("AdvectParams", size_of::<advect::AdvectParams>()),
    ("DepositParams", size_of::<deposit::DepositParams>()),
    ("VolumeMaterial", size_of::<volume::VolumeMaterial>()),
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
