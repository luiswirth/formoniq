//! Laws for [`studio::export`]: the derived supersampling factor stays
//! inside its pixel budget and its scale ceiling at every resolution and
//! only falls as the resolution rises, and a headless render of a known
//! view actually draws the scene, exercising the particle-advection
//! pipeline of a grade-1 field rather than clearing and presenting the
//! background.

use formoniq_studio::{
  demos::triforce_examples,
  export::{
    Displayed, EXPORT_FORMAT, EXPORT_SSAA_PIXEL_BUDGET, EXPORT_SSAA_SCALE_MAX, ExportSpec,
    ExportTarget, export_ssaa_scale, headless_context, render_at,
  },
  gallery::{MeshSource, Study},
  render::Renderer,
};

/// The derived supersampling factor stays inside the budget it exists to
/// respect, at every resolution: it never allocates more than
/// [`EXPORT_SSAA_PIXEL_BUDGET`] unless one sample per pixel already does, it
/// never exceeds [`EXPORT_SSAA_SCALE_MAX`], and it is never 0 (which would
/// allocate an empty target). Swept over the range a caller reaches, from a
/// thumbnail to well past 8K, rather than the one size the default happens to
/// be.
#[test]
fn the_export_supersampling_factor_respects_its_budget() {
  let sizes = [
    (1, 1),
    (64, 64),
    (640, 480),
    (1920, 1080),
    (3840, 2160),
    (7680, 4320),
    (30_000, 30_000),
  ];
  for size in sizes {
    let ssaa = export_ssaa_scale(size);
    assert!(
      (1..=EXPORT_SSAA_SCALE_MAX).contains(&ssaa),
      "{size:?}: {ssaa}"
    );
    let pixels = u64::from(size.0) * u64::from(size.1);
    let allocated = pixels * u64::from(ssaa) * u64::from(ssaa);
    assert!(
      allocated <= EXPORT_SSAA_PIXEL_BUDGET || ssaa == 1,
      "{size:?}: {ssaa}x allocates {allocated} px, over budget while it could step down"
    );
  }
  // The factor only falls as the resolution rises: a bigger export never
  // supersamples harder than a smaller one.
  assert!(export_ssaa_scale((640, 480)) >= export_ssaa_scale((3840, 2160)));
}

/// A headless render of a known view produces an image that is not a single
/// flat color, i.e. the frame graph actually drew the scene, rather than
/// clearing and presenting the background.
///
/// Pointed at grade 1, not the viewer's starting grade 0: a grade-1 field
/// reduces to a line field, whose particle advection is what this test's
/// stepping exercises. Grade 0 draws scalars only and would leave that path
/// untested.
///
/// Skipped, not failed, where no adapter exists: a machine without a GPU
/// cannot answer the question either way.
#[test]
fn headless_render_draws_the_scene() {
  let Some(ctx) = headless_context() else {
    eprintln!("no GPU adapter; skipping headless render test");
    return;
  };
  let spec = ExportSpec {
    study: Study::Cochains(triforce_examples()),
    mesh_source: MeshSource::Triforce,
    field: Some(0),
    size: (64, 64),
    frames: None,
    fps: 30,
  };
  // The premise of the test, checked rather than assumed: if this study ever
  // stopped carrying a line field, the render below would still pass while
  // silently no longer covering the particle advection pipeline.
  let scene = spec
    .study
    .build(&spec.mesh_source.build().expect("the triforce builds"));
  assert!(
    !scene.line_fields.is_empty(),
    "the triforce cochains are grade-1 line fields; without one the particle \
     advection pipeline is not what this test exercises"
  );

  let mut renderer = Renderer::new(&ctx, EXPORT_FORMAT, export_ssaa_scale(spec.size));
  let target = ExportTarget::new(&ctx, spec.size);
  let displayed = Displayed::build(&ctx, &spec).expect("the triforce scene builds");

  // Stepped, not merely drawn. A line field carries an advected population,
  // and with zero steps its compute pass never runs, so the dispatch, its
  // bind group and the layout the pipeline was built against would all go
  // unexercised while this test still passed. wgpu validates on submit, so a
  // mismatch here is a panic rather than a silent pass.
  let pixels = render_at(&ctx, &mut renderer, &target, &displayed, 0.0, 4);
  assert_eq!(pixels.len(), 64 * 64 * 4);
  let first = &pixels[..4];
  assert!(
    pixels.chunks_exact(4).any(|px| px != first),
    "every pixel is identical: the scene did not draw"
  );
}
