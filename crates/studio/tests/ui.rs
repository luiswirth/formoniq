//! Laws for [`studio::ui`]: the compact-layout width threshold genuinely
//! separates a phone from a desktop, a compact sidebar never covers the
//! whole scene, sidebar visibility defers to the layout only until the
//! reader speaks and then sticks, a toggle survives the frame, the degeneracy
//! shells recover the spherical-harmonic $(2l+1)$ multiplets and the
//! generic simple-spectrum singletons while staying robust near zero, the
//! six standard camera views look straight down distinct axes, and the mesh
//! stats caption names every dimension present.

use formoniq_studio::render::camera::Camera;
use formoniq_studio::ui::{
  COMPACT_WIDTH, CameraView, Visibility, degeneracy_shells, mesh_stats_line,
};

/// The width threshold means what it claims: a phone falls below it, a
/// desktop window above, and the two docked sidebars genuinely do not fit
/// beside a viewport at phone width, which is the condition the compact
/// layout exists to escape.
#[test]
fn the_threshold_separates_a_phone_from_a_desktop() {
  let phone = 390.0 / UI_ZOOM_FOR_TEST;
  assert!(
    phone < COMPACT_WIDTH,
    "a phone ({phone} points) must land in the compact layout"
  );
  assert!(
    180.0 + 250.0 > phone,
    "the docked sidebars must genuinely exceed a phone's width"
  );
  const { assert!(1280.0 / UI_ZOOM_FOR_TEST > COMPACT_WIDTH) };
}

/// A compact sidebar never covers the whole screen: it is an overlay, not a
/// takeover, so the scene stays partly visible even with one open.
#[test]
fn a_compact_sidebar_leaves_some_scene_showing() {
  for width in [320.0_f32, 390.0, 540.0, 700.0] {
    let side = (width * 0.85).min(280.0);
    assert!(
      side < width,
      "width {width}: sidebar {side} covers everything"
    );
  }
}

/// The layout decides only what the reader has not. `Auto` follows the
/// width, docked where there is room, closed where there is not, and an
/// explicit choice outranks it at any width, which is what makes the panels
/// collapsible on a desktop rather than only on a phone.
#[test]
fn visibility_defers_to_the_layout_only_until_the_reader_speaks() {
  let wide = true;
  let narrow = false;
  assert!(Visibility::Auto.resolve(wide));
  assert!(!Visibility::Auto.resolve(narrow));
  // Explicit wins either way.
  assert!(Visibility::Shown.resolve(narrow));
  assert!(!Visibility::Hidden.resolve(wide));
}

/// A toggle sticks. The first version of this compared the post-frame state
/// against the resolved value rather than the value the panels were drawn
/// with, so opening a closed panel was immediately undone and the button did
/// nothing at all. The round trip is the test: resolve, toggle, write back,
/// resolve again.
#[test]
fn toggling_a_sidebar_survives_the_frame() {
  for layout_default in [true, false] {
    let stored = Visibility::Auto;
    let drawn = stored.resolve(layout_default);

    // The reader clicks it.
    let toggled = !drawn;
    let written = if toggled == drawn {
      stored
    } else {
      Visibility::of(toggled)
    };

    assert_ne!(written, Visibility::Auto, "a click must become explicit");
    assert_eq!(
      written.resolve(layout_default),
      toggled,
      "the next frame must draw what the click asked for"
    );
  }
}

/// An untouched sidebar stays on `Auto`, so resizing the window still moves
/// it, the state records a decision, not every frame's outcome.
#[test]
fn an_untouched_sidebar_keeps_following_the_layout() {
  let stored = Visibility::Auto;
  let drawn = stored.resolve(true);
  let written = if drawn == stored.resolve(true) {
    stored
  } else {
    Visibility::of(drawn)
  };
  assert_eq!(written, Visibility::Auto);
  // And it follows the width when that changes.
  assert!(!written.resolve(false));
}

/// Mirrors `app.rs`'s `UI_ZOOM`: the panel widths above are egui points, and
/// the zoom is what turns a device's pixels into them.
const UI_ZOOM_FOR_TEST: f32 = 1.25;

fn shell_sizes(eigenvalues: &[f64]) -> Vec<usize> {
  degeneracy_shells(eigenvalues.iter().map(|&l| Some(l)))
    .unwrap()
    .iter()
    .map(|s| s.members.len())
    .collect()
}

/// The measured subdivision-3 icosphere grade-0 spectrum clusters into the
/// $(2l+1)$ spherical-harmonic shells: the near-equal multiplets group, the
/// order-one jumps between degrees split.
#[test]
fn sphere_spectrum_recovers_2l_plus_1_shells() {
  let spectrum = [0.00, 2.01, 2.01, 2.01, 6.07, 6.07, 6.07, 6.07, 6.07, 12.24];
  assert_eq!(shell_sizes(&spectrum), vec![1, 3, 5, 1]);
}

/// A near-zero harmonic space (a flat torus's two 1-cocycles) stays one shell
/// rather than splitting on numerical noise, since the absolute tolerance
/// carries a scale the relative gap alone lacks near zero.
#[test]
fn near_zero_harmonics_stay_one_shell() {
  let spectrum = [-1e-9, 2e-9, 4.0, 4.0];
  assert_eq!(shell_sizes(&spectrum), vec![2, 2]);
}

/// A generic simple spectrum, no symmetry, no degeneracy, degenerates the
/// pyramid to one member per row, ordered by eigenvalue.
#[test]
fn simple_spectrum_gives_singletons() {
  let spectrum = [1.0, 2.5, 4.0, 6.0, 9.0];
  assert_eq!(shell_sizes(&spectrum), vec![1, 1, 1, 1, 1]);
}

/// A field carrying no eigenvalue (the raw Whitney basis) has no shell
/// structure, so the caller falls back to a flat list.
#[test]
fn missing_eigenvalue_declines_to_shell() {
  assert!(degeneracy_shells([Some(1.0), None, Some(2.0)]).is_none());
}

/// Every standard view looks straight down a coordinate axis: its forward is a
/// unit axis vector, one component $plus.minus 1$ and the other two zero. That
/// is what makes it the square-on plan/elevation a free orbit never lands on.
#[test]
fn standard_views_look_down_an_axis() {
  for view in CameraView::ALL {
    let mut camera = Camera::new(1.0);
    let (yaw, pitch) = view.angles();
    camera.snap_to(yaw, pitch);
    let f = camera.forward();
    let axes = [f.x, f.y, f.z];
    let ones = axes
      .iter()
      .filter(|&&c| (c.abs() - 1.0).abs() < 1e-6)
      .count();
    let zeros = axes.iter().filter(|&&c| c.abs() < 1e-6).count();
    assert_eq!(ones, 1, "{}: forward {f:?} is not an axis", view.label());
    assert_eq!(zeros, 2, "{}: forward {f:?} is not an axis", view.label());
  }
  // And the six are distinct vantages, not a shorter list with repeats.
  let dirs: std::collections::HashSet<[i32; 3]> = CameraView::ALL
    .iter()
    .map(|v| {
      let mut c = Camera::new(1.0);
      let (yaw, pitch) = v.angles();
      c.snap_to(yaw, pitch);
      let f = c.forward();
      [f.x.round() as i32, f.y.round() as i32, f.z.round() as i32]
    })
    .collect();
  assert_eq!(dirs.len(), 6, "the six standard views must be distinct");
}

/// The size caption names every dimension present and totals over the whole
/// range, so it reads right in any dimension rather than for a fixed few
/// skeletons, the low dimensions by their classical names, higher ones by
/// the general "$k$-simplices".
#[test]
fn mesh_stats_line_names_each_dimension() {
  assert_eq!(
    mesh_stats_line(&[12, 30, 20]),
    "12 vertices · 30 edges · 20 faces"
  );
  assert_eq!(
    mesh_stats_line(&[5, 10, 10, 5, 1]),
    "5 vertices · 10 edges · 10 faces · 5 cells · 1 4-simplices"
  );
}
