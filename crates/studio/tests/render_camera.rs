//! Laws for [`studio::render::camera::Camera`]: the carried frame is
//! orthonormal and roll-free everywhere on the closed pitch range including
//! the poles, snapping and orbiting hold the pivot, orbiting about the eye
//! leaves it fixed, pitch saturates rather than tumbling over the top,
//! the view matches `look_at_rh` away from the poles, the projection stays
//! finite at the poles, and the reversed-Z depth is monotone, bottoms out
//! at the near and far planes, and resolves a fixed world-scale bias into
//! the far field.

use formoniq_studio::render::camera::{Camera, WORLD_UP};
use nalgebra::{Matrix4, Point3, Vector3};
use std::f32::consts::{FRAC_PI_2, PI};

/// Every $(psi, theta)$ of the closed range, poles included.
fn sweep() -> impl Iterator<Item = (f32, f32)> {
  (0..8).flat_map(|i| {
    (0..=8).map(move |j| {
      let yaw = -PI + 2.0 * PI * i as f32 / 8.0;
      let pitch = -FRAC_PI_2 + PI * j as f32 / 8.0;
      (yaw, pitch)
    })
  })
}

fn at(yaw: f32, pitch: f32) -> Camera {
  let mut camera = Camera::new(1.5);
  camera.yaw = yaw;
  camera.pitch = pitch;
  camera
}

/// The frame is an orthonormal right-handed basis on the closed pitch
/// range: total at the poles, which is the whole claim.
#[test]
fn frame_is_orthonormal_everywhere() {
  for (yaw, pitch) in sweep() {
    let c = at(yaw, pitch);
    let (f, r, u) = (c.forward(), c.right(), c.up());
    for v in [f, r, u] {
      assert!((v.norm() - 1.0).abs() < 1e-5, "yaw={yaw} pitch={pitch}");
    }
    assert!(f.dot(&r).abs() < 1e-5, "yaw={yaw} pitch={pitch}");
    assert!(f.dot(&u).abs() < 1e-5, "yaw={yaw} pitch={pitch}");
    assert!(r.dot(&u).abs() < 1e-5, "yaw={yaw} pitch={pitch}");
    assert!((r.cross(&f) - u).norm() < 1e-5, "yaw={yaw} pitch={pitch}");
  }
}

/// Roll-free: the screen's right axis stays level with the world, so the
/// horizon never tilts. Holds at the poles too, where "level" is the limit
/// `yaw` names and `look_at_rh` cannot.
#[test]
fn right_axis_is_level() {
  for (yaw, pitch) in sweep() {
    assert!(
      at(yaw, pitch).right().dot(&WORLD_UP).abs() < 1e-6,
      "yaw={yaw} pitch={pitch}"
    );
  }
}

/// Snapping top-down lands the view exactly on world $-z$ from any oblique
/// pose, and holds the pivot: the plane the flat view is for is square, and
/// centered where the perspective camera left off.
#[test]
fn snap_top_down_is_face_on() {
  for (yaw, pitch) in sweep() {
    let mut c = at(yaw, pitch);
    let pivot = c.pivot();
    c.snap_top_down();
    assert!((c.forward() - Vector3::new(0.0, 0.0, -1.0)).norm() < 1e-6);
    assert!((c.pivot() - pivot).norm() < 1e-4, "yaw={yaw} pitch={pitch}");
  }
}

/// Snapping to any orientation holds the pivot: the vantage changes while the
/// object stays framed, the invariant every canned pose is built on.
#[test]
fn snap_to_holds_the_pivot() {
  for (yaw, pitch) in sweep() {
    let mut c = at(0.4, -0.2);
    let pivot = c.pivot();
    c.snap_to(yaw, pitch);
    assert!((c.forward().norm() - 1.0).abs() < 1e-6);
    assert!((c.pivot() - pivot).norm() < 1e-4, "yaw={yaw} pitch={pitch}");
  }
}

/// Away from the poles the carried frame reproduces `look_at_rh` exactly:
/// this camera is its analytic continuation, not a different convention.
#[test]
fn view_matches_look_at() {
  for (yaw, pitch) in sweep() {
    if (pitch.abs() - FRAC_PI_2).abs() < 1e-3 {
      continue;
    }
    let c = at(yaw, pitch);
    let expected = Matrix4::look_at_rh(&c.eye, &c.pivot(), &WORLD_UP);
    let f = c.forward();
    let r = c.right();
    let u = c.up();
    let e = c.eye.coords;
    #[rustfmt::skip]
    let actual = Matrix4::new(
       r.x,  r.y,  r.z, -r.dot(&e),
       u.x,  u.y,  u.z, -u.dot(&e),
      -f.x, -f.y, -f.z,  f.dot(&e),
       0.0,  0.0,  0.0,  1.0,
    );
    assert!(
      (actual - expected).norm() < 1e-4,
      "yaw={yaw} pitch={pitch}\n{actual}\n{expected}"
    );
  }
}

/// The poles are ordinary points of the range, not excluded ones: a camera
/// looking straight down still has a finite view-projection.
#[test]
fn poles_are_finite() {
  for pitch in [-FRAC_PI_2, FRAC_PI_2] {
    for yaw in [-PI, 0.0, 1.0, PI] {
      let m = at(yaw, pitch).build_view_projection_matrix();
      assert!(m.iter().all(|x| x.is_finite()), "yaw={yaw} pitch={pitch}");
    }
  }
}

/// Orbiting holds the pivot fixed however far it is turned: the pivot is the
/// camera's own datum, so nothing that moves the eye can drift it.
#[test]
fn orbit_fixes_its_center() {
  let mut c = at(0.3, 0.2);
  let pivot = c.pivot();
  for _ in 0..40 {
    c.rotate(0.21, 0.13, pivot);
    assert!((c.pivot() - pivot).norm() < 1e-3);
  }
}

/// The center holds still on screen, not merely in space: its view-space
/// coordinate is conserved, so the grabbed point does not jump out from under
/// the cursor on the first pixel of the drag.
///
/// This is what a rigid rotation buys and what rebuilding the eye from
/// `forward` destroys, and it holds for an off-axis center too, which is
/// exactly the case that exposed the difference.
#[test]
fn orbit_pins_its_center_on_screen() {
  for center in [
    Point3::new(0.0, 0.0, 0.0),
    // Off the view axis: on-axis alone cannot tell the two formulations apart.
    Point3::new(1.7, -2.3, 0.9),
  ] {
    let mut c = at(0.3, 0.2);
    let view_space = |c: &Camera| c.frame().transpose() * (center - c.eye);
    let before = view_space(&c);
    for _ in 0..40 {
      c.rotate(0.21, 0.13, center);
      assert!(
        (view_space(&c) - before).norm() < 1e-3,
        "center {center:?} moved on screen"
      );
    }
  }
}

/// Rotating about the eye leaves it exactly where it was: looking is the same
/// primitive as orbiting, with the center brought in to zero radius.
#[test]
fn look_fixes_the_eye() {
  let mut c = at(0.3, 0.2);
  let eye = c.eye;
  for _ in 0..40 {
    c.rotate(0.21, 0.13, eye);
    assert!((c.eye - eye).norm() < 1e-4);
  }
}

/// Pitch saturates rather than tumbling over the top into an upside-down
/// view: the closed range is the whole range.
#[test]
fn pitch_saturates_at_the_poles() {
  let mut c = at(0.0, 0.0);
  for _ in 0..100 {
    let eye = c.eye;
    c.rotate(0.0, 0.5, eye);
  }
  assert!((c.pitch - -FRAC_PI_2).abs() < 1e-6);
  assert!(c.forward().z < -0.999);
}

/// The center of the viewport looks along the view direction under either
/// projection, the shared contract that lets one picking path serve both.
#[test]
fn center_ray_looks_forward() {
  for orthographic in [false, true] {
    let mut c = at(0.4, -0.3);
    c.orthographic = orthographic;
    let (origin, dir) = c.ray(0.0, 0.0);
    assert!((dir - c.forward()).norm() < 1e-5);
    assert!((origin - c.eye).norm() < 1e-5);
  }
}

/// The depth a point at `dist` along the view direction lands on, after the
/// divide, what the depth test actually compares.
fn depth_at(c: &Camera, dist: f32) -> f32 {
  let p = c.eye + c.forward() * dist;
  let clip = c.build_view_projection_matrix() * p.to_homogeneous();
  clip.z / clip.w
}

/// Reversed-Z, stated as the boundary condition it is: the near plane is $1$
/// and the far plane is $0$, under both projections.
#[test]
fn depth_is_reversed_at_the_planes() {
  for orthographic in [false, true] {
    let mut c = at(0.4, -0.3);
    c.orthographic = orthographic;
    assert!((depth_at(&c, c.znear) - 1.0).abs() < 1e-5, "near plane");
    assert!(depth_at(&c, c.zfar).abs() < 1e-5, "far plane");
  }
}

/// Nearer is larger, everywhere in the frustum, the law `CompareFunction::
/// Greater` and the clear to `DEPTH_CLEAR` are the two halves of.
#[test]
fn depth_decreases_with_distance() {
  for orthographic in [false, true] {
    let mut c = at(0.4, -0.3);
    c.orthographic = orthographic;
    let depths: Vec<f32> = (0..=64)
      .map(|i| c.znear + (c.zfar - c.znear) * i as f32 / 64.0)
      .map(|d| depth_at(&c, d))
      .collect();
    for w in depths.windows(2) {
      assert!(w[0] > w[1], "depth must decrease: {} then {}", w[0], w[1]);
    }
  }
}

/// The theorem the reversal exists for: a fixed world separation stays
/// resolvable in float32 depth however far out it is viewed.
///
/// This is what an unreversed buffer fails. Sweeping the eye out to the far
/// field, two points a wireframe's bias apart must still differ by many
/// float32 ulps, the quantity that decayed like $z^2$ before, and is the
/// z-fighting when it reaches zero.
#[test]
fn a_world_scale_bias_survives_depth_quantization() {
  let extent = 1.0_f32;
  // The skeleton marks' nudge: `4 * SKELETON_WIDTH_FRACTION * extent`.
  let bias = 4.0 * 0.004 * extent;

  let mut c = at(0.4, -0.3);
  c.znear = 1e-3 * extent;
  c.zfar = 1e3 * extent;

  for k in 1..=100 {
    let dist = k as f32 * extent;
    let (near, far) = (depth_at(&c, dist), depth_at(&c, dist + bias));
    let ulps = (near - far).abs() / (near.abs().max(far.abs()) * f32::EPSILON);
    assert!(
      ulps > 16.0,
      "bias unresolvable at {dist} extents: {ulps} ulps of separation"
    );
  }
}
