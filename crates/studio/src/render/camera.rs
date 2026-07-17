use nalgebra::{Matrix3, Matrix4, Orthographic3, Perspective3, Point3, Vector3};
use std::f32::consts::FRAC_PI_2;

/// Sends nalgebra's OpenGL-style clip space to wgpu's, *reversed*: $z in [-1, 1]$
/// with the near plane at $-1$ becomes $z in [0, 1]$ with the near plane at $1$
/// and the far plane at $0$.
///
/// The reversal is what makes the depth buffer usable at range, and it is not a
/// bias or a tolerance -- it is the observation that the two sources of error
/// cancel. Perspective depth is a hyperbola, $d(z) approx 1 - z_"near" \/ z$, so
/// its resolution decays like $z^2 \/ z_"near"$; float32's resolution decays in
/// the opposite direction, being densest at $0$. Unreversed the two compound,
/// concentrating precision where the hyperbola already had it and starving the
/// far field, where a wireframe's depth bias then loses to roundoff. Reversed
/// they very nearly cancel, and the relative precision is roughly uniform in $z$
/// -- which is why $z_"near"$ may stay aggressively small, as the framing
/// deliberately sets it, at almost no cost.
///
/// The flip must live in the matrix and not in the shader. As a row operation it
/// is $"row"_3 |-> "row"_4 - "row"_3$, acting on the clip-space coefficients
/// *before* the perspective divide; the same $1 - d$ applied afterwards, to the
/// already-quantized depth, would flip the picture and recover none of the
/// precision.
///
/// The orthographic branch inherits the reversal unchanged. Its depth is affine
/// in $z$, so reversing neither gains nor loses precision there -- but the sense
/// of the depth test is a property of the target, not of the projection, so
/// there is one constant and no case distinction.
#[rustfmt::skip]
pub const OPENGL_TO_REVERSED_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, -0.5, 0.5,
    0.0, 0.0, 0.0, 1.0,
);

/// World up. A roll-free camera needs a distinguished direction, and it cannot
/// come from the scene: a mesh in $RR^3$ has no canonical up, but the person
/// looking at it does. This is the viewer's axis, not the object's.
pub const WORLD_UP: Vector3<f32> = Vector3::new(0.0, 0.0, 1.0);

/// A roll-free camera: a position and a viewing direction, the direction given
/// in the spherical chart $(psi, theta)$ about [`WORLD_UP`].
///
/// **The eye is the primary state and the pivot is derived**, not the other way
/// round. An orbit camera that stores its target makes flying incoherent -- the
/// pivot drifts off whatever is being looked at, and a subsequent orbit swings
/// about a point with no relation to the scene. Here every gesture is a rotation
/// of $(psi, theta)$, and what distinguishes orbiting from looking is only the
/// *center* it is applied about ([`Self::rotate`]).
///
/// **Roll-free costs exactly two poles, and that is a theorem, not a defect.**
/// A roll-free framing is a choice of $u(d) perp d$ for every direction
/// $d in S^2$ -- a nowhere-zero section of $T S^2$ -- and the hairy-ball theorem
/// says none exists. So the singularity is forced; `pitch` is where it was put.
/// The alternative is a camera carrying the third degree of freedom, whose price
/// is holonomy: a closed loop of the cursor enclosing solid angle $Omega$
/// returns the frame rolled by $Omega$, which reads as unprompted drift.
///
/// This is *not* gimbal lock, and nothing here is clamped to avoid any. `pitch`
/// ranges over the closed $[-pi/2, pi/2]$, poles included, because the frame is
/// built forward from the angles ([`Self::right`]) rather than recovered from
/// the direction by `look_at`. The degeneracy is in that *inverse* map alone,
/// and `yaw` is carried state, so it is never lost. Beyond $|theta| = pi/2$
/// there is no new view to reach -- it is $psi + pi$ upside down, i.e. the roll
/// this camera declines.
pub struct Camera {
  /// The camera's world-space position: the primary state.
  pub eye: Point3<f32>,
  /// Azimuth $psi$ about [`WORLD_UP`]. Unbounded and never normalized -- it is
  /// the carried state that makes the frame total at the poles.
  pub yaw: f32,
  /// Elevation $theta in [-pi/2, pi/2]$, poles included.
  pub pitch: f32,
  /// Distance to the pivot: how far along [`Self::forward`] the point of
  /// interest sits. Not a degree of freedom of the view -- the eye and the
  /// angles already fix that -- but the scale the gestures that need a depth
  /// read: orbiting, panning, and the orthographic frustum, which has no focal
  /// distance of its own.
  pub pivot_distance: f32,
  pub aspect: f32,
  pub fovy: f32,
  pub znear: f32,
  pub zfar: f32,
  /// Orthographic vs. perspective. *Only* a projection, with no interaction
  /// model attached: every gesture means the same thing under both. A flat mesh
  /// viewed face-on wants parallel projection because nothing in it has depth
  /// for a vanishing point to act on -- which says nothing about how the mouse
  /// should behave.
  pub orthographic: bool,
}

impl Camera {
  pub fn new(aspect: f32) -> Self {
    Self {
      eye: Point3::new(-10.0, 0.0, 0.0),
      yaw: 0.0,
      pitch: 0.0,
      pivot_distance: 10.0,
      aspect,
      fovy: 45.0_f32.to_radians(),
      znear: 0.1,
      zfar: 100.0,
      orthographic: false,
    }
  }

  /// The viewing direction: the unit vector the camera looks along.
  pub fn forward(&self) -> Vector3<f32> {
    let (sy, cy) = self.yaw.sin_cos();
    let (sp, cp) = self.pitch.sin_cos();
    Vector3::new(cy * cp, sy * cp, sp)
  }

  /// The screen's right axis.
  ///
  /// This is the whole reason the camera is total at the poles. The roll-free
  /// right vector is $hat(f) times hat(z)$ normalized, which is
  /// $cos theta (sin psi, -cos psi, 0)$ -- and the $cos theta$ divides out. What
  /// remains is a function of `yaw` alone, defined on all of $[-pi/2, pi/2]$
  /// including where $hat(f) parallel hat(z)$ and the cross product vanishes.
  /// It is the analytic continuation of `look_at_rh`'s own right vector through
  /// its singularity, and it exists only because `yaw` was kept rather than
  /// recovered.
  pub fn right(&self) -> Vector3<f32> {
    let (sy, cy) = self.yaw.sin_cos();
    Vector3::new(sy, -cy, 0.0)
  }

  /// The screen's up axis. Not [`WORLD_UP`]: that is the axis the framing is
  /// *gauged* against, this is where up ends up on screen once pitched.
  pub fn up(&self) -> Vector3<f32> {
    self.right().cross(&self.forward())
  }

  /// The point of interest: where [`Self::pivot_distance`] along the view lands.
  /// Derived, never stored -- see the type's own note on why.
  pub fn pivot(&self) -> Point3<f32> {
    self.eye + self.forward() * self.pivot_distance
  }

  /// The camera's orientation as a rotation matrix, its columns the frame
  /// $(hat(r), hat(u), hat(f))$ -- the change of basis from view coordinates to
  /// world ones.
  pub fn frame(&self) -> Matrix3<f32> {
    Matrix3::from_columns(&[self.right(), self.up(), self.forward()])
  }

  /// Rotates the camera rigidly about `center` by a $(psi, theta)$ delta.
  ///
  /// The one rotation primitive, and the only thing separating the two idioms:
  /// orbiting passes the point being looked at (the eye swings around it),
  /// looking passes the eye itself (the view swings in place). They are the same
  /// rigid motion about different centers, which is why they compose without
  /// fighting.
  ///
  /// The eye's *offset* from the center is rotated, never rebuilt from
  /// [`Self::forward`]. Rebuilding it silently assumes the center lies on the
  /// view axis and snaps the eye onto that axis when it does not -- and an
  /// off-axis center is the whole point of a picked pivot. Rotating the offset
  /// makes this a rigid motion of the camera about `center`, whose conserved
  /// quantity is the center's own view-space coordinate: it holds still on
  /// screen, exactly where it was grabbed (`orbit_pins_its_center_on_screen`).
  pub fn rotate(&mut self, dyaw: f32, dpitch: f32, center: Point3<f32>) {
    let before = self.frame();
    // Both subtract, which is what makes the axes consistent: `forward` sweeps
    // toward $+y$ as `yaw` grows, i.e. to the *left* of the screen, so a
    // rightward drag must lower it -- matching pitch, where a drag up raises
    // the view.
    self.yaw -= dyaw;
    self.pitch = (self.pitch - dpitch).clamp(-FRAC_PI_2, FRAC_PI_2);
    // $R = F_1 F_0^T$ carries the old frame to the new one, whatever the two
    // are -- so it stays the exact rotation between them even on the frame
    // where the pitch clamp saturates and swallows part of the requested delta.
    let rotation = self.frame() * before.transpose();
    self.eye = center + rotation * (self.eye - center);
  }

  /// Half-width/height of the visible region at the pivot's depth: the
  /// orthographic frustum's bounds, and what turns a pixel drag into a
  /// world-space one. Derived from the `fovy`/`pivot_distance` a perspective
  /// camera would use there, so switching projection reframes nothing.
  pub fn ortho_half_extent(&self) -> (f32, f32) {
    let half_height = self.pivot_distance * (self.fovy / 2.0).tan();
    (half_height * self.aspect, half_height)
  }

  /// World-space size of one pixel at the pivot's depth, given the viewport
  /// height in pixels.
  pub fn world_per_pixel(&self, viewport_height: u32) -> f32 {
    let (_, half_height) = self.ortho_half_extent();
    2.0 * half_height / viewport_height.max(1) as f32
  }

  /// The world-space ray through a normalized device point ($x$ right, $y$ up,
  /// both in $[-1, 1]$), as an origin and a unit direction.
  ///
  /// Both projections answer this, and the difference between them is exactly
  /// what each one is: a perspective camera fans directions out of one origin,
  /// an orthographic one slides one direction across a plane of origins.
  pub fn ray(&self, ndc_x: f32, ndc_y: f32) -> (Point3<f32>, Vector3<f32>) {
    if self.orthographic {
      let (half_width, half_height) = self.ortho_half_extent();
      let origin =
        self.eye + self.right() * (ndc_x * half_width) + self.up() * (ndc_y * half_height);
      (origin, self.forward())
    } else {
      let tan_half = (self.fovy / 2.0).tan();
      let dir = self.forward()
        + self.right() * (ndc_x * tan_half * self.aspect)
        + self.up() * (ndc_y * tan_half);
      (self.eye, dir.normalize())
    }
  }

  pub fn build_view_projection_matrix(&self) -> Matrix4<f32> {
    let f = self.forward();
    let r = self.right();
    let u = self.up();
    let e = self.eye.coords;

    // Assembled from the carried frame rather than `look_at_rh(eye, target,
    // up)`: that call rebuilds `r` by normalizing $hat(f) times hat(z)$, which
    // is the one step that dies at the poles. Away from them the two agree
    // exactly (`view_matches_look_at`).
    #[rustfmt::skip]
    let view = Matrix4::new(
       r.x,  r.y,  r.z, -r.dot(&e),
       u.x,  u.y,  u.z, -u.dot(&e),
      -f.x, -f.y, -f.z,  f.dot(&e),
       0.0,  0.0,  0.0,  1.0,
    );

    let proj = if self.orthographic {
      let (half_width, half_height) = self.ortho_half_extent();
      Orthographic3::new(
        -half_width,
        half_width,
        -half_height,
        half_height,
        self.znear,
        self.zfar,
      )
      .into_inner()
    } else {
      Perspective3::new(self.aspect, self.fovy, self.znear, self.zfar).into_inner()
    };

    OPENGL_TO_REVERSED_WGPU_MATRIX * proj * view
  }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
  view_proj: [[f32; 4]; 4],
  /// World-space eye position, `w` unused (kept for uniform alignment): the
  /// wireframe shader's own use for it, to find the screen-facing
  /// perpendicular of a world-space-thick edge quad.
  eye: [f32; 4],
}

impl Default for CameraUniform {
  fn default() -> Self {
    Self::new()
  }
}

impl CameraUniform {
  pub fn new() -> Self {
    Self {
      view_proj: nalgebra::Matrix4::identity().into(),
      eye: [0.0; 4],
    }
  }

  pub fn update_view_proj(&mut self, camera: &Camera) {
    self.view_proj = camera.build_view_projection_matrix().into();
    let eye = camera.eye;
    self.eye = [eye.x, eye.y, eye.z, 1.0];
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::f32::consts::PI;

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

  /// The frame is an orthonormal right-handed basis on the *closed* pitch
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

  /// Orbiting holds the pivot fixed however far it is turned -- the property
  /// the old target-primary camera lost the moment anything moved the eye.
  #[test]
  fn orbit_fixes_its_center() {
    let mut c = at(0.3, 0.2);
    let pivot = c.pivot();
    for _ in 0..40 {
      c.rotate(0.21, 0.13, pivot);
      assert!((c.pivot() - pivot).norm() < 1e-3);
    }
  }

  /// The center holds still *on screen*, not merely in space: its view-space
  /// coordinate is conserved, so the grabbed point does not jump out from under
  /// the cursor on the first pixel of the drag.
  ///
  /// This is what a rigid rotation buys and what rebuilding the eye from
  /// `forward` destroys -- and it holds for an off-axis center too, which is
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
  /// projection -- the shared contract that lets one picking path serve both.
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
  /// divide -- what the depth test actually compares.
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

  /// Nearer is larger, everywhere in the frustum -- the law `CompareFunction::
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

  /// The theorem the reversal exists for: a fixed *world* separation stays
  /// resolvable in float32 depth however far out it is viewed.
  ///
  /// This is what an unreversed buffer fails. Sweeping the eye out to the far
  /// field, two points a wireframe's bias apart must still differ by many
  /// float32 ulps -- the quantity that decayed like $z^2$ before, and is the
  /// z-fighting when it reaches zero.
  #[test]
  fn a_world_scale_bias_survives_depth_quantization() {
    let extent = 1.0_f32;
    // The wireframe's nudge: `4 * WIREFRAME_WIDTH_FRACTION * extent`.
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
}
