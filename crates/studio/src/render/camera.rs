use nalgebra::{Matrix3, Matrix4, Orthographic3, Perspective3, Point3, Vector3};
use std::f32::consts::FRAC_PI_2;

/// Sends nalgebra's OpenGL-style clip space to wgpu's, reversed: $z in [-1, 1]$
/// with the near plane at $-1$ becomes $z in [0, 1]$ with the near plane at $1$
/// and the far plane at $0$.
///
/// The reversal is what makes the depth buffer usable at range, and it is not a
/// bias or a tolerance: it is the observation that the two sources of error
/// cancel. Perspective depth is a hyperbola, $d(z) approx 1 - z_"near" \/ z$, so
/// its resolution decays like $z^2 \/ z_"near"$; float32's resolution decays in
/// the opposite direction, being densest at $0$. Unreversed the two compound,
/// concentrating precision where the hyperbola already had it and starving the
/// far field, where a wireframe's depth bias then loses to roundoff. Reversed
/// they very nearly cancel, and the relative precision is roughly uniform in $z$,
/// which is why $z_"near"$ may stay aggressively small, as the framing
/// deliberately sets it, at almost no cost.
///
/// The flip must live in the matrix and not in the shader. As a row operation it
/// is $"row"_3 |-> "row"_4 - "row"_3$, acting on the clip-space coefficients
/// before the perspective divide; the same $1 - d$ applied afterwards, to the
/// already-quantized depth, would flip the picture and recover none of the
/// precision.
///
/// The orthographic branch inherits the reversal unchanged. Its depth is affine
/// in $z$, so reversing neither gains nor loses precision there, but the sense
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
/// The eye is the primary state and the pivot is derived, not the other way
/// round. An orbit camera that stores its target makes flying incoherent, the
/// pivot drifts off whatever is being looked at, and a subsequent orbit swings
/// about a point with no relation to the scene. Here every gesture is a rotation
/// of $(psi, theta)$, and what distinguishes orbiting from looking is only the
/// center it is applied about ([`Self::rotate`]).
///
/// Roll-free costs exactly two poles, and that is a theorem, not a defect.
/// A roll-free framing is a choice of $u(d) perp d$ for every direction
/// $d in S^2$, a nowhere-zero section of $T S^2$, and the hairy-ball theorem
/// says none exists. So the singularity is forced; `pitch` is where it was put.
/// The alternative is a camera carrying the third degree of freedom, whose price
/// is holonomy: a closed loop of the cursor enclosing solid angle $Omega$
/// returns the frame rolled by $Omega$, which reads as unprompted drift.
///
/// This is not gimbal lock, and nothing here is clamped to avoid any. `pitch`
/// ranges over the closed $[-pi/2, pi/2]$, poles included, because the frame is
/// built forward from the angles ([`Self::right`]) rather than recovered from
/// the direction by `look_at`. The degeneracy is in that inverse map alone,
/// and `yaw` is carried state, so it is never lost. Beyond $|theta| = pi/2$
/// there is no new view to reach: it is $psi + pi$ upside down, i.e. the roll
/// this camera declines.
pub struct Camera {
  /// The camera's world-space position: the primary state.
  pub eye: Point3<f32>,
  /// Azimuth $psi$ about [`WORLD_UP`]. Unbounded and never normalized: it is
  /// the carried state that makes the frame total at the poles.
  pub yaw: f32,
  /// Elevation $theta in [-pi/2, pi/2]$, poles included.
  pub pitch: f32,
  /// Distance to the pivot: how far along [`Self::forward`] the point of
  /// interest sits. Not a degree of freedom of the view, the eye and the
  /// angles already fix that, but the scale the gestures that need a depth
  /// read: orbiting, panning, and the orthographic frustum, which has no focal
  /// distance of its own.
  pub pivot_distance: f32,
  pub aspect: f32,
  pub fovy: f32,
  pub znear: f32,
  pub zfar: f32,
  /// Orthographic vs. perspective, selecting both the projection and the
  /// navigation it is paired with. A flat mesh viewed face-on wants parallel
  /// projection, nothing in it has depth for a vanishing point to act on,
  /// and wants a 2D-map interaction to match: it does not rotate, since tumbling
  /// a face-on plane only tilts it away, so it pans and zooms where a
  /// perspective view orbits and flies. The primitives are shared and only the
  /// input binding differs, which is chosen in the window's input handler, not
  /// here. This flag is the one bit both read.
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
  /// $cos theta (sin psi, -cos psi, 0)$, and the $cos theta$ divides out. What
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
  /// gauged against: this is where up ends up on screen once pitched.
  pub fn up(&self) -> Vector3<f32> {
    self.right().cross(&self.forward())
  }

  /// The point of interest: where [`Self::pivot_distance`] along the view lands.
  /// Derived, never stored, see the type's own note on why.
  pub fn pivot(&self) -> Point3<f32> {
    self.eye + self.forward() * self.pivot_distance
  }

  /// The camera's orientation as a rotation matrix, its columns the frame
  /// $(hat(r), hat(u), hat(f))$, the change of basis from view coordinates to
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
  /// The eye's offset from the center is rotated, never rebuilt from
  /// [`Self::forward`]. Rebuilding it silently assumes the center lies on the
  /// view axis and snaps the eye onto that axis when it does not, and an
  /// off-axis center is what a picked pivot is for. Rotating the offset
  /// makes this a rigid motion of the camera about `center`, whose conserved
  /// quantity is the center's own view-space coordinate: it holds still on
  /// screen, exactly where it was grabbed (`orbit_pins_its_center_on_screen`).
  pub fn rotate(&mut self, dyaw: f32, dpitch: f32, center: Point3<f32>) {
    let before = self.frame();
    // Both subtract, which is what makes the axes consistent: `forward` sweeps
    // toward $+y$ as `yaw` grows, i.e. to the left of the screen, so a
    // rightward drag must lower it, matching pitch, where a drag up raises
    // the view.
    self.yaw -= dyaw;
    self.pitch = (self.pitch - dpitch).clamp(-FRAC_PI_2, FRAC_PI_2);
    // $R = F_1 F_0^T$ carries the old frame to the new one, whatever the two
    // are, so it stays the exact rotation between them even on the frame
    // where the pitch clamp saturates and swallows part of the requested delta.
    let rotation = self.frame() * before.transpose();
    self.eye = center + rotation * (self.eye - center);
  }

  /// Reorients the camera to look straight down the world $-z$ onto the plane,
  /// keeping the point it was looking at fixed under the view.
  ///
  /// What entering orthographic mode needs: a face-on 2D view is only face-on
  /// from directly above, so the projection change alone is not enough, an
  /// oblique perspective pose reprojected orthographically is a skewed
  /// parallelogram, not the square plan the flat view is for. The pivot is held
  /// as the anchor, and `yaw` snaps to $pi/2$ so screen-right lands on world
  /// $+x$ ([`Self::right`]) and the plane keeps its own axes.
  pub fn snap_top_down(&mut self) {
    self.snap_to(FRAC_PI_2, -FRAC_PI_2);
  }

  /// Snaps the orientation to $(psi, theta)$ while holding the pivot fixed, so
  /// the object stays framed and only the vantage changes. The one primitive
  /// behind every canned pose, [`Self::snap_top_down`] and the axis-aligned
  /// standard views are each a choice of angles fed through here, the eye
  /// re-derived from the held pivot exactly as it is on entering the flat view.
  pub fn snap_to(&mut self, yaw: f32, pitch: f32) {
    let pivot = self.pivot();
    self.yaw = yaw;
    self.pitch = pitch;
    self.eye = pivot - self.forward() * self.pivot_distance;
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
