use nalgebra::{Matrix4, Orthographic3, Perspective3, Point3, Vector3};
use std::f32::consts::FRAC_PI_2;

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.5,
    0.0, 0.0, 0.0, 1.0,
);

pub struct Camera {
  pub target: Point3<f32>,
  pub distance: f32,
  pub yaw: f32,
  pub pitch: f32,
  pub aspect: f32,
  pub fovy: f32,
  pub znear: f32,
  pub zfar: f32,
  /// Top-down (orthographic, drag-to-pan) vs orbit (perspective,
  /// drag-to-orbit). The projection and the interaction model are one
  /// choice, not two independent toggles: a flat top view has no meaningful
  /// orbit (there is nothing on the far side to orbit around), and a
  /// perspective top-down view has no meaningful vanishing point (nothing
  /// has depth). `yaw`/`pitch` are unused in this mode -- the view looks
  /// straight down `target`'s own $z$, which is also exactly what avoids the
  /// orbit formula's degeneracy at the poles.
  pub top_down: bool,
}

impl Camera {
  pub fn new(aspect: f32) -> Self {
    Self {
      target: Point3::new(0.0, 0.0, 0.0),
      distance: 10.0,
      yaw: -FRAC_PI_2,
      pitch: FRAC_PI_2 - 0.1, // slightly tilted down from straight up
      aspect,
      fovy: 45.0_f32.to_radians(),
      znear: 0.1,
      zfar: 100.0,
      top_down: false,
    }
  }

  /// Half-width/height of the visible region at `target`'s depth: what an
  /// orthographic frustum uses for its bounds, and what drag-to-pan uses to
  /// turn a pixel delta into a world-space one. Derived from the same
  /// `fovy`/`distance` a perspective camera would use there, so an
  /// orthographic camera -- which has no focal distance of its own -- frames
  /// the same view a perspective one would at this distance.
  pub fn ortho_half_extent(&self) -> (f32, f32) {
    let half_height = self.distance * (self.fovy / 2.0).tan();
    (half_height * self.aspect, half_height)
  }

  pub fn build_view_projection_matrix(&self) -> Matrix4<f32> {
    let (eye, up) = if self.top_down {
      (
        self.target + Vector3::new(0.0, 0.0, self.distance),
        Vector3::y_axis(),
      )
    } else {
      let direction = Vector3::new(
        self.yaw.cos() * self.pitch.cos(),
        self.yaw.sin() * self.pitch.cos(),
        self.pitch.sin(),
      );
      (self.target - direction * self.distance, Vector3::z_axis())
    };
    let view = Matrix4::look_at_rh(&eye, &self.target, &up);

    let proj = if self.top_down {
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

    OPENGL_TO_WGPU_MATRIX * proj * view
  }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
  view_proj: [[f32; 4]; 4],
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
    }
  }

  pub fn update_view_proj(&mut self, camera: &Camera) {
    self.view_proj = camera.build_view_projection_matrix().into();
  }
}
