use nalgebra::{Matrix4, Perspective3, Point3, Vector3};
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
    }
  }

  pub fn build_view_projection_matrix(&self) -> Matrix4<f32> {
    let direction = Vector3::new(
      self.yaw.cos() * self.pitch.cos(),
      self.yaw.sin() * self.pitch.cos(),
      self.pitch.sin(),
    );

    let eye = self.target - direction * self.distance;
    let view = Matrix4::look_at_rh(&eye, &self.target, &Vector3::z_axis());
    let proj = Perspective3::new(self.aspect, self.fovy, self.znear, self.zfar);

    OPENGL_TO_WGPU_MATRIX * proj.into_inner() * view
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
