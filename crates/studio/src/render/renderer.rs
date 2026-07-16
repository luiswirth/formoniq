//! The frame graph: the passes, the transient targets they draw through, and
//! the one `render` that records them into a caller's `TextureView`.

use super::{
  camera::{Camera, CameraUniform},
  context::GpuContext,
  downsample::{DownsamplePass, SceneColorBinding},
  fill::FillPass,
  mesh::MeshBuffer,
  streamline::{StreamlineBuffer, StreamlinePass},
  uniform::{BoundsUniform, SegmentWidth, Uniforms, WaveUniform},
  wireframe::WireframePass,
  DEPTH_FORMAT, SSAA_SCALE, STREAMLINE_WIDTH_FRACTION, WIREFRAME_WIDTH_FRACTION,
};

/// The background the scene is cleared to.
const CLEAR_COLOR: wgpu::Color = wgpu::Color {
  r: 0.1,
  g: 0.1,
  b: 0.1,
  a: 1.0,
};

/// Everything one frame draws, as the caller states it: the baked geometry, the
/// material parameters of the field on display, and where and when it is seen
/// from. Borrowed, not owned -- the renderer holds no scene, no camera and no
/// clock between frames.
pub struct SceneView<'a> {
  pub mesh: &'a MeshBuffer,
  /// The traced curves of a line field, `None` for a scalar field. Their
  /// presence *is* the line-field mark: there is no branch to pick.
  pub streamlines: Option<&'a StreamlineBuffer>,
  /// The colormap range the fill normalizes against.
  pub field_min: f32,
  pub field_max: f32,
  /// The standing wave: peak displacement and $omega = sqrt(lambda)$. A field
  /// with no eigenvalue passes zeros and is drawn static by the same code.
  pub wave_amplitude: f32,
  pub wave_omega: f32,
  /// The object's own coordinate extent (its radius), which fixes the
  /// world-space width of every segment mark -- object-intrinsic, so a line
  /// reads the same thickness on any mesh however finely triangulated.
  pub extent: f32,
  pub camera: &'a Camera,
  /// The size of `target`, in pixels. The renderer allocates its own
  /// intermediates from this, so a window resize and an export resolution are
  /// the same input.
  pub size: (u32, u32),
  /// Seconds into the standing wave. An input, never state: the windowed loop
  /// passes wall-clock time, an exporter passes $t_k = k \/ "fps"$, and the
  /// frames are deterministic either way.
  pub time: f32,
}

/// The offscreen targets, at the supersampled resolution. Held by the renderer
/// and reallocated whenever the caller's target size changes, so neither the
/// window nor the exporter has to drive a resize.
struct Targets {
  size: (u32, u32),
  depth_view: wgpu::TextureView,
  scene_color_view: wgpu::TextureView,
  scene_color: SceneColorBinding,
}

impl Targets {
  /// The supersampled resolution every scene pass renders at: the caller's own
  /// size scaled by [`SSAA_SCALE`] per axis.
  fn supersampled(size: (u32, u32)) -> (u32, u32) {
    (size.0.max(1) * SSAA_SCALE, size.1.max(1) * SSAA_SCALE)
  }

  fn new(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    downsample: &DownsamplePass,
    size: (u32, u32),
  ) -> Self {
    let (width, height) = Self::supersampled(size);
    let extent = wgpu::Extent3d {
      width,
      height,
      depth_or_array_layers: 1,
    };
    let target = |label: &str, format| {
      device
        .create_texture(&wgpu::TextureDescriptor {
          label: Some(label),
          size: extent,
          mip_level_count: 1,
          sample_count: 1,
          dimension: wgpu::TextureDimension::D2,
          format,
          usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
          view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
    };
    let depth_view = target("Depth Texture", DEPTH_FORMAT);
    let scene_color_view = target("Scene Color Texture (supersampled)", format);
    let scene_color = downsample.bind(device, &scene_color_view);
    Self {
      size,
      depth_view,
      scene_color_view,
      scene_color,
    }
  }
}

/// The renderer: every pipeline, every uniform, and the frame graph, over any
/// target of the format it was built for.
///
/// It knows nothing of a window, a surface or a clock -- the target view, its
/// size and the time are all arguments to [`Self::render`] -- so the
/// interactive viewer and a headless export drive one implementation and cannot
/// drift.
pub struct Renderer {
  format: wgpu::TextureFormat,
  uniforms: Uniforms,
  fill: FillPass,
  wireframe: WireframePass,
  streamline: StreamlinePass,
  downsample: DownsamplePass,
  targets: Option<Targets>,
}

impl Renderer {
  pub fn new(ctx: &GpuContext, format: wgpu::TextureFormat) -> Self {
    let device = &ctx.device;
    let uniforms = Uniforms::new(device);
    Self {
      format,
      fill: FillPass::new(device, format, &uniforms),
      wireframe: WireframePass::new(device, format, &uniforms),
      streamline: StreamlinePass::new(device, format, &uniforms),
      downsample: DownsamplePass::new(device, format),
      uniforms,
      // Allocated on the first frame, from the size the caller renders at:
      // there is no size to guess at construction, and a window that never
      // opens should allocate nothing.
      targets: None,
    }
  }

  pub fn format(&self) -> wgpu::TextureFormat {
    self.format
  }

  fn write_uniforms(&self, queue: &wgpu::Queue, view: &SceneView) {
    let mut camera_uniform = CameraUniform::new();
    camera_uniform.update_view_proj(view.camera);
    self.uniforms.camera.write(queue, camera_uniform);
    self.uniforms.wave.write(
      queue,
      WaveUniform {
        time: view.time,
        amplitude: view.wave_amplitude,
        omega: view.wave_omega,
        _pad: 0.0,
      },
    );
    self.uniforms.bounds.write(
      queue,
      BoundsUniform {
        min_val: view.field_min,
        max_val: view.field_max,
        _pad1: 0.0,
        _pad2: 0.0,
      },
    );
    let width = |fraction: f32| SegmentWidth {
      half_width_world: fraction * view.extent,
      ..Default::default()
    };
    self
      .uniforms
      .wireframe_width
      .write(queue, width(WIREFRAME_WIDTH_FRACTION));
    self
      .uniforms
      .streamline_width
      .write(queue, width(STREAMLINE_WIDTH_FRACTION));
  }

  /// Records one frame of `view` into `target`.
  ///
  /// The caller supplies the encoder so it can compose its own passes (a
  /// windowed viewer's UI, an exporter's copy-to-buffer) after the scene in the
  /// same submission.
  pub fn render(
    &mut self,
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    target: &wgpu::TextureView,
    view: &SceneView,
  ) {
    self.write_uniforms(&ctx.queue, view);

    if self.targets.as_ref().is_none_or(|t| t.size != view.size) {
      self.targets = Some(Targets::new(
        &ctx.device,
        self.format,
        &self.downsample,
        view.size,
      ));
    }
    let targets = self.targets.as_ref().expect("just ensured");

    // The scene, at the supersampled resolution: the surface, then a line
    // field's ribbons over it, then the wireframe on top. One linear sequence
    // for every field -- the reduced grade picks the *marks* (a scalar field
    // carries no streamlines), never a branch through the graph.
    {
      let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Scene Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
          view: &targets.scene_color_view,
          resolve_target: None,
          ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(CLEAR_COLOR),
            store: wgpu::StoreOp::Store,
          },
          depth_slice: None,
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
          view: &targets.depth_view,
          depth_ops: Some(wgpu::Operations {
            load: wgpu::LoadOp::Clear(1.0),
            store: wgpu::StoreOp::Store,
          }),
          stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
      });

      self.fill.draw(&mut pass, &self.uniforms, view.mesh);
      if let Some(streamlines) = view.streamlines {
        self.streamline.draw(&mut pass, &self.uniforms, streamlines);
      }
      self.wireframe.draw(&mut pass, &self.uniforms, view.mesh);
    }

    // Antialiasing: box-filter the supersampled target down into the caller's
    // view.
    {
      let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Downsample Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
          view: target,
          resolve_target: None,
          ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(CLEAR_COLOR),
            store: wgpu::StoreOp::Store,
          },
          depth_slice: None,
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
      });
      self.downsample.draw(&mut pass, &targets.scene_color);
    }
  }
}
