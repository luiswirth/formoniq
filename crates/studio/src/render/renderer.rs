//! The frame graph: the passes, the transient targets they draw through, and
//! the one `render` that records a draw list into a caller's `TextureView`.

use super::{
  camera::Camera,
  context::GpuContext,
  downsample::{DownsamplePass, SceneColorBinding},
  fill::FillPass,
  item::{DrawList, RenderItem},
  segments::SegmentPass,
  uniform::{FrameUniform, SegmentMaterial, SurfaceMaterial, UniformBinding, UniformPool},
  DEPTH_FORMAT,
};

/// The background the scene is cleared to: a near-black the lit surface and the
/// light streamline halo both separate from.
///
/// Stated *linearly*, because that is what a clear value is -- [`wgpu::Color`]
/// is linear and an sRGB target encodes it on write, so the sRGB byte 0.1 would
/// name here comes out at roughly 0.35, a mid grey. This is the linear value
/// whose sRGB encoding is the dark 0.1 the background is meant to be:
/// $0.1^(2.2) approx 0.0079$.
const CLEAR_COLOR: wgpu::Color = wgpu::Color {
  r: 0.0079,
  g: 0.0079,
  b: 0.0079,
  a: 1.0,
};

/// One frame, as the caller states it: what to draw, and where and when it is
/// seen from. Borrowed, not owned -- the renderer holds no scene, no camera and
/// no clock between frames.
pub struct FrameView<'a> {
  /// The batches and their materials, in submission order.
  pub items: &'a DrawList<'a>,
  pub camera: &'a Camera,
  /// The size of `target`, in pixels. The renderer allocates its own
  /// intermediates from this, so a window resize and an export resolution are
  /// the same input.
  pub size: (u32, u32),
  /// Seconds into the standing wave. An input, never state: the windowed loop
  /// passes wall-clock time and an exporter passes the instant it means to
  /// render, and the frames are deterministic either way. Which instants those
  /// are is the caller's business, not this layer's.
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
  /// size scaled by the renderer's supersampling factor per axis.
  fn supersampled(size: (u32, u32), ssaa: u32) -> (u32, u32) {
    (size.0.max(1) * ssaa, size.1.max(1) * ssaa)
  }

  fn new(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    downsample: &DownsamplePass,
    size: (u32, u32),
    ssaa: u32,
  ) -> Self {
    let (width, height) = Self::supersampled(size, ssaa);
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
  ssaa: u32,
  frame: UniformBinding<FrameUniform>,
  surface_materials: UniformPool<SurfaceMaterial>,
  segment_materials: UniformPool<SegmentMaterial>,
  fill: FillPass,
  segments: SegmentPass,
  downsample: DownsamplePass,
  targets: Option<Targets>,
}

impl Renderer {
  /// A renderer for `format`, supersampling the scene pass by `ssaa` per axis.
  ///
  /// The factor is fixed here rather than per frame because it is baked into
  /// every pipeline as the WGSL `SSAA_SCALE` override; a caller that wants a
  /// different one builds a different renderer. See [`super::DEFAULT_SSAA_SCALE`]
  /// for the interactive choice.
  pub fn new(ctx: &GpuContext, format: wgpu::TextureFormat, ssaa: u32) -> Self {
    use wgpu::ShaderStages as Stages;
    let device = &ctx.device;
    // Every uniform here is visible in both stages: the fill pulses its colormap
    // by the same $cos(sqrt(lambda) t)$ that displaces it, and the segment marks
    // fade by it.
    let frame = UniformBinding::new(
      device,
      "frame",
      Stages::VERTEX_FRAGMENT,
      FrameUniform::default(),
    );
    let surface_materials = UniformPool::new(device, "surface material", Stages::VERTEX_FRAGMENT);
    let segment_materials = UniformPool::new(device, "segment material", Stages::VERTEX_FRAGMENT);
    Self {
      format,
      ssaa,
      fill: FillPass::new(device, format, &frame, &surface_materials, ssaa),
      segments: SegmentPass::new(device, format, &frame, &segment_materials, ssaa),
      downsample: DownsamplePass::new(device, format, ssaa),
      frame,
      surface_materials,
      segment_materials,
      // Allocated on the first frame, from the size the caller renders at:
      // there is no size to guess at construction, and a window that never
      // opens should allocate nothing.
      targets: None,
    }
  }

  pub fn format(&self) -> wgpu::TextureFormat {
    self.format
  }

  /// Writes the frame uniform and each item's material into its own binding.
  /// Returns, per item, the pool index it draws with: its position among the
  /// items of its own kind.
  fn write_uniforms(&mut self, ctx: &GpuContext, view: &FrameView) -> Vec<usize> {
    self
      .frame
      .write(&ctx.queue, FrameUniform::new(view.camera, view.time));

    let (mut nsurfaces, mut nsegments) = (0, 0);
    view
      .items
      .items
      .iter()
      .map(|item| match item {
        RenderItem::Surface(_, material) => {
          self
            .surface_materials
            .write(&ctx.device, &ctx.queue, nsurfaces, *material);
          nsurfaces += 1;
          nsurfaces - 1
        }
        RenderItem::Segments(_, material) => {
          self
            .segment_materials
            .write(&ctx.device, &ctx.queue, nsegments, *material);
          nsegments += 1;
          nsegments - 1
        }
      })
      .collect()
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
    view: &FrameView,
  ) {
    let materials = self.write_uniforms(ctx, view);

    if self.targets.as_ref().is_none_or(|t| t.size != view.size) {
      self.targets = Some(Targets::new(
        &ctx.device,
        self.format,
        &self.downsample,
        view.size,
        self.ssaa,
      ));
    }
    let targets = self.targets.as_ref().expect("just ensured");

    // The scene, at the supersampled resolution: the draw list, in the order
    // the caller gave it. One linear sequence for every field -- the reduced
    // grade and the baked dimension decide which *items* exist, never a branch
    // through the graph.
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

      let frame = self.frame.bind_group();
      for (item, &material) in view.items.items.iter().zip(&materials) {
        match item {
          RenderItem::Surface(batch, _) => self.fill.draw(
            &mut pass,
            frame,
            self.surface_materials.bind_group(material),
            batch,
          ),
          RenderItem::Segments(batch, _) => self.segments.draw(
            &mut pass,
            frame,
            self.segment_materials.bind_group(material),
            batch,
          ),
        }
      }
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
