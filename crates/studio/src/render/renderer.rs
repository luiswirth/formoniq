//! The frame graph: the passes, the transient targets they draw through, and
//! the one `render` that records a draw list into a caller's `TextureView`.

use super::{
  advect::AdvectPass,
  bloom::{BloomChain, BloomPass},
  camera::Camera,
  context::GpuContext,
  deposit::{dummy_read_bind_group, DepositPass},
  downsample::{DownsamplePass, SceneColorBinding},
  fill::FillPass,
  item::{DrawList, RenderItem},
  particles::ParticlePass,
  segments::SegmentPass,
  uniform::{
    FrameUniform, ParticleMaterial, PostUniform, SegmentMaterial, SurfaceMaterial, UniformBinding,
    UniformPool,
  },
  DEPTH_CLEAR, DEPTH_FORMAT, MASK_FORMAT, SCENE_FORMAT,
};

/// The background the scene is cleared to: a near-black the lit surface and the
/// light glyph halo both separate from.
///
/// Stated *linearly*, because that is what the scene target is -- radiance, in
/// [`SCENE_FORMAT`] -- and the resolve is what carries it across to the display.
/// This is the linear value whose sRGB encoding is the dark 0.1 the background
/// is meant to be: $0.1^(2.2) approx 0.0079$. The tone map bends it a little on
/// the way out, as it bends everything; a background this near zero barely
/// moves.
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
  /// Advection steps to take before this frame is drawn.
  ///
  /// The stateful counterpart of `time`, and separate from it on purpose: a
  /// standing wave is a *function* of an instant, so the renderer can evaluate
  /// it at any; a particle population is a simulation, so it can only be
  /// stepped to one. A caller that draws no particles passes 0 and nothing
  /// reads this.
  pub steps: u32,
  /// How this frame's radiance reaches the display. Stated by the caller, like
  /// `time`, so the window and an exporter cannot come to disagree about what a
  /// field looks like.
  pub post: PostUniform,
}

/// The offscreen targets, at the supersampled resolution. Held by the renderer
/// and reallocated whenever the caller's target size changes, so neither the
/// window nor the exporter has to drive a resize.
struct Targets {
  size: (u32, u32),
  depth_view: wgpu::TextureView,
  scene_color_view: wgpu::TextureView,
  mask_view: wgpu::TextureView,
  bloom: BloomChain,
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
    downsample: &DownsamplePass,
    bloom_pass: &BloomPass,
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
    let scene_color_view = target("Scene Color Texture (supersampled HDR)", SCENE_FORMAT);
    let mask_view = target("Scene Unbounded Mask Texture (supersampled)", MASK_FORMAT);
    let bloom = bloom_pass.chain(device, &scene_color_view, (width, height));
    let scene_color = downsample.bind(device, &scene_color_view, bloom.glow(), &mask_view);
    Self {
      size,
      depth_view,
      scene_color_view,
      mask_view,
      bloom,
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
  particle_materials: UniformPool<ParticleMaterial>,
  post: UniformBinding<PostUniform>,
  fill: FillPass,
  segments: SegmentPass,
  particles: ParticlePass,
  advect: AdvectPass,
  deposit: DepositPass,
  /// The atlas binding of a frame with no deposit: a 1x1 zero texture, which
  /// the fill's floor-1/gain-0 material makes the identity.
  dummy_deposit: wgpu::BindGroup,
  bloom: BloomPass,
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
    let particle_materials = UniformPool::new(device, "particle material", Stages::VERTEX_FRAGMENT);
    let post = UniformBinding::new(device, "post", Stages::FRAGMENT, PostUniform::default());
    Self {
      format,
      ssaa,
      // The scene passes draw into `SCENE_FORMAT`, never the caller's: the
      // resolve is the one pass that touches the render target, and therefore
      // the one that has to know what it is.
      fill: FillPass::new(device, SCENE_FORMAT, &frame, &surface_materials, ssaa),
      segments: SegmentPass::new(device, SCENE_FORMAT, &frame, &segment_materials, ssaa),
      particles: ParticlePass::new(device, SCENE_FORMAT, &frame, &particle_materials, ssaa),
      advect: AdvectPass::new(device),
      deposit: DepositPass::new(device),
      dummy_deposit: dummy_read_bind_group(device),
      bloom: BloomPass::new(device),
      downsample: DownsamplePass::new(device, format, &post, ssaa),
      frame,
      surface_materials,
      segment_materials,
      particle_materials,
      post,
      // Allocated on the first frame, from the size the caller renders at:
      // there is no size to guess at construction, and a window that never
      // opens should allocate nothing.
      targets: None,
    }
  }

  pub fn format(&self) -> wgpu::TextureFormat {
    self.format
  }

  /// Steps a population (and its trails) without drawing a frame: the burn-in.
  ///
  /// A fresh population sits stacked on its seeds and a fresh atlas is blank;
  /// the equilibrium both are meant to be seen in is reached only after the
  /// transient -- exactly a sampler's burn-in, and stepped through the same
  /// per-step advect/deposit pair a frame steps, so a warmed display is
  /// bit-identical to one that ran on screen for the same count. Submitted
  /// here rather than deferred, because the caller's next frame must already
  /// be in equilibrium.
  pub fn warmup(
    &self,
    ctx: &GpuContext,
    particles: &super::particles::ParticleBatch,
    deposit: Option<&super::deposit::DepositBatch>,
    steps: u32,
  ) {
    let mut encoder = ctx
      .device
      .create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Warmup Encoder"),
      });
    for _ in 0..steps {
      self.advect.dispatch(&mut encoder, particles, 1);
      if let Some(deposit) = deposit {
        self.deposit.record(&mut encoder, deposit);
      }
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
  }

  /// Writes the frame uniform and each item's material into its own binding.
  /// Returns, per item, the pool index it draws with: its position among the
  /// items of its own kind.
  fn write_uniforms(&mut self, ctx: &GpuContext, view: &FrameView) -> Vec<usize> {
    self
      .frame
      .write(&ctx.queue, FrameUniform::new(view.camera, view.time));
    self.post.write(&ctx.queue, view.post);

    let (mut nsurfaces, mut nsegments, mut nparticles) = (0, 0, 0);
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
        RenderItem::Particles(_, material) => {
          self
            .particle_materials
            .write(&ctx.device, &ctx.queue, nparticles, *material);
          nparticles += 1;
          nparticles - 1
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
        &self.downsample,
        &self.bloom,
        view.size,
        self.ssaa,
      ));
    }
    let targets = self.targets.as_ref().expect("just ensured");

    // The advection, before the scene pass reads what it writes. A compute pass
    // cannot be nested in a render pass, so this is the frame's first act
    // rather than an item's own business -- and it is still one linear
    // sequence: an item that advects nothing contributes no dispatch.
    //
    // The deposit is stepped *inside* the step loop, after each dispatch, not
    // once per frame: the trail must be a function of the step count alone, so
    // a frame owing several steps lays several splats, and a window and an
    // exporter that reach the same count show the same trail.
    for _ in 0..view.steps {
      for item in &view.items.items {
        if let RenderItem::Particles(batch, _) = item {
          self.advect.dispatch(encoder, batch, 1);
        }
      }
      if let Some(deposit) = view.items.deposit {
        self.deposit.record(encoder, deposit);
      }
    }

    // The scene, at the supersampled resolution: the draw list, in the order
    // the caller gave it. One linear sequence for every field -- the reduced
    // grade and the baked dimension decide which *items* exist, never a branch
    // through the graph.
    {
      let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Scene Pass"),
        color_attachments: &[
          Some(wgpu::RenderPassColorAttachment {
            view: &targets.scene_color_view,
            resolve_target: None,
            ops: wgpu::Operations {
              load: wgpu::LoadOp::Clear(CLEAR_COLOR),
              store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
          }),
          // Cleared to zero: nothing drawn this frame is unbounded until a
          // particle says otherwise.
          Some(wgpu::RenderPassColorAttachment {
            view: &targets.mask_view,
            resolve_target: None,
            ops: wgpu::Operations {
              load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
              store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
          }),
        ],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
          view: &targets.depth_view,
          depth_ops: Some(wgpu::Operations {
            load: wgpu::LoadOp::Clear(DEPTH_CLEAR),
            store: wgpu::StoreOp::Store,
          }),
          stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
      });

      let frame = self.frame.bind_group();
      let deposit = view
        .items
        .deposit
        .map_or(&self.dummy_deposit, |d| d.read_bind_group());
      for (item, &material) in view.items.items.iter().zip(&materials) {
        match item {
          RenderItem::Surface(batch, _) => self.fill.draw(
            &mut pass,
            frame,
            self.surface_materials.bind_group(material),
            deposit,
            batch,
          ),
          RenderItem::Segments(batch, _) => self.segments.draw(
            &mut pass,
            frame,
            self.segment_materials.bind_group(material),
            batch,
          ),
          RenderItem::Particles(batch, _) => self.particles.draw(
            &mut pass,
            frame,
            self.particle_materials.bind_group(material),
            batch,
          ),
        }
      }
    }

    // The glow: whatever the scene wrote above the display's range, blurred
    // wide. A frame with nothing over the threshold blooms nothing by
    // arithmetic, so the chain does not fork on what the draw list held.
    //
    // Skipped only when the resolve would multiply it by zero, which makes the
    // skip unobservable rather than a second code path: the stale glow is read
    // and scaled away either way.
    if view.post.bloom_intensity > 0.0 {
      self.bloom.render(encoder, &targets.bloom);
    }

    // The resolve: box-filter the supersampled target, add the glow, and tone
    // map the sum into the caller's view.
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
      self
        .downsample
        .draw(&mut pass, &targets.scene_color, self.post.bind_group());
    }
  }
}
