//! Headless rendering: the same frame graph the window drives, recorded into an
//! offscreen texture and read back to the CPU.
//!
//! Nothing here is a second renderer or a second display. A scene becomes a
//! mesh and field display through `display.rs`, exactly as `app.rs`
//! does it, and the only thing this module supplies that the window does not is
//! where the frame's time comes from: the standing wave's own period, sampled
//! at $t_k = k T \/ N$, instead of a clock. If
//! a material were constructed here, the two callers could disagree about what
//! a field looks like -- which is the whole reason the display layer is not in
//! `app.rs`.
//!
//! Native-only: there is no filesystem to write to on wasm, and no subprocess
//! to pipe to.

use std::io::Write;
use std::path::Path;

use crate::display::{default_camera, scene_extent, FieldDisplay, MeshDisplay};
use crate::gallery::{MeshSource, Study};
use crate::render::{FrameView, GpuContext, Renderer};
use crate::scene::Scene;
use crate::ui::Selection;

/// The texture format every export renders and encodes at.
///
/// `Srgb` because the window's swapchain is: the shaders write linear values
/// that the target's sRGB encoding converts on write, so an export target
/// without it would come out visibly darker than the same frame on screen. PNG
/// is an sRGB container, so the read-back bytes are already what it wants.
const EXPORT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// The supersampling factor an export renders at when its resolution leaves
/// room for it.
///
/// Higher than the window's [`crate::render::DEFAULT_SSAA_SCALE`] for the one
/// reason the window is lower: an export is not bound by a frame-rate budget,
/// so the only question is what the image should look like, and the answer to
/// that does not vary per invocation. It is therefore a property of exporting,
/// not a knob -- nobody asks for a worse still.
const EXPORT_SSAA_SCALE_MAX: u32 = 4;

/// The supersampled pixel count one export may allocate its scene pass at.
///
/// The scene pass holds a color and a depth target at the supersampled
/// resolution, 4 bytes per pixel each, so this budget is 8 bytes times the
/// count: a little over half a gigabyte at the value below, which is what a
/// 1080p export at the full [`EXPORT_SSAA_SCALE_MAX`] already costs. Above it
/// the factor steps down rather than the export dying: the cost is quadratic in
/// the factor, so a resolution high enough to exhaust a GPU is reachable by
/// asking for one (a 4K frame at 4x is 132 megapixels), and it is the pixels
/// that are wanted, not the samples per pixel -- a large export is oversampled
/// by its own size.
const EXPORT_SSAA_PIXEL_BUDGET: u64 = 64 << 20;

/// The largest supersampling factor `size` fits inside
/// [`EXPORT_SSAA_PIXEL_BUDGET`], never above [`EXPORT_SSAA_SCALE_MAX`] and
/// never below 1 -- an export resolution so large that even a single sample per
/// pixel exceeds the budget is the caller's own request, and is passed through
/// rather than silently shrunk into a smaller image than was asked for.
///
/// Derived, not asked for: the factor follows from the size and the hardware's
/// limits, both of which the code knows and the caller does not.
fn export_ssaa_scale(size: (u32, u32)) -> u32 {
  let pixels = u64::from(size.0.max(1)) * u64::from(size.1.max(1));
  (1..=EXPORT_SSAA_SCALE_MAX)
    .take_while(|&ssaa| pixels * u64::from(ssaa) * u64::from(ssaa) <= EXPORT_SSAA_PIXEL_BUDGET)
    .last()
    .unwrap_or(1)
}

/// What one export asks for: which scene, which field of it, and at what
/// resolution and cadence.
pub struct ExportSpec {
  pub study: Study,
  pub mesh_source: MeshSource,
  /// Which field of the built scene to show, indexed over the scene's scalar
  /// fields and then its line fields -- the picker's own order. `None` opens on
  /// the same first mode the viewer does.
  pub field: Option<usize>,
  pub size: (u32, u32),
  /// How many frames to sample the period with. `None` picks the count that
  /// makes playback at `fps` run at wall-clock speed.
  pub frames: Option<u32>,
  /// Playback rate of the written clip. It does not decide *which* instants are
  /// rendered -- the period does -- only how fast they are played back.
  pub fps: u32,
}

/// A GPU context with no surface: the same device the window gets, asked for
/// without anything to present to.
///
/// `None` when the machine offers no adapter at all (a CI container without a
/// software rasterizer), which callers treat as "cannot export here" rather
/// than as a failure.
pub fn headless_context() -> Option<GpuContext> {
  let instance = wgpu::Instance::default();
  let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
    power_preference: wgpu::PowerPreference::HighPerformance,
    compatible_surface: None,
    force_fallback_adapter: false,
  }))
  .ok()?;
  let (device, queue) =
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
  Some(GpuContext { device, queue })
}

/// The offscreen target and the buffer frames are read back through.
///
/// A texture-to-buffer copy requires each row to start at a
/// [`wgpu::COPY_BYTES_PER_ROW_ALIGNMENT`] boundary, so the buffer is padded to
/// that stride and the padding is dropped on the way out. The scene's own width
/// is almost never a multiple of 64 pixels, so this is the normal path, not an
/// edge case.
struct ExportTarget {
  texture: wgpu::Texture,
  view: wgpu::TextureView,
  buffer: wgpu::Buffer,
  size: (u32, u32),
  padded_bytes_per_row: u32,
}

impl ExportTarget {
  fn new(ctx: &GpuContext, size: (u32, u32)) -> Self {
    let (width, height) = (size.0.max(1), size.1.max(1));
    let unpadded_bytes_per_row = width * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;

    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
      label: Some("Export Target"),
      size: wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
      },
      mip_level_count: 1,
      sample_count: 1,
      dimension: wgpu::TextureDimension::D2,
      format: EXPORT_FORMAT,
      usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
      view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
      label: Some("Export Readback"),
      size: u64::from(padded_bytes_per_row) * u64::from(height),
      usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
      mapped_at_creation: false,
    });

    Self {
      texture,
      view,
      buffer,
      size: (width, height),
      padded_bytes_per_row,
    }
  }

  /// Renders one frame and returns it as tightly packed RGBA rows.
  fn frame(&self, ctx: &GpuContext, renderer: &mut Renderer, view: &FrameView) -> Vec<u8> {
    let mut encoder = ctx
      .device
      .create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Export Encoder"),
      });
    renderer.render(ctx, &mut encoder, &self.view, view);
    encoder.copy_texture_to_buffer(
      wgpu::TexelCopyTextureInfo {
        texture: &self.texture,
        mip_level: 0,
        origin: wgpu::Origin3d::ZERO,
        aspect: wgpu::TextureAspect::All,
      },
      wgpu::TexelCopyBufferInfo {
        buffer: &self.buffer,
        layout: wgpu::TexelCopyBufferLayout {
          offset: 0,
          bytes_per_row: Some(self.padded_bytes_per_row),
          rows_per_image: Some(self.size.1),
        },
      },
      wgpu::Extent3d {
        width: self.size.0,
        height: self.size.1,
        depth_or_array_layers: 1,
      },
    );
    ctx.queue.submit(std::iter::once(encoder.finish()));

    let slice = self.buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
      let _ = tx.send(r);
    });
    // The map completes on a queue callback, so the queue has to be pumped for
    // it to ever fire: unlike the windowed loop, nothing else here polls.
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv()
      .expect("the map callback always sends")
      .expect("mapping an export readback buffer cannot fail");

    let unpadded = (self.size.0 * 4) as usize;
    let pixels = slice
      .get_mapped_range()
      .chunks_exact(self.padded_bytes_per_row as usize)
      .flat_map(|row| row[..unpadded].to_vec())
      .collect();
    self.buffer.unmap();
    pixels
  }
}

/// One scene, built and displayed: everything an export needs that does not
/// change from frame to frame.
///
/// The mesh display is kept alive alongside the field display because the draw
/// list borrows both.
struct Displayed {
  mesh: MeshDisplay,
  field: FieldDisplay,
  camera: crate::render::camera::Camera,
  /// The selected field's standing-wave frequency $omega = sqrt(lambda)$, or
  /// `None` for a field that is not an eigenmode and so has no period.
  omega: Option<f64>,
}

impl Displayed {
  fn build(ctx: &GpuContext, spec: &ExportSpec) -> Result<Self, String> {
    let mesh_data = spec.mesh_source.build()?;
    let scene = spec.study.build(&mesh_data);

    let selection = match spec.field {
      None => crate::demos::default_selection(&scene),
      Some(index) => selection_at(&scene, index).ok_or_else(|| {
        format!(
          "--field {index} is out of range: this view has {} field(s)",
          scene.fields.len() + scene.line_fields.len()
        )
      })?,
    };

    let extent = scene_extent(&scene) as f32;
    let mesh = MeshDisplay::build(&ctx.device, &scene);
    let (field, attributes) = FieldDisplay::build(ctx, &scene, selection, extent);
    mesh.write_attributes(&ctx.queue, &attributes);
    let aspect = spec.size.0.max(1) as f32 / spec.size.1.max(1) as f32;
    let camera = default_camera(&scene, aspect);
    let omega = eigenvalue_of(&scene, selection).map(f64::sqrt);

    Ok(Self {
      mesh,
      field,
      camera,
      omega,
    })
  }

  /// The standing wave's period $T = 2 pi \/ omega$, or `None` for a field that
  /// does not oscillate: one that is not an eigenmode, and equally a harmonic
  /// mode ($lambda = 0$), whose period is unbounded rather than large.
  fn period(&self) -> Option<f64> {
    self
      .omega
      .filter(|&omega| omega > 1e-9)
      .map(|omega| std::f64::consts::TAU / omega)
  }

  /// The instants one clip renders: $t_k = k T \/ N$, for $k in {0, ..., N-1}$.
  ///
  /// The period is what is sampled, and `fps` is only the rate it is played
  /// back at -- which is the way round that makes the loop exact. Sampling
  /// $t_k = k \/ "fps"$ instead would divide the period by a number it has no
  /// reason to divide evenly: the wrap from the last frame to the first would
  /// jump by whatever phase the rounding left over, up to half a frame. Here
  /// $N$ divides $T$ by construction, so frame $N$ *is* frame $0$ and the clip
  /// closes on itself.
  ///
  /// `frames` therefore chooses how densely the period is sampled; its default
  /// is the count that makes playback at `fps` run at wall-clock speed. A field
  /// with no period does not move, so it is one still.
  fn frame_times(&self, spec: &ExportSpec) -> Vec<f32> {
    let Some(period) = self.period() else {
      return vec![0.0];
    };
    let n = spec
      .frames
      .unwrap_or_else(|| (period * f64::from(spec.fps)).round().max(1.0) as u32);
    (0..n.max(1))
      .map(|k| (f64::from(k) * period / f64::from(n.max(1))) as f32)
      .collect()
  }
}

/// The scene's fields in the picker's order -- scalars, then line fields --
/// indexed flat, which is what `--field N` names.
fn selection_at(scene: &Scene, index: usize) -> Option<Selection> {
  if index < scene.fields.len() {
    Some(Selection::Scalar(index))
  } else if index - scene.fields.len() < scene.line_fields.len() {
    Some(Selection::Line(index - scene.fields.len()))
  } else {
    None
  }
}

fn eigenvalue_of(scene: &Scene, selection: Selection) -> Option<f64> {
  match selection {
    Selection::Scalar(i) => scene.fields[i].eigenvalue,
    Selection::Line(i) => scene.line_fields[i].eigenvalue,
  }
}

/// Renders `spec` and writes it to `path`, as a PNG still or, for an `.mp4`, a
/// clip piped through `ffmpeg`.
pub fn export(spec: &ExportSpec, path: &Path) -> Result<(), String> {
  let ctx = headless_context().ok_or("no GPU adapter available for a headless render")?;
  let mut renderer = Renderer::new(&ctx, EXPORT_FORMAT, export_ssaa_scale(spec.size));
  let target = ExportTarget::new(&ctx, spec.size);
  let displayed = Displayed::build(&ctx, spec)?;

  let is_video = path
    .extension()
    .is_some_and(|e| e.eq_ignore_ascii_case("mp4"));
  if is_video {
    export_video(&ctx, &mut renderer, &target, &displayed, spec, path)
  } else {
    let pixels = render_at(&ctx, &mut renderer, &target, &displayed, 0.0, 0);
    write_png(&pixels, target.size, path)
  }
}

/// Renders one already-built frame -- the window's live camera, field and clock
/// time -- to a PNG at `path`. Unlike [`export`], nothing is rebuilt from a
/// spec: the caller's own draw list and camera are rendered exactly as they are
/// on screen, so the still is the current view.
///
/// The scene pass is still re-created at the export texture format with the
/// export supersampling rather than copied off the swapchain, so a window frame drawn
/// at the interactive SSAA budget is written at the higher export fidelity and
/// without the egui panels composited over it.
pub fn export_frame_png(ctx: &GpuContext, frame: &FrameView, path: &Path) -> Result<(), String> {
  let mut renderer = Renderer::new(ctx, EXPORT_FORMAT, export_ssaa_scale(frame.size));
  let target = ExportTarget::new(ctx, frame.size);
  let pixels = target.frame(ctx, &mut renderer, frame);
  write_png(&pixels, target.size, path)
}

/// One frame of the displayed scene at `time`, advecting `steps` further first.
/// The one place a `FrameView` is built here, so the still and the clip cannot
/// differ in anything but the instant they name.
///
/// `steps` is separate from `time` because a particle population is stepped, not
/// evaluated: the caller owns how far it has already gone. See
/// [`crate::display::steps_at`].
fn render_at(
  ctx: &GpuContext,
  renderer: &mut Renderer,
  target: &ExportTarget,
  displayed: &Displayed,
  time: f32,
  steps: u32,
) -> Vec<u8> {
  let items = displayed.field.draw_list(
    &displayed.mesh,
    crate::ui::MeshView::default(),
    crate::ui::FieldView::default(),
  );
  let frame = FrameView {
    items: &items,
    camera: &displayed.camera,
    size: target.size,
    time,
    steps,
    // An export has no viewer to ask, so it takes the display's own default --
    // the same rung the window opens on.
    post: crate::display::post_uniform(crate::ui::Post::default()),
  };
  target.frame(ctx, renderer, &frame)
}

fn write_png(pixels: &[u8], size: (u32, u32), path: &Path) -> Result<(), String> {
  image::save_buffer(path, pixels, size.0, size.1, image::ColorType::Rgba8)
    .map_err(|e| format!("writing {}: {e}", path.display()))
}

/// Pipes raw frames to an `ffmpeg` subprocess.
///
/// No encoder is vendored: `ffmpeg` is the one dependency an export of this kind
/// has that `cargo` cannot supply, so its absence is reported as itself rather
/// than as a broken pipe.
fn export_video(
  ctx: &GpuContext,
  renderer: &mut Renderer,
  target: &ExportTarget,
  displayed: &Displayed,
  spec: &ExportSpec,
  path: &Path,
) -> Result<(), String> {
  let times = displayed.frame_times(spec);
  let (width, height) = target.size;

  let mut child = std::process::Command::new("ffmpeg")
    .args(["-y", "-f", "rawvideo", "-pix_fmt", "rgba"])
    .args(["-s", &format!("{width}x{height}")])
    .args(["-r", &spec.fps.to_string()])
    .args(["-i", "-"])
    // yuv420p and even dimensions are what makes the result playable outside a
    // developer's own machine: the pixel format every H.264 decoder implements,
    // and the chroma subsampling that requires both axes to be even.
    .args(["-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    .args(["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"])
    .arg(path)
    .stdin(std::process::Stdio::piped())
    .stdout(std::process::Stdio::null())
    .stderr(std::process::Stdio::null())
    .spawn()
    .map_err(|e| match e.kind() {
      std::io::ErrorKind::NotFound => {
        "`ffmpeg` was not found on PATH; it is required to write an .mp4 (a .png \
         still needs no external tool)"
          .to_string()
      }
      _ => format!("starting ffmpeg: {e}"),
    })?;

  let mut stdin = child.stdin.take().expect("stdin was piped");
  // The particles are stepped across the clip rather than restarted per frame:
  // one population, carried forward, so consecutive frames are consecutive
  // instants of one flow. The times are increasing, so each frame owes the
  // difference.
  let mut steps_taken = 0;
  for (k, &time) in times.iter().enumerate() {
    let steps = crate::display::steps_at(time).saturating_sub(steps_taken);
    steps_taken += steps;
    let pixels = render_at(ctx, renderer, target, displayed, time, steps);
    stdin
      .write_all(&pixels)
      .map_err(|e| format!("piping frame {k} to ffmpeg: {e}"))?;
  }
  drop(stdin);

  let status = child
    .wait()
    .map_err(|e| format!("waiting for ffmpeg: {e}"))?;
  if !status.success() {
    return Err(format!("ffmpeg exited with {status}"));
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;

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
  /// flat color -- i.e. the frame graph actually drew the scene, rather than
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
      study: Study::Cochains(crate::demos::triforce_examples()),
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
    // and with zero steps its compute pass never runs -- so the dispatch, its
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
}
