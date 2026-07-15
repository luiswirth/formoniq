use std::sync::Arc;
use wgpu::util::DeviceExt;
use wgpu::{Device, Queue, RenderPipeline, Surface, SurfaceConfiguration};
use winit::{
  application::ApplicationHandler,
  event::*,
  event_loop::{ActiveEventLoop, EventLoop},
  window::{Window, WindowId},
};

use crate::render::{
  camera::{Camera, CameraUniform},
  mesh::{MeshBuffer, Vertex},
};
use crate::scene::Scene;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

pub mod render;
pub mod scene;
pub mod ui;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

// Icosphere subdivision depth. The Laplace-Beltrami eigensolve is dense in the
// vertex count, so keep this modest for an instant startup; bump for fidelity.
const SPHERE_SUBDIVISIONS: usize = 3;
const SPHERE_MODES: usize = 10;
// Which eigenmode colors the surface. Mode 0 is the constant harmonic; mode 4
// is the first grade-2 spherical harmonic.
const DISPLAY_MODE: usize = 4;

fn create_depth_texture(device: &Device, config: &SurfaceConfiguration) -> wgpu::TextureView {
  let size = wgpu::Extent3d {
    width: config.width,
    height: config.height,
    depth_or_array_layers: 1,
  };
  let desc = wgpu::TextureDescriptor {
    label: Some("Depth Texture"),
    size,
    mip_level_count: 1,
    sample_count: 1,
    dimension: wgpu::TextureDimension::D2,
    format: DEPTH_FORMAT,
    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
    view_formats: &[],
  };
  let texture = device.create_texture(&desc);
  texture.create_view(&wgpu::TextureViewDescriptor::default())
}

struct State<'a> {
  surface: Surface<'a>,
  device: Device,
  queue: Queue,
  config: SurfaceConfiguration,
  size: winit::dpi::PhysicalSize<u32>,
  render_pipeline: RenderPipeline,
  wireframe_pipeline: RenderPipeline,
  depth_view: wgpu::TextureView,

  camera: Camera,
  camera_uniform: CameraUniform,
  camera_buffer: wgpu::Buffer,
  camera_bind_group: wgpu::BindGroup,

  mesh_buffer: MeshBuffer,

  // kept alive to back bounds_bind_group's binding; never read directly
  #[allow(dead_code)]
  bounds_buffer: wgpu::Buffer,
  bounds_bind_group: wgpu::BindGroup,

  // Mouse state for orbit controls
  mouse_pressed: bool,
  last_mouse_pos: Option<winit::dpi::PhysicalPosition<f64>>,
}

impl<'a> State<'a> {
  async fn new(window: Arc<Window>) -> State<'a> {
    let size = window.inner_size();
    let instance = wgpu::Instance::default();

    let surface = instance.create_surface(window).unwrap();
    let adapter = instance
      .request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
        ..Default::default()
      })
      .await
      .unwrap();

    let (device, queue) = adapter
      .request_device(&wgpu::DeviceDescriptor::default())
      .await
      .unwrap();

    let config = surface
      .get_default_config(&adapter, size.width.max(1), size.height.max(1))
      .unwrap();
    surface.configure(&device, &config);

    // Real formoniq output: Laplace-Beltrami eigenfunctions on the unit
    // sphere (discrete spherical harmonics), colored by one mode.
    let scene = Scene::spherical_harmonics(SPHERE_SUBDIVISIONS, SPHERE_MODES);
    let field = &scene.fields[DISPLAY_MODE];
    let (field_min, field_max) = field.bounds();

    let mesh_buffer = MeshBuffer::new(&device, &scene.topology, &scene.coords, field.values());

    let mut camera = Camera::new(config.width as f32 / config.height as f32);
    camera.target = nalgebra::Point3::origin();
    camera.distance = 3.0;
    camera.pitch = 0.3;
    camera.yaw = -1.57;

    let mut camera_uniform = CameraUniform::new();
    camera_uniform.update_view_proj(&camera);

    let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Camera Buffer"),
      contents: bytemuck::cast_slice(&[camera_uniform]),
      usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let camera_bind_group_layout =
      device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[wgpu::BindGroupLayoutEntry {
          binding: 0,
          visibility: wgpu::ShaderStages::VERTEX,
          ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
          },
          count: None,
        }],
        label: Some("camera_bind_group_layout"),
      });

    let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      layout: &camera_bind_group_layout,
      entries: &[wgpu::BindGroupEntry {
        binding: 0,
        resource: camera_buffer.as_entire_binding(),
      }],
      label: Some("camera_bind_group"),
    });

    // Bounds uniform for fragment shader
    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct BoundsUniform {
      min_val: f32,
      max_val: f32,
      _pad1: f32,
      _pad2: f32,
    }

    let bounds_uniform = BoundsUniform {
      min_val: field_min,
      max_val: field_max,
      _pad1: 0.0,
      _pad2: 0.0,
    };

    let bounds_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Bounds Buffer"),
      contents: bytemuck::cast_slice(&[bounds_uniform]),
      usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bounds_bind_group_layout =
      device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[wgpu::BindGroupLayoutEntry {
          binding: 0,
          visibility: wgpu::ShaderStages::FRAGMENT,
          ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
          },
          count: None,
        }],
        label: Some("bounds_bind_group_layout"),
      });

    let bounds_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      layout: &bounds_bind_group_layout,
      entries: &[wgpu::BindGroupEntry {
        binding: 0,
        resource: bounds_buffer.as_entire_binding(),
      }],
      label: Some("bounds_bind_group"),
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
      label: Some("Shader"),
      source: wgpu::ShaderSource::Wgsl(include_str!("render/shader.wgsl").into()),
    });

    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some("Render Pipeline Layout"),
      bind_group_layouts: &[
        Some(&camera_bind_group_layout),
        Some(&bounds_bind_group_layout),
      ],
      immediate_size: 0,
    });

    let depth_stencil = Some(wgpu::DepthStencilState {
      format: DEPTH_FORMAT,
      depth_write_enabled: Some(true),
      depth_compare: Some(wgpu::CompareFunction::Less),
      stencil: wgpu::StencilState::default(),
      bias: wgpu::DepthBiasState::default(),
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
      label: Some("Render Pipeline"),
      layout: Some(&render_pipeline_layout),
      vertex: wgpu::VertexState {
        module: &shader,
        entry_point: Some("vs_main"),
        compilation_options: Default::default(),
        buffers: &[Some(Vertex::desc())],
      },
      fragment: Some(wgpu::FragmentState {
        module: &shader,
        entry_point: Some("fs_main"),
        compilation_options: Default::default(),
        targets: &[Some(wgpu::ColorTargetState {
          format: config.format,
          blend: Some(wgpu::BlendState::REPLACE),
          write_mask: wgpu::ColorWrites::ALL,
        })],
      }),
      primitive: wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::TriangleList,
        strip_index_format: None,
        front_face: wgpu::FrontFace::Ccw,
        cull_mode: None,
        polygon_mode: wgpu::PolygonMode::Fill,
        unclipped_depth: false,
        conservative: false,
      },
      depth_stencil: depth_stencil.clone(),
      multisample: wgpu::MultisampleState::default(),
      multiview_mask: None,
      cache: None,
    });

    // Wireframe pipeline — LineList with camera-only bind group
    let wireframe_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
      label: Some("Wireframe Shader"),
      source: wgpu::ShaderSource::Wgsl(include_str!("render/wireframe.wgsl").into()),
    });

    let wireframe_pipeline_layout =
      device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Wireframe Pipeline Layout"),
        bind_group_layouts: &[Some(&camera_bind_group_layout)],
        immediate_size: 0,
      });

    let wireframe_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
      label: Some("Wireframe Pipeline"),
      layout: Some(&wireframe_pipeline_layout),
      vertex: wgpu::VertexState {
        module: &wireframe_shader,
        entry_point: Some("vs_main"),
        compilation_options: Default::default(),
        buffers: &[Some(Vertex::desc())],
      },
      fragment: Some(wgpu::FragmentState {
        module: &wireframe_shader,
        entry_point: Some("fs_main"),
        compilation_options: Default::default(),
        targets: &[Some(wgpu::ColorTargetState {
          format: config.format,
          blend: Some(wgpu::BlendState::REPLACE),
          write_mask: wgpu::ColorWrites::ALL,
        })],
      }),
      primitive: wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::LineList,
        strip_index_format: None,
        front_face: wgpu::FrontFace::Ccw,
        cull_mode: None,
        polygon_mode: wgpu::PolygonMode::Fill,
        unclipped_depth: false,
        conservative: false,
      },
      depth_stencil,
      multisample: wgpu::MultisampleState::default(),
      multiview_mask: None,
      cache: None,
    });

    let depth_view = create_depth_texture(&device, &config);

    Self {
      surface,
      device,
      queue,
      config,
      size,
      render_pipeline,
      wireframe_pipeline,
      depth_view,
      camera,
      camera_uniform,
      camera_buffer,
      camera_bind_group,
      mesh_buffer,
      bounds_buffer,
      bounds_bind_group,
      mouse_pressed: false,
      last_mouse_pos: None,
    }
  }

  fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
    if new_size.width > 0 && new_size.height > 0 {
      self.size = new_size;
      self.config.width = new_size.width;
      self.config.height = new_size.height;
      self.surface.configure(&self.device, &self.config);
      self.depth_view = create_depth_texture(&self.device, &self.config);

      self.camera.aspect = self.config.width as f32 / self.config.height as f32;
      self.camera_uniform.update_view_proj(&self.camera);
      self.queue.write_buffer(
        &self.camera_buffer,
        0,
        bytemuck::cast_slice(&[self.camera_uniform]),
      );
    }
  }

  fn handle_input(&mut self, event: &WindowEvent) {
    match event {
      WindowEvent::MouseInput {
        state: button_state,
        button: winit::event::MouseButton::Left,
        ..
      } => {
        self.mouse_pressed = *button_state == ElementState::Pressed;
        if !self.mouse_pressed {
          self.last_mouse_pos = None;
        }
      }
      WindowEvent::CursorMoved { position, .. } => {
        if self.mouse_pressed {
          if let Some(last) = self.last_mouse_pos {
            let dx = (position.x - last.x) as f32;
            let dy = (position.y - last.y) as f32;
            self.camera.yaw += dx * 0.005;
            self.camera.pitch -= dy * 0.005;
            // Clamp pitch to avoid gimbal lock
            self.camera.pitch = self.camera.pitch.clamp(-1.5, 1.5);
            self.camera_uniform.update_view_proj(&self.camera);
            self.queue.write_buffer(
              &self.camera_buffer,
              0,
              bytemuck::cast_slice(&[self.camera_uniform]),
            );
          }
          self.last_mouse_pos = Some(*position);
        }
      }
      WindowEvent::MouseWheel { delta, .. } => {
        let scroll = match delta {
          winit::event::MouseScrollDelta::LineDelta(_, y) => *y,
          winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
        };
        self.camera.distance -= scroll;
        self.camera.distance = self.camera.distance.clamp(1.0, 100.0);
        self.camera_uniform.update_view_proj(&self.camera);
        self.queue.write_buffer(
          &self.camera_buffer,
          0,
          bytemuck::cast_slice(&[self.camera_uniform]),
        );
      }
      _ => {}
    }
  }

  fn render(&mut self) -> Result<(), ()> {
    let output = match self.surface.get_current_texture() {
      wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
      wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
        self.resize(self.size);
        return Ok(());
      }
      wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => return Ok(()),
      wgpu::CurrentSurfaceTexture::Validation => return Ok(()),
    };
    let view = output
      .texture
      .create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = self
      .device
      .create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Render Encoder"),
      });

    {
      let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Render Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
          view: &view,
          resolve_target: None,
          ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color {
              r: 0.1,
              g: 0.1,
              b: 0.1,
              a: 1.0,
            }),
            store: wgpu::StoreOp::Store,
          },
          depth_slice: None,
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
          view: &self.depth_view,
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

      // Draw filled triangles
      render_pass.set_pipeline(&self.render_pipeline);
      render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
      render_pass.set_bind_group(1, &self.bounds_bind_group, &[]);
      render_pass.set_vertex_buffer(0, self.mesh_buffer.vertex_buffer.slice(..));
      render_pass.set_index_buffer(
        self.mesh_buffer.index_buffer.slice(..),
        wgpu::IndexFormat::Uint32,
      );
      render_pass.draw_indexed(0..self.mesh_buffer.num_indices, 0, 0..1);

      // Draw wireframe edges on top
      render_pass.set_pipeline(&self.wireframe_pipeline);
      render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
      render_pass.set_vertex_buffer(0, self.mesh_buffer.vertex_buffer.slice(..));
      render_pass.set_index_buffer(
        self.mesh_buffer.wireframe_index_buffer.slice(..),
        wgpu::IndexFormat::Uint32,
      );
      render_pass.draw_indexed(0..self.mesh_buffer.num_wireframe_indices, 0, 0..1);
    }

    self.queue.submit(std::iter::once(encoder.finish()));
    self.queue.present(output);

    Ok(())
  }
}

#[derive(Default)]
struct App<'a> {
  window: Option<Arc<Window>>,
  state: Option<State<'a>>,
}

impl<'a> ApplicationHandler for App<'a> {
  fn resumed(&mut self, event_loop: &ActiveEventLoop) {
    if self.window.is_none() {
      let window = Arc::new(
        event_loop
          .create_window(Window::default_attributes())
          .unwrap(),
      );

      #[cfg(target_arch = "wasm32")]
      {
        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
          .and_then(|win| win.document())
          .and_then(|doc| {
            let dst = doc.get_element_by_id("wasm-example")?;
            let canvas = web_sys::Element::from(window.canvas()?);
            dst.append_child(&canvas).ok()?;
            Some(())
          })
          .expect("Couldn't append canvas to document body.");
      }

      let state = pollster::block_on(State::new(window.clone()));
      self.window = Some(window);
      self.state = Some(state);
    }
  }

  fn window_event(
    &mut self,
    event_loop: &ActiveEventLoop,
    _window_id: WindowId,
    event: WindowEvent,
  ) {
    if let Some(state) = &mut self.state {
      match event {
        WindowEvent::CloseRequested
        | WindowEvent::KeyboardInput {
          event:
            winit::event::KeyEvent {
              state: ElementState::Pressed,
              logical_key: winit::keyboard::Key::Named(winit::keyboard::NamedKey::Escape),
              ..
            },
          ..
        } => event_loop.exit(),
        WindowEvent::Resized(physical_size) => {
          state.resize(physical_size);
        }
        WindowEvent::RedrawRequested => {
          let _ = state.render();
        }
        other => {
          state.handle_input(&other);
        }
      }
    }
  }

  fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
    if let Some(window) = &self.window {
      window.request_redraw();
    }
  }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
  cfg_if::cfg_if! {
      if #[cfg(target_arch = "wasm32")] {
          std::panic::set_hook(Box::new(console_error_panic_hook::hook));
          console_log::init_with_level(log::Level::Warn).expect("Couldn't initialize logger");
      } else {
          env_logger::init();
      }
  }

  let event_loop = EventLoop::new().unwrap();
  let mut app = App::default();
  let _ = event_loop.run_app(&mut app);
}
