//! The volume pass: a field on a solid drawn as a participating medium.
//!
//! The one pass whose primitive is neither a triangle of the mesh nor a
//! billboard over one. An $n$-manifold reduces to `min(n, 2)` for the
//! rasterizer, and at $n >= 3$ that leaves the interior undrawn: a solid's
//! boundary is what a triangle can carry, and the inside is a medium the eye ray
//! integrates through.
//!
//! It runs *after* the scene pass rather than among its items, and that is not a
//! scheduling convenience. The march is clamped by the scene's depth buffer, and
//! a texture cannot be sampled in the pass that writes it as an attachment. The
//! ordering is also what makes the compositing right: front-to-back
//! emission-absorption needs no sorting because the ray is the depth order, and
//! the one thing it must know is where the opaque geometry cut it off.
//!
//! **Several volumes would be one march, not several passes.** Two overlapping
//! media composite correctly only if a single ray interleaves their samples in
//! depth order; running this pass twice blends one wholly over the other and is
//! wrong wherever they intersect. Nothing here forecloses that -- it is a longer
//! loop over more bound textures -- but it is not what a second `VolumeBatch`
//! would give.

use bytemuck::{Pod, Zeroable};

use super::{color_target, primitive, shader_module, uniform::UniformBinding};
use crate::volume::VolumeGrid;

/// The layout a [`VolumeBatch`]'s texture and sampler bind against.
///
/// A free function, like `deposit_read_layout`, because both the pass and the
/// batch need it and neither owns the other: WebGPU compares bind group layouts
/// structurally, so two identically-declared layouts are interchangeable and the
/// display layer can build a batch without reaching into the renderer.
pub fn volume_read_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
  device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
    label: Some("Volume Bind Group Layout"),
    entries: &[
      wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
          sample_type: wgpu::TextureSampleType::Float { filterable: true },
          view_dimension: wgpu::TextureViewDimension::D3,
          multisampled: false,
        },
        count: None,
      },
      wgpu::BindGroupLayoutEntry {
        binding: 1,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
      },
    ],
  })
}

/// How the medium is read: where its grid sits, how a texel decodes, and the two
/// scalars that turn a normalized field value into absorption and emission.
///
/// Mirrors `VolumeMaterial` in `preamble.wgsl` field for field; the layout test
/// checks the two agree in bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct VolumeMaterial {
  pub inv_view_proj: [[f32; 4]; 4],
  pub origin: [f32; 4],
  pub size: [f32; 4],
  pub value_min: f32,
  pub value_range: f32,
  pub min_val: f32,
  pub max_val: f32,
  pub diverging: f32,
  pub density: f32,
  pub emission: f32,
  pub step_size: f32,
  pub wave_omega: f32,
  /// Filled by the renderer from the frame, never by the display: see
  /// `inv_view_proj`.
  pub time: f32,
  /// Public only so a caller may spread `..Default::default()`; never read.
  pub _pad: [f32; 2],
}

impl Default for VolumeMaterial {
  fn default() -> Self {
    Self {
      inv_view_proj: nalgebra::Matrix4::identity().into(),
      origin: [0.0; 4],
      size: [1.0; 4],
      value_min: 0.0,
      value_range: 1.0,
      min_val: 0.0,
      max_val: 1.0,
      diverging: 0.0,
      density: 0.0,
      emission: 0.0,
      step_size: 1.0,
      wave_omega: 0.0,
      time: 0.0,
      _pad: [0.0; 2],
    }
  }
}

/// The sampled field, as a 3D texture, with the affine decoding that inverts the
/// bake's normalization.
///
/// Stored `R8Unorm`, which is a deliberate trade and the reason the material
/// carries `value_min`/`value_range`. Eight bits is 1/255 of the field's own
/// range, imperceptible in a fog whose opacity is an integral along a ray, and
/// it is the only widely *filterable* single-channel format: `R32Float` needs
/// the optional `float32-filterable` feature, without which the medium would be
/// blocky, and `Rgba16Float` costs eight bytes a voxel, which at this grid size
/// is tens of megabytes the web build should not pay.
pub struct VolumeBatch {
  bind_group: wgpu::BindGroup,
  /// The decoding the material needs, carried with the texture that determined
  /// it so a caller cannot pair one with the other's.
  pub value_min: f32,
  pub value_range: f32,
  /// The grid's world-space box, likewise.
  pub origin: [f32; 3],
  pub size: [f32; 3],
  /// A voxel's world-space diagonal: the length scale a march should not step
  /// past, since a longer step walks over voxels the bake resolved.
  pub voxel: f32,
}

impl VolumeBatch {
  pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, grid: &VolumeGrid) -> Self {
    let layout = volume_read_layout(device);
    // The normalization the texture stores. A field that vanishes identically
    // has no range to normalize against, so it encodes flat at zero rather than
    // dividing by one.
    let (lo, hi) = grid
      .values
      .iter()
      .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
        (lo.min(v), hi.max(v))
      });
    let (value_min, value_range) = if hi > lo { (lo, hi - lo) } else { (0.0, 1.0) };

    let extent = wgpu::Extent3d {
      width: grid.resolution[0] as u32,
      height: grid.resolution[1] as u32,
      depth_or_array_layers: grid.resolution[2] as u32,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
      label: Some("Volume Texture"),
      size: extent,
      mip_level_count: 1,
      sample_count: 1,
      dimension: wgpu::TextureDimension::D3,
      format: wgpu::TextureFormat::R8Unorm,
      usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
      view_formats: &[],
    });
    let texels: Vec<u8> = grid
      .values
      .iter()
      .map(|&v| (((v - value_min) / value_range).clamp(0.0, 1.0) * 255.0).round() as u8)
      .collect();
    queue.write_texture(
      wgpu::TexelCopyTextureInfo {
        texture: &texture,
        mip_level: 0,
        origin: wgpu::Origin3d::ZERO,
        aspect: wgpu::TextureAspect::All,
      },
      &texels,
      wgpu::TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: Some(extent.width),
        rows_per_image: Some(extent.height),
      },
      extent,
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    // Clamped, not repeated: the medium ends at the box, and a wrapped sample
    // would smear the far face's values back across the near one.
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
      label: Some("Volume Sampler"),
      address_mode_u: wgpu::AddressMode::ClampToEdge,
      address_mode_v: wgpu::AddressMode::ClampToEdge,
      address_mode_w: wgpu::AddressMode::ClampToEdge,
      mag_filter: wgpu::FilterMode::Linear,
      min_filter: wgpu::FilterMode::Linear,
      mipmap_filter: wgpu::MipmapFilterMode::Nearest,
      ..Default::default()
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: Some("Volume Bind Group"),
      layout: &layout,
      entries: &[
        wgpu::BindGroupEntry {
          binding: 0,
          resource: wgpu::BindingResource::TextureView(&view),
        },
        wgpu::BindGroupEntry {
          binding: 1,
          resource: wgpu::BindingResource::Sampler(&sampler),
        },
      ],
    });

    let voxel = (0..3)
      .map(|axis| grid.size[axis] / grid.resolution[axis] as f32)
      .fold(0.0f32, f32::max);
    Self {
      bind_group,
      value_min,
      value_range,
      origin: grid.origin,
      size: grid.size,
      voxel,
    }
  }
}

/// The pipeline and the bind group layouts; the texture lives in a
/// [`VolumeBatch`], the per-frame scalars in the material binding.
pub struct VolumePass {
  pipeline: wgpu::RenderPipeline,
  depth_layout: wgpu::BindGroupLayout,
}

impl VolumePass {
  pub fn new(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    material: &UniformBinding<VolumeMaterial>,
  ) -> Self {
    let layout = volume_read_layout(device);
    let depth_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
      label: Some("Volume Depth Bind Group Layout"),
      entries: &[wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
          sample_type: wgpu::TextureSampleType::Depth,
          view_dimension: wgpu::TextureViewDimension::D2,
          multisampled: false,
        },
        count: None,
      }],
    });
    let shader = shader_module(device, "Volume Shader", include_str!("volume.wgsl"));
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some("Volume Pipeline Layout"),
      bind_group_layouts: &[Some(material.layout()), Some(&layout), Some(&depth_layout)],
      immediate_size: 0,
    });
    // Premultiplied "over": the shader accumulates front-to-back, so what it
    // returns is already the medium's own radiance and the coverage it took.
    let blend = wgpu::BlendState {
      color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
      },
      alpha: wgpu::BlendComponent::REPLACE,
    };
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
      label: Some("Volume Pipeline"),
      layout: Some(&pipeline_layout),
      vertex: wgpu::VertexState {
        module: &shader,
        entry_point: Some("vs_main"),
        buffers: &[],
        compilation_options: wgpu::PipelineCompilationOptions::default(),
      },
      fragment: Some(wgpu::FragmentState {
        module: &shader,
        entry_point: Some("fs_main"),
        targets: &[color_target(format, blend)],
        compilation_options: wgpu::PipelineCompilationOptions::default(),
      }),
      primitive: primitive(),
      // No depth attachment: the medium reads the scene's depth as a texture and
      // writes none of its own, having no surface to put there.
      depth_stencil: None,
      multisample: wgpu::MultisampleState::default(),
      multiview_mask: None,
      cache: None,
    });
    Self {
      pipeline,
      depth_layout,
    }
  }

  /// The scene depth binding, rebuilt with the targets on every resize.
  pub fn bind_depth(&self, device: &wgpu::Device, depth: &wgpu::TextureView) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: Some("Volume Depth Bind Group"),
      layout: &self.depth_layout,
      entries: &[wgpu::BindGroupEntry {
        binding: 0,
        resource: wgpu::BindingResource::TextureView(depth),
      }],
    })
  }

  pub fn draw(
    &self,
    pass: &mut wgpu::RenderPass<'_>,
    material: &wgpu::BindGroup,
    depth: &wgpu::BindGroup,
    batch: &VolumeBatch,
  ) {
    pass.set_pipeline(&self.pipeline);
    pass.set_bind_group(0, material, &[]);
    pass.set_bind_group(1, &batch.bind_group, &[]);
    pass.set_bind_group(2, depth, &[]);
    pass.draw(0..3, 0..1);
  }
}
