use bytemuck::{Pod, Zeroable};
use manifold::{geometry::coord::mesh::MeshCoords, topology::complex::Complex};
use wgpu::util::DeviceExt;

use crate::{mesh3d, scene::VectorField};

/// The physical length of an arrow glyph, as a fraction of one lattice
/// cell's own spacing -- so neighboring arrows leave a gap regardless of how
/// densely the field was sampled. Matches the `plot/` Python tool's own
/// `length = 0.8 / quiver_count` convention.
const ARROW_LENGTH_FRACTION: f64 = 0.8;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
  pub position: [f32; 3],
  /// A surface vertex: the colormap scalar, the raw (signed) 0-form value,
  /// or the nodal `magnitude` field when the surface is the blank-canvas
  /// backdrop to a vector field's arrows. A glyph vertex: its arrow's
  /// magnitude relative to the field's peak (see `push_arrow_glyph`), which
  /// the shader uses to scale the standing-wave swing so a weak sample
  /// swings less than a strong one -- never the colormap, since glyphs are
  /// drawn flat and uncolored so the colormap always reads as "the surface".
  pub value: f32,
  /// A surface vertex: the standing-wave displacement at wave amplitude 1
  /// (`value` times its own outward unit normal), added to `position`,
  /// scaled by `wave.amplitude * cos(wave.omega * wave.time)` -- the
  /// classical scalar-membrane convention. A glyph vertex: its arrow's own
  /// root (tail), fixed at the sample point -- the shader scales the
  /// corner's offset from that root instead of translating it, see
  /// `shader.wgsl`.
  pub displacement: [f32; 3],
  /// Nonzero for a glyph vertex, zero for the surface: which branch of the
  /// fragment shader's coloring this vertex takes.
  pub is_glyph: f32,
}

impl Vertex {
  pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
    wgpu::VertexBufferLayout {
      array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
      step_mode: wgpu::VertexStepMode::Vertex,
      attributes: &[
        wgpu::VertexAttribute {
          offset: 0,
          shader_location: 0,
          format: wgpu::VertexFormat::Float32x3,
        },
        wgpu::VertexAttribute {
          offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
          shader_location: 1,
          format: wgpu::VertexFormat::Float32,
        },
        wgpu::VertexAttribute {
          offset: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
          shader_location: 2,
          format: wgpu::VertexFormat::Float32x3,
        },
        wgpu::VertexAttribute {
          offset: std::mem::size_of::<[f32; 7]>() as wgpu::BufferAddress,
          shader_location: 3,
          format: wgpu::VertexFormat::Float32,
        },
      ],
    }
  }
}

pub struct MeshBuffer {
  pub vertex_buffer: wgpu::Buffer,
  pub index_buffer: wgpu::Buffer,
  pub num_indices: u32,
  pub wireframe_index_buffer: wgpu::Buffer,
  pub num_wireframe_indices: u32,
}

impl MeshBuffer {
  pub fn new(
    device: &wgpu::Device,
    topology: &Complex,
    coords: &MeshCoords,
    zero_form: &[f64],
  ) -> Self {
    let (vertices, indices, wireframe_indices) =
      Self::surface_geometry(topology, coords, zero_form);
    Self::from_raw(device, &vertices, &indices, &wireframe_indices)
  }

  /// The mesh's own triangle surface, colored by a 0-form: the vertex/index
  /// lists both [`Self::new`] and [`Self::from_vector_field`] (as the blank
  /// canvas its arrows sit on) build their buffers from.
  fn surface_geometry(
    topology: &Complex,
    coords: &MeshCoords,
    zero_form: &[f64],
  ) -> (Vec<Vertex>, Vec<u32>, Vec<u32>) {
    assert_eq!(
      topology.dim(),
      2,
      "Only 2D meshes are supported for this MVP."
    );

    let nvertices = topology.skeleton_raw(0).len();
    assert_eq!(zero_form.len(), nvertices);

    // Triangle index triples, shared by the index buffer and the normal
    // computation below -- the normal is a property of this embedding's
    // triangles, not of any one mesh (e.g. the sphere).
    let triangles: Vec<[usize; 3]> = topology
      .cells()
      .handle_iter()
      .map(|cell| {
        let verts = &cell.simplex().vertices;
        assert_eq!(verts.len(), 3);
        [verts[0], verts[1], verts[2]]
      })
      .collect();

    let oriented_triangles = mesh3d::orient_triangles(&triangles);
    let vertex_normals = mesh3d::vertex_normals(&oriented_triangles, coords);

    let mut vertices = Vec::with_capacity(nvertices);
    for (i, &value) in zero_form.iter().enumerate() {
      let coord = coords.coord(i);
      let x = coord[0] as f32;
      let y = if coord.len() > 1 {
        coord[1] as f32
      } else {
        0.0
      };
      let z = if coord.len() > 2 {
        coord[2] as f32
      } else {
        0.0
      };
      let n = vertex_normals[i];
      let value = value as f32;

      vertices.push(Vertex {
        position: [x, y, z],
        value,
        displacement: [value * n.x as f32, value * n.y as f32, value * n.z as f32],
        is_glyph: 0.0,
      });
    }

    let indices: Vec<u32> = triangles.iter().flat_map(|t| t.map(|v| v as u32)).collect();

    // Edge indices (line list) from the 1-skeleton
    let mut edge_indices = Vec::new();
    for edge in topology.edges().handle_iter() {
      let verts = &edge.simplex().vertices;
      edge_indices.push(verts[0] as u32);
      edge_indices.push(verts[1] as u32);
    }

    (vertices, indices, edge_indices)
  }

  /// A grade-1 field: the surface colored by its nodal `magnitude` (the
  /// scalar readout of a vector field, the same role `MeshBuffer::new`'s
  /// 0-form plays for a genuine scalar field), with `samples` drawn as
  /// direction-only arrows on top -- flat, uncolored (`Vertex::is_glyph`),
  /// since color already lives on the surface underneath them. Direction is
  /// normalized for the glyph's shape: an arrow's *length* reads as
  /// legibility, not magnitude, matching the `plot/` Python tool's own
  /// `normalize_vectors` convention. Each arrow's magnitude relative to the
  /// field's peak is kept for its own standing-wave swing, so a sample where
  /// the field is small swings less than one where it's large.
  pub fn from_vector_field(
    device: &wgpu::Device,
    topology: &Complex,
    coords: &MeshCoords,
    field: &VectorField,
  ) -> Self {
    let (mut vertices, mut indices, wireframe_indices) =
      Self::surface_geometry(topology, coords, &field.magnitude);
    // The vector field's own animation lives entirely in the glyphs'
    // tangential motion below, not in a normal-displaced surface -- unlike a
    // genuine scalar eigenmode, `magnitude` is a derived proxy coloring the
    // canvas, not itself the solution of a scalar wave equation, so the
    // baked-in normal displacement `surface_geometry` gives every vertex
    // would be a claim this field never makes.
    for vertex in &mut vertices {
      vertex.displacement = [0.0; 3];
    }

    let mesh_width = coords.to_edge_lengths(topology).mesh_width_max();
    let length = ARROW_LENGTH_FRACTION * mesh_width / field.lattice_resolution.max(1) as f64;
    let peak_magnitude = field.max_magnitude();

    for sample in &field.samples {
      let magnitude = sample.vector.norm();
      if magnitude < 1e-12 {
        continue;
      }
      push_arrow_glyph(
        sample.position,
        sample.vector / magnitude,
        sample.normal,
        (magnitude / peak_magnitude) as f32,
        length,
        &mut vertices,
        &mut indices,
      );
    }

    Self::from_raw(device, &vertices, &indices, &wireframe_indices)
  }

  fn from_raw(
    device: &wgpu::Device,
    vertices: &[Vertex],
    indices: &[u32],
    wireframe_indices: &[u32],
  ) -> Self {
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Vertex Buffer"),
      contents: bytemuck::cast_slice(vertices),
      usage: wgpu::BufferUsages::VERTEX,
    });

    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Index Buffer"),
      contents: bytemuck::cast_slice(indices),
      usage: wgpu::BufferUsages::INDEX,
    });

    let wireframe_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Wireframe Index Buffer"),
      contents: bytemuck::cast_slice(wireframe_indices),
      usage: wgpu::BufferUsages::INDEX,
    });

    Self {
      vertex_buffer,
      index_buffer,
      num_indices: indices.len() as u32,
      wireframe_index_buffer,
      num_wireframe_indices: wireframe_indices.len() as u32,
    }
  }
}

/// Appends one flat arrow -- a shaft and a triangular head, fan-triangulated
/// from the tail -- pointing along `direction` from `origin`, lying in the
/// tangent plane perpendicular to `up`, to `vertices` and `indices`.
///
/// `up` is the sample's own cell normal, not assumed to be the ambient
/// $z$-axis, so the glyph stays tangent to a curved surface (the sphere's
/// grade-1 eigenmodes) rather than always lying flat in the $x$-$y$ plane.
/// Sitting exactly in that tangent plane, at the surface, would z-fight the
/// canvas underneath -- handled not by a geometric lift here (fragile: a
/// world-space offset shrinks to nothing in clip space at a distance or a
/// grazing angle) but by the same clip-space depth nudge the wireframe
/// pipeline already uses, keyed off `is_glyph` in the shader.
///
/// `relative_magnitude` is the sample's true magnitude divided by the
/// field's peak, baked into every corner's `value` so the shader can scale
/// the standing-wave swing per glyph -- `origin` is baked into every
/// corner's `displacement` as the arrow's fixed root, about which that swing
/// scales, rather than a translation applied to the whole rigid shape.
fn push_arrow_glyph(
  origin: na::Vector3<f64>,
  direction: na::Vector3<f64>,
  up: na::Vector3<f64>,
  relative_magnitude: f32,
  length: f64,
  vertices: &mut Vec<Vertex>,
  indices: &mut Vec<u32>,
) {
  let right = up.cross(&direction).normalize();

  let shaft_len = length * 0.6;
  let shaft_half_width = length * 0.06;
  let head_half_width = length * 0.18;

  let point = |u: f64, v: f64| origin + direction * u + right * v;
  let corners = [
    point(0.0, shaft_half_width),        // tail, left
    point(shaft_len, shaft_half_width),  // shaft/head junction, left
    point(shaft_len, head_half_width),   // head base, left
    point(length, 0.0),                  // tip
    point(shaft_len, -head_half_width),  // head base, right
    point(shaft_len, -shaft_half_width), // shaft/head junction, right
    point(0.0, -shaft_half_width),       // tail, right
  ];

  let base = vertices.len() as u32;
  let root = [origin.x as f32, origin.y as f32, origin.z as f32];
  vertices.extend(corners.iter().map(|p| Vertex {
    position: [p.x as f32, p.y as f32, p.z as f32],
    value: relative_magnitude,
    displacement: root,
    is_glyph: 1.0,
  }));
  for k in 1..corners.len() as u32 - 1 {
    indices.extend([base, base + k, base + k + 1]);
  }
}
