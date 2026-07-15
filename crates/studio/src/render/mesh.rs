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
  pub value: f32,       // 0-form scalar value
  pub normal: [f32; 3], // outward vertex normal, for standing-wave displacement
  pub _pad: f32,
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

      vertices.push(Vertex {
        position: [x, y, z],
        value: value as f32,
        normal: [n.x as f32, n.y as f32, n.z as f32],
        _pad: 0.0,
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

  /// A grade-1 field's samples as arrow glyphs, on top of the mesh's own
  /// surface shown blank (every vertex at the colormap's zero -- a canvas,
  /// not a field in its own right). Direction is normalized: an arrow's
  /// *length* reads as legibility, not magnitude, matching the `plot/`
  /// Python tool's own `normalize_vectors` convention -- magnitude is
  /// carried by `value` (the arrow's color) instead, which is the
  /// unambiguous channel since two nearby arrows of different length are
  /// otherwise hard to compare by eye.
  pub fn from_vector_field(
    device: &wgpu::Device,
    topology: &Complex,
    coords: &MeshCoords,
    field: &VectorField,
  ) -> Self {
    let nvertices = topology.skeleton_raw(0).len();
    let (mut vertices, mut indices, wireframe_indices) =
      Self::surface_geometry(topology, coords, &vec![0.0; nvertices]);

    let mesh_width = coords.to_edge_lengths(topology).mesh_width_max();
    let length = ARROW_LENGTH_FRACTION * mesh_width / field.lattice_resolution as f64;

    for sample in &field.samples {
      let magnitude = sample.vector.norm();
      if magnitude < 1e-12 {
        continue;
      }
      push_arrow_glyph(
        sample.position,
        sample.vector / magnitude,
        sample.normal,
        length,
        magnitude as f32,
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
fn push_arrow_glyph(
  origin: na::Vector3<f64>,
  direction: na::Vector3<f64>,
  up: na::Vector3<f64>,
  length: f64,
  value: f32,
  vertices: &mut Vec<Vertex>,
  indices: &mut Vec<u32>,
) {
  let right = up.cross(&direction).normalize();
  // Lifted a hair off the surface so the glyph doesn't z-fight the blank
  // canvas underneath it.
  let lift = up * (length * 0.02);

  let shaft_len = length * 0.6;
  let shaft_half_width = length * 0.06;
  let head_half_width = length * 0.18;

  let point = |u: f64, v: f64| origin + lift + direction * u + right * v;
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
  let n = [up.x as f32, up.y as f32, up.z as f32];
  vertices.extend(corners.iter().map(|p| Vertex {
    position: [p.x as f32, p.y as f32, p.z as f32],
    value,
    normal: n,
    _pad: 0.0,
  }));
  for k in 1..corners.len() as u32 - 1 {
    indices.extend([base, base + k, base + k + 1]);
  }
}
