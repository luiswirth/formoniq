use bytemuck::{Pod, Zeroable};
use manifold::{
  geometry::coord::{mesh::MeshCoords, vertex_curvature_radius},
  topology::complex::Complex,
};
use wgpu::util::DeviceExt;

use crate::{mesh3d, scene::LineField};

/// Fraction of the local curvature radius (see [`vertex_curvature_radius`])
/// allowed as peak normal displacement, mirroring `WAVE_AMPLITUDE_FRACTION`'s
/// role against the scene extent in `lib.rs`: kept below 1 so a vertex never
/// reaches its own focal point, where the normal offset would fold the
/// surface back on itself.
const CURVATURE_SAFETY_FRACTION: f32 = 0.9;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
  pub position: [f32; 3],
  /// Outward unit vertex normal: the axis of the scalar standing-wave
  /// displacement (the fill/wireframe shaders add `value * normal` scaled by
  /// the wave), and the surface normal the G-buffer shades the line-field
  /// surface with.
  pub normal: [f32; 3],
  /// The colormap scalar: a `ScalarField`'s signed 0-form value, or a
  /// `LineField`'s nodal magnitude tinting the surface underneath the LIC.
  pub value: f32,
  /// A `LineField`'s per-vertex unit ambient tangent direction, interpolated
  /// across each triangle and projected to screen space by the G-buffer pass;
  /// zero for a scalar field, which has no line to draw.
  pub direction: [f32; 3],
  /// The per-vertex cap on normal displacement magnitude: a fraction of the
  /// local curvature radius (`f32::INFINITY` where curvature imposes no
  /// bound), clamping the shared global `wave.amplitude` down wherever it
  /// would otherwise fold a thin or sharply curved feature.
  pub max_displacement: f32,
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
          format: wgpu::VertexFormat::Float32x3,
        },
        wgpu::VertexAttribute {
          offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
          shader_location: 2,
          format: wgpu::VertexFormat::Float32,
        },
        wgpu::VertexAttribute {
          offset: std::mem::size_of::<[f32; 7]>() as wgpu::BufferAddress,
          shader_location: 3,
          format: wgpu::VertexFormat::Float32x3,
        },
        wgpu::VertexAttribute {
          offset: std::mem::size_of::<[f32; 10]>() as wgpu::BufferAddress,
          shader_location: 4,
          format: wgpu::VertexFormat::Float32,
        },
      ],
    }
  }

  /// The wireframe pipeline's two per-edge-instance endpoint buffers: the
  /// same attribute layout as [`Self::desc`], but stepped once per drawn
  /// *instance* (an edge) instead of per vertex, and at locations shifted by
  /// five so both of an edge's endpoints reach the same vertex-shader
  /// invocation at once -- what a single `step_mode: Vertex` buffer can never
  /// expose, since it only ever holds the current vertex, never a neighbor.
  /// The six vertices of that invocation (drawn non-indexed) fan out into the
  /// edge's thickened quad; see `wireframe.wgsl`.
  pub fn desc_endpoint_a<'a>() -> wgpu::VertexBufferLayout<'a> {
    wgpu::VertexBufferLayout {
      step_mode: wgpu::VertexStepMode::Instance,
      ..Self::desc()
    }
  }
  pub fn desc_endpoint_b<'a>() -> wgpu::VertexBufferLayout<'a> {
    wgpu::VertexBufferLayout {
      array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
      step_mode: wgpu::VertexStepMode::Instance,
      attributes: &[
        wgpu::VertexAttribute {
          offset: 0,
          shader_location: 5,
          format: wgpu::VertexFormat::Float32x3,
        },
        wgpu::VertexAttribute {
          offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
          shader_location: 6,
          format: wgpu::VertexFormat::Float32x3,
        },
        wgpu::VertexAttribute {
          offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
          shader_location: 7,
          format: wgpu::VertexFormat::Float32,
        },
        wgpu::VertexAttribute {
          offset: std::mem::size_of::<[f32; 7]>() as wgpu::BufferAddress,
          shader_location: 8,
          format: wgpu::VertexFormat::Float32x3,
        },
        wgpu::VertexAttribute {
          offset: std::mem::size_of::<[f32; 10]>() as wgpu::BufferAddress,
          shader_location: 9,
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
  /// The edge list's two endpoints, as parallel per-instance vertex buffers
  /// (index `i` of each is one edge): bound together to the wireframe
  /// pipeline and drawn non-indexed, 6 vertices per instance. See
  /// [`Vertex::desc_endpoint_a`].
  pub wireframe_a_buffer: wgpu::Buffer,
  pub wireframe_b_buffer: wgpu::Buffer,
  pub num_wireframe_edges: u32,
}

impl MeshBuffer {
  /// The mesh's own triangle surface, colored by a 0-form and carrying no line
  /// direction: the scalar-field mark.
  pub fn new(
    device: &wgpu::Device,
    topology: &Complex,
    coords: &MeshCoords,
    zero_form: &[f64],
  ) -> Self {
    let (vertices, indices, edge_a, edge_b) = surface_geometry(topology, coords, zero_form, None);
    Self::from_raw(device, &vertices, &indices, &edge_a, &edge_b)
  }

  /// A line field: the surface colored by its nodal `magnitude` (the scalar
  /// readout of the field, the same role [`Self::new`]'s 0-form plays for a
  /// genuine scalar field), each vertex additionally carrying the field's unit
  /// tangent `direction`. The direction is what the G-buffer/LIC passes draw
  /// the streamlines from; the surface itself is not normal-displaced -- unlike
  /// a scalar eigenmode, the magnitude is a derived tint, not the solution of a
  /// scalar wave equation, so a baked-in displacement would be a claim this
  /// field never makes.
  pub fn from_line_field(
    device: &wgpu::Device,
    topology: &Complex,
    coords: &MeshCoords,
    field: &LineField,
  ) -> Self {
    let (vertices, indices, edge_a, edge_b) =
      surface_geometry(topology, coords, &field.magnitude, Some(&field.direction));
    Self::from_raw(device, &vertices, &indices, &edge_a, &edge_b)
  }

  fn from_raw(
    device: &wgpu::Device,
    vertices: &[Vertex],
    indices: &[u32],
    edge_a: &[Vertex],
    edge_b: &[Vertex],
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

    let wireframe_a_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Wireframe Endpoint A Buffer"),
      contents: bytemuck::cast_slice(edge_a),
      usage: wgpu::BufferUsages::VERTEX,
    });
    let wireframe_b_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Wireframe Endpoint B Buffer"),
      contents: bytemuck::cast_slice(edge_b),
      usage: wgpu::BufferUsages::VERTEX,
    });

    Self {
      vertex_buffer,
      index_buffer,
      num_indices: indices.len() as u32,
      wireframe_a_buffer,
      wireframe_b_buffer,
      num_wireframe_edges: edge_a.len() as u32,
    }
  }
}

/// The mesh's triangle surface as GPU vertices: one per mesh vertex, carrying
/// the colormap `value`, the outward unit normal, and an optional per-vertex
/// line `direction`. Returns the fill index list and the 1-skeleton edge
/// list, split into its two endpoint arrays (`edge_a[i]`/`edge_b[i]` are edge
/// `i`'s two ends) for the wireframe pass's per-instance buffers.
fn surface_geometry(
  topology: &Complex,
  coords: &MeshCoords,
  value: &[f64],
  direction: Option<&[na::Vector3<f64>]>,
) -> (Vec<Vertex>, Vec<u32>, Vec<Vertex>, Vec<Vertex>) {
  assert_eq!(
    topology.dim(),
    2,
    "Only 2D meshes are supported for this MVP."
  );

  let nvertices = topology.skeleton_raw(0).len();
  assert_eq!(value.len(), nvertices);

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
  let curvature_radius = vertex_curvature_radius(topology, coords);

  let mut vertices = Vec::with_capacity(nvertices);
  for (i, &value) in value.iter().enumerate() {
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
    let n = vertex_normals[i].normalize();
    let d = direction.map_or(na::Vector3::zeros(), |dir| dir[i]);
    let max_displacement = (CURVATURE_SAFETY_FRACTION as f64 * curvature_radius[i]) as f32;

    vertices.push(Vertex {
      position: [x, y, z],
      normal: [n.x as f32, n.y as f32, n.z as f32],
      value: value as f32,
      direction: [d.x as f32, d.y as f32, d.z as f32],
      max_displacement,
    });
  }

  let indices: Vec<u32> = triangles.iter().flat_map(|t| t.map(|v| v as u32)).collect();

  let mut edge_a = Vec::new();
  let mut edge_b = Vec::new();
  for edge in topology.edges().handle_iter() {
    let verts = &edge.simplex().vertices;
    edge_a.push(vertices[verts[0]]);
    edge_b.push(vertices[verts[1]]);
  }

  (vertices, indices, edge_a, edge_b)
}
