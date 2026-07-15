use bytemuck::{Pod, Zeroable};
use manifold::{dim3, geometry::coord::mesh::MeshCoords, topology::complex::Complex};
use wgpu::util::DeviceExt;

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

    let oriented_triangles = dim3::orient_triangles(&triangles);
    let vertex_normals = dim3::vertex_normals(&oriented_triangles, coords);

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

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Vertex Buffer"),
      contents: bytemuck::cast_slice(&vertices),
      usage: wgpu::BufferUsages::VERTEX,
    });

    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Index Buffer"),
      contents: bytemuck::cast_slice(&indices),
      usage: wgpu::BufferUsages::INDEX,
    });

    let wireframe_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Wireframe Index Buffer"),
      contents: bytemuck::cast_slice(&edge_indices),
      usage: wgpu::BufferUsages::INDEX,
    });

    Self {
      vertex_buffer,
      index_buffer,
      num_indices: indices.len() as u32,
      wireframe_index_buffer,
      num_wireframe_indices: edge_indices.len() as u32,
    }
  }
}
