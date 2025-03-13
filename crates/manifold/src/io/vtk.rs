use crate::{geometry::coord::VertexCoords, topology::skeleton::Skeleton};

use vtkio::{
  model::{
    Attributes, ByteOrder, CellType, Cells, UnstructuredGridPiece, Version, VertexNumbers, Vtk,
  },
  IOBuffer,
};

pub fn embedded_mesh_to_vtk(cells: &Skeleton, coords: &VertexCoords) -> Vtk {
  let cell_type = match cells.dim() {
    1 => CellType::Line,
    2 => CellType::Triangle,
    3 => CellType::Tetra,
    _ => panic!("Bad Mesh for VTK export"),
  };

  let points = IOBuffer::new(coords.matrix().iter().copied().collect());

  let connectivity = cells
    .simplex_iter()
    .flat_map(|simp| simp.vertices.clone())
    .map(|i| i as u64)
    .collect();
  let offsets = cells
    .simplex_iter()
    .map(|simp| simp.nvertices() as u64)
    .scan(0, |offset, nverts| {
      let this = *offset;
      *offset += nverts;
      Some(this)
    })
    .collect();

  let cell_verts = VertexNumbers::XML {
    connectivity,
    offsets,
  };
  let types = vec![cell_type; cells.len()];
  let cells = Cells { cell_verts, types };

  let grid = UnstructuredGridPiece {
    points,
    cells,
    data: Attributes::default(),
  };
  let data = grid.into();

  Vtk {
    version: Version::new((4, 2)),
    title: String::from("Formoniq VTK Export"),
    byte_order: ByteOrder::native(),
    data,
    file_path: None,
  }
}
