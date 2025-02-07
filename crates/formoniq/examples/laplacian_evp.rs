extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{
  fe::{
    evaluate_fe_function_at_cell_barycenters, evaluate_fe_function_cell_vertices, write_evaluations,
  },
  operators::FeFunction,
  problems::hodge_laplace,
};
use geometry::coord::{
  manifold::{cartesian::CartesianMeshInfo, dim3::TriangleSurface3D},
  write_coords,
};
use topology::{complex::TopologyComplex, simplex::write_simplicies};

use std::{
  fs::{self, File},
  io::BufWriter,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let path = "out/laplacian_evp";
  fs::remove_dir_all(path).unwrap();
  fs::create_dir_all(path).unwrap();

  let dim = 2;
  let grade = 1;
  let neigen = 10;

  let ncells_axis = 3;
  let (topology, coords) = CartesianMeshInfo::new_unit(dim, ncells_axis).compute_coord_cells();
  let triangle_mesh = TriangleSurface3D::from_coord_skeleton(topology.clone(), coords.clone());
  let topology = TopologyComplex::from_cell_skeleton(topology);
  let metric = coords.to_edge_lengths(&topology);

  let (eigenvals, eigenfuncs) =
    hodge_laplace::solve_hodge_laplace_evp(&topology, &metric, grade, neigen);
  for (ieigen, (&eigenval, eigenfunc)) in eigenvals.iter().zip(eigenfuncs.column_iter()).enumerate()
  {
    println!("eigenval={eigenval:.3}");
    let eigenfunc = FeFunction::new(grade, eigenfunc.into_owned());

    let cell_evals = evaluate_fe_function_cell_vertices(&eigenfunc, &topology, &coords);
    println!("{cell_evals:#?}");

    let file = File::create(format!("{path}/coords.txt"))?;
    let writer = BufWriter::new(file);
    write_coords(writer, &coords)?;

    let file = File::create(format!("{path}/cells.txt"))?;
    let writer = BufWriter::new(file);
    write_simplicies(writer, topology.cells().set_iter())?;

    let file = File::create(format!("{path}/evaluations.txt"))?;
    let writer = BufWriter::new(file);
    write_evaluations(writer, &cell_evals)?;

    let mut displacements = na::DVector::zeros(coords.nvertices());
    let cell_values = evaluate_fe_function_at_cell_barycenters(&eigenfunc, &topology, &coords);
    for cell in topology.cells().handle_iter() {
      let value = &cell_values[cell.kidx()];
      let norm = value.coeffs().norm();

      for vertex in cell.vertices() {
        displacements[vertex.kidx()] += norm / vertex.cocells().count() as f64;
      }
    }

    displacements = &displacements / displacements.max();
    if eigenval <= 1e-9 {
      displacements *= 0.0;
    }

    let mut surface = triangle_mesh.clone();
    surface
      .vertex_coords_mut()
      .coord_iter_mut()
      .zip(displacements.iter())
      .for_each(|(mut c, d)| c[2] = *d);
    //surface.displace_normal(&displacements);

    std::fs::write(
      format!("{path}/eigenfunc{ieigen}.obj"),
      surface.to_obj_string().as_bytes(),
    )
    .unwrap();
  }
  Ok(())
}
