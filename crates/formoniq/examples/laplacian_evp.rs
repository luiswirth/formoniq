extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use {
  formoniq::{
    fe::evaluate_fe_function_at_cell_barycenters, operators::FeFunction, problems::hodge_laplace,
  },
  manifold::{
    dim3::TriangleSurface3D, gen::cartesian::CartesianMeshInfo, topology::complex::Complex,
  },
};

use std::fs;

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
  let topology = Complex::from_cell_skeleton(topology);
  let metric = coords.to_edge_lengths(&topology);

  manifold::io::save_coords_to_file(&coords, format!("{path}/coords.txt"))?;
  manifold::io::save_cells_to_file(&topology, format!("{path}/cells.txt"))?;

  let (eigenvals, eigenfuncs) =
    hodge_laplace::solve_hodge_laplace_evp(&topology, &metric, grade, neigen);
  for (ieigen, (&eigenval, eigenfunc)) in eigenvals.iter().zip(eigenfuncs.column_iter()).enumerate()
  {
    println!("eigenval={eigenval:.3}");
    let eigenfunc = FeFunction::new(grade, eigenfunc.into_owned());

    formoniq::io::save_evaluations_to_file(
      &eigenfunc,
      &topology,
      &coords,
      format!("{path}/evaluations.txt"),
    )?;

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
      manifold::io::blender::to_obj_string(&surface).as_bytes(),
    )
    .unwrap();
  }
  Ok(())
}
