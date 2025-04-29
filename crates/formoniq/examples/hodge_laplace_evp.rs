use {
  common::linalg::nalgebra::Vector, ddf::cochain::Cochain, ddf::whitney::form::WhitneyForm,
  formoniq::problems::hodge_laplace, manifold::geometry::coord::simplex::SimplexCoords,
};

use std::{
  fs::{self, File},
  io::Write,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let out_path = "out/laplacian_evp";
  let _ = fs::remove_dir_all(out_path);
  fs::create_dir_all(out_path).unwrap();

  let grade = 1;
  let neigen = 10;

  let mut path = String::new();
  println!("Enter mesh file path.");
  std::io::stdin().read_line(&mut path)?;
  path = path.trim().to_string();
  println!("{path}");

  let obj_file = fs::read(path)?;

  let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&obj_file);
  let metric = coords.to_edge_lengths(&topology);

  fs::write(
    format!("{out_path}/mesh.obj"),
    manifold::io::blender::coord_complex2obj(&topology, &coords),
  )?;

  let (eigenvals, _, eigenfuncs) =
    hodge_laplace::solve_hodge_laplace_evp(&topology, &metric, grade, neigen);
  for (ieigen, (&eigenval, eigenfunc)) in eigenvals.iter().zip(eigenfuncs.column_iter()).enumerate()
  {
    println!("ieigen={ieigen}, eigenval={eigenval:.3}");
    let eigen_cochain = Cochain::new(grade, eigenfunc.into_owned());

    let mut vertex_values = vec![Vector::zeros(3); topology.vertices().len()];
    for cell in topology.cells().handle_iter() {
      let cell_coords = SimplexCoords::from_simplex_and_coords(&cell, &coords);
      let whitney = WhitneyForm::new(eigen_cochain.clone(), &topology, &coords);
      for (vertex_coord, &ivertex) in cell_coords.vertices.coord_iter().zip(&cell.vertices) {
        let vertex_value = whitney.eval_known_cell(cell, vertex_coord);
        vertex_values[ivertex] += vertex_value.coeffs();
      }
    }

    let mut file = File::create(format!("{out_path}/eigen{ieigen}.txt"))?;
    writeln!(&mut file, "x y z vx vy vz")?;
    for (vertex_value, vertex_coord) in vertex_values.iter().zip(coords.coord_iter()) {
      writeln!(
        file,
        "{} {} {} {} {} {}",
        vertex_coord[0],
        vertex_coord[1],
        vertex_coord[2],
        vertex_value[0],
        vertex_value[1],
        vertex_value[2],
      )?;
    }
  }
  Ok(())
}
