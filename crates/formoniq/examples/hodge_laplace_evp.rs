use {
  ddf::cochain::Cochain, ddf::whitney::form::WhitneyForm, exterior::exterior_dim,
  formoniq::problems::hodge_laplace, manifold::geometry::coord::simplex::SimplexCoords,
};

use std::{
  fs::{self, File},
  io::Write,
  path::PathBuf,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let out_path = "out/laplacian_evp";
  let _ = fs::remove_dir_all(out_path);
  fs::create_dir_all(out_path).unwrap();

  println!("Enter mesh file path.");
  let mut path = String::new();
  std::io::stdin().read_line(&mut path)?;
  let path: PathBuf = path.trim().to_string().into();

  let file_ext = path.extension().expect("Missing file extension.");

  let (topology, coords) = match file_ext.to_str().unwrap() {
    "msh" => {
      let gmsh_file = fs::read(path)?;
      manifold::io::gmsh::gmsh2coord_complex(&gmsh_file)
    }
    "obj" => {
      let obj_file = fs::read(path)?;
      let obj_string = String::from_utf8(obj_file)?;
      manifold::io::blender::obj2coord_complex(&obj_string)
    }
    _ => panic!("Unkown file extension."),
  };

  let ambient_dim = coords.dim();
  let metric = coords.to_edge_lengths(&topology);

  fs::write(
    format!("{out_path}/mesh.obj"),
    manifold::io::blender::coord_complex2obj(&topology, &coords),
  )?;

  println!("Enter exterior grade.");
  let mut grade = String::new();
  std::io::stdin().read_line(&mut grade)?;
  let grade: usize = grade.trim().parse()?;

  println!("Enter number of eigens.");
  let mut neigen = String::new();
  std::io::stdin().read_line(&mut neigen)?;
  let neigen: usize = neigen.trim().parse()?;

  let (eigenvals, _, eigenfuncs) =
    hodge_laplace::solve_hodge_laplace_evp(&topology, &metric, grade, neigen);
  for (ieigen, (&eigenval, eigenfunc)) in eigenvals.iter().zip(eigenfuncs.column_iter()).enumerate()
  {
    println!("ieigen={ieigen}, eigenval={eigenval:.3}");
    let eigen_cochain = Cochain::new(grade, eigenfunc.into_owned());

    let mut file = File::create(format!("{out_path}/eigen{ieigen}.txt"))?;

    // Write header
    for i in 0..ambient_dim {
      write!(&mut file, "x{i} ")?;
    }
    for i in 0..exterior_dim(ambient_dim, grade) {
      write!(&mut file, "v{i} ")?;
    }
    writeln!(&mut file)?;

    for cell in topology.cells().handle_iter() {
      let cell_coords = SimplexCoords::from_simplex_and_coords(&cell, &coords);
      let whitney = WhitneyForm::new(eigen_cochain.clone(), &topology, &coords);
      let cell_value = whitney.eval_known_cell(cell, &cell_coords.barycenter());

      for coord in cell_coords.barycenter().iter() {
        write!(file, "{coord:.6} ")?;
      }
      for comp in cell_value.coeffs() {
        write!(file, "{comp:.6} ")?;
      }
      writeln!(file)?;
    }
  }
  Ok(())
}
