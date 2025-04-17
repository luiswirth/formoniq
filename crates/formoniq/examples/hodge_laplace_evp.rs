use {ddf::cochain::Cochain, formoniq::problems::hodge_laplace};

use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let out_path = "out/laplacian_evp";
  let _ = fs::remove_dir_all(out_path);
  fs::create_dir_all(out_path).unwrap();

  let grade = 1;
  let neigen = 10;

  for mesh_file in fs::read_dir("/home/luis/thesis/mesh/gmsh/torus/")? {
    let mesh_file = mesh_file?;
    let file_name = mesh_file.file_name().into_string().unwrap();
    let out_path = format!("{out_path}/{file_name}");
    let _ = fs::remove_dir_all(&out_path);
    fs::create_dir_all(&out_path).unwrap();

    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&fs::read(mesh_file.path())?);
    fs::write(
      format!("{out_path}/mesh.obj"),
      manifold::io::blender::coord_complex2obj(&topology, &coords),
    )?;

    let metric = coords.to_edge_lengths(&topology);

    let (eigenvals, _, eigenfuncs) =
      hodge_laplace::solve_hodge_laplace_evp(&topology, &metric, grade, neigen);
    for (ieigen, (&eigenval, eigenfunc)) in
      eigenvals.iter().zip(eigenfuncs.column_iter()).enumerate()
    {
      println!("ieigen={ieigen}, eigenval={eigenval:.3}");
      let _eigenfunc = Cochain::new(grade, eigenfunc.into_owned());
    }
  }
  Ok(())
}
