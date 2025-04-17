use {
  ddf::{cochain::Cochain, reconstruct::reconstruct_at_mesh_cells_barycenters},
  formoniq::problems::hodge_laplace,
};

use std::fs;
use std::io::BufWriter;
use std::io::Write;

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

    //  manifold::io::blender::obj2coord_complex(&fs::read_to_string("/home/luis/dl/torus-naked.obj")?);
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
      println!("eigenval={eigenval:.3}");
      let eigenfunc = Cochain::new(grade, eigenfunc.into_owned());
      let cell_values = reconstruct_at_mesh_cells_barycenters(&eigenfunc, &topology, &coords);

      let file = fs::File::create(format!("{out_path}/sol{ieigen}_facevectors.txt"))?;
      let mut writer = BufWriter::new(file);
      for cell_value in cell_values {
        for comp in cell_value.coeffs() {
          write!(writer, "{comp:.6} ")?;
        }
        writeln!(writer)?;
      }
    }
  }
  Ok(())
}
