extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use {
  formoniq::problems::hodge_laplace,
  manifold::{gen::cartesian::CartesianMeshInfo, topology::complex::Complex},
  whitney::cochain::Cochain,
};

use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let path = "out/laplacian_evp";
  let _ = fs::remove_dir_all(path);
  fs::create_dir_all(path).unwrap();

  let dim = 2;
  let grade = 1;
  let neigen = 3;

  let (topology, coords) =
    manifold::io::gmsh::gmsh2coord_complex(&fs::read("/home/luis/annulus/annulus1.msh")?);

  //let ncells_axis = 10;
  //let (topology, coords) = CartesianMeshInfo::new_unit(dim, ncells_axis).compute_coord_complex();
  let metric = coords.to_edge_lengths(&topology);

  manifold::io::save_skeleton_to_file(&topology, dim, format!("{path}/cells.skel"))?;
  manifold::io::save_skeleton_to_file(&topology, 1, format!("{path}/edges.skel"))?;
  manifold::io::save_coords_to_file(&coords, format!("{path}/vertices.coords"))?;

  let (eigenvals, eigenfuncs) =
    hodge_laplace::solve_hodge_laplace_evp(&topology, &metric, grade, neigen);
  for (ieigen, (&eigenval, eigenfunc)) in eigenvals.iter().zip(eigenfuncs.column_iter()).enumerate()
  {
    println!("eigenval={eigenval:.3}");
    let eigenfunc = Cochain::new(grade, eigenfunc.into_owned());
    whitney::io::save_cochain_to_file(&eigenfunc, format!("{path}/eigenfunc{ieigen}.cochain"))?;
  }
  Ok(())
}
