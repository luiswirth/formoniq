extern crate nalgebra as na;

use {
  ddf::cochain::cochain_projection,
  exterior::{field::DiffFormClosure, MultiForm},
  manifold::{
    gen::cartesian::CartesianMeshInfo, geometry::coord::CoordRef, topology::complex::Complex,
  },
};

use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let path = "out/visual";
  let _ = fs::remove_dir_all(path);
  fs::create_dir_all(path).unwrap();

  let dim = 2;
  let grade = 1;

  let ncells_axis = 10;
  let (topology, coords) = CartesianMeshInfo::new_unit(dim, ncells_axis).compute_coord_cells();
  let topology = Complex::from_cells(topology);
  let _metric = coords.to_edge_lengths(&topology);

  manifold::io::save_skeleton_to_file(&topology, dim, format!("{path}/cells.skel"))?;
  manifold::io::save_skeleton_to_file(&topology, 1, format!("{path}/edges.skel"))?;
  manifold::io::save_coords_to_file(&coords, format!("{path}/vertices.coords"))?;

  let form = |p: CoordRef| {
    let x = 0.5 - p[0];
    let y = 0.5 - p[1];
    let comps = na::dvector![-y, x].normalize();
    MultiForm::line(comps)
  };
  let form = DiffFormClosure::new(Box::new(form), dim, grade);
  let cochain = cochain_projection(&form, &topology, &coords);
  ddf::io::save_cochain_to_file(&cochain, format!("{path}/proj.cochain"))?;

  Ok(())
}
