extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use {
  exterior::{field::DifferentialFormClosure, MultiForm},
  manifold::{
    gen::cartesian::CartesianMeshInfo, geometry::coord::CoordRef, topology::complex::Complex,
  },
  whitney::discretize_form_on_mesh,
};

use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let path = "out/visual";
  let _ = fs::remove_dir_all(path);
  fs::create_dir_all(path).unwrap();

  let dim = 2;
  let grade = 1;

  let ncells_axis = 1;
  let (topology, coords) = CartesianMeshInfo::new_unit(dim, ncells_axis).compute_coord_cells();
  let topology = Complex::from_cell_skeleton(topology);
  let _metric = coords.to_edge_lengths(&topology);

  manifold::io::save_coords_to_file(&coords, format!("{path}/coords.txt"))?;
  manifold::io::save_cells_to_file(&topology, format!("{path}/cells.txt"))?;

  let form = |p: CoordRef| {
    let _x = p[0];
    let _y = p[1];
    let comps = na::dvector![1.0, 0.0];
    MultiForm::from_grade1(comps)
  };
  let form = DifferentialFormClosure::new(Box::new(form), dim, grade);
  let cochain = discretize_form_on_mesh(&form, &topology, &coords);

  for (simplex, coeff) in topology.skeleton(grade).set_iter().zip(cochain.coeffs()) {
    let simplex = simplex.vertices.indices();
    println!("W[{simplex:?}] = {coeff}");
  }

  //let mut cochain = Cochain::zero(grade, &topology);
  //cochain.coeffs[15] = 1.0;

  formoniq::io::save_evaluations_to_file(
    &cochain,
    &topology,
    &coords,
    format!("{path}/evaluations.txt"),
  )?;

  Ok(())
}
