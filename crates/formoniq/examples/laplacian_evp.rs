extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{
  fe::evaluate_fe_function_at_dof_barycenters, operators::FeFunction, problems::hodge_laplace,
};
use geometry::coord::manifold::{cartesian::CartesianMesh, dim3::TriangleSurface3D};

use std::fs;

fn main() {
  let path = "out/laplacian_evp";
  fs::remove_dir_all(path).unwrap();
  fs::create_dir_all(path).unwrap();

  let dim = 2;
  let grade = 0;
  let neigen = 10;

  let ncells_axis = 4;
  let coord_mesh = CartesianMesh::new_unit(dim, ncells_axis).compute_coord_manifold();
  let triangle_mesh = TriangleSurface3D::from_coord_skeleton(coord_mesh.clone());
  let coord_mesh = coord_mesh.into_coord_complex();
  let metric_mesh = coord_mesh.to_metric_complex();

  let dof_skeleton = metric_mesh.topology().skeleton(grade);

  let (eigenvals, eigenfuncs) = hodge_laplace::solve_hodge_laplace_evp(&metric_mesh, grade, neigen);
  for (ieigen, (&eigenval, eigenfunc)) in eigenvals.iter().zip(eigenfuncs.column_iter()).enumerate()
  {
    println!("eigenval={eigenval:.3}");
    let eigenfunc = FeFunction::new(grade, eigenfunc.into_owned());

    let mut displacements = na::DVector::zeros(coord_mesh.coords().nvertices());
    let dof_values = evaluate_fe_function_at_dof_barycenters(&eigenfunc, &coord_mesh);
    for dof_simp in dof_skeleton.iter() {
      let value = &dof_values[dof_simp.kidx()];
      let norm = value.coeffs().norm();

      for vertex in dof_simp.vertices() {
        displacements[vertex.kidx()] += norm;
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
}
