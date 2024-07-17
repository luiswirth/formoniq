extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{
  assemble::{assemble_galmat, assemble_galvec, fix_dof_coeffs},
  fe::{laplacian_neg_elmat, lumped_mass_elmat, LoadElvec},
  matrix::FaerCholesky,
  mesh::hyperbox::{HyperBoxDirichletBcMap, HyperBoxMesh},
  space::FeSpace,
};

use std::{f64::consts::TAU, fmt::Write, rc::Rc};

// $u(x, t) = sin(x_1) sin(x_2) ... sin(x_d) cos(sqrt(d) t)$

fn main() {
  tracing_subscriber::fmt::init();

  let dim: usize = 2;
  let nboxes_per_dim = 50;
  let domain_length = TAU;
  let mesh_width = domain_length / nboxes_per_dim as f64;

  let final_time = TAU / (dim as f64).sqrt();

  // CFL condition (with 5% margin)
  let mut timestep = 0.95 * mesh_width / (dim as f64).sqrt();
  let mut nsteps = (final_time / timestep).ceil() as usize;

  let anim_fps = 30;
  let anim_duration = 5;
  let anim_nframes = anim_fps * anim_duration;
  if nsteps < anim_nframes {
    nsteps = anim_nframes;
    timestep = final_time / nsteps as f64;
  }

  let mesh = HyperBoxMesh::new_unit_scaled(dim, domain_length, nboxes_per_dim);

  let space = Rc::new(FeSpace::new(mesh.mesh().clone()));

  let mut galmat_laplacian = assemble_galmat(&space, laplacian_neg_elmat);
  let mut galmat_mass = assemble_galmat(&space, lumped_mass_elmat);
  let mut galvec = assemble_galvec(&space, LoadElvec::new(|_| 0.0));

  let dirichlet_bc = HyperBoxDirichletBcMap::new(&mesh, |_| 0.0);

  fix_dof_coeffs(dirichlet_bc.clone(), &mut galmat_laplacian, &mut galvec);
  fix_dof_coeffs(dirichlet_bc, &mut galmat_mass, &mut galvec);

  let galmat_laplacian = galmat_laplacian.to_nalgebra();
  let galmat_mass = galmat_mass.to_nalgebra();

  let galmat_mass_cholesky = FaerCholesky::new(galmat_mass.clone());

  let mut file = std::fs::File::create("out/wavesol.txt").unwrap();
  std::io::Write::write_all(
    &mut file,
    format!(
      "{} {} {} {} {}\n",
      dim,
      domain_length,
      mesh.nnodes_per_dim(),
      final_time,
      nsteps
    )
    .as_bytes(),
  )
  .unwrap();

  let mut mu = na::DVector::from_iterator(
    space.ndofs(),
    (0..space.ndofs()).map(|idof| {
      let inode = idof;
      let x = mesh.nodes().coord(inode);
      (0..dim).map(|idim| x[idim].sin()).product()
    }),
  );
  let mut nu = na::DVector::zeros(space.ndofs());

  for istep in 0..nsteps {
    let _t = istep as f64 / (nsteps - 1) as f64 * final_time;
    println!("Solving wave equation at step={istep}/{nsteps}...");

    let rhs = &galmat_mass * nu + timestep * (&galvec - &galmat_laplacian * &mu);
    nu = galmat_mass_cholesky.solve(&rhs).column(0).into();
    mu += timestep * &nu;

    let contents: String = mu.row_iter().fold(String::new(), |mut s, v| {
      let _ = writeln!(s, "{}", v[0]);
      s
    });
    std::io::Write::write_all(&mut file, contents.as_bytes()).unwrap();
  }
}
