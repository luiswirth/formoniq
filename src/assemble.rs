use std::rc::Rc;

use crate::space::FeSpace;

/// Assembly algorithm for the Galerkin Matrix in Lagrangian (0-form) FE.
/// `cell_dim`: The simplex dimension we are assembling over.
pub fn assemble_galmat_lagrangian(space: Rc<FeSpace>) -> nas::CscMatrix<f64> {
  let mesh = space.mesh();
  let cell_dim = mesh.dim_intrinsic();

  // Lagrangian (0-form) has dofs associated with the nodes.
  let mut galmat = nas::CooMatrix::new(space.ndofs(), space.ndofs());
  for (icell, _) in mesh.dsimplicies(cell_dim).iter().enumerate() {
    let cell_geo = mesh.coordinate_simplex((cell_dim, icell));
    let elmat = cell_geo.elmat();
    for (ilocal, iglobal) in space
      .dof_indices_global((cell_dim, icell))
      .iter()
      .copied()
      .enumerate()
    {
      for (jlocal, jglobal) in space
        .dof_indices_global((cell_dim, icell))
        .iter()
        .copied()
        .enumerate()
      {
        galmat.push(iglobal, jglobal, elmat[(ilocal, jlocal)]);
      }
    }
  }
  nas::CscMatrix::from(&galmat)
}
