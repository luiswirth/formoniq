use std::rc::Rc;

use crate::{mesh::SimplexEntity, space::FeSpace, Dim};

struct ElmatProvider;

/// Assembly algorithm for the Galerkin Matrix in Lagrangian (0-form) FE.
/// `cell_dim`: The simplex dimension we are assembling over.
pub fn assemble_galmat_lagrangian<ElmatProvider>(
  cell_dim: Dim,
  trial_space: Rc<FeSpace>,
  test_space: Rc<FeSpace>,
  elmat_provider: ElmatProvider,
) -> nas::CscMatrix<f64>
where
  ElmatProvider: Fn(&SimplexEntity) -> na::DMatrix<f64>,
{
  assert!(Rc::ptr_eq(trial_space.mesh(), test_space.mesh()));
  let mesh = trial_space.mesh();

  // Lagrangian (0-form) has dofs associated with the nodes.
  let mut galmat = nas::CooMatrix::new(test_space.ndofs(0), trial_space.ndofs(0));
  for entity in mesh.skeleton(cell_dim).simplicies() {
    let elmat = elmat_provider(entity);
    for (ilocal, iglobal) in test_space
      .dof_indices_global(*entity, 0)
      .iter()
      .copied()
      .enumerate()
    {
      for (jlocal, jglobal) in trial_space
        .dof_indices_global(*entity, 0)
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
