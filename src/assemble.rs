use crate::{space::DofHandler, Dim};

struct ElmatProvider;

pub fn assemble_matrix_lagrangian(
  cell_dim: Dim,
  dofh_trial: DofHandler,
  dofh_test: DofHandler,
  elmat_provider: ElmatProvider,
) -> na::DMatrix<f64> {
  todo!()
}
