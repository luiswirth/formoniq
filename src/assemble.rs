use crate::{
  fe::{ElmatProvider, ElvecProvider},
  mesh::SimplicialManifold,
  space::{DofIdx, FeSpace},
  sparse::SparseMatrix,
};

pub fn assemble_galmat_raw(
  ndofs: usize,
  elmat: na::DMatrix<f64>,
  cells_dofs: &[Vec<DofIdx>],
) -> SparseMatrix {
  let mut galmat = SparseMatrix::zeros(ndofs, ndofs);
  for cell_dofs in cells_dofs {
    for (ilocal, &iglobal) in cell_dofs.iter().enumerate() {
      for (jlocal, &jglobal) in cell_dofs.iter().enumerate() {
        galmat.push(iglobal, jglobal, elmat[(ilocal, jlocal)]);
      }
    }
  }
  galmat
}

/// Assembly algorithm for the Galerkin Matrix.
pub fn assemble_galmat(space: &FeSpace, elmat: impl ElmatProvider) -> SparseMatrix {
  let mut galmat = SparseMatrix::zeros(space.ndofs(), space.ndofs());
  for cell in space.mesh().cells() {
    let cell = cell.as_cell_complex();
    let elmat = elmat.eval(&cell);
    let dof_faces = &cell.faces()[space.rank()];

    for (ilocal, iglobal) in dof_faces.iter().copied().enumerate() {
      for (jlocal, jglobal) in dof_faces.iter().copied().enumerate() {
        galmat.push(iglobal, jglobal, elmat[(ilocal, jlocal)]);
      }
    }
  }
  galmat
}

/// Assembly algorithm for the Galerkin Vector.
pub fn assemble_galvec(space: &FeSpace, elvec: impl ElvecProvider) -> na::DVector<f64> {
  let mut galvec = na::DVector::zeros(space.ndofs());
  for cell in space.mesh().cells() {
    let cell = cell.as_cell_complex();
    let elvec = elvec.eval(&cell);
    let dof_faces = &cell.faces()[space.rank()];
    for (ilocal, iglobal) in dof_faces.iter().copied().enumerate() {
      galvec[iglobal] += elvec[ilocal];
    }
  }
  galvec
}

pub fn drop_boundary_dofs_galmat(mesh: &SimplicialManifold, galmat: &mut SparseMatrix) {
  drop_dofs_galmat(&mesh.boundary_nodes(), galmat)
}

pub fn drop_dofs_galmat(dofs: &[DofIdx], galmat: &mut SparseMatrix) {
  assert!(galmat.nrows() == galmat.ncols());
  let ndofs_old = galmat.ncols();
  let ndofs_new = ndofs_old - dofs.len();

  let (_, _, mut triplets) = std::mem::take(galmat).into_parts();
  for (ndrops, mut idof) in dofs.iter().copied().enumerate() {
    idof -= ndrops;
    let mut itriplet = 0;
    while itriplet < triplets.len() {
      let mut triplet = triplets[itriplet];
      let r = &mut triplet.0;
      let c = &mut triplet.1;

      if *r == idof || *c == idof {
        triplets.remove(itriplet);
      } else {
        if *r > idof {
          *r -= 1;
        }
        if *c > idof {
          *c -= 1;
        }

        itriplet += 1;
      }
    }
  }

  *galmat = SparseMatrix::new(ndofs_new, ndofs_new, triplets);
}

pub fn drop_dofs_galvec(dofs: &[DofIdx], galvec: &mut na::DVector<f64>) {
  *galvec = std::mem::take(galvec).remove_rows_at(dofs);
}

pub fn reintroduce_zeroed_boundary_dofs_galsols(
  mesh: &SimplicialManifold,
  galsols: &mut na::DMatrix<f64>,
) {
  reintroduce_zeroed_dofs_galsols(&mesh.boundary_nodes(), galsols)
}

pub fn reintroduce_zeroed_dofs_galsols(dofs: &[DofIdx], galsols: &mut na::DMatrix<f64>) {
  let mut galsol_owned = std::mem::take(galsols);
  for (ninserts, &dof) in dofs.iter().enumerate() {
    galsol_owned = galsol_owned.insert_row(ninserts + dof, 0.0);
  }
  *galsols = galsol_owned;
}
