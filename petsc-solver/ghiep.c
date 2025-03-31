#include "petscsys.h"
#include <slepceps.h>

static char help[] = "Generalized Hermitian Indefinite Eigenvalue Problem (GHIEP).\n";

int main(int argc, char **argv) {
  PetscErrorCode ierr;
  ierr = SlepcInitialize(&argc, &argv, NULL, help); if (ierr) return ierr;

  Mat A, B;
  PetscViewer viewerA, viewerB;

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, "in/A.bin", FILE_MODE_READ, &viewerA); CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD, &A); CHKERRQ(ierr);
  ierr = MatSetFromOptions(A); CHKERRQ(ierr);
  ierr = MatLoad(A, viewerA); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewerA); CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, "in/B.bin", FILE_MODE_READ, &viewerB); CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD, &B); CHKERRQ(ierr);
  ierr = MatSetFromOptions(B); CHKERRQ(ierr);
  ierr = MatLoad(B, viewerB); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewerB); CHKERRQ(ierr);

  EPS eps;
  ierr = EPSCreate(PETSC_COMM_WORLD, &eps); CHKERRQ(ierr);
  ierr = EPSSetOperators(eps, A, B); CHKERRQ(ierr);
  ierr = EPSSetProblemType(eps, EPS_GHIEP); CHKERRQ(ierr);
  ierr = EPSSetFromOptions(eps); CHKERRQ(ierr);
  ierr = EPSSolve(eps); CHKERRQ(ierr);

  PetscInt npairs;
  ierr = EPSGetConverged(eps, &npairs); CHKERRQ(ierr);

  PetscScalar kr;
  
  Vec xr;
  PetscInt xr_size;
  ierr = MatCreateVecs(A, NULL, &xr); CHKERRQ(ierr);
  VecGetSize(xr, &xr_size);

  PetscViewer viewer_eigenvals;
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, "out/eigenvals.bin", FILE_MODE_WRITE, &viewer_eigenvals); CHKERRQ(ierr);
  PetscViewerBinaryWrite(viewer_eigenvals, &npairs, 1, PETSC_INT);

  PetscViewer viewer_eigenvecs;
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, "out/eigenvecs.bin", FILE_MODE_WRITE, &viewer_eigenvecs); CHKERRQ(ierr);
  PetscViewerBinaryWrite(viewer_eigenvecs, &xr_size, 1, PETSC_INT);
  PetscViewerBinaryWrite(viewer_eigenvecs, &npairs, 1, PETSC_INT);

  for (PetscInt i = 0; i < npairs; ++i) {
    ierr = EPSGetEigenpair(eps, i, &kr, NULL, xr, NULL); CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(viewer_eigenvals, &kr, 1, PETSC_SCALAR); CHKERRQ(ierr);
    ierr = VecView(xr, viewer_eigenvecs); CHKERRQ(ierr);
  }

  ierr = PetscViewerDestroy(&viewer_eigenvecs); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer_eigenvals); CHKERRQ(ierr);
  ierr = EPSDestroy(&eps); CHKERRQ(ierr);
  ierr = MatDestroy(&A); CHKERRQ(ierr);
  ierr = MatDestroy(&B); CHKERRQ(ierr);
  ierr = VecDestroy(&xr); CHKERRQ(ierr);

  ierr = SlepcFinalize();
  return ierr;
}
