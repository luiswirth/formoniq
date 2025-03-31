#include <petscksp.h>

static char help[] = "Solve a symmetric/indefinite (saddle-point) linear system A x = b using PETSc.\n\
Reads A from 'in/A.bin' and b from 'in/b.bin', solves for x, writes to 'out/x.bin'.\n\n";

int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc, &argv, (char*)0, help); if (ierr) return ierr;

  /* --- 1) Read the matrix A from a binary file --- */
  Mat            A;
  PetscViewer    viewerA;
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"in/A.bin",FILE_MODE_READ,&viewerA);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD, &A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatLoad(A, viewerA);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewerA);CHKERRQ(ierr);

  /* --- 2) Read the RHS vector b from a binary file --- */
  Vec            b;
  PetscViewer    viewerB;
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"in/b.bin",FILE_MODE_READ,&viewerB);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD, &b);CHKERRQ(ierr);
  ierr = VecLoad(b, viewerB);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewerB);CHKERRQ(ierr);

  /* --- 3) Create a solution vector x with same layout as b --- */
  Vec            x;
  ierr = VecDuplicate(b, &x);CHKERRQ(ierr);

  /* --- 4) Create a KSP (Krylov Subspace) solver --- */
  KSP            ksp;
  ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);CHKERRQ(ierr);

  /*
     Because we have a symmetric or Hermitian *indefinite* matrix,
     we often choose MINRES or SYMMLQ, etc. from the command line:
       -ksp_type minres  -pc_type lu
     or try a block preconditioner for a saddle-point system, etc.
  */
  ierr = KSPSetOperators(ksp, A, A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /* --- 5) Solve A x = b --- */
  ierr = KSPSolve(ksp, b, x);CHKERRQ(ierr);

  /* --- 6) Write the solution vector x to a binary file --- */
  PetscViewer    viewerX;
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"out/x.bin",FILE_MODE_WRITE,&viewerX);CHKERRQ(ierr);
  ierr = VecView(x, viewerX);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewerX);CHKERRQ(ierr);

  /* --- 7) Clean up and finalize PETSc --- */
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
