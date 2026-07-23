# iterative

Iterative solving of sparse linear systems,
built around a single object: an **approximate inverse** `B ≈ A⁻¹`.

The one abstraction earns three names depending on where it is plugged in:

- iterated alone, it is a **solver**
- wrapped in a Krylov method, it is a **preconditioner**
- sitting on a level of a multigrid hierarchy, it is a **smoother**

They are the same object.
`Jacobi` is a poor solver, a poor preconditioner and an excellent smoother:
one fact (it damps high-frequency error and only that) read against three job descriptions.

## Interfaces

- `LinearOperator`:
  apply `A`, the only thing a Krylov method asks of its system matrix.
  Matrix-free by default.
- `ApproxInverse`:
  apply `B ≈ A⁻¹`.
  The central trait.
  Solvers, preconditioners and smoothers all are one.
- `SelfAdjoint`:
  a marker on `ApproxInverse` asserting `B` is a fixed symmetric positive-definite operator.
  Conjugate gradients accepts a preconditioner only through this bound,
  so a non-symmetric approximate inverse is rejected at compile time
  rather than silently breaking convergence.

Entry-needing preconditioners (a diagonal, a triangular sweep)
take the assembled sparse matrix at construction.
Everything else is matrix-free.
Consumers are generic over the operator and the preconditioner, monomorphized,
with no dynamic dispatch on the apply path.

The crate is discretization-agnostic:
it knows nothing of meshes or differential forms.
A geometric multigrid hierarchy, or a structure-preserving (auxiliary-space) preconditioner,
is supplied from above as an implementor of these traits.
