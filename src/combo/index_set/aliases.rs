use super::variants::*;
use super::IndexSet;

pub use algebraic_topology::*;
mod algebraic_topology {
  use super::*;

  pub type SimplexFace<B, O, S> = IndexSet<B, O, S>;

  pub type LocalSimplex<O, S> = SimplexFace<Local, O, S>;
  pub type RefSimplexFace = SimplexFace<Local, Sorted, Unsigned>;

  pub type MeshContainerCell = SimplexFace<Unspecified, Ordered, Signed>;
  pub type MeshContainerSimplex = SimplexFace<Unspecified, Sorted, Unsigned>;
  pub type MeshSimplex<O, S> = SimplexFace<Global, O, S>;
}

pub use exterior_algebra::*;
mod exterior_algebra {
  use super::*;

  pub type WedgeTerm<B, O, S> = IndexSet<B, O, S>;
  pub type FormComponent<B, O, S> = WedgeTerm<B, O, S>;
  pub type CanonicalFormComponent = FormComponent<Local, Sorted, Unsigned>;
}
