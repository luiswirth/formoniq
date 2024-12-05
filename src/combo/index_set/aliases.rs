use super::variants::*;
use super::IndexSet;

pub use algebraic_topology::*;
mod algebraic_topology {
  use super::*;

  pub type Simplex<B, O, S> = IndexSet<B, O, S>;

  pub type RefSimplex = Simplex<Local, Sorted, Unsigned>;
  pub type LocalSimplex<O, S> = Simplex<Local, O, S>;
  pub type GlobalSimplex<O, S> = Simplex<Global, O, S>;

  pub type MeshSimplex = Simplex<Unspecified, Sorted, Unsigned>;
  pub type MeshCell = Simplex<Unspecified, Ordered, Signed>;
}

pub use exterior_algebra::*;
mod exterior_algebra {
  use super::*;

  pub type WedgeTerm<O, S> = IndexSet<Local, O, S>;

  pub type FormComponent<O, S> = WedgeTerm<O, S>;
  pub type CanonicalFormComponent = FormComponent<Sorted, Unsigned>;
}
