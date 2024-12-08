//! Combinatorial structures relevant for exterior algebra.

use super::variants::*;
use super::IndexSet;

pub type ExteriorProdTerm<O, S> = IndexSet<Local, O, S>;

pub type FormComponent<O, S> = ExteriorProdTerm<O, S>;
pub type CanonicalFormComponent = FormComponent<Sorted, Unsigned>;
