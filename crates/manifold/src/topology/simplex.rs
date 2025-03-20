use super::VertexIdx;
use crate::Dim;

use multi_index::{binomial, sign::Sign};

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct Simplex {
  pub vertices: Vec<VertexIdx>,
}
impl Simplex {
  pub fn new(vertices: Vec<VertexIdx>) -> Self {
    Self { vertices }
  }
  pub fn nvertices(&self) -> usize {
    self.vertices.len()
  }
  pub fn dim(&self) -> Dim {
    self.nvertices() - 1
  }

  pub fn subsimps(&self, sub_dim: Dim) -> impl Iterator<Item = Self> {
    itertools::Itertools::combinations(self.clone().into_iter(), sub_dim + 1).map(Self::from)
  }
  pub fn boundary(&self) -> impl Iterator<Item = SignedSimplex> {
    let mut sign = Sign::from_parity(self.nvertices() - 1);
    self.subsimps(self.dim() - 1).map(move |simp| {
      let this_sign = sign;
      sign.flip();
      SignedSimplex::new(simp, this_sign)
    })
  }

  pub fn iter(&self) -> std::slice::Iter<'_, usize> {
    self.vertices.iter()
  }

  pub fn with_sign(self, sign: Sign) -> SignedSimplex {
    SignedSimplex::new(self, sign)
  }

  pub fn global_to_local_subsimp(&self, sub_global: &Self) -> Simplex {
    let sub_local = sub_global
      .iter()
      .map(|iglobal| {
        self
          .iter()
          .position(|iother| iglobal == iother)
          .expect("Not a subset.")
      })
      .collect();
    Simplex::new(sub_local)
  }
}

impl IntoIterator for Simplex {
  type Item = usize;
  type IntoIter = std::vec::IntoIter<Self::Item>;
  fn into_iter(self) -> Self::IntoIter {
    self.vertices.into_iter()
  }
}

impl From<Vec<usize>> for Simplex {
  fn from(vertices: Vec<usize>) -> Self {
    Self::new(vertices)
  }
}
impl<const N: usize> From<[usize; N]> for Simplex {
  fn from(vertices: [usize; N]) -> Self {
    Self::new(vertices.to_vec())
  }
}

impl Simplex {
  pub fn standard(dim: Dim) -> Self {
    Self::new((0..dim + 1).collect())
  }
  pub fn single(v: usize) -> Self {
    Self::new(vec![v])
  }

  pub fn supsimps(&self, sup_dim: Dim, root: &Self) -> impl Iterator<Item = Self> + use<'_> {
    root.subsimps(sup_dim).filter(|sup| self.is_subsimp_of(sup))
  }
  pub fn anti_boundary(&self, root: &Simplex) -> impl Iterator<Item = SignedSimplex> + use<'_> {
    let mut sign = Sign::from_parity(self.nvertices() + 1);
    self.supsimps(self.dim() + 1, root).map(move |sup| {
      let this_sign = sign;
      sign.flip();
      SignedSimplex::new(sup, this_sign)
    })
  }
}

impl Simplex {
  pub fn sort(&mut self) {
    self.vertices.sort_unstable()
  }
  pub fn sorted(mut self) -> Self {
    self.sort();
    self
  }
  pub fn is_sorted(&self) -> bool {
    self.vertices.is_sorted()
  }

  pub fn is_subsimp_of(&self, other: &Self) -> bool {
    self.sub_cmp(other).map(|o| o.is_le()).unwrap_or(false)
  }
  pub fn is_supsimp_of(&self, other: &Self) -> bool {
    self.sub_cmp(other).map(|o| o.is_ge()).unwrap_or(false)
  }
  /// Subset partial order relation.
  pub fn sub_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    assert!(self.is_sorted());
    assert!(other.is_sorted());

    use std::cmp::Ordering as O;
    let mut is_le = true;
    let mut is_ge = true;

    let mut this = self.iter().peekable();
    let mut other = other.iter().peekable();
    while let (Some(self_v), Some(other_v)) = (this.peek(), other.peek()) {
      match self_v.cmp(other_v) {
        O::Equal => {
          this.next();
          other.next();
        }
        O::Less => {
          is_le = false;
          this.next();
        }
        O::Greater => {
          is_ge = false;
          other.next();
        }
      }
    }

    if this.next().is_some() {
      is_le = false;
    }
    if other.next().is_some() {
      is_ge = false;
    }

    match (is_le, is_ge) {
      (true, true) => Some(O::Equal),
      (true, false) => Some(O::Less),
      (false, true) => Some(O::Greater),
      _ => None,
    }
  }
}

pub fn graded_subsimps(dim_cell: Dim) -> impl Iterator<Item = impl Iterator<Item = Simplex>> {
  (0..=dim_cell).map(move |d| standard_subsimps(dim_cell, d))
}
pub fn standard_subsimps(dim_cell: Dim, dim_sub: Dim) -> impl Iterator<Item = Simplex> {
  Simplex::standard(dim_cell).subsimps(dim_sub)
}

pub fn nsubsimplicies(dim_cell: Dim, dim_sub: Dim) -> usize {
  binomial(dim_cell + 1, dim_sub + 1)
}
pub fn nedges(dim_cell: Dim) -> usize {
  nsubsimplicies(dim_cell, 1)
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct SignedSimplex {
  pub simplex: Simplex,
  pub sign: Sign,
}
impl SignedSimplex {
  pub fn new(simplex: Simplex, sign: Sign) -> Self {
    Self { simplex, sign }
  }
}
