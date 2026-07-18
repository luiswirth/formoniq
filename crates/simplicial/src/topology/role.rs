//! Role proofs on simplices: dimension preconditions as types.
//!
//! Several operations are only meaningful on a simplex in a specific role --
//! the metric of a *cell*, the boundary status of a *facet*, the length of an
//! *edge*. A role is a proposition about a simplex's dimension relative to its
//! complex, and [`Roled`] is the witness: the check happens once, where the
//! witness is built, and a signature demanding the witness states its own
//! precondition instead of trusting a comment.
//!
//! Roles are propositions, not a partition. A simplex may carry several at
//! once -- the edge of a 1-complex is an [`Edge`] and a [`Cell`], and both
//! witnesses coexist. What cannot happen is *constructing* a role the simplex
//! does not carry. Dimension stays fully runtime: the witness is a phantom
//! checked against `complex.dim()`, never a const generic.
//!
//! The witness is pure topology -- a predicate on dimensions, no metric --
//! which is why this module sits in `topology`. Operations keyed on a role
//! that need geometry stay on the geometry side, consuming the proof.

use super::handle::{KSimplexIdx, SimplexIdx, SimplexRef, SkeletonRef};
use crate::Dim;

use rayon::iter::ParallelIterator;

use std::{marker::PhantomData, ops::Deref};

/// How a role pins a simplex's dimension: absolutely, or relative to the
/// complex by codimension. The one datum every role carries, and
/// [`dim_in`](Self::dim_in) the one predicate they all share.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoleDim {
  Dim(Dim),
  Codim(Dim),
}

impl RoleDim {
  /// The dimension a role pinned this way has in a `top`-dimensional complex,
  /// `None` where the complex has no such dimension (`Codim(1)` in a
  /// 0-complex, `Dim(1)` in a 0-complex). The checked form is what keeps the
  /// degenerate boundary total: a point has no facets, not an underflow.
  pub const fn dim_in(self, top: Dim) -> Option<Dim> {
    match self {
      RoleDim::Dim(dim) => {
        if dim <= top {
          Some(dim)
        } else {
          None
        }
      }
      RoleDim::Codim(codim) => top.checked_sub(codim),
    }
  }
}

/// A role: a proposition about a simplex's dimension in its complex, pinned
/// by its [`RoleDim`].
pub trait SimplexRole {
  /// The dimension the role pins, absolutely or by codimension.
  const DIM: RoleDim;
  const NAME: &'static str;

  /// Whether a `dim`-simplex of a `top`-dimensional complex carries this
  /// role: exactly when `dim` is the role's dimension there.
  fn admits(dim: Dim, top: Dim) -> bool {
    Self::DIM.dim_in(top) == Some(dim)
  }
}

/// The role markers: each is one [`RoleDim`], everything else is shared.
pub mod roles {
  use super::{RoleDim, SimplexRole};

  /// $dim = 0$.
  pub struct Vertex;
  /// $dim = 1$.
  pub struct Edge;
  /// Codimension 0: the top-dimensional simplices, the charts of the atlas.
  pub struct Cell;
  /// Codimension 1: shared by at most two cells (the manifold property).
  pub struct Facet;
  /// Codimension 2: the hinges where Regge curvature concentrates.
  pub struct Ridge;

  impl SimplexRole for Vertex {
    const DIM: RoleDim = RoleDim::Dim(0);
    const NAME: &'static str = "vertex";
  }
  impl SimplexRole for Edge {
    const DIM: RoleDim = RoleDim::Dim(1);
    const NAME: &'static str = "edge";
  }
  impl SimplexRole for Cell {
    const DIM: RoleDim = RoleDim::Codim(0);
    const NAME: &'static str = "cell";
  }
  impl SimplexRole for Facet {
    const DIM: RoleDim = RoleDim::Codim(1);
    const NAME: &'static str = "facet";
  }
  impl SimplexRole for Ridge {
    const DIM: RoleDim = RoleDim::Codim(2);
    const NAME: &'static str = "ridge";
  }
}

/// A simplex together with the proof that it carries role `R`.
///
/// A `Roled` is a [`SimplexRef`] plus a phantom witness -- no representation of
/// its own. It derefs to the plain ref, so all navigation is inherited; what
/// the role adds are the operations whose precondition it proves.
pub struct Roled<'m, R: SimplexRole> {
  inner: SimplexRef<'m>,
  _role: PhantomData<fn() -> R>,
}

pub type Vertex<'m> = Roled<'m, roles::Vertex>;
pub type Edge<'m> = Roled<'m, roles::Edge>;
pub type Cell<'m> = Roled<'m, roles::Cell>;
pub type Facet<'m> = Roled<'m, roles::Facet>;
pub type Ridge<'m> = Roled<'m, roles::Ridge>;

impl<'m, R: SimplexRole> Roled<'m, R> {
  /// A proof established by construction -- navigation that only produces this
  /// role -- rather than by checking.
  pub(crate) fn trusted(simplex: SimplexRef<'m>) -> Self {
    debug_assert!(R::admits(simplex.dim(), simplex.complex().dim()));
    Self {
      inner: simplex,
      _role: PhantomData,
    }
  }
  /// Forget the proof.
  pub fn get(self) -> SimplexRef<'m> {
    self.inner
  }
}

/// Obtaining a proof from a bare ref.
impl<'m> SimplexRef<'m> {
  /// The proof that this simplex carries role `R`, if it does: the one check.
  pub fn as_role<R: SimplexRole>(self) -> Option<Roled<'m, R>> {
    R::admits(self.dim(), self.complex().dim()).then_some(Roled {
      inner: self,
      _role: PhantomData,
    })
  }
  /// The asserting form of [`as_role`](Self::as_role), for trusted boundaries:
  /// panics if the simplex does not carry the role.
  pub fn role<R: SimplexRole>(self) -> Roled<'m, R> {
    self.as_role().unwrap_or_else(|| {
      panic!(
        "{:?} of a {}-complex is not a {}",
        self.idx(),
        self.complex().dim(),
        R::NAME
      )
    })
  }
}

impl<'m, R: SimplexRole> Deref for Roled<'m, R> {
  type Target = SimplexRef<'m>;
  fn deref(&self) -> &SimplexRef<'m> {
    &self.inner
  }
}
impl<R: SimplexRole> Clone for Roled<'_, R> {
  fn clone(&self) -> Self {
    *self
  }
}
impl<R: SimplexRole> Copy for Roled<'_, R> {}
impl<R: SimplexRole> PartialEq for Roled<'_, R> {
  fn eq(&self, other: &Self) -> bool {
    self.inner == other.inner
  }
}
impl<R: SimplexRole> Eq for Roled<'_, R> {}
impl<R: SimplexRole> std::hash::Hash for Roled<'_, R> {
  fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
    self.inner.hash(state);
  }
}
impl<R: SimplexRole> std::fmt::Debug for Roled<'_, R> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_tuple(R::NAME).field(&self.inner).finish()
  }
}

/// What the cell proof unlocks: the navigation whose meaning presupposes top
/// dimension.
impl<'m> Cell<'m> {
  /// The facets of a cell are facets *of the complex*: the codim-1 faces of a
  /// codim-0 simplex. A proof-preserving override of [`SimplexRef::facets`].
  pub fn facets(self) -> impl Iterator<Item = Facet<'m>> {
    self.get().facets().map(Roled::trusted)
  }
  /// The neighbors: the cells sharing a facet with this one (the dual-graph
  /// adjacency). Two distinct cells share at most one facet, so no
  /// deduplication is needed.
  pub fn neighbors(self) -> impl Iterator<Item = Cell<'m>> {
    self.facets().filter_map(move |facet| {
      let (a, b) = facet.adjacent_cells();
      if a == self {
        b
      } else {
        Some(a)
      }
    })
  }
}

/// What the facet proof unlocks: the two-sidedness the manifold property
/// grants exactly in codimension 1.
impl<'m> Facet<'m> {
  /// The at most two cells this facet bounds -- exactly one iff it lies on the
  /// boundary. The manifold property, checked at
  /// [`Complex::from_cells`](super::complex::Complex::from_cells), is what
  /// makes the pair total.
  pub fn adjacent_cells(self) -> (Cell<'m>, Option<Cell<'m>>) {
    let mut cells = self.get().cells();
    let first = cells.next().expect("a facet bounds at least one cell");
    (first, cells.next())
  }
  /// Whether this facet lies on the boundary: it bounds a single cell.
  pub fn is_boundary(self) -> bool {
    self.adjacent_cells().1.is_none()
  }
}

/// What the edge proof unlocks: the pair structure of grade 1. The metric
/// side adds the length the proof keys
/// ([`EdgeRefExt`](crate::geometry::metric::mesh::EdgeRefExt)).
impl<'m> Edge<'m> {
  /// The two endpoints, in vertex order: the type states that an edge has
  /// exactly two.
  pub fn endpoints(self) -> (Vertex<'m>, Vertex<'m>) {
    let complex = self.complex();
    let vertices = &self.simplex().vertices;
    let vertex = |v| Roled::trusted(SimplexIdx::new(0, v).handle(complex));
    (vertex(vertices[0]), vertex(vertices[1]))
  }
}

/// What the ridge proof unlocks: the hinge structure the manifold property
/// grants in codimension 2.
impl<'m> Ridge<'m> {
  /// The two facets of `cell` that contain this ridge: a cell has one facet
  /// per dropped vertex, and exactly the two dropping a vertex outside the
  /// ridge retain it.
  fn hinge_facets(self, cell: Cell<'m>) -> impl Iterator<Item = Facet<'m>> {
    cell
      .facets()
      .filter(move |facet| self.simplex().is_subsimplex_of(facet.simplex()))
  }

  /// The fan of the hinge: the incident cells in adjacency order around the
  /// ridge, consecutive ones sharing a facet that contains it. Closed (the
  /// last cell neighboring the first) iff the ridge is interior; an open fan
  /// runs from boundary to boundary. The codimension-2 successor of
  /// [`Facet::adjacent_cells`], and the walk along which a Regge deficit
  /// angle sums its dihedral angles.
  pub fn fan(self) -> Vec<Cell<'m>> {
    let mut incident = self.get().cells();
    let first = incident.next().expect("every simplex lies in a cell");
    // Start at a boundary end if there is one, so an open fan runs end to end.
    let start = std::iter::once(first)
      .chain(incident)
      .find(|&cell| self.hinge_facets(cell).any(|facet| facet.is_boundary()))
      .unwrap_or(first);

    let mut fan = vec![start];
    let mut entered = self.hinge_facets(start).find(|facet| facet.is_boundary());
    loop {
      let cell = *fan.last().unwrap();
      let exit = self
        .hinge_facets(cell)
        .find(|facet| entered.is_none_or(|e| e.idx() != facet.idx()))
        .expect("a cell has two facets containing each of its ridges");
      let (a, b) = exit.adjacent_cells();
      match if a == cell { b } else { Some(a) } {
        // The boundary: the open fan ends.
        None => break,
        // Around the hinge: the closed fan ends.
        Some(next) if next == start => break,
        Some(next) => {
          fan.push(next);
          entered = Some(exit);
        }
      }
    }
    fan
  }
}

/// A skeleton whose simplices all carry role `R`: the source of role proofs
/// for whole-skeleton iteration, so the dominant walks
/// ([`cells`](super::complex::Complex::cells),
/// [`facets`](super::complex::Complex::facets)) yield their proofs for free.
pub struct RoledSkeleton<'m, R: SimplexRole> {
  inner: SkeletonRef<'m>,
  _role: PhantomData<fn() -> R>,
}

impl<'m, R: SimplexRole> RoledSkeleton<'m, R> {
  pub(crate) fn trusted(inner: SkeletonRef<'m>) -> Self {
    Self {
      inner,
      _role: PhantomData,
    }
  }
  /// Forget the proof.
  pub fn get(self) -> SkeletonRef<'m> {
    self.inner
  }
  pub fn handle_by_kidx(&self, kidx: KSimplexIdx) -> Roled<'m, R> {
    Roled::trusted(self.inner.handle_by_kidx(kidx))
  }
  pub fn handle_iter(&self) -> impl ExactSizeIterator<Item = Roled<'m, R>> + '_ {
    self.inner.handle_iter().map(Roled::trusted)
  }
  pub fn handle_par_iter(&self) -> impl ParallelIterator<Item = Roled<'m, R>> + '_ {
    self.inner.handle_par_iter().map(Roled::trusted)
  }
}
impl<'m, R: SimplexRole> Deref for RoledSkeleton<'m, R> {
  type Target = SkeletonRef<'m>;
  fn deref(&self) -> &SkeletonRef<'m> {
    &self.inner
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::{gen::cartesian::CartesianMeshInfo, topology::complex::Complex};

  /// The role predicates, swept over all dimensions and grades: a role is
  /// admitted exactly on its dimension, and roles coexist where their
  /// dimensions coincide (the edge of a 1-complex is an edge and a cell) --
  /// propositions, not a partition.
  #[test]
  fn roles_are_admitted_exactly_on_their_dimension() {
    for top in 0..=4 {
      let complex = Complex::standard(top);
      for dim in 0..=top {
        for simplex in complex.skeleton(dim).handle_iter() {
          assert_eq!(simplex.as_role::<roles::Vertex>().is_some(), dim == 0);
          assert_eq!(simplex.as_role::<roles::Edge>().is_some(), dim == 1);
          assert_eq!(simplex.as_role::<roles::Cell>().is_some(), dim == top);
          assert_eq!(simplex.as_role::<roles::Facet>().is_some(), dim + 1 == top);
          assert_eq!(simplex.as_role::<roles::Ridge>().is_some(), dim + 2 == top);
        }
      }
    }
  }

  /// A face carries no cell proof: asserting one is a contract violation, and
  /// the type -- not a convention -- is what says so.
  #[test]
  #[should_panic(expected = "is not a cell")]
  fn a_face_is_not_a_cell() {
    let complex = Complex::standard(2);
    let edge = complex.skeleton(1).handle_iter().next().unwrap();
    edge.role::<roles::Cell>();
  }

  /// Dual-graph adjacency is symmetric, and a cell has exactly one neighbor
  /// per interior facet.
  #[test]
  fn neighboring_is_symmetric_and_facet_induced() {
    for dim in 1..=3 {
      let (complex, _) = CartesianMeshInfo::new_unit(dim, 2).compute_coord_complex();
      for cell in complex.cells().handle_iter() {
        let interior = cell.facets().filter(|f| !f.is_boundary()).count();
        assert_eq!(cell.neighbors().count(), interior);
        for neighbor in cell.neighbors() {
          assert!(neighbor.neighbors().any(|back| back == cell));
        }
      }
    }
  }

  /// The generic accessor is total: `None` exactly where the complex has no
  /// simplices of the role's dimension, never an underflow.
  #[test]
  fn role_skeletons_exist_exactly_where_their_dimension_does() {
    for top in 0..=4 {
      let complex = Complex::standard(top);
      assert!(complex.role_skeleton::<roles::Vertex>().is_some());
      assert_eq!(complex.role_skeleton::<roles::Edge>().is_some(), top >= 1);
      assert!(complex.role_skeleton::<roles::Cell>().is_some());
      assert_eq!(complex.role_skeleton::<roles::Facet>().is_some(), top >= 1);
      assert_eq!(complex.role_skeleton::<roles::Ridge>().is_some(), top >= 2);
    }
  }

  /// An edge's endpoints are its two vertices, in order, with their proofs.
  #[test]
  fn edge_endpoints_are_its_vertices() {
    for top in 1..=4 {
      let complex = Complex::standard(top);
      for edge in complex.edges().handle_iter() {
        let (a, b) = edge.endpoints();
        assert_eq!(vec![a.kidx(), b.kidx()], edge.simplex().vertices);
      }
    }
  }

  /// The fan of a ridge: every incident cell exactly once, consecutive cells
  /// sharing a facet containing the ridge, closed exactly on interior ridges.
  #[test]
  fn ridge_fans_walk_the_hinge() {
    for dim in 2..=3 {
      let (complex, _) = CartesianMeshInfo::new_unit(dim, 2).compute_coord_complex();
      for ridge in complex
        .role_skeleton::<roles::Ridge>()
        .unwrap()
        .handle_iter()
      {
        let fan = ridge.fan();
        let hinged = |a: Cell, b: Cell| {
          a.facets().any(|f| {
            ridge.simplex().is_subsimplex_of(f.simplex()) && b.facets().any(|g| g.idx() == f.idx())
          })
        };

        let mut fanned: Vec<_> = fan.iter().map(|cell| cell.idx()).collect();
        fanned.sort_unstable();
        assert!(fanned.windows(2).all(|w| w[0] != w[1]));
        let mut incident: Vec<_> = ridge.get().cells().map(|cell| cell.idx()).collect();
        incident.sort_unstable();
        assert_eq!(fanned, incident);

        for pair in fan.windows(2) {
          assert!(hinged(pair[0], pair[1]));
        }
        let interior = !ridge
          .get()
          .cofaces(dim - 1)
          .any(|f| f.role::<roles::Facet>().is_boundary());
        if interior {
          assert!(hinged(*fan.last().unwrap(), fan[0]));
        }
      }
    }
  }

  /// Every facet of the standard simplex is boundary: it bounds the one cell.
  #[test]
  fn standard_complex_is_all_boundary() {
    for top in 1..=4 {
      let complex = Complex::standard(top);
      for facet in complex.facets().handle_iter() {
        assert!(facet.is_boundary());
        let (cell, other) = facet.adjacent_cells();
        assert_eq!(cell.dim(), top);
        assert!(other.is_none());
      }
    }
  }
}
