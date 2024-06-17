//! This module defined the objects necessary for a Triangulation.
//! A mesh plays the role of a container of mesh entities (cells,faces,edges,vertices).
//! It must allows the unique identification of the entities.
//! It must allow from traversal of the cells -> must induce a global numbering on all entities.
//! It must represent mesh topology (incidence).
//! It must represent mesh geometry (location, shape)

pub mod util;

use crate::{simplex::Simplex, Dim};

use std::rc::Rc;

pub type NodeId = usize;
/// SimplexDim, SimplexDimIdx
pub type EntityId = (Dim, usize);

pub type Node = na::DVector<f64>;

/// All simplicies of a Mesh are connected to this MeshNodes object.
/// It contains all the Nodes of the Mesh.
#[derive(Debug, Clone)]
pub struct MeshNodes(Rc<Vec<Node>>);
impl MeshNodes {
  pub fn new(nodes: Vec<Node>) -> Self {
    assert!(!nodes.is_empty(), "Mesh nodes may not be empty.");
    let dim_ambient = nodes[0].len();
    assert!(
      nodes.iter().all(|n| dim_ambient == n.len()),
      "All mesh nodes must have the same dimension."
    );
    let nodes = Rc::new(nodes);
    Self(nodes)
  }

  pub fn nodes(&self) -> &[Node] {
    &self.0
  }
  pub fn dim_ambient(&self) -> Dim {
    self.0[0].len()
  }
}
impl<'n> std::ops::Index<usize> for MeshNodes {
  type Output = Node;

  fn index(&self, i: usize) -> &Self::Output {
    &self.0[i]
  }
}

/// Not supposed to be mutated, once created.
/// Lifetime 'n of nodes
#[derive(Debug)]
pub struct SimplexEntity {
  /// sorted vertices of the Simplex
  vertices: Vec<NodeId>,
}

impl SimplexEntity {
  pub fn new(vertices: Vec<NodeId>) -> Self {
    assert!(!vertices.is_empty(), "Simplex may not be empty.");
    Self { vertices }
  }
  pub fn dim_intrinsic(&self) -> Dim {
    self.vertices.len() - 1
  }
  pub fn nvertices(&self) -> usize {
    self.vertices.len()
  }
  pub fn vertices(&self) -> &[NodeId] {
    &self.vertices
  }
}

/// A k-Skeleton, is a collection of only k-simplicies.
#[derive(Debug)]
pub struct Skeleton {
  simplicies: Vec<SimplexEntity>,
}
impl Skeleton {
  pub fn new(simplicies: Vec<SimplexEntity>) -> Self {
    assert!(!simplicies.is_empty(), "Skeleton may not be empty.");
    let dim = simplicies[0].dim_intrinsic();
    assert!(
      simplicies.iter().all(|s| s.dim_intrinsic() == dim),
      "Simplicies in Skeleton must all have same dimension."
    );
    Self { simplicies }
  }

  pub fn dim(&self) -> Dim {
    self.simplicies[0].dim_intrinsic()
  }
  pub fn simplicies(&self) -> &[SimplexEntity] {
    &self.simplicies
  }
  pub fn nsimplicies(&self) -> usize {
    self.simplicies.len()
  }
}

/// A `Triangulation` or a Simplicial Mesh
/// It's a pure simplicial complex.
#[derive(Debug)]
pub struct Triangulation {
  nodes: MeshNodes,
  skeletons: Vec<Skeleton>,
}
impl Triangulation {
  pub fn from_skeletons(nodes: MeshNodes, skeletons: Vec<Skeleton>) -> Self {
    assert!(!skeletons.is_empty(), "Triangulation may not be empty.");
    assert!(
      skeletons.iter().enumerate().all(|(i, s)| s.dim() == i),
      "Skeletons in Triangulation must have their index be equal to the dimension."
    );
    Self { nodes, skeletons }
  }
  pub fn nskeletons(&self) -> usize {
    self.skeletons.len()
  }
  /// The dimension of highest dimensional [`Skeleton`]
  pub fn dim_intrinsic(&self) -> Dim {
    self.skeletons.len() - 1
  }
  pub fn dim_ambient(&self) -> Dim {
    self.nodes.dim_ambient()
  }
  pub fn nodes(&self) -> &MeshNodes {
    &self.nodes
  }
  pub fn skeletons(&self) -> &[Skeleton] {
    &self.skeletons
  }
  pub fn entity_by_id(&self, id: EntityId) -> &SimplexEntity {
    &self.skeletons[id.0].simplicies()[id.1]
  }
  pub fn coordinate_simplex(&self, id: EntityId) -> Simplex {
    let entity = self.entity_by_id(id);
    let mut vertices = na::DMatrix::zeros(self.nodes().dim_ambient(), entity.nvertices());
    for (i, &v) in entity.vertices().iter().enumerate() {
      vertices.column_mut(i).copy_from(&self.nodes.nodes()[v]);
    }
    Simplex::new(vertices)
  }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Orientation {
  #[default]
  Pos = 1,
  Neg = -1,
}
impl Orientation {
  /// Simplex orientation might change when permuting the vertices.
  /// This depends on the parity of the number of swaps.
  /// Even permutations preserve the orientation.
  /// Odd permutations invert the orientation.
  /// Based on the number
  pub fn from_permutation_parity(n: usize) -> Self {
    match n % 2 {
      0 => Self::Pos,
      1 => Self::Neg,
      _ => unreachable!(),
    }
  }
}
impl std::ops::Neg for Orientation {
  type Output = Self;

  fn neg(self) -> Self::Output {
    match self {
      Self::Pos => Self::Neg,
      Self::Neg => Self::Pos,
    }
  }
}
impl std::ops::Mul for Orientation {
  type Output = Self;

  fn mul(self, other: Self) -> Self::Output {
    match self == other {
      true => Self::Pos,
      false => Self::Neg,
    }
  }
}
impl std::ops::MulAssign for Orientation {
  fn mul_assign(&mut self, other: Self) {
    *self = *self * other;
  }
}
impl From<Orientation> for char {
  fn from(o: Orientation) -> Self {
    match o {
      Orientation::Pos => '+',
      Orientation::Neg => '-',
    }
  }
}
impl std::fmt::Display for Orientation {
  fn fmt(&self, fmt: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
    write!(fmt, "{}", char::from(*self))
  }
}
