//! This module defined the objects necessary for a Triangulation.

pub mod util;

use std::rc::Rc;

pub type NodeId = usize;
pub type EntityId = usize;

// A mesh plays the role of a container of mesh entities (cells,faces,edges,vertices).
// It must allows the unique identification of the entities.
// It must allow from traversal of the cells -> must induce a global numbering on all entities.
// It must represent mesh topology (incidence).
// It must represent mesh geometry (location, shape)

/// All simplicies of a Mesh are connected to this MeshNodes object.
/// It contains all the Nodes of the Mesh.
#[derive(Debug, Clone)]
pub struct MeshNodes(Rc<Vec<na::DVector<f64>>>);
impl MeshNodes {
  pub fn new(nodes: Vec<na::DVector<f64>>) -> Self {
    assert!(!nodes.is_empty(), "Mesh nodes may not be empty.");
    let dim_ambient = nodes[0].len();
    assert!(nodes.iter().all(|n| dim_ambient == n.len()));
    let nodes = Rc::new(nodes);
    Self(nodes)
  }
  pub fn new_unchecked(nodes: Vec<na::DVector<f64>>) -> Self {
    let nodes = Rc::new(nodes);
    Self(nodes)
  }

  pub fn nodes(&self) -> &[na::DVector<f64>] {
    &self.0
  }
  pub fn dim_ambient(&self) -> usize {
    self.0[0].len()
  }
}
impl std::cmp::PartialEq for MeshNodes {
  fn eq(&self, other: &Self) -> bool {
    Rc::ptr_eq(&self.0, &other.0)
  }
}
impl std::cmp::Eq for MeshNodes {}

/// Not supposed to be mutated, once created.
#[derive(Debug, PartialEq, Eq)]
pub struct Simplex {
  /// connection to mesh
  mesh_nodes: MeshNodes,
  /// sorted vertices of the Simplex
  vertices: Vec<NodeId>,
}

impl Simplex {
  pub fn new(mesh_nodes: MeshNodes, vertices: Vec<NodeId>) -> Self {
    assert!(!vertices.is_empty(), "Simplex may not be empty.");
    Self {
      mesh_nodes,
      vertices,
    }
  }
  pub fn dim(&self) -> usize {
    self.vertices.len() - 1
  }
  pub fn vertices(&self) -> &[NodeId] {
    &self.vertices
  }

  /// all direct proper (one dim lower) faces
  pub fn subs(&self) -> Vec<Simplex> {
    if self.dim() == 0 {
      return Vec::new();
    }
    (0..self.vertices.len())
      .map(|i| {
        let mesh_nodes = self.mesh_nodes.clone();
        let mut vertices = self.vertices.clone();
        vertices.remove(i);
        Self {
          mesh_nodes,
          vertices,
        }
      })
      .collect()
  }
}
impl std::fmt::Display for Simplex {
  fn fmt(&self, fmt: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
    write!(fmt, "Simplex[")?;
    for v in self.vertices() {
      write!(fmt, " x_{},", v)?;
    }
    write!(fmt, "]")
  }
}

/// A k-Skeleton, is a collection of only k-simplicies.
#[derive(Debug)]
pub struct Skeleton {
  simplicies: Vec<Simplex>,
}
impl Skeleton {
  pub fn new(simplicies: Vec<Simplex>) -> Self {
    assert!(!simplicies.is_empty(), "Skeleton may not be empty.");
    let dim = simplicies[0].dim();
    assert!(
      simplicies.iter().all(|s| s.dim() == dim),
      "Simplicies in Skeleton must all have same dimension."
    );
    Self { simplicies }
  }

  pub fn dim(&self) -> usize {
    self.simplicies[0].dim()
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
  /// The dimension of highest dimensional [`Skeleton`]
  pub fn dim(&self) -> usize {
    self.skeletons.len() - 1
  }
  pub fn nodes(&self) -> &MeshNodes {
    &self.nodes
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
