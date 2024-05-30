use std::rc::Rc;

pub type NodeId = usize;

/// All simplicies of a Mesh are connected to this MeshNodes object.
/// It contains all the Nodes of the Mesh.
#[derive(Debug)]
pub struct MeshNodes(Vec<na::DVector<f64>>);
impl MeshNodes {
  pub fn new(nodes: Vec<na::DVector<f64>>) -> Self {
    Self(nodes)
  }

  pub fn nodes(&self) -> &[na::DVector<f64>] {
    &self.0
  }
}

/// Not supposed to be mutated, once created.
#[derive(Debug)]
pub struct Simplex {
  /// connection to mesh
  mesh_nodes: Rc<MeshNodes>,
  /// sorted nodes of the Simplex
  nodes: Vec<NodeId>,
  orientation: Orientation,
}
impl PartialEq for Simplex {
  fn eq(&self, other: &Self) -> bool {
    self.cmp(other) == Some(Orientation::Pos)
  }
}

impl Simplex {
  pub fn new(mesh_nodes: Rc<MeshNodes>, nodes: Vec<NodeId>) -> Self {
    Self::new_with_orientation(mesh_nodes, nodes, Default::default())
  }
  pub fn new_with_orientation(
    mesh_nodes: Rc<MeshNodes>,
    mut nodes: Vec<NodeId>,
    mut orientation: Orientation,
  ) -> Self {
    assert!(!nodes.is_empty(), "Simplex may not be empty.");
    let swaps = sort_count(&mut nodes);
    orientation *= Orientation::from(swaps);
    Self {
      mesh_nodes,
      nodes,
      orientation,
    }
  }
  pub fn dim(&self) -> usize {
    self.nodes.len() - 1
  }
  pub fn vertices(&self) -> &[NodeId] {
    &self.nodes
  }

  pub fn cmp(&self, other: &Self) -> Option<Orientation> {
    if Rc::ptr_eq(&self.mesh_nodes, &other.mesh_nodes) || self.nodes != other.nodes {
      return None;
    }
    Some(match self.orientation == other.orientation {
      true => Orientation::Pos,
      false => Orientation::Neg,
    })
  }

  /// all direct proper (one dim lower) faces
  pub fn subs(&self) -> Vec<Simplex> {
    if self.dim() == 0 {
      return Vec::new();
    }
    (0..self.nodes.len())
      .map(|i| {
        let mesh_nodes = Rc::clone(&self.mesh_nodes);
        let mut nodes = self.nodes.clone();
        nodes.remove(i);
        let orientation = Orientation::from(i);
        Self {
          mesh_nodes,
          orientation,
          nodes,
        }
      })
      .collect()
  }
}
impl std::fmt::Display for Simplex {
  fn fmt(&self, fmt: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
    write!(fmt, "Simplex({})[", self.orientation)?;
    for v in self.vertices() {
      write!(fmt, " x_{},", v)?;
    }
    write!(fmt, "]")
  }
}

/// Not supposed to be mutated, once created.
#[derive(Debug)]
pub struct Triangulation {
  /// Simplicies of highest intrinsic dimension.
  /// Subsimplicies are not stored, but computed.
  simplicies: Vec<Simplex>,
}
impl Triangulation {
  pub fn new(simplicies: Vec<Simplex>) -> Self {
    assert!(!simplicies.is_empty(), "Triangulation may not be empty.");
    let dim_intrinsic = simplicies[0].dim();
    assert!(
      simplicies.iter().all(|s| s.dim() == dim_intrinsic),
      "Intrinsic dimension of Simplicies in Triangulation must match."
    );

    Self { simplicies }
  }
  pub fn dim_intrinsic(&self) -> usize {
    self.simplicies[0].dim()
  }
  pub fn simplicies(&self) -> &[Simplex] {
    &self.simplicies
  }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Orientation {
  #[default]
  Pos,
  Neg,
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

  fn mul(self, rhs: Self) -> Self::Output {
    Self::from(self == rhs)
  }
}
impl std::ops::MulAssign for Orientation {
  fn mul_assign(&mut self, rhs: Self) {
    *self = Self::from(*self == rhs);
  }
}
impl From<bool> for Orientation {
  fn from(b: bool) -> Self {
    match b {
      true => Self::Pos,
      false => Self::Neg,
    }
  }
}
impl From<usize> for Orientation {
  fn from(n: usize) -> Self {
    match n % 2 {
      0 => Self::Pos,
      1 => Self::Neg,
      _ => unreachable!(),
    }
  }
}
impl std::fmt::Display for Orientation {
  fn fmt(&self, fmt: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
    write!(
      fmt,
      "{}",
      match self {
        Self::Pos => '+',
        Self::Neg => '-',
      }
    )
  }
}

// TODO: horribly inefficent -> better algorithm
/// sorts the slice and counts the swaps
/// bubble sort
pub fn sort_count<T: Ord>(s: &mut [T]) -> usize {
  let mut nswaps = 0;
  let mut swapped = true;
  while swapped {
    swapped = false;
    for i in 1..s.len() {
      if s[i - 1] > s[i] {
        s.swap(i - 1, i);
        nswaps += 1;
        swapped = true;
      }
    }
  }
  nswaps
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn unit_squared_2d_triangulation() {
    let nodes = MeshNodes::new(vec![
      na::DVector::from_column_slice(&[0.0, 0.0]),
      na::DVector::from_column_slice(&[1.0, 0.0]),
      na::DVector::from_column_slice(&[1.0, 1.0]),
      na::DVector::from_column_slice(&[0.0, 1.0]),
    ]);
    let nodes = Rc::new(nodes);
    let unit_triangle = Simplex::new(nodes.clone(), vec![0, 1, 3]);
    assert_eq!(unit_triangle.dim(), 2);
    let other_triangle = Simplex::new(nodes, vec![1, 2, 3]);
    assert_eq!(other_triangle.dim(), 2);
    let triangulation = Triangulation::new(vec![unit_triangle, other_triangle]);
    assert_eq!(triangulation.dim_intrinsic(), 2);
    let edges: Vec<_> = triangulation
      .simplicies()
      .iter()
      .map(|s| s.subs())
      .flatten()
      .collect();
    for e in &edges {
      println!("{e}");
    }
    let vertices: Vec<_> = edges.iter().map(|s| s.subs()).flatten().collect();
    for v in &vertices {
      println!("{v}");
    }
  }
}
