#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Orientation {
  #[default]
  Pos,
  Neg,
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

/// Not supposed to be mutated, once created.
#[derive(Debug)]
pub struct Simplex {
  vertices: Vec<na::DVector<f64>>,
  orientation: Orientation,
}

impl Simplex {
  pub fn new(vertices: Vec<na::DVector<f64>>) -> Self {
    Self::new_with_orientation(vertices, Default::default())
  }
  pub fn new_with_orientation(vertices: Vec<na::DVector<f64>>, orientation: Orientation) -> Self {
    assert!(!vertices.is_empty(), "Simplex may not be empty.");
    let dim_ambient = vertices[0].len();
    assert!(
      vertices.iter().all(|s| s.len() == dim_ambient),
      "Vertices in simplex must have same dimension."
    );
    Self {
      vertices,
      orientation,
    }
  }
  pub fn dim_ambient(&self) -> usize {
    self.vertices[0].len()
  }
  pub fn dim_intrinsic(&self) -> usize {
    self.vertices.len() - 1
  }
  pub fn vertices(&self) -> &[na::DVector<f64>] {
    &self.vertices
  }

  /// all direct proper (one dim lower) faces
  pub fn subs(&self) -> Vec<Simplex> {
    if self.dim_intrinsic() == 0 {
      return Vec::new();
    }
    (0..self.vertices.len())
      .map(|i| {
        let orientation = Orientation::from(i);
        let mut points = self.vertices.clone();
        points.remove(i);
        Simplex {
          orientation,
          vertices: points,
        }
      })
      .collect()
  }
}
impl std::fmt::Display for Simplex {
  fn fmt(&self, fmt: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
    write!(fmt, "Simplex({})[\n", self.orientation)?;
    for (i, v) in self.vertices().iter().enumerate() {
      write!(fmt, "  x_{}: [", i)?;
      for c in v {
        write!(fmt, " {c},")?;
      }
      write!(fmt, "],\n")?;
    }
    write!(fmt, "]\n")
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
    let dim_ambient = simplicies[0].dim_ambient();
    assert!(
      simplicies.iter().all(|s| s.dim_ambient() == dim_ambient),
      "Ambient dimension of Simplicies in Triangulation must match."
    );
    let dim_intrinsic = simplicies[0].dim_intrinsic();
    assert!(
      simplicies
        .iter()
        .all(|s| s.dim_intrinsic() == dim_intrinsic),
      "Intrinsic dimension of Simplicies in Triangulation must match."
    );

    Self { simplicies }
  }
  pub fn dim_ambient(&self) -> usize {
    self.simplicies[0].dim_ambient()
  }
  pub fn dim_intrinsic(&self) -> usize {
    self.simplicies[0].dim_intrinsic()
  }
  pub fn simplicies(&self) -> &[Simplex] {
    &self.simplicies
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn unit_squared_2d_triangulation() {
    let unit_triangle = Simplex::new(vec![
      na::DVector::from_column_slice(&[0.0, 0.0]),
      na::DVector::from_column_slice(&[1.0, 0.0]),
      na::DVector::from_column_slice(&[0.0, 1.0]),
    ]);
    assert_eq!(unit_triangle.dim_ambient(), 2);
    assert_eq!(unit_triangle.dim_intrinsic(), 2);
    let other_triangle = Simplex::new(vec![
      na::DVector::from_column_slice(&[1.0, 1.0]),
      na::DVector::from_column_slice(&[0.0, 1.0]),
      na::DVector::from_column_slice(&[1.0, 0.0]),
    ]);
    assert_eq!(other_triangle.dim_ambient(), 2);
    assert_eq!(other_triangle.dim_intrinsic(), 2);
    let triangulation = Triangulation::new(vec![unit_triangle, other_triangle]);
    assert_eq!(triangulation.dim_ambient(), 2);
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
