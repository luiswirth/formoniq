pub mod geometry;
pub mod mesh;
pub mod simplex;

pub use geometry::CellGramians;

use metric::CausalType;
use simplicial::linalg::Vector;

pub type EdgeIdx = usize;

/// A column of signed squared edge lengths: the Regge datum, read per edge.
///
/// [`MeshLengthsSq`](mesh::MeshLengthsSq) carries one over the whole
/// 1-skeleton and [`SimplexLengthsSq`](simplex::SimplexLengthsSq) one over a
/// single simplex's own edges, which are two scopes of the same data, so what
/// an entry *means* is written once: the sign is the causal character and the
/// magnitude is the length.
///
/// This is not a trait over geometry *representations*. Those are the concrete
/// [`MeshLengthsSq`](mesh::MeshLengthsSq) and [`CellGramians`], and a source
/// converts into one at the boundary of the API rather than being dispatched
/// on (see [`geometry`]).
pub trait LengthsSq {
  /// The signed squared lengths, in edge order.
  fn lengths_sq(&self) -> &Vector;

  fn nedges(&self) -> usize {
    self.lengths_sq().len()
  }
  /// The signed squared length of an edge: the Regge primitive, its sign the
  /// causal character.
  fn length_sq(&self, iedge: EdgeIdx) -> f64 {
    self.lengths_sq()[iedge]
  }
  /// The magnitude $sqrt(abs(s))$ of an edge. On an indefinite metric this is
  /// meaningful only together with [`Self::causal_type`]. It is never NaN.
  fn length(&self, iedge: EdgeIdx) -> f64 {
    self.length_sq(iedge).abs().sqrt()
  }
  /// The causal character of an edge: the sign of its squared length.
  fn causal_type(&self, iedge: EdgeIdx) -> CausalType {
    CausalType::from_norm_sq(self.length_sq(iedge))
  }
  fn iter(&self) -> impl ExactSizeIterator<Item = f64> {
    self.lengths_sq().iter().copied()
  }

  /// The largest edge magnitude, and the smallest.
  ///
  /// Zero on a column with no edges, where there is no length to take an
  /// extremum over: a point carries the trivial geometry, not an error.
  fn max_length(&self) -> f64 {
    self.iter().map(f64::abs).fold(0.0, f64::max).sqrt()
  }
  fn min_length(&self) -> f64 {
    self
      .iter()
      .map(f64::abs)
      .reduce(f64::min)
      .unwrap_or(0.0)
      .sqrt()
  }
}
