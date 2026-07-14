//! A chart of the atlas.

use super::{Bary, MeshPoint, Transition};
use crate::{
  topology::{
    complex::Complex,
    handle::{SimplexIdx, SimplexRef},
  },
  Dim,
};

use common::combo::Combination;

/// A chart of the piecewise-affine atlas: a cell of the complex, together with
/// the barycentric coordinates it carries.
///
/// **Top-dimensional by construction.** The charts of the atlas are the cells
/// and nothing else -- a face carries no chart, so there is no frame on one in
/// which to express the value of a section, and a point of a face is instead
/// carried by a supporting cell (see [`MeshPoint`]). That contract used to be an
/// assertion repeated wherever a cell was needed; here it is a type, checked once
/// when the chart is made and never again.
///
/// The chart *map* itself is not data: given the cell, the barycentric
/// coordinates are canonical. Nor is the chart's local structure -- the reference
/// vertices, the barycentric differentials, the volume -- since it depends on the
/// dimension alone and not on the cell. That is the fact that every chart of the
/// atlas is the *same* chart up to the labelling of its vertices, and it is why
/// the element matrices are computed once on the reference cell and reused on
/// every cell of the mesh. The labelling is what differs, and the labelling is
/// precisely what a [`Transition`] is made of.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Chart<'m> {
  cell: SimplexRef<'m>,
}

impl<'m> Chart<'m> {
  /// The chart of a cell. Panics if the simplex is not a cell.
  pub fn new(cell: SimplexRef<'m>) -> Self {
    assert_eq!(
      cell.dim(),
      cell.complex().dim(),
      "The charts of the atlas are the cells: a face carries no chart."
    );
    Self { cell }
  }

  /// The cell this is the chart of: the domain of the chart, in the complex.
  pub fn cell(self) -> SimplexRef<'m> {
    self.cell
  }
  pub fn idx(self) -> SimplexIdx {
    self.cell.idx()
  }
  pub fn complex(self) -> &'m Complex {
    self.cell.complex()
  }
  /// The dimension of the manifold, which is that of every chart of it.
  pub fn dim(self) -> Dim {
    self.cell.dim()
  }

  /// The point of the manifold with the given barycentric coordinates.
  pub fn point(self, bary: Bary) -> MeshPoint {
    MeshPoint::new(self.idx(), bary)
  }
  /// The barycenter of the cell, as a point of the manifold.
  pub fn barycenter(self) -> MeshPoint {
    MeshPoint::barycenter(self.idx())
  }
  /// The point given by the barycentric coordinates it has on a face of this
  /// cell, identified by the face's local vertex positions.
  pub fn point_on_face(self, positions: &Combination, face_bary: &Bary) -> MeshPoint {
    MeshPoint::on_face(self.idx(), positions, face_bary)
  }

  /// The transition map into another chart, defined on the face the two share.
  pub fn transition_to(self, target: Chart<'m>) -> Transition {
    Transition::new(self, target)
  }
}

/// The chart of a cell: `cell.chart()`.
pub trait ChartExt<'m> {
  fn chart(self) -> Chart<'m>;
}
impl<'m> ChartExt<'m> for SimplexRef<'m> {
  fn chart(self) -> Chart<'m> {
    Chart::new(self)
  }
}

#[cfg(test)]
mod test {
  use super::*;

  /// A face carries no chart: making one is a contract violation, and it is the
  /// type -- not a convention -- that says so.
  #[test]
  #[should_panic(expected = "a face carries no chart")]
  fn a_face_has_no_chart() {
    let complex = Complex::standard(2);
    let edge = complex.skeleton(1).handle_iter().next().unwrap();
    Chart::new(edge);
  }
}
