//! A chart of the atlas.

use super::{Bary, MeshPoint, Transition};
use crate::topology::role::Cell;

use multiindex::Combination;

/// A chart of the piecewise-affine atlas *is* a cell of the complex, and the
/// type identity states it: the [`Cell`] witness carries the whole contract
/// (top-dimensional by construction -- a face carries no chart, so there is no
/// frame on one in which to express the value of a section, and a point of a
/// face is instead carried by a supporting cell, see [`MeshPoint`]).
///
/// The chart *map* itself is not data: given the cell, the barycentric
/// coordinates are canonical. Nor is the chart's local structure -- the
/// reference vertices, the barycentric differentials, the volume -- since it
/// depends on the dimension alone and not on the cell. That is the fact that
/// every chart of the atlas is the *same* chart up to the labelling of its
/// vertices, and it is why the element matrices are computed once on the
/// reference cell and reused on every cell of the mesh. The labelling is what
/// differs, and the labelling is precisely what a [`Transition`] is made of.
pub type Chart<'m> = Cell<'m>;

/// What makes a cell a chart: the atlas operations, on the [`Cell`] witness.
pub trait ChartExt<'m> {
  /// The point of the manifold with the given barycentric coordinates.
  fn point(self, bary: Bary) -> MeshPoint;
  /// The barycenter of the cell, as a point of the manifold.
  fn barycenter(self) -> MeshPoint;
  /// The point given by the barycentric coordinates it has on a face of this
  /// cell, identified by the face's local vertex positions.
  fn point_on_face(self, positions: &Combination, face_bary: &Bary) -> MeshPoint;
  /// The transition map into another chart, defined on the face the two share.
  fn transition_to(self, target: Chart<'m>) -> Transition;
}

impl<'m> ChartExt<'m> for Cell<'m> {
  fn point(self, bary: Bary) -> MeshPoint {
    MeshPoint::new(self.idx(), bary)
  }
  fn barycenter(self) -> MeshPoint {
    MeshPoint::barycenter(self.idx())
  }
  fn point_on_face(self, positions: &Combination, face_bary: &Bary) -> MeshPoint {
    MeshPoint::on_face(self.idx(), positions, face_bary)
  }
  fn transition_to(self, target: Chart<'m>) -> Transition {
    Transition::new(self, target)
  }
}
