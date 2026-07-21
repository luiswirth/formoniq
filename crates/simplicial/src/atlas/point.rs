//! A point of the simplicial manifold, in the barycentric chart.

use super::{
  Bary, BaryRef, Chart, Local, LocalRef, Transition, bary2local, barycenter_bary,
  face_bary_to_cell_bary, is_bary_inside, local2bary,
};
use crate::{
  Dim,
  topology::{
    complex::Complex,
    handle::{SimplexIdx, SimplexRef},
  },
};

use multiindex::Combination;

/// The weight below which a barycentric coordinate counts as vanishing, and the
/// point as lying on the opposite face.
///
/// A tolerance is unavoidable here: whether a point lies *on* a face is an
/// equality test on floating-point weights, and it is what decides whether a
/// [`Transition`] into a neighbouring chart exists at all. It is the same
/// question [`super::is_bary_inside`] asks of the closed cell,
/// and it is answered with the same tolerance.
pub const BARY_EPS: f64 = 1e-12;

/// A point of the simplicial manifold: a cell together with the barycentric
/// coordinates of the point within it.
///
/// The intrinsic notion of a point, defined on any geometry -- coordinates,
/// Regge edge lengths or bare cell metrics alike. Points on a shared face have
/// more than one such representation, one per incident cell, and the
/// [`Transition`] maps are exactly what relates them.
///
/// The cell must be a **cell** of the complex the point is used with, because
/// the charts of the atlas are the cells and nothing else. A point of a face is
/// represented by a supporting cell and the barycentric coordinates it has *in
/// that cell*, exactly as when a form is integrated over a face. A
/// [`SimplexIdx`] of lower dimension is not a stricter case to be
/// supported: a face carries no chart, so there is no frame in which to express a
/// value there. Since a `MeshPoint` stores an index and not a handle, it cannot
/// know its [`Complex`] and cannot check that itself; the contract is a type at
/// the one place a point meets a complex, [`chart`](Self::chart).
#[derive(Debug, Clone, PartialEq)]
pub struct MeshPoint {
  cell: SimplexIdx,
  bary: Bary,
}

impl MeshPoint {
  /// The barycentric weights must be affine, $sum_i lambda_i = 1$: this is the
  /// one boundary where raw weights enter a point, and it is the contract that
  /// keeps a `MeshPoint` on the affine hull of its cell. A weight vector off the
  /// hull is not a point of the manifold. The check is a `debug_assert` with
  /// tolerance rather than a hard type gate, because the weights are only ever
  /// floating-point affine and legitimate drift must not be rejected.
  pub fn new(cell: SimplexIdx, bary: Bary) -> Self {
    assert_eq!(bary.dim(), cell.dim() + 1, "Wrong number of barycentrics.");
    debug_assert!(
      approx::relative_eq!(bary.view().sum(), 1.0, epsilon = 1e-9),
      "Barycentric weights must sum to one."
    );
    Self { cell, bary }
  }
  /// From the local (cartesian) coordinates of the cell chart.
  pub fn from_local<'a>(cell: SimplexIdx, local: impl Into<LocalRef<'a>>) -> Self {
    Self::new(cell, local2bary(local))
  }
  /// The barycenter of a cell.
  pub fn barycenter(cell: SimplexIdx) -> Self {
    Self::new(cell, barycenter_bary(cell.dim()))
  }
  /// The point of a cell's chart given by the barycentric coordinates it has on
  /// one of the cell's faces, identified by its local vertex positions.
  ///
  /// The face of a cell has no chart of its own (only cells do), so a point of a
  /// face is always carried by a supporting cell -- and this is the map that
  /// puts it there. Pure combinatorics: scatter the weights onto the positions.
  pub fn on_face<'a>(
    cell: SimplexIdx,
    positions: &Combination,
    face_bary: impl Into<BaryRef<'a>>,
  ) -> Self {
    Self::new(
      cell,
      face_bary_to_cell_bary(cell.dim(), positions, face_bary),
    )
  }

  /// The index of the cell whose chart this point is expressed in.
  pub fn cell_idx(&self) -> SimplexIdx {
    self.cell
  }

  /// The [`Chart`] this point lives in.
  ///
  /// The single crossing from a point to the complex it belongs to, and hence
  /// the one place the atlas contract is enforced: the stored index must
  /// prove the [`Cell`](crate::topology::role::Cell) role here, since
  /// index-level data carries no proof of its own.
  pub fn chart<'m>(&self, complex: &'m Complex) -> Chart<'m> {
    self.cell.handle(complex).role()
  }

  /// The dimension of the manifold, which is that of the containing cell.
  pub fn dim(&self) -> Dim {
    self.cell.dim()
  }
  pub fn bary(&self) -> BaryRef<'_> {
    self.bary.as_view()
  }
  /// The local (cartesian) coordinates of the cell chart.
  pub fn local(&self) -> Local {
    bary2local(&self.bary)
  }
  /// Whether the point lies in the closed cell, rather than in the affine
  /// extension of the chart beyond it.
  pub fn is_inside(&self) -> bool {
    is_bary_inside(&self.bary)
  }

  /// The local vertex positions of the face whose interior the point lies in:
  /// those with nonvanishing barycentric weight.
  ///
  /// The support determines everything about how the point is shared: it is the
  /// smallest face carrying the point, hence exactly the set of cells in whose
  /// charts the point is representable are the cells containing that face.
  pub fn support_positions(&self) -> Combination {
    Combination::from_increasing((0..self.bary.dim()).filter(|&i| self.bary[i].abs() > BARY_EPS))
  }

  /// The face of the complex whose interior the point lies in: the smallest
  /// simplex carrying it. A point in the interior of a cell supports the cell
  /// itself; a vertex of the mesh supports that vertex.
  pub fn support<'m>(&self, complex: &'m Complex) -> SimplexRef<'m> {
    let cell = self.chart(complex);
    let face = cell.simplex().select(self.support_positions());
    complex.skeleton(face.dim()).handle_by_simplex(&face)
  }

  /// The same point of the manifold, seen in another chart.
  ///
  /// `None` when the point is not in the overlap of the two charts -- that is,
  /// when its [`support`](Self::support) is not a face of the target cell, so
  /// there is no representation of it there. See [`Transition`].
  pub fn transition_to(&self, target: Chart) -> Option<Self> {
    Transition::new(self.chart(target.complex()), target).apply(self)
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::topology::complex::Complex;

  /// The charts of the atlas are the cells: resolving a point whose simplex is a
  /// face, not a cell, is a contract violation and not a supported case.
  ///
  /// There is no frame on a face in which to express a value -- which is why a
  /// point of a face is carried by a *supporting cell* instead.
  #[test]
  #[should_panic(expected = "is not a cell")]
  fn a_point_of_a_face_has_no_chart() {
    let complex = Complex::standard(2);
    let edge = complex.skeleton(1).handle_iter().next().unwrap();
    let point = MeshPoint::barycenter(edge.idx());
    point.chart(&complex);
  }

  /// The support of a point is the smallest face carrying it: the barycenter of
  /// a face supports that face, and an interior point supports the whole cell.
  #[test]
  fn support_is_the_smallest_carrying_face() {
    for dim in 1..=3 {
      let complex = Complex::standard(dim);
      let cell = complex.cells().handle_iter().next().unwrap();

      for face_dim in 0..=dim {
        for face in cell.faces(face_dim) {
          let positions = face.simplex().relative_to(cell.simplex());
          let point = MeshPoint::on_face(cell.idx(), &positions, &barycenter_bary(face_dim));
          assert_eq!(point.support(&complex), face);
        }
      }
    }
  }
}
