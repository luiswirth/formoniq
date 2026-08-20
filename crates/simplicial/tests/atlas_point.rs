//! A [`MeshPoint`]: the checked constructor decides affineness, a point
//! resolves only through a cell's own chart, and its support is the smallest
//! carrying face.

use simplicial::Dim;
use simplicial::atlas::{Bary, MeshPoint, barycenter_bary};
use simplicial::linalg::Vector;
use simplicial::topology::complex::Complex;

/// The checked constructor decides affineness of the weights: it accepts what
/// the structural constructors build and rejects a weight vector off the
/// affine hull.
///
/// Affine and not convex, so a point extrapolated outside the cell is
/// accepted: it is a point of the chart's extension, which is what makes the
/// hypothesis the sum and not the range.
#[test]
fn new_checked_decides_affineness() {
  for dim in (0..=3usize).map(Dim::from) {
    let cell = Complex::unit(dim)
      .cells()
      .handle_iter()
      .next()
      .unwrap()
      .idx();
    assert!(MeshPoint::barycenter(cell).is_valid());

    if dim.index() > 0 {
      let mut outside = Vector::zeros(dim.index() + 1);
      outside[0] = 2.0;
      outside[1] = -1.0;
      assert!(MeshPoint::new_checked(cell, Bary::new(outside)).is_some());
    }

    let off_hull = Vector::zeros(dim.index() + 1);
    assert!(MeshPoint::new_checked(cell, Bary::new(off_hull)).is_none());
  }
}

/// The charts of the atlas are the cells: resolving a point whose simplex is a
/// face, not a cell, is a contract violation and not a supported case.
///
/// There is no frame on a face in which to express a value, which is why a
/// point of a face is carried by a supporting cell instead.
#[test]
#[should_panic(expected = "is not a cell")]
fn a_point_of_a_face_has_no_chart() {
  let complex = Complex::unit(Dim::new(2));
  let edge = complex.skeleton(Dim::new(1)).handle_iter().next().unwrap();
  let point = MeshPoint::barycenter(edge.idx());
  point.chart(&complex);
}

/// The support of a point is the smallest face carrying it: the barycenter of
/// a face supports that face, and an interior point supports the whole cell.
#[test]
fn support_is_the_smallest_carrying_face() {
  for dim in (1..=3usize).map(Dim::from) {
    let complex = Complex::unit(dim);
    let cell = complex.cells().handle_iter().next().unwrap();

    for face_dim in dim.range_inclusive() {
      for face in cell.faces(face_dim) {
        let positions = face.simplex().relative_to(cell.simplex());
        let point = MeshPoint::on_face(cell.idx(), &positions, &barycenter_bary(face_dim));
        assert_eq!(point.support(&complex), face);
      }
    }
  }
}
