//! Indexing on [`SkeletonVec`] and [`ComplexVec`]: by simplex, by
//! within-grade position, and by [`SimplexIdx`].

use simplicial::topology::data::{ComplexData, ComplexVec, SkeletonData, SkeletonVec};
use simplicial::topology::handle::SimplexIdx;

#[test]
fn skeleton_vec_indexing() {
  let edges = SkeletonVec::new(1, vec![10.0, 20.0, 30.0]);
  assert_eq!(edges[2usize], 30.0);
  assert_eq!(edges[SimplexIdx::new(1, 1)], 20.0);
  assert_eq!(edges.at_id(SimplexIdx::new(1, 0)), &10.0);
  assert_eq!(edges.grade(), 1);
  assert_eq!(edges.len(), 3);
}

#[test]
fn complex_vec_indexing() {
  let data = ComplexVec::new(vec![
    SkeletonVec::new(0, vec![1, 2, 3]),
    SkeletonVec::new(1, vec![4, 5]),
  ]);
  assert_eq!(data[SimplexIdx::new(0, 2)], 3);
  assert_eq!(data[SimplexIdx::new(1, 0)], 4);
  assert_eq!(data.at(SimplexIdx::new(1, 1)), &5);
  assert_eq!(data.dim(), 1);
}
