//! Refinement is a homeomorphism of the underlying polyhedron, so the
//! topological invariants do not see it.

mod common;

use common::{annulus, two_sphere};
use simplicial::Dim;
use simplicial::mesher::grid::CartesianTopology;

/// The Euler characteristic is a topological invariant, so Freudenthal
/// refinement leaves it alone: $chi(K) = chi("refine"_R (K))$, although every
/// $f_k$ in the alternating sum $sum_k (-1)^k f_k$ changes.
///
/// Stated on complexes of three different characteristics ($chi = 1$ on the
/// contractible cube, $2$ on the sphere, $0$ on the annulus), since an
/// implementation returning a constant would pass on any one of them, and the
/// raw face counts are asserted to move so that the invariance is a claim
/// about cancellation rather than about nothing having happened.
#[test]
fn euler_characteristic_is_invariant_under_refinement() {
  let cubes = (1..=3usize)
    .map(Dim::from)
    .map(|dim| CartesianTopology::cube(dim, 2).triangulate());

  for coarse in cubes.chain([two_sphere(), annulus()]) {
    let dim = coarse.dim();
    for r in 2..=3 {
      let fine = coarse.refine(r);
      let fine = fine.complex();
      assert_eq!(
        fine.euler_characteristic(),
        coarse.euler_characteristic(),
        "dim {dim}, R = {r}"
      );
      assert!(
        fine.nsimplices(dim) > coarse.nsimplices(dim),
        "dim {dim}, R = {r}: the refinement did nothing"
      );
    }
  }
}
