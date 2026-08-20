//! A [`Subcomplex`](simplicial::topology::subcomplex::Subcomplex): the
//! boundary's homology, the trace as a cochain map, and the relative chain
//! group as the kernel of the trace.

use simplicial::Dim;
use simplicial::linalg::{CooMatrix, CsrMatrix, Matrix};
use simplicial::mesher::grid::CartesianTopology;

/// The boundary of the n-cube is a closed manifold with the homology of
/// the (n-1)-sphere.
#[test]
fn boundary_of_cube_is_sphere() {
  for dim in (1..=3usize).map(Dim::from) {
    let topology = CartesianTopology::cube(dim, 2).triangulate();
    let boundary = topology.boundary_complex().unwrap();
    assert!(!boundary.complex().has_boundary());
    for k in dim.range() {
      // S^(n-1) betti numbers. The 0-sphere is two points.
      let expected = if dim == 1 {
        2
      } else {
        usize::from(k == 0 || k == dim - 1)
      };
      assert_eq!(
        boundary.complex().betti_number(k),
        expected,
        "dim={dim} k={k}"
      );
    }
  }
}

/// The trace is a cochain map: $"tr" compose dif = dif compose "tr"$.
#[test]
fn trace_is_cochain_map() {
  for dim in (2..=3usize).map(Dim::from) {
    let topology = CartesianTopology::cube(dim, 2).triangulate();
    let boundary = topology.boundary_complex().unwrap();
    for k in (dim - 1).range() {
      let trace_k = CsrMatrix::from(&boundary.trace_operator(k));
      let trace_kk = CsrMatrix::from(&boundary.trace_operator(k + 1));
      let dif_parent = CsrMatrix::from(&topology.coboundary_operator(k));
      let dif_boundary = CsrMatrix::from(&boundary.complex().coboundary_operator(k));

      let tr_dif = Matrix::from(&CooMatrix::from(&(trace_kk * dif_parent)));
      let dif_tr = Matrix::from(&CooMatrix::from(&(dif_boundary * trace_k)));
      assert_eq!(tr_dif, dif_tr);
    }
  }
}

/// Exactness of $0 -> C(K, diff K) -> C(K) -> C(diff K) -> 0$: the relative
/// chain group is the kernel of the trace, hence the complement of the
/// inclusion, coordinate for coordinate and not merely in dimension.
///
/// The two sides reach the same selection by different routes, the inclusion
/// through the renumbered boundary complex and the relative basis through
/// the boundary facets of the parent, so their agreement is the sequence
/// being exact rather than a tautology.
#[test]
fn the_relative_complex_is_the_kernel_of_the_trace() {
  for dim in (1..=3usize).map(Dim::from) {
    let topology = CartesianTopology::cube(dim, 2).triangulate();
    let boundary = topology.boundary_complex().unwrap();
    for k in dim.range_inclusive() {
      let inclusion = boundary.inclusion(k);
      assert_eq!(inclusion.len(), boundary.complex().nsimplices(k));
      assert_eq!(inclusion.total(), topology.nsimplices(k));
      assert_eq!(&inclusion.complement(), &topology.interior_selection(k));
    }
  }
}
