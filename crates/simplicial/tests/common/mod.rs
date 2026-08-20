//! Fixture complexes shared by the topology law tests: homology,
//! cohomology and orientation all check laws against the same combinatorial
//! objects, so the objects live here once.

use simplicial::mesher::grid::CartesianTopology;
use simplicial::topology::complex::Complex;
use simplicial::topology::simplex::Simplex;
use simplicial::topology::skeleton::Skeleton;

/// A combinatorial 2-sphere: the boundary of a tetrahedron, whose four
/// triangles are the facets of the 3-simplex.
///
/// $b_2 = 1$, and no coordinates are needed to say so: a sphere is a
/// combinatorial object here, not a subdivided icosahedron.
pub fn two_sphere() -> Complex {
  Complex::from_cells(Skeleton::new(vec![
    Simplex::new(vec![0, 1, 2]),
    Simplex::new(vec![0, 1, 3]),
    Simplex::new(vec![0, 2, 3]),
    Simplex::new(vec![1, 2, 3]),
  ]))
}

/// A square annulus: the $3 times 3$ grid with the middle box removed, so
/// $b_1 = 1$.
///
/// The middle box is named by its index, not by a barycenter: the Kuhn
/// generator emits `factorial(dim)` cells per box in colex order, so box
/// $(1, 1)$ of a $3 times 3$ grid is index 4 and its cells are 8 and 9. That
/// is combinatorics, which is why this test needs no embedding.
pub fn annulus() -> Complex {
  let grid = CartesianTopology::cube(2, 3);
  let middle_box = 4;
  let per_box = multiindex::factorial(2);
  let removed = middle_box * per_box..(middle_box + 1) * per_box;
  let cells: Vec<Simplex> = grid
    .cell_skeleton()
    .iter()
    .enumerate()
    .filter(|(i, _)| !removed.contains(i))
    .map(|(_, s)| s.clone())
    .collect();
  Complex::from_cells(Skeleton::new(cells))
}

/// A spread of complexes with nontrivial homology across the grades.
pub fn test_complexes() -> Vec<Complex> {
  let mut complexes: Vec<Complex> = (1..=3)
    .map(|dim| CartesianTopology::cube(dim, 2).triangulate())
    .collect();
  complexes.push(two_sphere());
  complexes.push(annulus());
  complexes
}

/// Whether a chain is a cycle: $diff_k z = 0$.
pub fn is_cycle(complex: &Complex, chain: &simplicial::topology::chain::Chain) -> bool {
  chain.boundary(complex).coeffs().iter().all(|&c| c == 0)
}

/// Whether a cochain is a cocycle: $dif^k z = 0$, over $ZZ$ hence exactly.
pub fn is_cocycle(complex: &Complex, cochain: &simplicial::topology::chain::Cochain<i64>) -> bool {
  cochain.dif(complex).coeffs().iter().all(|&c| c == 0)
}
