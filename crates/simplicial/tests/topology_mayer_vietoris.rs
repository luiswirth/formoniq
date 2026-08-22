//! Mayer-Vietoris: a space covered by two subcomplexes has its cohomology
//! determined by theirs and by that of their intersection, through the long
//! exact sequence
//!
//! $dots.c -> H^k (K) -> H^k (A) plus.circle H^k (B) -> H^k (A inter B) -> H^(k+1) (K) -> dots.c$
//!
//! Exactness is checked the way any long exact sequence is checkable from
//! dimensions alone: the alternating sum of the dimensions of an exact
//! sequence vanishes, so
//!
//! $sum_k (-1)^k (b^k (K) - b^k (A) - b^k (B) + b^k (A inter B)) = 0.$
//!
//! That is a weaker statement than exactness, and it is the one a Betti
//! number can carry. It is not weak: the cover is chosen so that every piece
//! is contractible or a disjoint union of contractible pieces, and the whole
//! is not, which is exactly the situation the sequence exists to handle. The
//! generator of $H^1$ of the circle and of $H^2$ of the sphere is invisible
//! in each piece and appears only through the connecting map.

use simplicial::topology::complex::Complex;
use simplicial::topology::simplex::Simplex;
use simplicial::topology::skeleton::Skeleton;

fn complex(cells: impl IntoIterator<Item = Vec<usize>>) -> Complex {
  // A skeleton stores every simplex in colex vertex order, and orientation is
  // a gauge no Betti number reads, so the cells are sorted rather than wound.
  let cells = cells
    .into_iter()
    .map(|mut cell| {
      cell.sort_unstable();
      Simplex::new(cell)
    })
    .collect();
  Complex::from_cells(Skeleton::new(cells))
}

/// A cover $K = A union B$ together with the Betti numbers each piece is
/// expected to carry, all padded to the same length.
struct Cover {
  name: &'static str,
  whole: (Complex, Vec<usize>),
  left: (Complex, Vec<usize>),
  right: (Complex, Vec<usize>),
  overlap: (Complex, Vec<usize>),
}

fn covers() -> Vec<Cover> {
  // The circle as two arcs overlapping in two of its edges: each arc is
  // contractible, the overlap is two disjoint edges, hence two contractible
  // components, and it is that disconnection which produces $H^1$ of the
  // circle. Vertices are renumbered per piece, since a complex owns a
  // contiguous vertex range.
  let circle = Cover {
    name: "circle covered by two arcs",
    whole: (complex((0..6).map(|i| vec![i, (i + 1) % 6])), vec![1, 1]),
    left: (complex((0..4).map(|i| vec![i, i + 1])), vec![1, 0]),
    right: (complex((0..4).map(|i| vec![i, i + 1])), vec![1, 0]),
    overlap: (complex([vec![0, 1], vec![2, 3]]), vec![2, 0]),
  };

  // The 2-sphere as the boundary of a tetrahedron, covered by three of its
  // triangles and the fourth: two disks glued along a circle. Neither disk
  // sees $H^2$, and the fundamental class appears through the connecting map
  // out of $H^1$ of the equator.
  let sphere = Cover {
    name: "sphere covered by two disks",
    whole: (
      complex([vec![0, 1, 2], vec![0, 1, 3], vec![0, 2, 3], vec![1, 2, 3]]),
      vec![1, 0, 1],
    ),
    left: (
      complex([vec![0, 1, 2], vec![0, 1, 3], vec![0, 2, 3]]),
      vec![1, 0, 0],
    ),
    right: (complex([vec![0, 1, 2]]), vec![1, 0, 0]),
    overlap: (complex([vec![0, 1], vec![0, 2], vec![1, 2]]), vec![1, 1, 0]),
  };

  vec![circle, sphere]
}

/// The Mayer-Vietoris sequence of every cover is exact, read through the
/// vanishing of its alternating sum of dimensions.
#[test]
fn the_mayer_vietoris_sequence_of_a_cover_is_exact() {
  for cover in covers() {
    let Cover {
      name,
      whole,
      left,
      right,
      overlap,
    } = cover;

    let pieces = [whole, left, right, overlap];
    let betti: Vec<Vec<usize>> = pieces
      .iter()
      .map(|(complex, expected)| {
        let mut betti = complex.betti_numbers();
        betti.resize(expected.len(), 0);
        assert_eq!(&betti, expected, "{name}: Betti numbers");
        betti
      })
      .collect();

    let alternating: isize = (0..betti[0].len())
      .map(|k| {
        let term =
          betti[0][k] as isize - betti[1][k] as isize - betti[2][k] as isize + betti[3][k] as isize;
        if k % 2 == 0 { term } else { -term }
      })
      .sum();
    assert_eq!(alternating, 0, "{name}: the sequence is not exact");
  }
}
