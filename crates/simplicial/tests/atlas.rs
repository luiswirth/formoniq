//! The reference chart: the lattice construction, the two coordinate systems
//! (barycentric and local cartesian) and their differentials.

use approx::assert_relative_eq;
use multiindex::binomial;
use simplicial::Dim;
use simplicial::atlas::{
  Local, bary2local, barycenter_bary, face_bary_to_cell_bary, is_bary_inside, local2bary,
  unit_bary, unit_difbary, unit_difbarys, unit_face_spanning_vectors, unit_lattice,
  unit_lattice_bary, unit_lattice_interior, unit_lattice_interior_bary, unit_vertices,
};
use simplicial::linalg::{RowVector, Vector};

/// The lattice has $binom(R + n, n)$ points, each a composition of $R$.
#[test]
fn lattice_is_a_composition_set() {
  for dim in (0..=4usize).map(Dim::from) {
    for refinement in 1..=5 {
      let lattice: Vec<_> = unit_lattice(dim, refinement).collect();
      assert_eq!(
        lattice.len(),
        binomial(refinement + dim.index(), dim.index())
      );
      for point in &lattice {
        assert_eq!(point.len(), dim + 1);
        assert_eq!(point.iter().sum::<usize>(), refinement);
      }
      let unique: std::collections::HashSet<_> = lattice.iter().collect();
      assert_eq!(unique.len(), lattice.len());
    }
  }
}

/// $L_1^n$ is the vertex set of the reference cell, in the order
/// [`unit_vertices`] places it: the lattice extends the vertices rather than
/// merely containing them.
#[test]
fn lattice_at_refinement_one_is_the_vertices() {
  for dim in (0..=4usize).map(Dim::from) {
    let vertices = unit_vertices(dim);
    for (ivertex, bary) in unit_lattice_bary(dim, 1).enumerate() {
      assert_relative_eq!(bary2local(&bary).view(), &vertices.column(ivertex));
    }
  }
}

/// The weights are barycentric, and every lattice point is a point of the
/// closed cell.
#[test]
fn lattice_bary_lies_in_the_cell() {
  for dim in (0..=4usize).map(Dim::from) {
    for refinement in 1..=5 {
      for bary in unit_lattice_bary(dim, refinement) {
        assert_relative_eq!(bary.view().sum(), 1.0);
        assert!(is_bary_inside(&bary));
      }
    }
  }
}

/// The lattice closes on the faces: the points vanishing on vertex `i` are,
/// with that weight dropped, exactly the facet's own lattice at the same $R$.
/// This is what lets two cells agree on a shared facet combinatorially.
#[test]
fn lattice_restricts_to_the_facet_lattice() {
  for dim in (1..=4usize).map(Dim::from) {
    for refinement in 1..=5 {
      let facet: std::collections::HashSet<_> = unit_lattice(dim - 1, refinement).collect();
      for ivertex in 0..=dim.index() {
        let restricted: std::collections::HashSet<_> = unit_lattice(dim, refinement)
          .filter(|k| k[ivertex] == 0)
          .map(|mut k| {
            k.remove(ivertex);
            k
          })
          .collect();
        assert_eq!(restricted, facet);
      }
    }
  }
}

/// The interior is exactly the lattice minus every face: $binom(R-1, n)$
/// points, each on no face, and none of the boundary ones missed.
#[test]
fn lattice_interior_is_the_lattice_off_the_faces() {
  for dim in (0..=4usize).map(Dim::from) {
    for refinement in 1..=6 {
      let interior: Vec<_> = unit_lattice_interior(dim, refinement).collect();
      let expected: Vec<_> = unit_lattice(dim, refinement)
        .filter(|k| k.iter().all(|&k| k >= 1))
        .collect();
      assert_eq!(interior, expected);
      assert_eq!(
        interior.len(),
        refinement
          .checked_sub(1)
          .map_or(0, |r| binomial(r, dim.index()))
      );
    }
  }
}

/// The base case of the interior: $R = n + 1$ spends one unit on each part and
/// leaves the barycenter alone, and below that there is no inside to have.
#[test]
fn lattice_interior_bottoms_out_at_the_barycenter() {
  for dim in (0..=4usize).map(Dim::from) {
    for refinement in 0..=dim.index() {
      assert_eq!(unit_lattice_interior(dim, refinement).count(), 0);
    }
    let base: Vec<_> = unit_lattice_interior_bary(dim, dim.index() + 1).collect();
    assert_eq!(base.len(), 1);
    assert_relative_eq!(base[0].view(), barycenter_bary(dim).view());
  }
}

/// The two chart coordinate systems are mutually inverse.
#[test]
fn bary_local_roundtrip() {
  for dim in (0..=4usize).map(Dim::from) {
    let local = Local::from_iterator(dim.index(), (0..dim.index()).map(|i| 0.1 * (i + 1) as f64));
    let bary = local2bary(&local);
    assert_relative_eq!(bary.sum(), 1.0, epsilon = 1e-12);
    assert_relative_eq!(bary2local(&bary).vector(), local.vector(), epsilon = 1e-12);
  }
}

/// The rows of the barycentric differential are the individual $dif lambda_i$,
/// and they sum to zero: $sum_i lambda_i = 1$ is constant.
#[test]
fn unit_difbarys_rows_are_difbary_and_sum_to_zero() {
  for dim in (0..=4usize).map(Dim::from) {
    let difbarys = unit_difbarys(dim);
    for ivertex in 0..=dim.index() {
      assert_relative_eq!(
        difbarys.row(ivertex).into_owned(),
        unit_difbary(dim, ivertex)
      );
    }
    assert_relative_eq!(difbarys.row_sum(), RowVector::zeros(dim.index()));
  }
}

/// The barycentric coordinate functions are dual to the reference vertices,
/// $lambda_i (e_j) = delta_(i j)$, and the differentials are their gradients.
#[test]
fn unit_barys_are_dual_to_unit_vertices() {
  for dim in (0..=4usize).map(Dim::from) {
    let vertices = unit_vertices(dim);
    for (j, vertex) in vertices.column_iter().enumerate() {
      let local = Local::new(vertex.into_owned());
      for i in 0..=dim.index() {
        let expected = f64::from(i == j);
        assert_relative_eq!(unit_bary(i, &local), expected, epsilon = 1e-12);
      }
    }
  }
}

/// A face's spanning vectors are the differences of the reference vertices it
/// selects, and a point of the face has the face's weights on those positions.
#[test]
fn face_bary_scatters_onto_the_face() {
  let cell_dim = 3;
  for face_dim in 0..=cell_dim {
    for positions in multiindex::combinations(cell_dim + 1, face_dim + 1) {
      let face_bary = barycenter_bary(face_dim);
      let bary = face_bary_to_cell_bary(cell_dim, &positions, &face_bary);

      assert_relative_eq!(bary.sum(), 1.0, epsilon = 1e-12);
      for i in 0..=cell_dim {
        let on_face = positions.iter().any(|p| p == i);
        assert_eq!(bary[i] != 0.0, on_face);
      }

      // The face's barycenter, expressed in the cell chart, is the mean of the
      // reference vertices the face selects.
      let vertices = unit_vertices(cell_dim);
      let spanning = unit_face_spanning_vectors(cell_dim, &positions);
      assert_eq!(spanning.ncols(), face_dim);
      let mean = positions
        .iter()
        .map(|p| vertices.column(p).into_owned())
        .sum::<Vector>()
        / (face_dim + 1) as f64;
      assert_relative_eq!(bary2local(&bary).vector(), &mean, epsilon = 1e-12);
    }
  }
}
