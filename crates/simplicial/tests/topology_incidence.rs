//! [`FaceIncidence`]: the cell-to-face relation in its two readings, forward
//! and converse, checked to be transposes and to agree with navigation.

use simplicial::Dim;
use simplicial::topology::complex::Complex;
use simplicial::topology::incidence::{FaceIncidence, Incidence};

/// Single cells and refined meshes at each dimension.
///
/// The one-cell case is where every face has valence one, and the point
/// complex is the degenerate end of it, its only cell being its only face.
/// The refined meshes carry interior faces shared by several cells, which is
/// what a converse reading has to get right.
fn meshes() -> impl Iterator<Item = (String, Complex)> {
  let single = (0..=3).map(|dim| (format!("unit {dim}"), Complex::unit(dim)));
  let refined = (1..=3).flat_map(|dim| {
    (2..=3).map(move |refinement| {
      (
        format!("dim {dim} refined {refinement}"),
        Complex::unit(dim).refine(refinement).into_complex(),
      )
    })
  });
  single.chain(refined)
}

/// The two readings are transposes: a face appears at position $p$ of cell
/// $c$ in one exactly when $(c, p)$ appears at that face in the other.
///
/// Checked both ways round, since one inclusion alone would pass on a
/// converse reading that dropped incidences.
#[test]
fn the_two_readings_are_transposes() {
  for (name, complex) in meshes() {
    for grade in (0..=complex.dim().index()).map(Dim::from) {
      let incidence = FaceIncidence::new(&complex, grade);

      let mut forward = Vec::new();
      for cell in 0..incidence.ncells() {
        for (position, &face) in incidence.cell_faces(cell).iter().enumerate() {
          forward.push((face, Incidence { cell, position }));
        }
      }
      let mut converse = Vec::new();
      for face in 0..incidence.nfaces() {
        converse.extend(incidence.face_cells(face).iter().map(|&i| (face, i)));
      }

      forward.sort_unstable();
      converse.sort_unstable();
      assert_eq!(forward, converse, "{name} at grade {grade}");
    }
  }
}

/// The forward reading agrees with navigating the complex, which is what it
/// is a materialization of.
#[test]
fn the_forward_reading_is_the_face_navigation() {
  for (name, complex) in meshes() {
    for grade in (0..=complex.dim().index()).map(Dim::from) {
      let incidence = FaceIncidence::new(&complex, grade);
      for cell in complex.cells().handle_iter() {
        let navigated: Vec<_> = cell.faces(grade).map(|f| f.kidx()).collect();
        assert_eq!(
          navigated,
          incidence.cell_faces(cell.kidx()),
          "{name} at grade {grade}"
        );
      }
    }
  }
}

/// Every face of the complex is reached, and only through cells that
/// actually contain it. Together with the transpose law this pins the
/// relation: no incidence invented, none dropped.
#[test]
fn every_face_is_covered_by_the_cells_containing_it() {
  for (name, complex) in meshes() {
    for grade in (0..=complex.dim().index()).map(Dim::from) {
      let incidence = FaceIncidence::new(&complex, grade);
      for face in complex.skeleton(grade).handle_iter() {
        let cells: Vec<_> = incidence
          .face_cells(face.kidx())
          .iter()
          .map(|i| i.cell)
          .collect();
        let cofaces: Vec<_> = face
          .cofaces(complex.dim())
          .map(|c| c.kidx())
          .collect::<std::collections::BTreeSet<_>>()
          .into_iter()
          .collect();
        assert!(!cells.is_empty(), "{name} at grade {grade}: face unreached");
        assert_eq!(cells, cofaces, "{name} at grade {grade}");
      }
    }
  }
}

/// Total at the degenerate boundary: the point complex, whose one cell is
/// its one face, related to itself at position zero.
#[test]
fn the_point_complex_relates_its_cell_to_itself() {
  let incidence = FaceIncidence::new(&Complex::unit(0), 0);
  assert_eq!((incidence.ncells(), incidence.nfaces()), (1, 1));
  assert_eq!(incidence.nlocal(), 1);
  assert_eq!(incidence.cell_faces(0), [0]);
  assert_eq!(
    incidence.face_cells(0),
    [Incidence {
      cell: 0,
      position: 0
    }]
  );
  assert_eq!(incidence.max_valence(), 1);
}
