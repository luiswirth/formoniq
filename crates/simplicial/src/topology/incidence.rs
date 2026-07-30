//! The incidence between the cells and a skeleton, materialized both ways
//! round.

use super::{complex::Complex, handle::KSimplexIdx};
use crate::Dim;

/// A place in the complex where a cell meets one of its faces: the cell, and
/// the position of the face within that cell's own enumeration.
///
/// The position is what a converse lookup cannot do without.
/// [`SimplexRef::cofaces`](super::handle::SimplexRef::cofaces) already answers which cells contain a simplex, but
/// anything indexed by a cell's faces (a row of a local matrix, a local degree
/// of freedom) needs to know where among them this one sits, and recovering
/// that means searching the cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Incidence {
  pub cell: KSimplexIdx,
  pub position: usize,
}

/// The incidence relation between the cells of a complex and their
/// $k$-dimensional faces, stored in both of its readings.
///
/// One relation $I subset.eq Delta_n times Delta_k$, held twice: `cell_faces`
/// groups it by cell, `face_cells` by face, and neither is derived from the
/// other at lookup time. They are the two orders of the same sparse boolean
/// matrix, transposes of each other, which is the law
/// [`Self::face_cells`] rests on.
///
/// The reason to materialize it is the converse reading. Grouping by cell is
/// what [`SimplexRef::faces`](super::handle::SimplexRef::faces) already iterates; grouping by face is a
/// scatter-free traversal of the same data, letting a computation that would
/// otherwise accumulate into shared faces make each face the sole property of
/// one task. The forward reading is kept alongside it precisely so the two
/// directions are one object rather than two code paths.
///
/// Every cell has the same number of $k$-faces, $binom(n+1, k+1)$, so the
/// forward reading is a flat array with a fixed stride. The converse is not:
/// the number of cells at a face is the face's valence, which the mesh
/// decides, so it carries offsets.
#[derive(Debug, Clone)]
pub struct FaceIncidence {
  grade: Dim,
  nlocal: usize,
  ncells: usize,
  /// Cell-major, stride [`Self::nlocal`].
  faces: Vec<KSimplexIdx>,
  /// Offsets into [`Self::incidences`], one per face plus the total.
  offsets: Vec<usize>,
  /// Face-major, grouped by [`Self::offsets`].
  incidences: Vec<Incidence>,
}

impl FaceIncidence {
  /// Walk the cells once and invert as you go.
  ///
  /// Total at the extremes: at `grade` equal to the complex dimension every
  /// cell is its own single face, and on the point complex that face is the
  /// cell itself.
  pub fn new(complex: &Complex, grade: impl Into<Dim>) -> Self {
    let grade = grade.into();
    let cells = complex.cells();
    // A skeleton is never empty, so there is always a cell to read the stride
    // from.
    let ncells = cells.len();
    let nfaces = complex.skeleton(grade).len();

    let mut faces = Vec::new();
    let mut valences = vec![0usize; nfaces];
    for cell in cells.handle_iter() {
      for face in cell.faces(grade) {
        faces.push(face.kidx());
        valences[face.kidx()] += 1;
      }
    }
    let nlocal = faces.len() / ncells;

    let mut offsets = Vec::with_capacity(nfaces + 1);
    let mut total = 0;
    for valence in &valences {
      offsets.push(total);
      total += valence;
    }
    offsets.push(total);

    // Fill each face's run left to right, `filled` tracking how far each has
    // got. Cells are visited in order, so every run ends up cell-ascending.
    let mut filled = vec![0usize; nfaces];
    let mut incidences = vec![
      Incidence {
        cell: 0,
        position: 0
      };
      total
    ];
    for (cell, cell_faces) in faces.chunks(nlocal).enumerate() {
      for (position, &face) in cell_faces.iter().enumerate() {
        incidences[offsets[face] + filled[face]] = Incidence { cell, position };
        filled[face] += 1;
      }
    }

    Self {
      grade,
      nlocal,
      ncells,
      faces,
      offsets,
      incidences,
    }
  }

  /// The grade $k$ of the faces related.
  pub fn grade(&self) -> Dim {
    self.grade
  }
  /// The number of $k$-faces of one cell, $binom(n+1, k+1)$.
  pub fn nlocal(&self) -> usize {
    self.nlocal
  }
  pub fn ncells(&self) -> usize {
    self.ncells
  }
  pub fn nfaces(&self) -> usize {
    self.offsets.len() - 1
  }

  /// The forward reading: the global indices of a cell's $k$-faces, in the
  /// order [`SimplexRef::faces`](super::handle::SimplexRef::faces) enumerates them.
  pub fn cell_faces(&self, cell: KSimplexIdx) -> &[KSimplexIdx] {
    &self.faces[cell * self.nlocal..(cell + 1) * self.nlocal]
  }

  /// The converse reading: every place a face meets a cell, cell-ascending.
  ///
  /// The valence of the face is the length, so a boundary face of a manifold
  /// mesh has one and an interior one has as many as share it.
  pub fn face_cells(&self, face: KSimplexIdx) -> &[Incidence] {
    &self.incidences[self.offsets[face]..self.offsets[face + 1]]
  }

  /// The forward reading flat, cell-major with stride [`Self::nlocal`]: the
  /// whole relation as one array.
  pub fn faces_flat(&self) -> &[KSimplexIdx] {
    &self.faces
  }

  /// The largest valence any face has, the width a fixed-shape traversal of
  /// the converse reading must pad to.
  pub fn max_valence(&self) -> usize {
    (0..self.nfaces())
      .map(|face| self.face_cells(face).len())
      .max()
      .unwrap_or(0)
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::topology::complex::Complex;

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
}
