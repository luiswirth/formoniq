//! Coherent orientation of the cells, and the orientability test.
//!
//! A [`Skeleton`](super::skeleton::Skeleton) stores every simplex in its
//! canonical colex vertex order, so each cell carries an orientation chosen by
//! the indexing convention and unrelated to its neighbors'. That choice is a
//! gauge: flipping cell $K$ replaces its basis Whitney form
//! $omega_K |-> -omega_K$, so every assembled operator transforms by the
//! diagonal congruence $A |-> S A S$ with $S = "diag"(plus.minus 1)$, and the
//! chain complex, the spectrum and the homology are all invariant. Nothing in
//! the FEEC core needs the gauge fixed.
//!
//! What does need it is any question asked about the manifold as a whole
//! rather than cell by cell: a global volume form, hence the Hodge star of a
//! top-grade form, hence $integral_M$. Those are the same question, and this
//! module answers it.
//!
//! A coherent orientation is an assignment $sigma: "cells" -> {plus.minus 1}$
//! such that adjacent cells induce opposite orientations on the facet they
//! share:
//!
//! $sigma_(K_1) partial[F, K_1] + sigma_(K_2) partial[F, K_2] = 0$
//!
//! for every interior facet $F$. Equivalently $partial_n (sum_K sigma_K K)$ is
//! supported on the boundary facets alone, on a closed connected manifold
//! that chain is the fundamental class, the generator of $H_n (K; ZZ) tilde.eq
//! ZZ$. So orientability is not an extra structure bolted on: it is the
//! statement $H_n != 0$, and the propagation here is the constructive form of
//! it (see [`homology`](super::homology) for the rank-theoretic form).
//!
//! The condition is a constraint between neighbors, so it propagates: fix
//! $sigma$ on one cell and every cell reachable through interior facets
//! follows. The complex is orientable iff the walk never returns to a cell
//! demanding the sign opposite to the one already assigned, which is exactly
//! how a Möbius band or a Klein bottle fails. Orientability is a property of
//! each connected component, and the sign of $sigma$ on a component is free;
//! this is the usual ambiguity of a fundamental class, not an arbitrary choice
//! the code could avoid.

use super::{
  complex::Complex,
  handle::{KSimplexIdx, SimplexIdx},
  role::{Cell, roles},
};

use multiindex::Sign;

/// A coherent orientation of a [`Complex`]: one [`Sign`] per cell, relative to
/// the cell's own colex vertex order, such that adjacent cells induce opposite
/// orientations on their shared facet.
///
/// Obtainable only by a coherence check that can fail: intrinsically from
/// [`Complex::orientation`], or by validating an externally supplied candidate
/// with [`Complex::orient_by`]. Both return `None` on a non-orientable complex,
/// so holding one is the proof that the mesh is orientable, in the sense of
/// the type-level witnesses in [`role`](super::role). Code that needs a global volume form should take an
/// `&Orientation` rather than re-deriving or assuming the property.
///
/// Fixed only up to a global sign per connected component.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Orientation {
  /// Indexed by cell kidx.
  signs: Vec<Sign>,
}

impl Orientation {
  /// The cell's sign relative to its stored colex vertex order: `Pos` if that
  /// order already agrees with the coherent orientation, `Neg` if the cell's
  /// frame is flipped against it.
  ///
  /// This is the factor a top-grade quantity read in the cell's own frame, a
  /// Hodge star $star: Lambda^n -> Lambda^0$, an integral over the cell,
  /// must be multiplied by to be comparable across cells.
  pub fn sign(&self, cell: Cell<'_>) -> Sign {
    self.signs[cell.idx().kidx]
  }

  /// The signs by cell kidx, in the skeleton's colex order.
  pub fn signs(&self) -> &[Sign] {
    &self.signs
  }

  /// The reversed orientation: the other generator. On a disconnected complex
  /// this flips every component at once, which is one of the $2^c$ coherent
  /// orientations, not the only other one.
  pub fn reversed(&self) -> Self {
    Self {
      signs: self.signs.iter().map(|s| s.other()).collect(),
    }
  }
}

impl Complex {
  /// A coherent orientation of the cells, or `None` if the complex is
  /// non-orientable.
  ///
  /// Computed by propagating the induced-orientation constraint across
  /// interior facets, restarting at an unvisited cell for each connected
  /// component. Linear in the number of (cell, facet) incidences, and cached
  /// after the first call.
  ///
  /// A 0-complex is orientable: it has no facets, so the constraint set is
  /// empty and every cell keeps `Pos`. This is the degenerate boundary
  /// answering on the same code path, not a guard.
  pub fn orientation(&self) -> Option<&Orientation> {
    self
      .orientation_cache()
      .get_or_init(|| self.compute_orientation())
      .as_ref()
  }

  /// Whether a coherent orientation exists. The Möbius band and the Klein
  /// bottle are the smallest failures.
  pub fn is_orientable(&self) -> bool {
    self.orientation().is_some()
  }

  /// Validate an externally supplied orientation: one [`Sign`] per cell,
  /// relative to that cell's colex order, as a mesh file's winding gives it.
  ///
  /// `None` unless the candidate is coherent, adjacent cells inducing
  /// opposite orientations on every shared facet, so the witness still means
  /// what it means and a miswound file cannot forge one. This is the only thing
  /// an external source adds over [`Complex::orientation`]: which of the $2^c$
  /// generators is intended, a choice the intrinsic computation cannot make
  /// because an orientation is fixed only up to a global sign per component.
  ///
  /// # Panics
  /// If the candidate does not carry exactly one sign per cell.
  pub fn orient_by(&self, signs: Vec<Sign>) -> Option<Orientation> {
    let dim = self.dim();
    assert_eq!(
      signs.len(),
      self.nsimplices(dim),
      "an orientation candidate needs one sign per cell"
    );
    let induced = |cell: Cell<'_>, facet: SimplexIdx| -> Sign {
      cell
        .get()
        .boundary()
        .find(|(_, sub)| sub.idx() == facet)
        .expect("a facet of a cell appears in its boundary")
        .0
    };
    let facets = self.role_skeleton::<roles::Facet>()?;
    let coherent = facets.handle_iter().all(|facet| {
      let (a, b) = facet.adjacent_cells();
      let Some(b) = b else {
        // A boundary facet has one side and constrains nothing.
        return true;
      };
      let (ka, kb) = (a.idx().kidx, b.idx().kidx);
      signs[ka] * induced(a, facet.idx()) == (signs[kb] * induced(b, facet.idx())).other()
    });
    coherent.then_some(Orientation { signs })
  }

  fn compute_orientation(&self) -> Option<Orientation> {
    let ncells = self.nsimplices(self.dim());
    let mut signs: Vec<Option<Sign>> = vec![None; ncells];

    let cell = |kidx: KSimplexIdx| -> Cell<'_> {
      SimplexIdx::new(self.dim(), kidx)
        .handle(self)
        .role::<roles::Cell>()
    };

    // The sign the cell's stored frame induces on one of its facets: the entry
    // $diff_n [F, K]$ of the boundary operator, which is where the orientation
    // convention already lives.
    let induced = |k: KSimplexIdx, facet: SimplexIdx| -> Sign {
      cell(k)
        .get()
        .boundary()
        .find(|(_, sub)| sub.idx() == facet)
        .expect("a facet of a cell appears in its boundary")
        .0
    };

    for start in 0..ncells {
      if signs[start].is_some() {
        continue;
      }
      signs[start] = Some(Sign::Pos);
      let mut stack = vec![start];
      while let Some(k) = stack.pop() {
        let sigma = signs[k].expect("a cell is signed before it is pushed");
        for facet in cell(k).facets() {
          let (a, b) = facet.adjacent_cells();
          let Some(other) = (if a.idx() == cell(k).idx() { b } else { Some(a) }) else {
            // A boundary facet constrains nothing: it has one side.
            continue;
          };
          let j = other.idx().kidx;
          // Opposite induced orientations on the shared facet:
          // $sigma_j = -sigma_k dot diff[F, K] dot diff[F, J]$.
          let want = (sigma * induced(k, facet.idx()) * induced(j, facet.idx())).other();
          match signs[j] {
            None => {
              signs[j] = Some(want);
              stack.push(j);
            }
            // The walk closed a loop demanding both signs: non-orientable.
            Some(have) if have != want => return None,
            Some(_) => {}
          }
        }
      }
    }

    Some(Orientation {
      signs: signs
        .into_iter()
        .map(|s| s.expect("every cell is reached by its component's walk"))
        .collect(),
    })
  }
}
