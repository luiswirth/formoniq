//! Uniform refinement of the simplicial complex.
//!
//! Refinement is where a single mesh becomes a nested family of meshes, and it
//! respects the topology/geometry split: this module produces only the refined
//! *topology* and the affine, metric-free record of how each new cell sits
//! inside a coarse one (a [`Child`]). Transporting a geometry across that record
//! -- pulling each coarse cell's metric back onto its children, or placing the
//! new vertices of an embedding -- is the geometry layer's job (see
//! `geometry::refine`), keyed off the same [`Subdivision`].
//!
//! Every cell is subdivided by the one reference pattern
//! ([`ReferenceRefinement`](crate::atlas::refine::ReferenceRefinement)),
//! relabelled onto the cell's vertices. New vertices
//! shared between cells are identified by the coarse simplex they are supported
//! on together with their barycentric weights there -- a key both incident
//! charts compute identically -- so the refined mesh closes up conformingly, and
//! [`Complex::from_cells`] rederives every skeleton and boundary operator from
//! the new cells, its manifold check standing as the conformity assertion.

use super::{
  VertexIdx, complex::Complex, data::SkeletonVec, handle::KSimplexIdx, ordering::CellOrdering,
  simplex::Simplex, skeleton::Skeleton,
};
use crate::{
  atlas::{LocalCartesian, SimplexCoords, ref_refinement},
  linalg::{Matrix, Vector},
};

use std::collections::HashMap;

/// A refinement of a [`Complex`]: the refined topology together with the affine
/// provenance every geometry needs to follow it.
#[derive(Debug, Clone)]
pub struct Subdivision {
  complex: Complex,
  refinement: usize,
  /// Per refined cell (in the refined complex's colex order): its coarse parent
  /// and the affine map into that parent's chart.
  children: SkeletonVec<Child>,
  /// The number of coarse vertices, which keep their labels in the refined
  /// complex: the refined vertices $0..\"ncoarse\"$ are the coarse ones.
  ncoarse_vertices: usize,
  /// Per *new* refined vertex (labels $"ncoarse"..$): a coarse cell it lies in
  /// and its barycentric coordinates there, enough to place it from any
  /// coarse-cell data. A coarse vertex needs no birth -- it maps to itself.
  new_births: Vec<VertexBirth>,
  /// The refined complex's own vertex ordering: each child in the order the
  /// reference pattern emitted its corners. Refining the next level in *this*
  /// ordering is what makes refinement compose.
  ordering: CellOrdering,
}

/// The provenance of a refined cell: which coarse cell it subdivides, and its
/// affine embedding into that cell's chart.
#[derive(Debug, Clone)]
pub struct Child {
  /// The coarse cell (by kidx) this child subdivides.
  pub parent: KSimplexIdx,
  /// The Jacobian of the child's affine embedding into the parent chart's local
  /// frame, its columns ordered by the child's sorted global vertices -- the
  /// same basis the refined cell's own metric is read in. Metric-free, and the
  /// map a geometry is transported (pulled back) along.
  pub jacobian: Matrix,
}

/// The birth of a new refined vertex, as an affine combination of coarse
/// vertices: pairs of a coarse vertex (global index) and its weight, the weights
/// summing to one. Metric-free and chart-consistent -- the combination is the
/// same read from any incident coarse cell -- so it places the new vertex in
/// *any* coarse-vertex data (an embedding, a field) by a weighted sum.
#[derive(Debug, Clone)]
pub struct VertexBirth {
  pub combination: Vec<(VertexIdx, f64)>,
}

impl Complex {
  /// The uniform $R$-fold refinement of the mesh: every cell edgewise-subdivided
  /// into $R^n$ children by the one reference pattern. $R = 1$ is the identity.
  ///
  /// Purely topological and affine; carries no geometry. The returned
  /// [`Subdivision`] holds the refined [`Complex`] and the provenance a geometry
  /// is transported along.
  ///
  /// Refines in the colex ordering, the one a mesh carries implicitly. Prefer
  /// [`Complex::refine_with`] when an ordering is available: composing *this*
  /// with itself does not give the $R R'$-fold refinement above dimension two,
  /// because each level re-derives a child's order by sorting rather than
  /// inheriting it.
  pub fn refine(&self, refinement: usize) -> Subdivision {
    self.refine_with(&CellOrdering::colex(self), refinement)
  }

  /// The uniform $R$-fold refinement, subdividing each cell in the order the
  /// given [`CellOrdering`] gives its vertices.
  ///
  /// The ordering is what a refinement tower needs: with it, refinement
  /// composes, $"refine"_(R') compose "refine"_R = "refine"_(R R')$, so a tower
  /// stays inside the family its generator produced and every child is similar
  /// to its parent. The refined complex's own ordering rides on the returned
  /// [`Subdivision`] ([`Subdivision::ordering`]), which is what makes the next
  /// level possible.
  ///
  /// # Panics
  /// If the ordering is not an ordering of this complex's cells. It must also
  /// be face-consistent ([`CellOrdering::is_face_consistent`]) -- two cells that
  /// order a shared face differently subdivide it differently, and the result
  /// is non-conforming; that is checked by `from_cells` when the refined
  /// complex is built, not assumed here.
  pub fn refine_with(&self, ordering: &CellOrdering, refinement: usize) -> Subdivision {
    assert_eq!(
      ordering.ncells(),
      self.cells().len(),
      "an ordering must cover exactly this complex's cells"
    );
    let dim = self.dim();
    let pattern = ref_refinement(dim, refinement);
    let ncoarse_vertices = self.vertices().len();

    // Global labels for the new (non-coarse) vertices, keyed by the coarse
    // simplex a lattice point is supported on and its weights there -- the key
    // both incident charts agree on.
    let mut new_index: HashMap<Vec<(VertexIdx, usize)>, usize> = HashMap::new();
    let mut new_births: Vec<VertexBirth> = Vec::new();

    // The refined cells, and the provenance of each in construction order;
    // reindexed to the final colex order once the complex is built.
    let mut cells: Vec<Simplex> = Vec::new();
    let mut provenance: Vec<(Simplex, Child)> = Vec::new();
    // Each child's corners in the pattern's own order: the refined mesh's
    // ordering, and the whole of what a child inherits from its parent.
    let mut words: Vec<Vec<VertexIdx>> = Vec::new();

    for cell in self.cells().handle_iter() {
      let cverts = ordering.word(cell);

      let resolve = |pv: usize,
                     new_index: &mut HashMap<Vec<(VertexIdx, usize)>, usize>,
                     new_births: &mut Vec<VertexBirth>|
       -> VertexIdx {
        let k = &pattern.vertices()[pv];
        // Sorted by vertex, so the key is canonical: the ordering permutes
        // `cverts`, and two cells sharing this lattice point must still agree
        // on its label or the refined mesh tears along their common face.
        let mut support: Vec<(VertexIdx, usize)> = k
          .iter()
          .enumerate()
          .filter(|&(_, &w)| w > 0)
          .map(|(i, &w)| (cverts[i], w))
          .collect();
        support.sort_unstable();
        // Supported on a single coarse vertex: it *is* that vertex.
        if support.len() == 1 {
          return support[0].0;
        }
        let scale = (refinement as f64).recip();
        let combination = support
          .iter()
          .map(|&(v, w)| (v, w as f64 * scale))
          .collect();
        *new_index.entry(support).or_insert_with(|| {
          let idx = ncoarse_vertices + new_births.len();
          new_births.push(VertexBirth { combination });
          idx
        })
      };

      // Where each ordered parent vertex sits in the stored (sorted) cell: the
      // permutation the ordering is, and what re-expresses a lattice point in
      // the frame the parent's metric is read in.
      let sorted = &cell.simplex().vertices;
      let slot: Vec<usize> = cverts
        .iter()
        .map(|v| sorted.binary_search(v).expect("a word permutes its cell"))
        .collect();

      for child in pattern.children() {
        // Each corner as (global vertex, its parent-local coordinate). The
        // stored cell sorts its vertices globally, and the induced metric reads
        // that sorted order as its basis, so the child's realization in the
        // parent frame must too -- hence the sort, and hence the lattice weights
        // are read through `slot` rather than through the pattern's own frame,
        // whose axes follow the *ordering*. The two agree only when the ordering
        // is the colex one. This is why the Jacobian is per-cell and not pure
        // reference data.
        let mut corners: Vec<(VertexIdx, Vector)> = child
          .iter()
          .map(|&pv| {
            let global = resolve(pv, &mut new_index, &mut new_births);
            let weights = &pattern.vertices()[pv];
            let mut local = Vector::zeros(dim);
            let scale = (refinement as f64).recip();
            for (i, &weight) in weights.iter().enumerate() {
              // Barycentric weight on sorted vertex `slot[i]`; the reference
              // frame places vertex 0 at the origin and vertex j at e_(j-1).
              if slot[i] > 0 {
                local[slot[i] - 1] += weight as f64 * scale;
              }
            }
            (global, local)
          })
          .collect();
        words.push(corners.iter().map(|&(global, _)| global).collect());
        corners.sort_by_key(|&(global, _)| global);

        // The child realized in the parent chart's local frame, in the stored
        // (sorted) vertex order; its linear part is the affine map the parent's
        // metric is pulled back along. The degenerate 0-cell is the empty map,
        // handled by the realization itself -- no special case here.
        let local: Vec<Vector> = corners.iter().map(|(_, x)| x.clone()).collect();
        let jacobian =
          SimplexCoords::<LocalCartesian>::new(Matrix::from_columns(&local)).linear_transform();

        let simplex = Simplex::new(corners.iter().map(|&(global, _)| global).collect());
        cells.push(simplex.clone());
        provenance.push((
          simplex,
          Child {
            parent: cell.kidx(),
            jacobian,
          },
        ));
      }
    }

    let complex = Complex::from_cells(Skeleton::new(cells));

    // Reindex the provenance to the refined complex's colex cell order.
    let cell_skeleton = complex.skeleton_raw(dim);
    let mut children: Vec<Option<Child>> = (0..complex.nsimplices(dim)).map(|_| None).collect();
    for (simplex, child) in provenance {
      let kidx = cell_skeleton.kidx_by_simplex(&simplex);
      children[kidx] = Some(child);
    }
    let children = SkeletonVec::new(
      dim,
      children
        .into_iter()
        .map(|c| c.expect("every refined cell has a provenance"))
        .collect(),
    );

    // Matched by vertex set, which also validates that the words name exactly
    // the refined cells.
    let ordering = CellOrdering::new(&complex, words);

    Subdivision {
      complex,
      refinement,
      children,
      ncoarse_vertices,
      new_births,
      ordering,
    }
  }
}

impl Subdivision {
  pub fn complex(&self) -> &Complex {
    &self.complex
  }
  pub fn into_complex(self) -> Complex {
    self.complex
  }
  pub fn refinement(&self) -> usize {
    self.refinement
  }
  /// The refined complex's vertex ordering, inherited from the pattern.
  ///
  /// Feed it to [`Complex::refine_with`] to take the next level: that is what
  /// makes a tower agree with a single refinement of the product refinement.
  pub fn ordering(&self) -> &CellOrdering {
    &self.ordering
  }
  /// The provenance of each refined cell, keyed by refined-cell kidx.
  pub fn children(&self) -> &SkeletonVec<Child> {
    &self.children
  }
  pub fn ncoarse_vertices(&self) -> usize {
    self.ncoarse_vertices
  }
  pub fn nvertices(&self) -> usize {
    self.ncoarse_vertices + self.new_births.len()
  }
  /// The births of the new refined vertices (labels $"ncoarse"..$).
  pub fn new_births(&self) -> &[VertexBirth] {
    &self.new_births
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::mesher::cartesian::CartesianGrid;
  use crate::topology::data::SkeletonData;

  /// Refinement counts match the reference pattern, and the refined complex is
  /// a valid manifold complex (its `from_cells` manifold check is the conformity
  /// assertion). Vertices grow by the number of new lattice points per cell,
  /// deduplicated across shared faces.
  #[test]
  fn refine_counts_and_conformity() {
    for dim in 1..=3 {
      let (coarse, _) = CartesianGrid::new_unit(dim, 2).triangulate();
      for r in 1..=3 {
        let sub = coarse.refine(r);
        let fine = sub.complex();

        // R^n children per coarse cell.
        assert_eq!(
          fine.nsimplices(dim),
          coarse.nsimplices(dim) * r.pow(dim as u32)
        );
        // Coarse vertices keep their labels; refinement only adds vertices.
        assert!(sub.nvertices() >= coarse.vertices().len());
        assert_eq!(fine.vertices().len(), sub.nvertices());
        // The child provenance covers every refined cell.
        assert_eq!(sub.children().len(), fine.nsimplices(dim));
        // Boundary of the boundary vanishes: a valid chain complex was built.
        for k in 1..dim {
          use crate::linalg::CsrMatrix;
          let d0 = CsrMatrix::from(&fine.coboundary_operator(k - 1));
          let d1 = CsrMatrix::from(&fine.coboundary_operator(k));
          assert!((d1 * d0).values().iter().all(|&v| v == 0.0));
        }
      }
    }
  }

  /// A single triangle refines (red) into 4 triangles over 6 vertices; a single
  /// tetrahedron into 8 over 10. The classical Bank/Bey counts.
  #[test]
  fn red_refinement_classical_counts() {
    let tri = Complex::standard(2).refine(2);
    assert_eq!(tri.complex().nsimplices(2), 4);
    assert_eq!(tri.nvertices(), 6);

    let tet = Complex::standard(3).refine(2);
    assert_eq!(tet.complex().nsimplices(3), 8);
    assert_eq!(tet.nvertices(), 10);
  }

  /// Totality: a 0-complex refines to itself, and $R = 1$ is the identity on
  /// the cell count for every dimension.
  #[test]
  fn refine_degenerate_and_identity() {
    let points = Complex::standard(0).refine(3);
    assert_eq!(points.complex().nsimplices(0), 1);
    assert_eq!(points.nvertices(), 1);

    for dim in 0..=3 {
      let (coarse, _) = CartesianGrid::new_unit(dim, 2).triangulate();
      let identity = coarse.refine(1);
      assert_eq!(identity.complex().nsimplices(dim), coarse.nsimplices(dim));
      assert_eq!(identity.nvertices(), coarse.vertices().len());
    }
  }
}
