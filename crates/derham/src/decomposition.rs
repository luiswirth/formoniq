//! The geometric decomposition of a finite element space:
//!
//!   $P^-_r Lambda^k (K) = plus.circle.big_(f subset.eq K) mono("int")_f$,
//!
//! a direct sum over every subsimplex of the cell, each summand the piece whose
//! support is all of $f$ and no smaller face. A basis function attached to $f$
//! is shared by exactly the cells containing $f$, which is what makes the global
//! space a subspace of $H Lambda^k$ rather than a direct sum of unrelated cell
//! spaces. The local-to-global map is a consequence of the decomposition.
//!
//! See Arnold, Falk and Winther, *Geometric decompositions and local bases for
//! spaces of finite element differential forms*.
//!
//! # Consistent numbering without agreement
//!
//! Two cells sharing a face give that face's degrees of freedom the same global
//! index, with no communication and no orientation convention. A dof attached to
//! $f$ is named by data intrinsic to $f$: a subsimplex of $f$ and a monomial in
//! $f$'s own barycentric coordinates, both in positions within $f$. A
//! [`Skeleton`](simplicial::topology::skeleton::Skeleton) stores every simplex
//! colex-sorted in the global vertex numbering, so both cells see $f$'s vertices
//! in the same order.
//!
//! Nothing here consults a
//! [`CellOrdering`](simplicial::topology::ordering::CellOrdering).

use crate::polynomial::{PolyDegree, PolyForm, whitney};
use multialgebra::ExteriorGrade;
use multiindex::{Combination, Dim, MonoIndex, Repetition, combinations};
use simplicial::topology::{complex::Complex, handle::SimplexIdx};

/// One degree of freedom of the trimmed space $P^-_r Lambda^k$, named
/// intrinsically to the simplex it is attached to.
///
/// The pair $(sigma, alpha)$ of the basis function $lambda^alpha W_sigma$, both
/// in the positions of the attachment simplex rather than of any cell
/// containing it, so two cells meeting at $f$ produce the same pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dof {
  /// The subsimplex carrying the Whitney form, in positions within the
  /// attachment simplex.
  pub blade: Combination,
  /// The monomial's exponents, in positions within the attachment simplex.
  pub monomial: MonoIndex,
}

/// The degrees of freedom whose support is the *whole* of a $d$-simplex: the
/// summand $mono("int")_f$ that $f$ contributes, and nothing any face of $f$
/// already contributes.
///
/// A pair $(sigma, alpha)$ lands here when $sigma union "supp"(alpha)$ is all
/// $d+1$ vertices. The further condition $alpha_i = 0$ for $i$ below the
/// smallest vertex of $sigma$ makes the family independent rather than merely
/// spanning.
pub fn interior_dofs(
  simplex_dim: impl Into<Dim>,
  degree: impl Into<PolyDegree>,
  grade: impl Into<ExteriorGrade>,
) -> Vec<Dof> {
  let (simplex_dim, degree, grade) = (simplex_dim.into(), degree.into(), grade.into());
  if degree.get() < 1 || grade.get() < 0 || simplex_dim.get() < 0 {
    return Vec::new();
  }
  let nvertices = (simplex_dim + 1).index();
  let full = Combination::full(nvertices);

  let mut dofs = Vec::new();
  for blade in combinations(nvertices, grade.index() + 1) {
    let Some(lowest) = blade.iter().next() else {
      continue;
    };
    for monomial in MonoIndex::all(Repetition::Allowed, nvertices, degree.index() - 1) {
      if monomial.iter().any(|symbol| symbol < lowest) {
        continue;
      }
      // Together they touch every vertex, so no proper face carries it.
      let support = monomial.iter().fold(blade, |support, symbol| {
        if support.contains(symbol) {
          support
        } else {
          support.inserted(symbol)
        }
      });
      if support == full {
        dofs.push(Dof { blade, monomial });
      }
    }
  }
  dofs
}

/// The geometric decomposition of $P^-_r Lambda^k$ on cells of a given
/// dimension.
///
/// A property of the dimension, degree and grade, never of a particular cell,
/// so it is built once and reused on every cell.
#[derive(Debug, Clone)]
pub struct GeometricDecomposition {
  cell_dim: Dim,
  degree: PolyDegree,
  grade: ExteriorGrade,
  /// The local basis, in the order it is emitted: for each subsimplex of the
  /// cell, ascending in dimension then in colex, each of its interior dofs.
  local: Vec<LocalDof>,
  /// How many dofs a subsimplex of each dimension carries, indexed by
  /// dimension.
  per_dim: Vec<usize>,
}

/// A local basis function's place in the decomposition.
#[derive(Debug, Clone)]
pub struct LocalDof {
  /// The subsimplex of the cell it attaches to, in the cell's vertex positions.
  pub attachment: Combination,
  /// Its index among that simplex's own degrees of freedom.
  pub index_within: usize,
  /// The dof, named intrinsically to the attachment simplex.
  pub dof: Dof,
}

impl GeometricDecomposition {
  pub fn new(
    cell_dim: impl Into<Dim>,
    degree: impl Into<PolyDegree>,
    grade: impl Into<ExteriorGrade>,
  ) -> Self {
    let (cell_dim, degree, grade) = (cell_dim.into(), degree.into(), grade.into());
    let nvertices = (cell_dim + 1).index();

    let per_dim: Vec<usize> = (0..nvertices)
      .map(|d| interior_dofs(d, degree, grade).len())
      .collect();

    let mut local = Vec::new();
    for subsimplex_dim in 0..nvertices {
      let dofs = interior_dofs(subsimplex_dim, degree, grade);
      for attachment in combinations(nvertices, subsimplex_dim + 1) {
        for (index_within, dof) in dofs.iter().enumerate() {
          local.push(LocalDof {
            attachment,
            index_within,
            dof: dof.clone(),
          });
        }
      }
    }

    Self {
      cell_dim,
      degree,
      grade,
      local,
      per_dim,
    }
  }

  pub fn cell_dim(&self) -> Dim {
    self.cell_dim
  }
  pub fn degree(&self) -> PolyDegree {
    self.degree
  }
  pub fn grade(&self) -> ExteriorGrade {
    self.grade
  }
  /// The local basis, in the order the element matrices index it.
  pub fn local(&self) -> &[LocalDof] {
    &self.local
  }
  /// The number of degrees of freedom on one cell.
  pub fn ndofs_local(&self) -> usize {
    self.local.len()
  }
  /// How many degrees of freedom a subsimplex of the given dimension carries.
  pub fn dofs_per_simplex(&self, simplex_dim: impl Into<Dim>) -> usize {
    let simplex_dim = simplex_dim.into();
    simplex_dim
      .index_in(self.cell_dim)
      .and_then(|d| self.per_dim.get(d).copied())
      .unwrap_or(0)
  }

  /// The basis function of a local degree of freedom, on the reference cell.
  ///
  /// The name $(sigma, alpha)$ lifted from the attachment simplex's positions
  /// into the cell's.
  pub fn basis_function(&self, local: &LocalDof) -> PolyForm {
    let blade = local.attachment.select(local.dof.blade);
    let monomial = MonoIndex::new(
      Repetition::Allowed,
      local
        .dof
        .monomial
        .iter()
        .map(|symbol| local.attachment.index_at(symbol)),
    );

    let form = whitney(self.cell_dim, blade);
    let coefficient = PolyForm::monomial(self.cell_dim, monomial, Combination::empty());
    PolyForm::from_tensor(coefficient.tensor().product(form.tensor()))
  }

  /// The whole local basis of the reference cell, paired with the subsimplex
  /// each function is attached to.
  pub fn local_basis(&self) -> Vec<(Combination, PolyForm)> {
    self
      .local
      .iter()
      .map(|local| (local.attachment, self.basis_function(local)))
      .collect()
  }

  /// The exterior derivative on the reference cell, from this space's basis to
  /// that of the next grade at the same polynomial degree.
  ///
  /// Well defined because the trimmed complex closes: $dif$ maps
  /// $P^-_r Lambda^k$ into $P_(r-1) Lambda^(k+1) subset P^-_r Lambda^(k+1)$.
  ///
  /// Metric-free, exact, and the same for every cell, so computed once and
  /// scattered.
  ///
  /// Obtained by expanding $dif u_i$ in the target basis in the faithful
  /// representation, where equality of components is equality of forms. The
  /// degrees are matched first by [`PolyForm::raise`], $dif$ dropping the
  /// polynomial degree by one.
  ///
  /// # Panics
  /// If the target is not this space at one grade higher.
  pub fn dif_matrix(&self, target: &Self) -> multialgebra::Matrix {
    assert_eq!(
      target.degree, self.degree,
      "dif keeps the polynomial degree"
    );
    assert_eq!(target.grade, self.grade + 1, "dif raises the grade by one");
    assert_eq!(target.cell_dim, self.cell_dim);

    let columns = |basis: &[(multiindex::Combination, PolyForm)]| {
      let components: Vec<multialgebra::Vector> = basis
        .iter()
        .map(|(_, form)| form.faithful_components())
        .collect();
      let nrows = components.first().map_or(0, |c| c.len());
      multialgebra::Matrix::from_fn(nrows, components.len(), |i, j| components[j][i])
    };

    let target_basis = target.local_basis();
    let image: Vec<(multiindex::Combination, PolyForm)> = self
      .local_basis()
      .into_iter()
      .map(|(attachment, form)| (attachment, form.dif().raise()))
      .collect();

    let target_columns = columns(&target_basis);
    let image_columns = columns(&image);
    if target_columns.ncols() == 0 || image_columns.ncols() == 0 {
      return multialgebra::Matrix::zeros(target_columns.ncols(), image_columns.ncols());
    }
    target_columns
      .svd(true, true)
      .solve(&image_columns, 1e-9)
      .expect("the trimmed complex closes, so the image lies in the target span")
  }

  /// The number of global degrees of freedom on a complex: the decomposition
  /// summed over every simplex of it.
  pub fn ndofs(&self, complex: &Complex) -> usize {
    self
      .cell_dim
      .range_inclusive()
      .map(|d| complex.nsimplices(d) * self.dofs_per_simplex(d))
      .sum()
  }

  /// Where a simplex dimension's block starts in the global numbering.
  ///
  /// Dofs are grouped by the dimension of the simplex they attach to, then by
  /// that simplex's index in its skeleton, then by their index within it. Only
  /// the innermost index is forced, having to be intrinsic to the simplex.
  pub fn block_offset(&self, complex: &Complex, simplex_dim: Dim) -> usize {
    simplex_dim
      .range()
      .map(|d| complex.nsimplices(d) * self.dofs_per_simplex(d))
      .sum()
  }

  /// The global index of each local degree of freedom of a cell, in the order
  /// of [`Self::local`]: the local-to-global map assembly scatters through.
  pub fn local_to_global(&self, complex: &Complex, cell: SimplexIdx) -> Vec<usize> {
    let cell_simplex = complex.skeleton_raw(cell.dim).simplex_by_kidx(cell.kidx);
    let cell_vertices = cell_simplex.vertices.to_vec();

    self
      .local
      .iter()
      .map(|local| {
        // The attachment simplex in global vertices, what both cells agree on.
        let vertices: Vec<usize> = local
          .attachment
          .iter()
          .map(|position| cell_vertices[position])
          .collect();
        // From the vertex list directly: a `Combination` is a 64-bit bitset
        // over the indices it holds, so routing global vertex numbers through
        // one would cap the mesh at 64 vertices.
        let simplex_dim = Dim::from(vertices.len() - 1);
        let simplex = simplicial::topology::simplex::Simplex::new(vertices);
        let kidx = complex.skeleton_raw(simplex_dim).kidx_by_simplex(&simplex);

        self.block_offset(complex, simplex_dim)
          + kidx * self.dofs_per_simplex(simplex_dim)
          + local.index_within
      })
      .collect()
  }
}

/// The local-to-global map of a finite element space, materialized once.
///
/// The map assembly scatters a local matrix through. Materializing it is what
/// lets one assembly routine serve every polynomial degree; the first-order
/// case, a dof per $k$-simplex, is this map and not a different mechanism.
#[derive(Debug, Clone)]
pub struct CellDofs {
  ndofs: usize,
  ndofs_local: usize,
  /// Cell-major, `ndofs_local` entries each.
  indices: Vec<usize>,
}

impl CellDofs {
  pub fn new(decomposition: &GeometricDecomposition, complex: &Complex) -> Self {
    let cell_dim = decomposition.cell_dim();
    let ncells = complex.nsimplices(cell_dim);
    let ndofs_local = decomposition.ndofs_local();

    let mut indices = Vec::with_capacity(ncells * ndofs_local);
    for kidx in 0..ncells {
      let cell = SimplexIdx {
        dim: cell_dim,
        kidx,
      };
      indices.extend(decomposition.local_to_global(complex, cell));
    }

    Self {
      ndofs: decomposition.ndofs(complex),
      ndofs_local,
      indices,
    }
  }

  pub fn ndofs(&self) -> usize {
    self.ndofs
  }
  pub fn ndofs_local(&self) -> usize {
    self.ndofs_local
  }
  /// The global indices of one cell's degrees of freedom, in local order.
  pub fn cell(&self, kidx: usize) -> &[usize] {
    &self.indices[kidx * self.ndofs_local..(kidx + 1) * self.ndofs_local]
  }
  pub fn ncells(&self) -> usize {
    self
      .indices
      .len()
      .checked_div(self.ndofs_local)
      .unwrap_or(0)
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::polynomial::trimmed_dim;

  /// The decomposition accounts for the space exactly: summing each
  /// subsimplex's own degrees of freedom gives $dim P^-_r Lambda^k$.
  ///
  /// The direct sum being direct. Double-counting or dropping a face still
  /// produces a usable-looking local basis, and only this count catches it.
  #[test]
  fn the_decomposition_accounts_for_the_whole_space() {
    for n in 1..=4 {
      for r in 1..=3 {
        for k in 0..=n {
          let decomposition = GeometricDecomposition::new(n, r, k);
          let total: usize = (0..=n)
            .map(|d| multiindex::binomial(n + 1, d + 1) * decomposition.dofs_per_simplex(d))
            .sum();
          assert_eq!(total, trimmed_dim(n, r, k), "n={n} r={r} k={k}");
          assert_eq!(decomposition.ndofs_local(), trimmed_dim(n, r, k));
        }
      }
    }
  }

  /// A degree of freedom lives on the simplex it says it does, and on no
  /// smaller one: its support touches every vertex of the attachment.
  ///
  /// What conforming assembly rests on. Were a function attached to a face
  /// supported on less, identifying it across cells would glue together things
  /// that are not equal.
  #[test]
  fn a_dof_is_supported_on_its_whole_attachment() {
    for d in 0..=3 {
      for r in 1..=3 {
        for k in 0..=d {
          for dof in interior_dofs(d, r, k) {
            let support = dof.monomial.iter().fold(dof.blade, |support, symbol| {
              if support.contains(symbol) {
                support
              } else {
                support.inserted(symbol)
              }
            });
            assert_eq!(support, Combination::full(d + 1));
          }
        }
      }
    }
  }

  /// At $r = 1$ the decomposition is the Whitney one: exactly one degree of
  /// freedom on each $k$-simplex and none anywhere else.
  #[test]
  fn the_first_order_decomposition_is_the_whitney_one() {
    for n in 1..=4 {
      for k in 0..=n {
        let decomposition = GeometricDecomposition::new(n, 1, k);
        for d in 0..=n {
          let expected = usize::from(d == k);
          assert_eq!(
            decomposition.dofs_per_simplex(d),
            expected,
            "n={n} k={k} d={d}"
          );
        }
      }
    }
  }

  /// The local basis the decomposition builds spans the same space as the
  /// basis built directly, and is independent.
  #[test]
  fn the_decomposed_basis_is_a_basis() {
    for n in 1..=3 {
      for r in 1..=3 {
        for k in 0..=n {
          let decomposition = GeometricDecomposition::new(n, r, k);
          let basis = decomposition.local_basis();
          assert_eq!(basis.len(), trimmed_dim(n, r, k));

          let ncomponents = basis[0].1.tensor().components().len();
          let matrix = multialgebra::Matrix::from_fn(ncomponents, basis.len(), |i, j| {
            basis[j].1.tensor().components()[i]
          });
          assert_eq!(
            matrix.rank(1e-9),
            basis.len(),
            "the decomposed basis is dependent at n={n} r={r} k={k}"
          );
        }
      }
    }
  }

  /// The reference exterior derivative reproduces $dif$ pointwise: expanding
  /// $dif u_i$ in the target basis and evaluating agrees with evaluating
  /// $dif u_i$ directly.
  ///
  /// The matrix comes from a linear solve, so this says the solve found the
  /// expansion and not merely a small residual.
  #[test]
  fn the_reference_dif_expands_the_derivative() {
    for n in 1..=3 {
      for r in 1..=3 {
        for k in 0..n {
          let source = GeometricDecomposition::new(n, r, k);
          let target = GeometricDecomposition::new(n, r, k + 1);
          let matrix = source.dif_matrix(&target);
          assert_eq!(matrix.nrows(), target.ndofs_local());
          assert_eq!(matrix.ncols(), source.ndofs_local());

          let source_basis = source.local_basis();
          let target_basis = target.local_basis();
          for (column, (_, form)) in source_basis.iter().enumerate() {
            let derivative = form.dif();
            for point in probe_points(n) {
              let direct = derivative.at_bary(&point);
              let expanded: multialgebra::Vector = (0..target_basis.len())
                .map(|row| matrix[(row, column)] * target_basis[row].1.at_bary(&point).components())
                .fold(
                  multialgebra::Vector::zeros(direct.components().len()),
                  |acc, v| acc + v,
                );
              approx::assert_relative_eq!(direct.components(), &expanded, epsilon = 1e-9);
            }
          }
        }
      }
    }
  }

  /// $dif compose dif = 0$ on the reference cell, at every degree.
  #[test]
  fn the_reference_dif_squares_to_zero() {
    for n in 2..=3 {
      for r in 1..=3 {
        for k in 0..(n - 1) {
          let a = GeometricDecomposition::new(n, r, k);
          let b = GeometricDecomposition::new(n, r, k + 1);
          let c = GeometricDecomposition::new(n, r, k + 2);
          let composed = b.dif_matrix(&c) * a.dif_matrix(&b);
          approx::assert_relative_eq!(composed.amax(), 0.0, epsilon = 1e-9);
        }
      }
    }
  }

  /// At $r = 1$ the reference exterior derivative is the simplicial coboundary:
  /// the incidence matrix of the cell's own faces, entries in ${-1, 0, 1\}$.
  ///
  /// Ties the polynomial construction to the combinatorial one, fixing the signs
  /// as well as the pattern.
  #[test]
  fn the_first_order_reference_dif_is_the_coboundary() {
    for n in 1..=3 {
      for k in 0..n {
        let source = GeometricDecomposition::new(n, 1, k);
        let target = GeometricDecomposition::new(n, 1, k + 1);
        let matrix = source.dif_matrix(&target);
        let expected =
          simplicial::topology::simplex::unit_boundary_operator(Dim::from(n), Dim::from(k + 1))
            .transpose();
        approx::assert_relative_eq!(matrix, expected, epsilon = 1e-9);
      }
    }
  }

  fn probe_points(cell_dim: usize) -> Vec<simplicial::atlas::Bary> {
    (0..3)
      .map(|seed| {
        let weights: Vec<f64> = (0..=cell_dim)
          .map(|i| 1.0 + ((3 * i + 2 * seed) % 5) as f64)
          .collect();
        let total: f64 = weights.iter().sum();
        simplicial::atlas::Bary::new(multialgebra::Vector::from_vec(
          weights.into_iter().map(|w| w / total).collect(),
        ))
      })
      .collect()
  }

  /// Two cells sharing a face give that face's degrees of freedom the same
  /// global indices.
  ///
  /// The conformity condition. It holds without the cells agreeing on anything,
  /// a dof being named intrinsically to its attachment.
  #[test]
  fn cells_sharing_a_face_agree_on_its_dofs() {
    use simplicial::mesher::cartesian::CartesianGrid;

    for n in 1..=3 {
      let (topology, _) = CartesianGrid::new_unit(Dim::from(n), 2).triangulate();
      for r in 1..=3 {
        for k in 0..=n {
          let decomposition = GeometricDecomposition::new(n, r, k);

          // Global index -> the (attachment vertices, dof) it stands for.
          let mut seen: std::collections::HashMap<usize, (Vec<usize>, Dof)> =
            std::collections::HashMap::new();
          for (kidx, cell) in topology.skeleton_raw(Dim::from(n)).iter().enumerate() {
            let cell_idx = SimplexIdx {
              dim: Dim::from(n),
              kidx,
            };
            let cell_vertices = cell.vertices.to_vec();
            let globals = decomposition.local_to_global(&topology, cell_idx);

            for (local, global) in decomposition.local().iter().zip(globals) {
              let vertices: Vec<usize> = local
                .attachment
                .iter()
                .map(|position| cell_vertices[position])
                .collect();
              let entry = (vertices, local.dof.clone());
              if let Some(previous) = seen.get(&global) {
                assert_eq!(
                  previous, &entry,
                  "global dof {global} means two different things at n={n} r={r} k={k}"
                );
              } else {
                seen.insert(global, entry);
              }
            }
          }

          // Every global index is used, and none is out of range.
          assert_eq!(
            seen.len(),
            decomposition.ndofs(&topology),
            "n={n} r={r} k={k}"
          );
          assert!(seen.keys().all(|&g| g < decomposition.ndofs(&topology)));
        }
      }
    }
  }
}
