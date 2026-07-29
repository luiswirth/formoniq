//! The trimmed finite element complex $P^-_r Lambda^k$, at any polynomial
//! degree.
//!
//! [`WhitneyComplex`](crate::whitney_complex) is its $r = 1$ case, not a
//! different construction: the same geometric decomposition, assembly and
//! exterior derivative at a degree of one.
//!
//! It implements the same [`HilbertComplex`], so every problem written against
//! that interface runs at any degree unchanged.
//!
//! The geometry is the intrinsic Regge primitive of any signature; nothing here
//! assumes an embedding or a definite metric.

use crate::{
  assemble::{GalMat, assemble_galmat_dofs, assemble_galvec_dofs, scatter_local_operator},
  operators::{CellQuadrature, ElVecProvider, SourceElVec, TrimmedMassElmat},
  whitney_complex::HilbertComplex,
};

use derham::{
  decomposition::{CellDofs, GeometricDecomposition},
  interpolate::samples::LsfSamples,
  section::Section,
};
use multialgebra::ExteriorGrade;
use multiindex::Degree;
use simplicial::{
  Dim,
  atlas::SimplexQuadRule,
  geometry::metric::mesh::MeshLengthsSq,
  linalg::{CooMatrix, CsrMatrix, Vector},
  topology::complex::Complex,
};

use std::collections::HashSet;

/// The polynomial degree of a trimmed space: $r$ in $P^-_r Lambda^k$.
pub type PolyDegree = Degree;

/// The discrete Hilbert complex of trimmed polynomial forms,
///
/// $P^-_r Lambda^0 -> P^-_r Lambda^1 -> dots.c -> P^-_r Lambda^n$,
///
/// with the $L^2 Lambda^k$ inner products.
///
/// The exterior derivative is exact and metric-free at every degree; the metric
/// enters only through the mass matrices.
///
/// Degrees of freedom are numbered by the [`GeometricDecomposition`]. The
/// per-grade decompositions and dof maps are built once, a solve assembling
/// several matrices over the same space.
pub struct TrimmedComplex<'a> {
  topology: &'a Complex,
  geometry: &'a MeshLengthsSq,
  degree: PolyDegree,
  /// Per grade $0 <= k <= n$.
  decompositions: Vec<GeometricDecomposition>,
  dofs: Vec<CellDofs>,
  /// Which degrees of freedom are constrained to vanish, per grade: empty for
  /// the absolute (natural) boundary condition.
  constrained: Vec<HashSet<usize>>,
}

impl<'a> TrimmedComplex<'a> {
  /// The full complex, with the absolute (natural) boundary condition.
  pub fn new(
    topology: &'a Complex,
    geometry: &'a MeshLengthsSq,
    degree: impl Into<PolyDegree>,
  ) -> Self {
    let degree = degree.into();
    let dim = topology.dim();
    let decompositions: Vec<_> = dim
      .range_inclusive()
      .map(|grade| GeometricDecomposition::new(dim, degree, grade))
      .collect();
    let dofs = decompositions
      .iter()
      .map(|decomposition| CellDofs::new(decomposition, topology))
      .collect();
    let constrained = dim.range_inclusive().map(|_| HashSet::new()).collect();

    Self {
      topology,
      geometry,
      degree,
      decompositions,
      dofs,
      constrained,
    }
  }

  /// The relative complex of the pair $(K, diff K)$: the essential
  /// (homogeneous Dirichlet) boundary condition.
  ///
  /// A degree of freedom is constrained exactly when the simplex it is attached
  /// to lies on the boundary. The trace of a basis function onto a face
  /// vanishes unless the function is attached to that face or a subsimplex of
  /// it, so this drops precisely the functions with nonzero boundary trace.
  pub fn relative(mut self) -> Self {
    let boundary: Vec<HashSet<usize>> = self
      .topology
      .dim()
      .range_inclusive()
      .map(|grade| {
        self
          .topology
          .boundary_simplices(grade)
          .into_iter()
          .map(|idx| idx.kidx)
          .collect()
      })
      .collect();

    self.constrained = self
      .topology
      .dim()
      .range_inclusive()
      .map(|grade| self.constrained_dofs(grade, &boundary))
      .collect();
    self
  }

  /// The global dofs attached to a constrained simplex, at one grade.
  fn constrained_dofs(&self, grade: ExteriorGrade, boundary: &[HashSet<usize>]) -> HashSet<usize> {
    let Some(index) = grade.index_in(self.dim()) else {
      return HashSet::new();
    };
    let decomposition = &self.decompositions[index];
    let mut constrained = HashSet::new();
    for attachment_dim in self.dim().range_inclusive() {
      let per = decomposition.dofs_per_simplex(attachment_dim);
      if per == 0 {
        continue;
      }
      let offset = decomposition.block_offset(self.topology, attachment_dim);
      for &kidx in &boundary[attachment_dim.index()] {
        for within in 0..per {
          constrained.insert(offset + kidx * per + within);
        }
      }
    }
    constrained
  }

  pub fn dim(&self) -> Dim {
    self.topology.dim()
  }
  pub fn topology(&self) -> &'a Complex {
    self.topology
  }
  pub fn geometry(&self) -> &'a MeshLengthsSq {
    self.geometry
  }
  pub fn poly_degree(&self) -> PolyDegree {
    self.degree
  }

  fn decomposition(&self, grade: ExteriorGrade) -> Option<&GeometricDecomposition> {
    grade
      .index_in(self.dim())
      .map(|index| &self.decompositions[index])
  }
  fn cell_dofs(&self, grade: ExteriorGrade) -> Option<&CellDofs> {
    grade.index_in(self.dim()).map(|index| &self.dofs[index])
  }

  /// The number of degrees of freedom before constraints.
  fn ndofs_full(&self, grade: ExteriorGrade) -> usize {
    self.cell_dofs(grade).map_or(0, CellDofs::ndofs)
  }

  /// The unconstrained degrees of freedom, ascending: the rows of the
  /// inclusion.
  fn free_dofs(&self, grade: ExteriorGrade) -> Vec<usize> {
    let Some(index) = grade.index_in(self.dim()) else {
      return Vec::new();
    };
    (0..self.ndofs_full(grade))
      .filter(|dof| !self.constrained[index].contains(dof))
      .collect()
  }

  /// The mass matrix before constraints.
  fn mass_full(&self, grade: ExteriorGrade) -> GalMat {
    let (Some(decomposition), Some(dofs)) = (self.decomposition(grade), self.cell_dofs(grade))
    else {
      return GalMat::new(0, 0);
    };
    let elmat = TrimmedMassElmat::new(decomposition);
    assemble_galmat_dofs(self.topology, self.geometry, dofs, dofs, |metric, chart| {
      elmat.eval(metric, chart)
    })
  }

  /// The exterior derivative before constraints.
  fn dif_full(&self, grade: ExteriorGrade) -> CsrMatrix {
    let (Some(source), Some(target)) = (self.decomposition(grade), self.decomposition(grade + 1))
    else {
      return CsrMatrix::zeros(self.ndofs_full(grade + 1), self.ndofs_full(grade));
    };
    let local = source.dif_matrix(target);
    scatter_local_operator(
      self.topology,
      self.cell_dofs(grade + 1).expect("target grade is in range"),
      self.cell_dofs(grade).expect("source grade is in range"),
      &local,
    )
  }
}

impl TrimmedComplex<'_> {
  /// The load vector $[integral_K inner(f, v_i) vol]_i$ of an analytic source,
  /// restricted to the unconstrained degrees of freedom.
  ///
  /// Quadrature, not an exact integral: the source is an arbitrary field,
  /// unlike the mass matrix where both factors are polynomial. `quad_degree`
  /// defaults to twice the space's polynomial degree, exact for a polynomial
  /// source of that degree.
  pub fn source_vector<F>(&self, source: &F, quad_degree: Option<usize>) -> Vector
  where
    F: Sync + Section,
  {
    let grade = source.grade();
    let (Some(decomposition), Some(dofs)) = (self.decomposition(grade), self.cell_dofs(grade))
    else {
      return Vector::zeros(0);
    };
    let quad_degree = quad_degree.unwrap_or(2 * self.degree.index());
    let rule = || SimplexQuadRule::degree(self.dim(), quad_degree);
    let quad = CellQuadrature::new(self.dim(), Some(rule()));
    let shapes = LsfSamples::trimmed(decomposition, quad.nodes());
    let elvec = SourceElVec::with_shapes(source, Some(rule()), shapes);

    let full = assemble_galvec_dofs(self.topology, self.geometry, dofs, |metric, chart| {
      elvec.eval(metric, chart)
    });
    self.inclusion(grade).transpose() * full
  }
}

impl HilbertComplex for TrimmedComplex<'_> {
  fn dim(&self) -> Dim {
    TrimmedComplex::dim(self)
  }

  fn ndofs(&self, grade: impl Into<ExteriorGrade>) -> usize {
    let grade = grade.into();
    grade.index_in(self.dim()).map_or(0, |index| {
      self.ndofs_full(grade) - self.constrained[index].len()
    })
  }

  fn mass(&self, grade: impl Into<ExteriorGrade>) -> GalMat {
    let grade = grade.into();
    let inclusion = self.inclusion(grade);
    let mass = CsrMatrix::from(&self.mass_full(grade));
    GalMat::from(&(inclusion.transpose() * mass * inclusion))
  }

  fn dif(&self, grade: impl Into<ExteriorGrade>) -> CsrMatrix {
    let grade = grade.into();
    self.inclusion(grade + 1).transpose() * self.dif_full(grade) * self.inclusion(grade)
  }

  fn codif_dif(&self, grade: impl Into<ExteriorGrade>) -> GalMat {
    let grade = grade.into();
    let dif = self.dif(grade);
    let mass = CsrMatrix::from(&self.mass(grade + 1));
    GalMat::from(&(dif.transpose() * mass * dif))
  }

  /// The harmonic space is a topological invariant, so it is the Betti number
  /// at every polynomial degree: the trimmed complex has the cohomology of the
  /// manifold, and raising $r$ does not change the topology it resolves.
  fn harmonic_dim(&self, grade: impl Into<ExteriorGrade>) -> usize {
    let grade = grade.into();
    if !grade.in_range(self.dim()) {
      return 0;
    }
    let index = grade.index();
    if self.constrained[index].is_empty() {
      self.topology.betti_number(grade)
    } else {
      self.topology.relative_betti_number(grade)
    }
  }

  fn inclusion(&self, grade: impl Into<ExteriorGrade>) -> CsrMatrix {
    let grade = grade.into();
    let free = self.free_dofs(grade);
    let mut coo = CooMatrix::new(self.ndofs_full(grade), free.len());
    for (constrained_index, &full) in free.iter().enumerate() {
      coo.push(full, constrained_index, 1.0);
    }
    CsrMatrix::from(&coo)
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::whitney_complex::WhitneyComplex;
  use simplicial::mesher::cartesian::CartesianGrid;

  /// At $r = 1$ the trimmed complex is the Whitney complex: the same dof
  /// counts, mass matrices and exterior derivative.
  ///
  /// Stated on the assembled global operators rather than the reference cell, so
  /// the numbering and the scattering are checked too.
  #[test]
  fn the_first_order_trimmed_complex_is_the_whitney_complex() {
    for dim in 1..=3 {
      let (topology, coords) = CartesianGrid::new_unit(Dim::from(dim), 2).triangulate();
      let geometry = coords.to_edge_lengths_sq(&topology);

      let whitney = WhitneyComplex::new(&topology, &geometry);
      let trimmed = TrimmedComplex::new(&topology, &geometry, 1);

      for grade in 0..=dim {
        assert_eq!(
          HilbertComplex::ndofs(&trimmed, grade),
          whitney.ndofs(grade),
          "dof count at dim {dim} grade {grade}"
        );

        let expected = CsrMatrix::from(&whitney.mass(grade));
        let actual = CsrMatrix::from(&HilbertComplex::mass(&trimmed, grade));
        let difference = &actual - &expected;
        assert!(
          difference.values().iter().all(|v| v.abs() < 1e-10),
          "mass at dim {dim} grade {grade}"
        );

        let expected = whitney.dif(grade);
        let actual = HilbertComplex::dif(&trimmed, grade);
        let difference = &actual - &expected;
        assert!(
          difference.values().iter().all(|v| v.abs() < 1e-10),
          "dif at dim {dim} grade {grade}"
        );
      }
    }
  }

  /// The discrete complex is a complex at every polynomial degree:
  /// $dif compose dif = 0$ globally, after assembly.
  #[test]
  fn the_assembled_complex_is_a_complex() {
    for dim in 2..=3 {
      let (topology, coords) = CartesianGrid::new_unit(Dim::from(dim), 2).triangulate();
      let geometry = coords.to_edge_lengths_sq(&topology);
      for degree in 1..=3 {
        let trimmed = TrimmedComplex::new(&topology, &geometry, degree);
        for grade in 0..(dim - 1) {
          let composed =
            HilbertComplex::dif(&trimmed, grade + 1) * HilbertComplex::dif(&trimmed, grade);
          assert!(
            composed.values().iter().all(|v| v.abs() < 1e-9),
            "dif dif at dim {dim} degree {degree} grade {grade}"
          );
        }
      }
    }
  }

  /// The mass matrix is symmetric positive definite on a Riemannian geometry,
  /// at every degree: it is an $L^2$ inner product on a space of functions that
  /// are not identically zero.
  #[test]
  fn the_mass_matrix_is_positive_definite() {
    for dim in 1..=3 {
      let (topology, coords) = CartesianGrid::new_unit(Dim::from(dim), 1).triangulate();
      let geometry = coords.to_edge_lengths_sq(&topology);
      for degree in 1..=3 {
        let trimmed = TrimmedComplex::new(&topology, &geometry, degree);
        for grade in 0..=dim {
          let mass =
            nalgebra::DMatrix::from(&CsrMatrix::from(&HilbertComplex::mass(&trimmed, grade)));
          if mass.nrows() == 0 {
            continue;
          }
          approx::assert_relative_eq!(mass, mass.transpose(), epsilon = 1e-10);
          let smallest = mass
            .symmetric_eigenvalues()
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));
          assert!(
            smallest > 1e-12,
            "mass is not positive definite at dim {dim} degree {degree} grade {grade}: {smallest:e}"
          );
        }
      }
    }
  }
}
