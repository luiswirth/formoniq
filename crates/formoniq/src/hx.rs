//! Hiptmair-Xu auxiliary-space preconditioning of the grade-$k$ Hodge-Laplace
//! problem, the FEEC wiring of [`iterative::AuxiliarySpace`].
//!
//! The operator is the coercive $H Lambda^k (dif)$ Gram matrix
//! $A_k = M_k + D_k^T M_(k+1) D_k$ ([`HilbertComplex::hdif_gram`], SPD on a
//! Riemannian geometry). Its conditioning is not the mesh resolution but the
//! large near-kernel of $dif$: the range of $dif_(k-1)$, on which $A_k$ acts as
//! the bare mass while the rest of the spectrum carries the up-stiffness that
//! grows like $h^(-2)$. A pointwise smoother cannot see that near-kernel, so
//! plain multigrid stalls at grade $>= 1$. The structure-preserving fix moves it
//! onto auxiliary spaces where a nodal solver is effective, the additive
//! preconditioner
//!
//! $ B = S + Pi_"grad" A_(k-1)^(-1) Pi_"grad"^T + Pi_"vec" L_"vec"^(-1) Pi_"vec"^T, $
//!
//! with $S$ a weighted-Jacobi smoother of $A_k$ and two corrections realizing the
//! regular decomposition $omega = dif phi + "(vector-nodal remainder)"$:
//!
//! - the *gradient* space, tied in by $Pi_"grad" = D_(k-1)$ (the discrete
//!   exterior derivative, [`HilbertComplex::dif`]) with auxiliary operator the
//!   grade-$(k-1)$ problem $A_(k-1)$: it captures the $dif$-range part of the
//!   near-kernel exactly, since $A_k dif phi = M_k dif phi$ there;
//! - the *vector-nodal* space $[cal(W) Lambda^0]^(binom(N,k))$, tied in by the
//!   interpolation $Pi_"vec"$ of [`vector_nodal_prolongation`] with auxiliary
//!   operator $L_"vec"$, the block diagonal of $binom(N,k)$ copies of the scalar
//!   nodal problem $A_0$: it captures the regular remainder, whose components in
//!   a fixed frame decouple into scalar nodal Poisson solves.
//!
//! This is general in dimension and grade (invariant: no $k = 1$ special case):
//! the classical $H("curl")$ and $H("div")$ preconditioners are $N = 3$, $k = 1$
//! and $k = 2$. It is total on the degenerate boundary: at grade $0$ there is no
//! gradient space ($Lambda^(-1) = 0$) and the preconditioner is a plain smoothed
//! nodal solve; where a space is empty the correction is simply absent.
//!
//! No established theory bounds the contraction uniformly for arbitrary $(k, N)$
//! on arbitrary geometry, so the mesh-independence here is an empirical claim the
//! examples measure, not a proven one.
//!
//! ## Extrinsic ingredient (invariant 2)
//!
//! The vector-nodal space needs the constant $k$-covectors $e_I$ to be the *same*
//! covectors at every vertex, i.e. one global frame. On an intrinsic Regge
//! manifold there is none without a discrete connection, only the per-cell frames
//! glued by [`Transition`](simplicial::atlas). [`vector_nodal_prolongation`]
//! therefore takes [`MeshCoords`] and reads the frame off the embedding, which is
//! what makes the $binom(N,k)$ auxiliary copies decouple into scalar Laplacians.
//! It is the one embedding-dependent ingredient of the solver; the operator and
//! the gradient correction are intrinsic. A fully intrinsic, connection-based
//! vector-nodal space is future work.

use exterior::exterior_power;
use iterative::{ApproxInverse, AuxiliarySpace, Jacobi, SelfAdjoint};
use simplicial::{
  geometry::coord::mesh::MeshCoords,
  linalg::{CooMatrix, CsrMatrix, Matrix, Vector},
  topology::complex::Complex,
};

use crate::{
  linalg::DirectInverse,
  whitney_complex::{HilbertComplex, WhitneyComplex},
};

use exterior::ExteriorGrade;

/// The vector-nodal interpolation
/// $Pi: [cal(W) Lambda^0]^(binom(N,k)) -> cal(W) Lambda^k$, the de Rham map of a
/// nodal field valued in the constant ambient $k$-covectors.
///
/// A vector-nodal basis element is $phi_a e_I$: the nodal hat at vertex $a$ times
/// the constant ambient basis $k$-covector $e_I$ ($N$ the ambient dimension,
/// $I$ a colex $k$-subset). Its Whitney-$k$ coefficient on a $k$-simplex $sigma$
/// is $integral_sigma phi_a e_I$, and since $phi_a$ restricts to $sigma$ as its
/// own barycentric coordinate (zero unless $a in sigma$) and $e_I$ is constant,
/// this is $1/(k+1)! angle.l e_I, tau_sigma angle.r$ with $tau_sigma$ the ambient
/// tangent $k$-blade of $sigma$ in colex vertex order: the tangent blade carries
/// the cell Jacobian, and $integral phi_a$ over the standard $k$-simplex is its
/// volume $1/k!$ times $1/(k+1)$. The map is thus assembled per $k$-simplex, no
/// containing cell consulted.
///
/// Columns are grouped $I$-major, `col = I * nvertices + a`, so the auxiliary
/// operator is the block diagonal of $binom(N,k)$ copies of the scalar nodal
/// problem, one contiguous vertex block per covector.
///
/// Reads the ambient frame off `coords` (see the module note on invariant 2).
/// Total on the boundary: at grade $0$ it is the identity (one covector, each
/// simplex a vertex); where $binom(N,k) = 0$ it has no columns.
pub fn vector_nodal_prolongation(
  topology: &Complex,
  coords: &MeshCoords,
  grade: impl Into<ExteriorGrade>,
) -> CsrMatrix {
  let grade = grade.into();
  let k = grade.index();
  let ambient = coords.dim().index();
  let nvertices = coords.nvertices();
  let ksimplices = topology.skeleton(grade);
  let nrows = ksimplices.len();
  let ncovectors = num_minors(ambient, k);
  let ncols = ncovectors * nvertices;

  // Integral of a nodal hat over the standard k-simplex: vol(1/k!) times 1/(k+1).
  let factor = 1.0 / (1..=k + 1).product::<usize>() as f64;
  let mut coo = CooMatrix::new(nrows, ncols);
  for (row, simp) in ksimplices.handle_iter().enumerate() {
    let vertices: Vec<usize> = simp.simplex().iter().collect();
    let blade = tangent_blade(coords, &vertices, ambient, k);
    for &a in &vertices {
      for (comp, &value) in blade.iter().enumerate() {
        coo.push(row, comp * nvertices + a, factor * value);
      }
    }
  }
  CsrMatrix::from(&coo)
}

/// The ambient tangent $k$-blade $v_1 wedge dots.c wedge v_k$ of a simplex,
/// its edge vectors from vertex $0$ (colex order) reduced by the $k$-th exterior
/// power to the column of $binom(N,k)$ minors in colex.
fn tangent_blade(coords: &MeshCoords, vertices: &[usize], ambient: usize, k: usize) -> Vector {
  let base = coords.coord(vertices[0]).view();
  let edges = Matrix::from_fn(ambient, k, |i, j| {
    coords.coord(vertices[j + 1]).view()[i] - base[i]
  });
  exterior_power(&edges, k).column(0).into_owned()
}

/// $binom(N, k)$, the dimension of $Lambda^k RR^N$: the number of covectors, and
/// the row count of a tangent blade. Zero off range.
fn num_minors(ambient: usize, k: usize) -> usize {
  if k > ambient {
    return 0;
  }
  (0..k).fold(1usize, |acc, i| acc * (ambient - i) / (i + 1))
}

/// A grade-$k$ Hodge-Laplace solver preconditioned by the Hiptmair-Xu
/// auxiliary-space preconditioner, on a single mesh (no refinement tower).
///
/// Owns the operator $A_k$ and the preconditioner $B$; [`solve`](Self::solve)
/// runs $B$-preconditioned CG. The auxiliary blocks are direct faer solves here,
/// which validates the HX structure and its iteration-count reduction; recursive
/// multigrid on the blocks is what makes it scale, and is the benchmark's job.
pub struct GradeKHodgeHx {
  operator: CsrMatrix,
  preconditioner: AuxiliarySpace<Jacobi>,
}

/// The classic damped-Jacobi weight, damping the upper half of the spectrum.
const SMOOTHER_WEIGHT: f64 = 2.0 / 3.0;

impl GradeKHodgeHx {
  /// Assemble the operator and the auxiliary-space preconditioner for grade
  /// `grade` on the given Whitney complex, reading the ambient frame off
  /// `coords`.
  ///
  /// # Panics
  /// If an auxiliary operator is not SPD (a non-Riemannian geometry), which the
  /// direct block solves require.
  pub fn new(
    complex: &WhitneyComplex,
    coords: &MeshCoords,
    grade: impl Into<ExteriorGrade>,
  ) -> Self {
    let grade = grade.into();
    let operator = complex.hdif_gram(grade);
    let smoother = Jacobi::weighted(&operator, SMOOTHER_WEIGHT);
    let mut preconditioner = AuxiliarySpace::new(smoother);

    // Gradient correction: Pi_grad = D_{k-1}, auxiliary operator A_{k-1}. Absent
    // at grade 0, where the lower space Lambda^{-1} is trivial.
    let lower = grade - 1;
    if complex.ndofs(lower) > 0 {
      let prolong = complex.dif(lower);
      preconditioner =
        preconditioner.with_correction(prolong, Box::new(direct(complex.hdif_gram(lower))));
    }

    // Vector-nodal correction: Pi_vec, auxiliary operator the block diagonal of
    // C(N,k) scalar nodal problems A_0. The blocks are identical, so one factor
    // is shared across them. Absent where the space has no columns.
    let prolong = vector_nodal_prolongation(complex.topology(), coords, grade);
    if prolong.ncols() > 0 {
      let ncovectors = prolong.ncols() / coords.nvertices();
      let blocks = ReplicatedBlock::new(direct(complex.hdif_gram(0)), ncovectors);
      preconditioner = preconditioner.with_correction(prolong, Box::new(blocks));
    }

    Self {
      operator,
      preconditioner,
    }
  }

  /// The assembled operator $A_k$.
  pub fn operator(&self) -> &CsrMatrix {
    &self.operator
  }

  /// The auxiliary-space preconditioner $B$.
  pub fn preconditioner(&self) -> &AuxiliarySpace<Jacobi> {
    &self.preconditioner
  }

  /// Solve $A_k x = "rhs"$ by $B$-preconditioned CG.
  pub fn solve(&self, rhs: &Vector, stop: iterative::StopCriterion) -> (Vector, iterative::Report) {
    iterative::krylov::cg(&self.operator, &self.preconditioner, rhs, stop)
  }
}

/// The direct SPD inverse of an auxiliary operator, expecting positive
/// definiteness (the Riemannian case the auxiliary solves require).
fn direct(operator: CsrMatrix) -> DirectInverse {
  DirectInverse::try_new(operator).expect("auxiliary operator must be SPD")
}

/// One factorization applied to each of `count` contiguous blocks of equal size:
/// the block-diagonal inverse of `count` identical copies of an operator,
/// sharing a single factor. The vector-nodal auxiliary operator is exactly this,
/// $binom(N,k)$ copies of the scalar nodal problem in a fixed frame.
struct ReplicatedBlock {
  inverse: DirectInverse,
  count: usize,
}

impl ReplicatedBlock {
  fn new(inverse: DirectInverse, count: usize) -> Self {
    Self { inverse, count }
  }
}

impl ApproxInverse for ReplicatedBlock {
  fn dim(&self) -> usize {
    self.inverse.dim() * self.count
  }
  fn apply(&self, r: &Vector) -> Vector {
    let block = self.inverse.dim();
    let mut out = Vector::zeros(self.dim());
    for i in 0..self.count {
      let piece = self.inverse.apply(&r.rows(i * block, block).into_owned());
      out.rows_mut(i * block, block).copy_from(&piece);
    }
    out
  }
}

impl SelfAdjoint for ReplicatedBlock {}

#[cfg(test)]
mod tests {
  use super::*;
  use derham::{
    cochain::Cochain,
    interpolate::interpolant::WhitneyInterpolant,
    project::derham_map,
    section::{CoordFieldExt, Wedge},
  };
  use exterior::ExteriorElement;
  use glatt::field::DiffFormClosure;
  use simplicial::mesher::cartesian::CartesianGrid;

  /// Column $(a, I)$ of $Pi_"vec"$ is the de Rham map of $phi_a e_I$: the
  /// assembly is validated against the tested [`derham_map`], the primitive it
  /// stands in for. Swept over dimension and grade so it is one statement, not a
  /// fixed case.
  #[test]
  fn vector_nodal_prolongation_is_the_derham_map() {
    for dim in 1..=3 {
      let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      let nvertices = coords.nvertices();
      for grade in 1..=dim {
        let pi = vector_nodal_prolongation(&topology, &coords, grade);
        let ncovectors = num_minors(dim, grade);
        for covector in 0..ncovectors {
          // The constant ambient basis k-covector e_I, pulled onto the mesh.
          let e_i = DiffFormClosure::new(
            move |_| ExteriorElement::new(unit(ncovectors, covector), dim, grade),
            dim,
            grade,
          );
          for a in 0..nvertices {
            let pulled = e_i.pullback_on(&topology, &coords);
            let hat = WhitneyInterpolant::new(Cochain::new(0, unit(nvertices, a)), &topology);
            let reference = derham_map(&Wedge::new(hat, pulled), &topology, 2);
            let assembled = &pi * unit(pi.ncols(), covector * nvertices + a);
            let err = (reference.coeffs() - &assembled).norm();
            assert!(
              err < 1e-9,
              "dim {dim} grade {grade} covector {covector} vertex {a}: \
               Pi_vec column disagrees with derham map, err {err}"
            );
          }
        }
      }
    }
  }

  fn unit(n: usize, i: usize) -> Vector {
    let mut v = Vector::zeros(n);
    v[i] = 1.0;
    v
  }

  /// HX-preconditioned CG reaches the same solution as the direct solve of the
  /// same grade-$k$ system: the preconditioner changes the path, not the fixed
  /// point. Swept over the grades of a 2D and a 3D mesh.
  #[test]
  fn hx_cg_matches_the_direct_solve() {
    use iterative::StopCriterion;
    for dim in 2..=3 {
      let (topology, coords) = CartesianGrid::new_unit(dim, 3).triangulate();
      let geometry = coords.to_edge_lengths_sq(&topology);
      let complex = WhitneyComplex::new(&topology, &geometry);
      for grade in 1..=dim {
        let hx = GradeKHodgeHx::new(&complex, &coords, grade);
        let n = hx.operator().nrows();
        let rhs = Vector::from_fn(n, |i, _| ((i * i + 1) as f64).cos());
        let (x, report) = hx.solve(&rhs, StopCriterion::rtol(1e-10));
        assert!(
          report.converged,
          "dim {dim} grade {grade}: HX-CG did not converge"
        );
        let direct = DirectInverse::try_new(hx.operator().clone()).unwrap();
        let err = (&x - direct.apply(&rhs)).norm();
        assert!(
          err < 1e-7,
          "dim {dim} grade {grade}: HX-CG disagrees with direct, err {err}"
        );
      }
    }
  }
}
