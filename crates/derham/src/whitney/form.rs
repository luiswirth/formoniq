use {
  common::{
    combo::{factorial_f64, Combination, Sign},
    linalg::nalgebra::Matrix,
  },
  exterior::{ExteriorGrade, MultiForm, MultiVector},
  simplicial::{
    atlas::{ref_difbarys, BaryRef},
    Dim,
  },
};

/// The Whitney form $W_sigma$ of a DOF subsimplex, on the reference cell: the
/// basis of the lowest-order trimmed space $P^-_1 Lambda^k$, dual to the
/// degrees of freedom, $integral_tau W_sigma = delta_(sigma tau)$.
///
/// Work in the formal barycentric space $Lambda(RR^(n+1))$, where the
/// vertex set $sigma$ is a blade $e_sigma$ and the barycentric coordinates
/// $lambda(x)$ are a vector. Then the Whitney form is the pullback along
/// the barycentric coordinate map of the Koszul contraction of the blade:
///
/// $W_sigma = k! med lambda^* (iota_(lambda(x)) e_sigma)
///   = k! sum_i (-1)^i lambda_(sigma_i)
///     dif lambda_(sigma_0) wedge dots.c hat(dif lambda_(sigma_i)) dots.c wedge dif lambda_(sigma_k)$
///
/// The contraction $iota_lambda$ is the Koszul operator $kappa$ of FEEC.
///
/// Purely combinatorial: the barycentric differentials of the reference cell
/// are the constant [`ref_difbarys`], so a Whitney form depends on nothing but
/// the cell dimension and the DOF vertex set -- no coordinates, no metric.
/// This is what lets them live on a bare Regge manifold.
#[derive(Debug, Clone)]
pub struct WhitneyForm {
  cell_dim: Dim,
  /// The local vertex set of the DOF subsimplex.
  dof_simp: Combination,
  /// The differential of the barycentric coordinate map
  /// $lambda: RR^n -> RR^(n+1)$: the rows are the $dif lambda_i$.
  difbarys: Matrix,
}
impl WhitneyForm {
  pub fn standard(cell_dim: Dim, dof_simp: Combination) -> Self {
    Self {
      cell_dim,
      dof_simp,
      difbarys: ref_difbarys(cell_dim),
    }
  }

  pub fn cell_dim(&self) -> Dim {
    self.cell_dim
  }
  pub fn grade(&self) -> ExteriorGrade {
    self.dof_simp.card() - 1
  }
  pub fn nvertices(&self) -> usize {
    self.cell_dim + 1
  }

  /// The DOF vertex set as a blade in the formal barycentric space
  /// $Lambda^(k+1) (RR^(n+1))$.
  fn barycentric_blade(&self) -> MultiForm {
    MultiForm::from_blade_signed(self.nvertices(), Sign::Pos, self.dof_simp)
  }

  /// The value at a point of the reference cell, in its reference frame.
  pub fn at_bary<'a>(&self, bary: impl Into<BaryRef<'a>>) -> MultiForm {
    let bary = MultiVector::line(bary.into().view().into_owned());
    let koszul = self.barycentric_blade().interior_product(&bary);
    factorial_f64(self.grade()) * koszul.pullback(&self.difbarys)
  }

  /// The constant exterior derivative
  /// $dif W_sigma = (k+1)! med lambda^* (e_sigma)
  /// = (k+1)! dif lambda_(sigma_0) wedge dots.c wedge dif lambda_(sigma_k)$.
  ///
  /// Vanishes automatically for the top grade, where $Lambda^(k+1) (RR^n)$
  /// is the zero space.
  pub fn dif(&self) -> MultiForm {
    factorial_f64(self.grade() + 1) * self.barycentric_blade().pullback(&self.difbarys)
  }
}
