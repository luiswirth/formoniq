use {
  exterior::{ExteriorGrade, MultiForm, MultiVector},
  multiindex::{Combination, Sign, factorial_f64},
  simplicial::linalg::Matrix,
  simplicial::{
    Dim,
    atlas::{BaryRef, ref_difbarys},
  },
};

/// The *local* shape function of the Whitney form $W_sigma$: its restriction to
/// one cell, indexed by the DOF subsimplex's local vertex positions and written
/// in the reference barycentric frame. The basis of the lowest-order trimmed
/// space $P^-_1 Lambda^k$, dual to the degrees of freedom,
/// $integral_tau W_sigma = delta_(sigma tau)$.
///
/// The Whitney form itself is *global*: the $lambda_i$ of Whitney's
/// construction are the barycentric coordinates of the whole complex, so
/// $W_sigma$ is indexed by a simplex of the mesh and supported on its star.
/// That object gets no type of its own because it is the special case of
/// [`WhitneyInterpolant`](super::interpolant::WhitneyInterpolant) at a cochain
/// with a single unit degree of freedom. What lives here is the piece an
/// element integral consumes, and the distinction is load-bearing: the local
/// shape function of a fixed grade is reference data, one object for the whole
/// mesh, while the global form is one object per simplex.
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
pub struct WhitneyLsf {
  cell_dim: Dim,
  /// The local vertex set of the DOF subsimplex.
  dof_simp: Combination,
  /// The differential of the barycentric coordinate map
  /// $lambda: RR^n -> RR^(n+1)$: the rows are the $dif lambda_i$.
  difbarys: Matrix,
}
impl WhitneyLsf {
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
    (self.dof_simp.card() - 1).into()
  }
  pub fn nvertices(&self) -> usize {
    (self.cell_dim + 1).index()
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
    factorial_f64(self.grade().index()) * koszul.pullback(&self.difbarys)
  }

  /// The constant exterior derivative
  /// $dif W_sigma = (k+1)! med lambda^* (e_sigma)
  /// = (k+1)! dif lambda_(sigma_0) wedge dots.c wedge dif lambda_(sigma_k)$.
  ///
  /// Vanishes automatically for the top grade, where $Lambda^(k+1) (RR^n)$
  /// is the zero space.
  pub fn dif(&self) -> MultiForm {
    factorial_f64(self.grade().index() + 1) * self.barycentric_blade().pullback(&self.difbarys)
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use exterior::{exterior_bases, multiform_gramian};
  use gramian::{Gramian, Metric};
  use multiindex::combinations;
  use simplicial::atlas::{SimplexQuadRule, refsimp_vol};
  use simplicial::linalg::Vector;

  /// A non-diagonal metric of signature $(n - q, q)$, so the law is not read on
  /// an orthonormal frame where terms cancel for the wrong reason.
  fn skewed_metric(dim: usize, q: usize) -> Metric {
    let j = Matrix::from_fn(dim, dim, |i, k| match i.cmp(&k) {
      std::cmp::Ordering::Equal => 1.0,
      std::cmp::Ordering::Greater => ((3 * i + 5 * k) % 4) as f64 / 8.0,
      std::cmp::Ordering::Less => 0.0,
    });
    Metric::new(Gramian::pseudo_euclidean(dim - q, q).pullback(&j))
  }

  /// A corollary of $delta compose kappa = 0$ on constant forms, since
  /// $W_sigma = k! lambda^* (kappa e_sigma)$: on a constant form
  /// $diff_i (kappa omega)_(j_1 dots j_k) = omega_(i j_1 dots j_k)$, so
  /// $delta kappa omega$ contracts the symmetric $g^(i j_1)$ into two
  /// alternating slots. Lowest order only, where $diff kappa = id$.
  ///
  /// This is why the weak Lie derivative has no volume contribution from
  /// $dif iota_v$: Cartan's second term is supported on $diff K$ alone.
  ///
  /// Tested by adjointness rather than through a formula for $delta$. The
  /// bubble $b = product_i lambda_i$ vanishes on every facet, so $phi = b c$
  /// kills the boundary term and $dif phi = dif b wedge c$ needs no star.
  #[test]
  fn whitney_forms_are_coclosed() {
    for dim in (1..=3).map(Dim::from) {
      let difbarys = ref_difbarys(dim);
      let nvertices = (dim + 1).index();
      let qr = SimplexQuadRule::degree(dim, nvertices + 1);

      // $dif b = sum_i (product_(j != i) lambda_j) dif lambda_i$.
      let bubble_dif = |bary: BaryRef| {
        let mut coeffs = Vector::zeros(dim.index());
        for i in 0..nvertices {
          let weight: f64 = (0..nvertices)
            .filter(|&j| j != i)
            .map(|j| bary.view()[j])
            .product();
          coeffs += weight * difbarys.row(i).transpose();
        }
        MultiForm::line(coeffs)
      };

      for q in 0..=dim.index() {
        let metric = skewed_metric(dim.index(), q);
        for grade in 1..=dim.index() {
          let inner = multiform_gramian(&metric, grade);
          for dof_simp in combinations(nvertices, grade + 1) {
            let whitney = WhitneyLsf::standard(dim, dof_simp);
            for blade in exterior_bases(dim, grade - 1) {
              let c = MultiForm::from_blade_signed(dim, Sign::Pos, blade);
              let integral = qr.integrate_ref(
                &|bary: BaryRef| {
                  let dif_phi = bubble_dif(bary).wedge(&c);
                  inner.inner(dif_phi.coeffs(), whitney.at_bary(bary).coeffs())
                },
                refsimp_vol(dim),
              );
              assert!(
                integral.abs() < 1e-12,
                "dim {dim:?} grade {grade} q {q} dof {dof_simp:?} blade {blade:?}: {integral}"
              );
            }
          }
        }
      }
    }
  }
}
