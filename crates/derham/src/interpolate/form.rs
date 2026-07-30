use {
  multialgebra::{ExteriorGrade, Tensor, Variance, exterior_dim, tensor::Transport},
  multiindex::{Combination, Sign, factorial_f64},
  simplicial::linalg::Matrix,
  simplicial::{
    Dim,
    atlas::{BaryRef, unit_difbarys},
    topology::simplex::unit_subsimps,
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
/// are the constant [`unit_difbarys`], so a Whitney form depends on nothing but
/// the cell dimension and the DOF vertex set -- no coordinates, no metric.
/// This is what lets them live on a bare Regge manifold.
#[derive(Debug, Clone)]
pub struct WhitneyLsf {
  cell_dim: Dim,
  /// The local vertex set of the DOF subsimplex.
  dof_simp: Combination,
  /// $lambda^*$ at grade $k$, the pullback along the barycentric coordinate
  /// map: what carries a formal barycentric blade to a reference $k$-form.
  bary_pullback: Transport,
  /// The same at grade $k+1$, where [`Self::dif`] lands.
  bary_pullback_dif: Transport,
}
impl WhitneyLsf {
  pub fn unit(cell_dim: Dim, dof_simp: Combination) -> Self {
    // The differential of the barycentric coordinate map
    // $lambda: RR^n -> RR^(n+1)$: the rows are the $dif lambda_i$.
    let difbarys = unit_difbarys(cell_dim);
    let grade = Dim::from(dof_simp.card() - 1);
    let covariant = |grade| Tensor::one_alternating(grade, Variance::Covariant, cell_dim);
    Self {
      cell_dim,
      dof_simp,
      bary_pullback: Transport::new(&covariant(grade), &difbarys),
      bary_pullback_dif: Transport::new(&covariant(grade + 1), &difbarys),
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
  fn barycentric_blade(&self) -> Tensor {
    Tensor::from_blade_signed(
      self.nvertices(),
      Sign::Pos,
      self.dof_simp,
      Variance::Covariant,
    )
  }

  /// The value at a point of the reference cell, in its reference frame.
  pub fn at_bary<'a>(&self, bary: impl Into<BaryRef<'a>>) -> Tensor {
    let bary = Tensor::line(bary.into().view().into_owned(), Variance::Contravariant);
    let koszul = self.barycentric_blade().interior_product(&bary);
    factorial_f64(self.grade().index()) * self.bary_pullback.pullback(&koszul)
  }

  /// The constant exterior derivative
  /// $dif W_sigma = (k+1)! med lambda^* (e_sigma)
  /// = (k+1)! dif lambda_(sigma_0) wedge dots.c wedge dif lambda_(sigma_k)$.
  ///
  /// Vanishes automatically for the top grade, where $Lambda^(k+1) (RR^n)$
  /// is the zero space.
  pub fn dif(&self) -> Tensor {
    factorial_f64(self.grade().index() + 1)
      * self.bary_pullback_dif.pullback(&self.barycentric_blade())
  }
}

/// The whole Whitney basis of one grade as a single linear map
/// $C: RR^(Delta_k) -> Lambda^k (RR^(n+1)) times.circle RR^(n+1)$, sending a
/// degree of freedom to the components of $W_sigma$ on the products
/// $dif lambda_I lambda_v$.
///
/// The deletion formula read as a matrix: column $sigma$ carries exactly $k+1$
/// nonzeros, the entry $(-1)^i k!$ at the row $(sigma without sigma_i, sigma_i)$.
/// It is the matrix of the Koszul contraction $kappa$ on blades, and summing its
/// blocks over the vertex index collapses $kappa$ to $iota_bb(1)$, giving $k!$
/// times the boundary operator (test `koszul_collapses_to_the_boundary_operator`).
///
/// Combinatorial, and by [`WhitneyLsf`] the entire cell-independent content of
/// the basis: a metric reaches an $L^2$ product only through the form it is
/// pulled back from ([`pullback`](Self::pullback)). A higher-order trimmed
/// space $P^-_r Lambda^k$ enlarges this map and changes nothing else.
#[derive(Debug, Clone)]
pub struct WhitneyExpansion {
  cell_dim: Dim,
  grade: ExteriorGrade,
  dofs: Vec<Combination>,
}
impl WhitneyExpansion {
  pub fn new(cell_dim: impl Into<Dim>, grade: impl Into<ExteriorGrade>) -> Self {
    let (cell_dim, grade) = (cell_dim.into(), grade.into());
    let dofs = unit_subsimps(cell_dim, grade).collect();
    Self {
      cell_dim,
      grade,
      dofs,
    }
  }

  pub fn cell_dim(&self) -> Dim {
    self.cell_dim
  }
  pub fn grade(&self) -> ExteriorGrade {
    self.grade
  }
  /// The degrees of freedom, in colex order: the columns of the map.
  pub fn dofs(&self) -> &[Combination] {
    &self.dofs
  }

  /// The pullback $C^top (H times.circle Q) C$ along the basis of a bilinear
  /// form that factors into a part $H$ on the barycentric $k$-blades and a part
  /// $Q$ on the barycentric coordinates.
  ///
  /// Every $L^2$ product of Whitney forms on an affine cell has this shape,
  /// because the integrand does: the blades are constant and the coordinates
  /// carry the whole $x$-dependence. With $H = Lambda^k (dif lambda)
  /// (Lambda^k g^(-1)) Lambda^k (dif lambda)^top$ and $Q$ the barycentric
  /// [`unit_bary_gramian`](simplicial::atlas::unit_bary_gramian), the result is
  /// the Hodge mass matrix at unit volume.
  ///
  /// The factors are taken apart because their tensor product is the one thing
  /// worth not forming: $C$ has $k+1$ nonzeros per column, so the sum here runs
  /// over $(k+1)^2$ terms per entry, where the product has
  /// $(n+1)^2 binom(n+1,k)^2$.
  pub fn pullback(&self, blade: &Matrix, bary: &Matrix) -> Matrix {
    let mut pullback = Matrix::zeros(self.dofs.len(), self.dofs.len());
    for (i, asimp) in self.dofs.iter().enumerate() {
      for (j, bsimp) in self.dofs.iter().enumerate() {
        pullback[(i, j)] = asimp
          .deletions()
          .flat_map(|a| bsimp.deletions().map(move |b| (a, b)))
          .map(|((asign, avertex, ablade), (bsign, bvertex, bblade))| {
            (asign * bsign).as_f64()
              * blade[(ablade.rank(), bblade.rank())]
              * bary[(avertex, bvertex)]
          })
          .sum();
      }
    }
    factorial_f64(self.grade.index()).powi(2) * pullback
  }

  /// The map written out, with the blade index in colex order and the vertex
  /// index running fastest: the row order of the tensor product
  /// [`pullback`](Self::pullback) takes apart.
  pub fn matrix(&self) -> Matrix {
    let nvertices = (self.cell_dim + 1).index();
    let nblades = exterior_dim(nvertices, self.grade);
    let scale = factorial_f64(self.grade.index());
    let mut coeffs = Matrix::zeros(nblades * nvertices, self.dofs.len());
    for (j, dof) in self.dofs.iter().enumerate() {
      for (sign, vertex, blade) in dof.deletions() {
        coeffs[(blade.rank() * nvertices + vertex, j)] = sign.as_f64() * scale;
      }
    }
    coeffs
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use approx::assert_relative_eq;
  use metric::Metric;
  use metric::tensor::inner;
  use multialgebra::{exterior_bases, exterior_dim};
  use multiindex::combinations;
  use simplicial::atlas::{SimplexQuadRule, unit_simplex_volume};
  use simplicial::linalg::Vector;
  use simplicial::topology::simplex::unit_boundary_operator;

  /// A non-diagonal metric of signature $(n - q, q)$, so the law is not read on
  /// an orthonormal frame where terms cancel for the wrong reason.
  fn skewed_metric(dim: usize, q: usize) -> Metric {
    let j = Matrix::from_fn(dim, dim, |i, k| match i.cmp(&k) {
      std::cmp::Ordering::Equal => 1.0,
      std::cmp::Ordering::Greater => ((3 * i + 5 * k) % 4) as f64 / 8.0,
      std::cmp::Ordering::Less => 0.0,
    });
    Metric::pseudo_euclidean(dim - q, q).pullback(&j)
  }

  /// $C^top (H times.circle Q) C$ computed on the factors agrees with the same
  /// pullback formed through the explicit map and the explicit tensor product.
  ///
  /// The two sides are different objects, not two spellings of one: the right
  /// is the definition, the left the evaluation that never leaves the factors.
  /// Random asymmetric factors, so a transposed index is not invisible.
  #[test]
  fn whitney_pullback_is_the_kronecker_sandwich() {
    for dim in (0..=4).map(Dim::from) {
      let nvertices = (dim + 1).index();
      for grade in 0..=dim.index() {
        let expansion = WhitneyExpansion::new(dim, grade);
        let nblades = exterior_dim(nvertices, grade);
        let entry = |i: usize, j: usize| ((7 * i + 3 * j + 1) % 11) as f64 - 5.0;
        let blade = Matrix::from_fn(nblades, nblades, entry);
        let bary = Matrix::from_fn(nvertices, nvertices, entry);

        let matrix = expansion.matrix();
        let expected = matrix.transpose() * blade.kronecker(&bary) * &matrix;
        assert_relative_eq!(expansion.pullback(&blade, &bary), expected);
      }
    }
  }

  /// Summing the blocks of $C$ over the vertex index collapses the Koszul
  /// contraction $kappa$ to $iota_bb(1)$, and $iota_bb(1)$ is the simplicial
  /// boundary: the result is $k!$ times $diff$.
  ///
  /// This is the $kappa$ half of a correspondence whose $dif$ half is Stokes,
  /// $R compose dif = dif compose R$. The two operators of the exterior
  /// algebra have the two operators of the chain complex as their shadows,
  /// and forgetting the vertex weights is the map that takes one to the
  /// other. It is why the deletion formula of a Whitney form and the boundary
  /// of a simplex are the same combinatorics rather than an analogy.
  ///
  /// At grade 0 the collapse is the *augmentation* onto the empty simplex,
  /// which [`unit_boundary_operator`] deliberately drops, so the law is read
  /// there against the all-ones row it must be.
  #[test]
  fn koszul_collapses_to_the_boundary_operator() {
    for dim in (0..=4).map(Dim::from) {
      let nvertices = (dim + 1).index();
      for grade in 0..=dim.index() {
        let expansion = WhitneyExpansion::new(dim, grade);
        let matrix = expansion.matrix();
        let ndofs = expansion.dofs().len();
        let nblades = exterior_dim(nvertices, grade);

        let collapsed = Matrix::from_fn(nblades, ndofs, |blade, dof| {
          (0..nvertices)
            .map(|vertex| matrix[(blade * nvertices + vertex, dof)])
            .sum()
        });

        let scale = factorial_f64(grade);
        let expected = if grade == 0 {
          Matrix::from_element(1, ndofs, scale)
        } else {
          scale * unit_boundary_operator(dim, grade.into())
        };
        assert_relative_eq!(collapsed, expected);
      }
    }
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
      let difbarys = unit_difbarys(dim);
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
        Tensor::line(coeffs, Variance::Covariant)
      };

      for q in 0..=dim.index() {
        let metric = skewed_metric(dim.index(), q);
        for grade in 1..=dim.index() {
          for dof_simp in combinations(nvertices, grade + 1) {
            let whitney = WhitneyLsf::unit(dim, dof_simp);
            for blade in exterior_bases(dim, grade - 1) {
              let c = Tensor::from_blade_signed(dim, Sign::Pos, blade, Variance::Covariant);
              let integral = qr.integrate_unit(
                &|bary: BaryRef| {
                  let dif_phi = bubble_dif(bary).wedge(&c);
                  inner(&dif_phi, &whitney.at_bary(bary), &metric)
                },
                unit_simplex_volume(dim),
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
