use {
  multialgebra::{
    ExteriorGrade, Factor, Slot, Tensor, Variance,
    tensor::{Slots, Transport, covariant_slots, one_alternating, tensor_dim, tensor_strides},
  },
  multiindex::{Combination, Sign, factorial_f64},
  simplicial::linalg::Matrix,
  simplicial::{
    Dim,
    atlas::{BaryRef, unit_difbarys},
    topology::simplex::unit_subsimps,
  },
};

/// The local shape function of the Whitney form $W_sigma$: its restriction to
/// one cell, indexed by the DOF subsimplex's local vertex positions and written
/// in the reference barycentric frame. The basis of the lowest-order trimmed
/// space $P^-_1 Lambda^k$, dual to the degrees of freedom,
/// $integral_tau W_sigma = delta_(sigma tau)$.
///
/// The Whitney form itself is global: the $lambda_i$ of Whitney's
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
/// the cell dimension and the DOF vertex set, no coordinates, no metric.
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
    let covariant = |grade| one_alternating(grade, Variance::Covariant, cell_dim);
    Self {
      cell_dim,
      dof_simp,
      bary_pullback: Transport::new(&covariant(grade), &difbarys),
      bary_pullback_dif: Transport::new(&covariant(grade + 1), &difbarys),
    }
  }

  /// The whole basis of one grade on the reference cell, in the colex order of
  /// the DOF vertex sets, which is the order the faces of a cell come in.
  ///
  /// Reference data: every chart of the atlas is the same chart up to the
  /// labelling of its vertices, so this family is built once and read on every
  /// cell of the mesh.
  pub fn basis(
    cell_dim: impl Into<Dim>,
    grade: impl Into<ExteriorGrade>,
  ) -> impl Iterator<Item = Self> {
    let cell_dim = cell_dim.into();
    unit_subsimps(cell_dim, grade.into()).map(move |dof_simp| Self::unit(cell_dim, dof_simp))
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
/// $C: RR^(Delta_k) -> Lambda^k (RR^(n+1)) times.o RR^(n+1)$, sending a
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
  /// The codomain $Lambda^k (RR^(n+1))^* times.o (RR^(n+1))^*$ as a shape,
  /// so the component layout is [`tensor_strides`] rather than arithmetic
  /// written out here.
  slots: Slots,
}
impl WhitneyExpansion {
  pub fn new(cell_dim: impl Into<Dim>, grade: impl Into<ExteriorGrade>) -> Self {
    let (cell_dim, grade) = (cell_dim.into(), grade.into());
    let dofs = unit_subsimps(cell_dim, grade).collect();
    let nvertices = (cell_dim + 1).index();
    Self {
      cell_dim,
      grade,
      dofs,
      // Uniformly covariant: $lambda_v$ is a coordinate functional on the formal
      // barycentric space and $dif lambda_I$ a $k$-form on it, so both slots are
      // dual. The two symmetries coincide at degree one.
      slots: covariant_slots(
        [Factor::alternating(grade), Factor::alternating(1)],
        nvertices,
      ),
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
  /// The product space this map lands in, whose per-slot Gram matrices are what
  /// [`pullback`](Self::pullback) consumes.
  pub fn slots(&self) -> &[Slot] {
    &self.slots
  }

  /// The nonzeros of the column at `dof`: the deletion formula
  /// $W_sigma = k! sum_i (-1)^i lambda_(sigma_i) dif lambda_(sigma without sigma_i)$,
  /// as $(dif lambda$ blade, $lambda$ vertex, coefficient$)$.
  ///
  /// The single definition of this map. Both readings of it, the explicit
  /// [`matrix`](Self::matrix) and the factored [`pullback`](Self::pullback),
  /// consume this rather than restating it, so the two cannot drift apart. The
  /// $k!$ lives here, which is why the pullback of a bilinear form carries
  /// $(k!)^2$ without anyone squaring anything.
  fn column(&self, dof: Combination) -> impl Iterator<Item = (Combination, usize, f64)> {
    let scale = factorial_f64(self.grade.index());
    dof
      .deletions()
      .map(move |(sign, vertex, blade)| (blade, vertex, sign.as_f64() * scale))
  }

  /// The pullback $C^top (H times.o Q) C$ along the basis of a bilinear
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
  /// Not forming $H times.o Q$ is the general rule
  /// ([`apply_factorwise`](multialgebra::tensor::apply_factorwise)), and not
  /// what is special here. What is special is that $C$ is sparse: the
  /// deletion formula gives it $k+1$ nonzeros per column, so an entry is a sum
  /// over $(k+1)^2$ pairs rather than over the $(n+1) binom(n+1,k)$ dimensions
  /// of the product space. That sparsity is a property of the Whitney basis, not
  /// of the tensor product, so it is spent here and cannot be spent by the
  /// algebra: a generic factored application would hand back dense columns and
  /// be slower than this.
  pub fn pullback(&self, blade: &Matrix, bary: &Matrix) -> Matrix {
    Matrix::from_fn(self.dofs.len(), self.dofs.len(), |i, j| {
      self
        .column(self.dofs[i])
        .flat_map(|a| self.column(self.dofs[j]).map(move |b| (a, b)))
        .map(|((ablade, avertex, avalue), (bblade, bvertex, bvalue))| {
          avalue * bvalue * blade[(ablade.rank(), bblade.rank())] * bary[(avertex, bvertex)]
        })
        .sum()
    })
  }

  /// The map written out, its rows laid out on the basis of
  /// [`slots`](Self::slots): both indices in colex order, the blade index (the
  /// first slot) running fastest.
  ///
  /// That layout is [`tensor_strides`] and not arithmetic of its own, which is
  /// what makes this agree with the Kronecker product
  /// [`pullback`](Self::pullback) takes apart. Written by hand the two would
  /// have the same shape and different meanings, which no shape check catches.
  pub fn matrix(&self) -> Matrix {
    let strides = tensor_strides(&self.slots);
    let mut coeffs = Matrix::zeros(tensor_dim(&self.slots), self.dofs.len());
    for (j, &dof) in self.dofs.iter().enumerate() {
      for (blade, vertex, value) in self.column(dof) {
        coeffs[(blade.rank() * strides[0] + vertex * strides[1], j)] = value;
      }
    }
    coeffs
  }
}
