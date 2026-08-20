//! The local Whitney shape function and [`WhitneyExpansion`]: the pullback
//! computed on the factors agrees with the explicit Kronecker sandwich, the
//! Koszul contraction collapses to the boundary operator, and a Whitney
//! form is coclosed.

use approx::assert_relative_eq;
use derham::interpolate::form::{WhitneyExpansion, WhitneyLsf};
use metric::Metric;
use metric::tensor::inner;
use multialgebra::tensor::{factorwise_kronecker, tensor_strides};
use multialgebra::{Tensor, Variance, exterior_bases, exterior_dim};
use multiindex::{Dim, Sign, combinations, factorial_f64};
use simplicial::atlas::{BaryRef, SimplexQuadRule, unit_difbarys, unit_simplex_volume};
use simplicial::linalg::{Matrix, Vector};
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
      let product = factorwise_kronecker(&[blade.clone(), bary.clone()]);
      let expected = matrix.transpose() * product * &matrix;
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
/// At grade 0 the collapse is the augmentation onto the empty simplex,
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

      let strides = tensor_strides(expansion.slots());
      let collapsed = Matrix::from_fn(nblades, ndofs, |blade, dof| {
        (0..nvertices)
          .map(|vertex| matrix[(blade * strides[0] + vertex * strides[1], dof)])
          .sum()
      });

      let scale = factorial_f64(grade);
      let expected = if grade == 0 {
        Matrix::from_element(1, ndofs, scale)
      } else {
        scale * unit_boundary_operator(dim, grade)
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
