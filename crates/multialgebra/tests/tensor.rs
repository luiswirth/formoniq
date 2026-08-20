//! The [`Tensor`] product algebra: the induced Gramian, the product, the
//! Koszul transfer and the exterior special case.

use approx::assert_relative_eq;
use multialgebra::tensor::{covariant_slots, factorwise_kronecker, tensor_dim, tensor_strides};
use multialgebra::{Degree, Factor, Matrix, Symmetry, Tensor, Vector};
use multiindex::{Sign, factorial};

fn probe(nrows: usize, ncols: usize, seed: usize) -> Matrix {
  Matrix::from_fn(nrows, ncols, |i, j| {
    ((7 * i + 3 * j + 5 * seed + 1) % 11) as f64 - 5.0
  })
}
/// A map of full column rank, so a pulled-back metric stays non-degenerate
/// and the law is tested on a metric rather than on a singular form.
fn probe_map(nrows: usize, ncols: usize, seed: usize) -> Matrix {
  Matrix::from_fn(nrows, ncols, |i, j| {
    ((seed + 3 * i + 7 * j) % 5) as f64 / 5.0 + if i == j { 1.0 } else { 0.0 }
  })
}
/// A symmetric positive definite matrix: a bilinear form, which is all the
/// induced form asks for.
fn probe_metric(dim: usize) -> Matrix {
  let a = probe(dim, dim, 2);
  a.transpose() * &a + Matrix::identity(dim, dim)
}

/// A homogeneous polynomial form $"Sym"^r times.circle Lambda^k$ with
/// deterministic components.
fn poly_form(dim: usize, r: usize, k: usize, seed: usize) -> Tensor {
  let factors = covariant_slots([Factor::symmetric(r), Factor::alternating(k)], dim);
  let len = tensor_dim(&factors);
  Tensor::new(
    factors,
    Vector::from_fn(len, |i, _| ((seed + 5 * i) % 7) as f64 - 3.0),
  )
}

/// The induced Gramian is the pullback of the Gramian: measuring the images
/// under $F(A)$ is measuring the originals in the pulled-back metric,
/// $"Gram"_F (A^T G A) = F(A)^T "Gram"_F (G) F(A)$.
///
/// This is the law that decides the normalization, and it is stated on a
/// rectangular map so the two ends cannot coincide and hide a factor. The
/// alternating side is the familiar Cauchy-Binet statement; the symmetric one
/// is its permanental counterpart, and it is what pins $"per"$ with nothing
/// in front of it against the $"per" \/ d!$ that the symmetrized-tensor
/// convention would give.
#[test]
fn the_induced_gramian_is_the_pullback_of_the_gramian() {
  for degree in 0..=3 {
    for symmetry in [Symmetry::Alternating, Symmetry::Symmetric] {
      for &(target, source) in &[(3, 2), (4, 3), (3, 3), (4, 2)] {
        let factor = Factor::new(symmetry, degree);
        let map = probe_map(target, source, 1);
        let metric = probe_metric(target);

        let pulled = map.transpose() * (&metric) * &map;
        let induced = factor.induced(&map);
        assert_relative_eq!(
          &factor.induced_form(&pulled),
          &(induced.transpose() * &factor.induced_form(&metric) * &induced),
          epsilon = 1e-7
        );
      }
    }
  }
}

/// Under a Euclidean metric the alternating basis is orthonormal and the
/// symmetric one orthogonal with $norm(x^alpha)^2 = alpha!$.
///
/// The multiplicity a repeated slot carries, which is exactly what the
/// alternating side lacks because it forbids the repetition. Not a loose
/// normalization: the two follow from the one convention, and the previous
/// law is what forces it.
#[test]
fn the_euclidean_gramian_reads_off_the_multiplicities() {
  for dim in 1..=4 {
    let euclidean: Matrix = Matrix::identity(dim, dim);
    for degree in 0..=3 {
      let alternating = Factor::alternating(degree);
      let identity = Matrix::identity(alternating.multidim(dim), alternating.multidim(dim));
      assert_relative_eq!(&alternating.induced_form(&euclidean), &identity);

      let symmetric = Factor::symmetric(degree);
      let gramian = symmetric.induced_form(&euclidean);
      for (i, index) in symmetric.basis(dim).enumerate() {
        for (j, other) in symmetric.basis(dim).enumerate() {
          let expected = if i == j {
            (0..dim)
              .map(|symbol| factorial(index.multiplicity(symbol)))
              .product::<usize>() as f64
          } else {
            0.0
          };
          assert_relative_eq!(gramian[(i, j)], expected);
          let _ = &other;
        }
      }
    }
  }
}

/// On a single factor the product is the merge of the tensor, which is what
/// ties the algebra structure back to the two primitives:
/// $a b = "merge"_0 (a times.circle b)$.
///
/// Stated where it is true. With one factor a side there is no reordering, so
/// no Koszul sign, and the factorwise product and the seam merge coincide;
/// with more factors they genuinely differ, and it is the factorwise one that
/// is the algebra.
#[test]
fn on_one_factor_the_product_is_a_merge_of_a_tensor() {
  let dim = 3;
  for symmetry in [Symmetry::Alternating, Symmetry::Symmetric] {
    for left_degree in 0..=2 {
      for right_degree in 0..=2 {
        let build = |degree, seed| {
          let factors = covariant_slots([Factor::new(symmetry, degree)], dim);
          let len = tensor_dim(&factors);
          Tensor::new(
            factors,
            Vector::from_fn(len, |i, _| ((seed + 5 * i) % 7) as f64 - 3.0),
          )
        };
        let (a, b) = (build(left_degree, 1), build(right_degree, 2));
        let fused = a.product(&b);
        let composed = a.tensor(&b).merge(0);
        assert_eq!(fused.slots(), composed.slots());
        assert_relative_eq!(fused.components(), composed.components(), epsilon = 1e-12);
      }
    }
  }
}

/// The product is graded-commutative in the Koszul sense:
/// $b a = (-1)^(abs(a) abs(b)) a b$, where the degree that counts is the
/// alternating one, symmetric factors being even.
///
/// Checked on a mixed shape, $"Sym" times.circle Lambda$, which is where the
/// sign is a real claim: on one factor it is the wedge's antisymmetry, and on
/// a purely symmetric shape it is plain commutativity, so neither alone
/// exercises the rule.
#[test]
fn the_product_is_koszul_graded_commutative() {
  let dim = 3;
  for left in 0..=2 {
    for right in 0..=2 {
      let a = poly_form(dim, 1, left, 1);
      let b = poly_form(dim, 2, right, 2);
      let sign = Sign::from_parity(left * right).as_f64();
      assert_relative_eq!(
        a.product(&b).components(),
        &(sign * b.product(&a)).components(),
        epsilon = 1e-12
      );
    }
  }
}

/// The product is associative, and the scalar tensor is its unit.
#[test]
fn the_product_is_an_associative_algebra() {
  let dim = 3;
  let unit = Tensor::new(
    covariant_slots([Factor::symmetric(0), Factor::alternating(0)], dim),
    Vector::from_element(1, 1.0),
  );
  let a = poly_form(dim, 1, 1, 3);
  assert_relative_eq!(a.product(&unit).components(), a.components());
  assert_relative_eq!(unit.product(&a).components(), a.components());

  let b = poly_form(dim, 1, 1, 4);
  let c = poly_form(dim, 2, 1, 5);
  assert_relative_eq!(
    a.product(&b).product(&c).components(),
    a.product(&b.product(&c)).components(),
    epsilon = 1e-9
  );
}

/// Transferring twice in the same direction vanishes, both ways round:
/// $dif compose dif = 0$ and $kappa compose kappa = 0$.
///
/// One law for two operators, which is the point of [`Tensor::transfer`],
/// they are the same operation in opposite directions, so nilpotency is one
/// statement about it rather than two coincidences.
#[test]
fn transferring_twice_in_one_direction_vanishes() {
  for dim in 1..=4 {
    for r in 0..=3 {
      for k in 0..=dim {
        let form = poly_form(dim, r, k, 1);
        // Sym -> Lambda twice: the exterior derivative.
        let twice = form.transfer(0, 1).transfer(0, 1);
        assert_relative_eq!(twice.components().amax(), 0.0, epsilon = 1e-12);
        // Lambda -> Sym twice: the Koszul operator.
        let twice = form.transfer(1, 0).transfer(1, 0);
        assert_relative_eq!(twice.components().amax(), 0.0, epsilon = 1e-12);
      }
    }
  }
}

/// The Koszul homotopy formula: on homogeneous $"Sym"^r times.circle
/// Lambda^k$, $dif kappa + kappa dif = (r + k) id$.
///
/// The identity the whole polynomial de Rham complex rests on: it is what
/// makes that complex exact, and hence what the trimmed spaces
/// $P^-_r Lambda^k$ are cut out by. Checking it here checks that both
/// directions of the transfer carry the right signs and the right
/// multiplicities, which no weaker law does: nilpotency alone passes on an
/// operator scaled by anything.
#[test]
fn the_koszul_homotopy_formula_holds() {
  for dim in 1..=4 {
    for r in 0..=3 {
      for k in 0..=dim {
        let form = poly_form(dim, r, k, 2);
        let dif_then_koszul = form.transfer(0, 1).transfer(1, 0);
        let koszul_then_dif = form.transfer(1, 0).transfer(0, 1);
        let sum = dif_then_koszul + koszul_then_dif;
        let expected = (r + k) as f64 * form.clone();
        assert_relative_eq!(sum.components(), expected.components(), epsilon = 1e-9);
        if r + k > 0 {
          assert!(
            expected.components().amax() > 0.0,
            "the law would hold vacuously"
          );
        }
      }
    }
  }
}

/// What the pullback law does and does not pin.
///
/// A global constant cancels: scaling the Gramian by any $c$ scales both
/// sides of $"Gram"(A^* g) = F(A)^* "Gram"(g)$ equally, so functoriality
/// alone leaves the overall factor free and cannot choose between $"per"$ and
/// $"per" \/ d!$. What it does pin is the shape, a normalization
/// depending on the multi-index, such as the $alpha!$ that
/// [`Factor::induced`] carries, breaks the law outright.
///
/// The remaining constant is fixed by siblinghood rather than by this law:
/// $Lambda$ and $"Sym"$ take the same construction, so they take the same
/// factor, and $Lambda$'s is one. Recording the distinction here because a
/// law that passes for a whole family of candidates is not evidence for any
/// one of them.
#[test]
fn the_pullback_law_pins_the_shape_and_not_the_constant() {
  let factor = Factor::symmetric(2);
  let (target, source) = (4, 3);
  let map = probe_map(target, source, 1);
  let metric = probe_metric(target);
  let pulled = map.transpose() * (&metric) * &map;
  let induced = factor.induced(&map);

  let holds = |rescale: &dyn Fn(&Factor, &Matrix) -> Matrix| {
    let lhs = rescale(&factor, &pulled);
    let rhs = rescale(&factor, &metric);
    (lhs - induced.transpose() * rhs * &induced).amax() < 1e-7
  };

  // Any global constant survives, so the law cannot choose one.
  for constant in [1.0, 0.5, 1.0 / 6.0] {
    assert!(holds(&move |factor: &Factor, g: &Matrix| {
      factor.induced_form(g) * constant
    }));
  }
  // A per-index factor does not.
  assert!(!holds(&|factor: &Factor, g: &Matrix| {
    let scale: Vec<f64> = factor
      .basis(g.nrows())
      .map(|index| {
        (0..g.nrows())
          .map(|symbol| factorial(index.multiplicity(symbol)))
          .product::<usize>() as f64
      })
      .collect();
    Matrix::from_fn(scale.len(), scale.len(), |i, j| {
      factor.induced_form(g)[(i, j)] / scale[i]
    })
  }));
}

/// The two families give the same Gramian at degrees zero and one, where
/// there is no symmetry to impose and $det$ and $"per"$ of a $1 times 1$
/// block agree.
#[test]
fn the_gramians_coincide_below_degree_two() {
  for dim in 1..=4 {
    let metric = probe_metric(dim);
    for degree in 0..=1 {
      assert_relative_eq!(
        &Factor::alternating(degree).induced_form(&metric),
        &Factor::symmetric(degree).induced_form(&metric)
      );
    }
  }
}

/// [`factorwise_kronecker`] lays a per-factor matrix out in the order the
/// strides imply: the entry at a pair of flat component indices is the
/// product of the per-factor entries at the ranks those indices decompose
/// into.
///
/// The law tying the two conventions together. Reversing the Kronecker order
/// produces a matrix of the same shape, symmetric and positive definite when
/// the operands are, so nothing but this catches it.
#[test]
fn factorwise_kronecker_follows_the_stride_order() {
  for dim in 1..=3 {
    let factors = [Factor::symmetric(2), Factor::alternating(1)];
    let strides = tensor_strides(&covariant_slots(factors, dim));
    let dims: Vec<usize> = factors.iter().map(|f| f.multidim(dim)).collect();

    let per_factor: Vec<Matrix> = dims
      .iter()
      .enumerate()
      .map(|(f, &d)| Matrix::from_fn(d, d, |i, j| (1 + f + 3 * i + 7 * j) as f64))
      .collect();
    let combined = factorwise_kronecker(&per_factor);

    assert_eq!(combined.nrows(), dims.iter().product::<usize>());
    for row_a in 0..dims[0] {
      for row_b in 0..dims[1] {
        for col_a in 0..dims[0] {
          for col_b in 0..dims[1] {
            let row = row_a * strides[0] + row_b * strides[1];
            let col = col_a * strides[0] + col_b * strides[1];
            assert_eq!(
              combined[(row, col)],
              per_factor[0][(row_a, col_a)] * per_factor[1][(row_b, col_b)],
              "dim {dim}: the kronecker order disagrees with the strides"
            );
          }
        }
      }
    }
  }
}

/// An element of the exterior algebra is a single alternating factor, which
/// is strictly stronger than being alternating in every factor: a tensor
/// product of two exterior elements is not one.
#[test]
fn exterior_is_single_slot_not_merely_alternating() {
  let dim = 3;
  let single: Tensor = Tensor::zero(covariant_slots([Factor::alternating(2)], dim));
  assert!(single.is_exterior());
  assert!(single.is_alternating());
  assert_eq!(single.single().map(|s| s.degree()), Some(Degree::from(2)));

  let paired: Tensor = Tensor::zero(covariant_slots(
    [Factor::alternating(1), Factor::alternating(1)],
    dim,
  ));
  assert!(paired.is_alternating());
  assert!(
    !paired.is_exterior(),
    "two slots are not the exterior algebra"
  );
  assert!(paired.single().is_none());

  let mixed: Tensor = Tensor::zero(covariant_slots(
    [Factor::symmetric(2), Factor::alternating(1)],
    dim,
  ));
  assert!(!mixed.is_alternating());
  assert!(!mixed.is_symmetric());
  assert!(!mixed.is_exterior());

  // The scalar is vacuously both, and is not a slot.
  let scalar = Tensor::scalar(1.0);
  assert!(scalar.is_alternating() && scalar.is_symmetric());
  assert!(scalar.single().is_none());
}
