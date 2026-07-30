//! The metric-dependent laws of the exterior algebra.
//!
//! They live here rather than in `multialgebra` because they need a metric, and
//! that is the whole content of the crate split: the wedge and the contraction
//! are stated one crate down, where no metric exists to leak in.

use approx::assert_relative_eq;
use metric::Metric;
use metric::tensor::{TensorExt, inner, multiform_metric, multivector_metric, tensor_gramian};
use multialgebra::tensor::Slots;
use multialgebra::tensor::{pairing, wedge_pairing};
use multialgebra::{Factor, Matrix, Slot, Tensor, Variance, Vector, exterior_bases, exterior_dim};
use multiindex::Sign;

fn probe_matrix(nrows: usize, ncols: usize, seed: usize) -> Matrix {
  Matrix::from_fn(nrows, ncols, |i, j| {
    ((seed + 3 * i + 7 * j) % 5) as f64 / 5.0 + if i == j { 1.0 } else { 0.0 }
  })
}

fn probe_element(dim: usize, grade: usize, seed: usize, variance: Variance) -> Tensor {
  Tensor::new(
    Tensor::one_alternating(grade, variance, dim),
    Vector::from_fn(exterior_dim(dim, grade), |i, _| {
      ((seed + 5 * i) % 7) as f64 - 3.0
    }),
  )
}

fn probe_metric(dim: usize) -> Metric {
  let a = probe_matrix(dim, dim, 5);
  Metric::new(
    Variance::Covariant,
    a.transpose() * &a + Matrix::identity(dim, dim),
  )
}
fn probe_pseudo_metric(dim: usize, q: usize) -> Metric {
  let j = Matrix::from_fn(dim, dim, |i, jj| {
    if i == jj {
      1.0
    } else if i > jj {
      ((3 * i + 5 * jj) % 4) as f64 / 8.0
    } else {
      0.0
    }
  });
  Metric::pseudo_euclidean(dim - q, q).pullback(&j)
}

/// $star star = (-1)^(k(n-k)) (-1)^q$ on any signature, for both variances.
#[test]
fn hodge_star_involution() {
  for dim in 1..=4 {
    for q in 0..=dim {
      for metric in [
        Metric::pseudo_euclidean(dim - q, q),
        probe_pseudo_metric(dim, q),
      ] {
        for grade in 0..=dim {
          let sign = Sign::from_parity(grade * (dim - grade)) * Sign::from_parity(q);

          let form = probe_element(dim, grade, 2, Variance::Covariant);
          let twice = form.star(&metric, Sign::Pos).star(&metric, Sign::Pos);
          assert_relative_eq!(
            twice.components(),
            &(sign.as_f64() * form).components(),
            epsilon = 1e-12
          );

          let vector = probe_element(dim, grade, 3, Variance::Contravariant);
          let twice = vector.star(&metric, Sign::Pos).star(&metric, Sign::Pos);
          assert_relative_eq!(
            twice.components(),
            &(sign.as_f64() * vector).components(),
            epsilon = 1e-12
          );
        }
      }
    }
  }
}

/// $alpha wedge star beta = inner(alpha, beta) vol$: the defining property,
/// tying wedge, inner product and star together on every signature.
#[test]
fn wedge_with_star_is_inner_times_volume() {
  for dim in 1..=4 {
    for q in 0..=dim {
      for metric in [
        Metric::pseudo_euclidean(dim - q, q),
        probe_pseudo_metric(dim, q),
      ] {
        for grade in 0..=dim {
          let alpha = probe_element(dim, grade, 3, Variance::Covariant);
          let beta = probe_element(dim, grade, 4, Variance::Covariant);
          let wedge = alpha.wedge(&beta.star(&metric, Sign::Pos));
          assert_eq!(wedge.grade(), dim);
          assert_relative_eq!(
            wedge[0],
            inner(&alpha, &beta, &metric) * metric.det_sqrt(),
            epsilon = 1e-12
          );
        }
      }
    }
  }
}

/// For the Euclidean metric the star is the signed complement of each blade.
#[test]
fn hodge_star_euclidean_is_signed_complement() {
  for dim in 1..=4 {
    let euclidean = Metric::euclidean(dim);
    for grade in 0..=dim {
      for blade in exterior_bases(dim, grade) {
        let element = Tensor::from_blade_signed(dim, Sign::Pos, blade, Variance::Covariant);
        let star = element.star(&euclidean, Sign::Pos);
        let (sign, complement) = blade.complement_signed(dim);
        let expected = Tensor::from_blade_signed(dim, sign, complement, Variance::Covariant);
        assert_relative_eq!(star.components(), expected.components());
      }
    }
  }
}

/// The musical isomorphisms are inverse and turn the pairing into the inner
/// product.
#[test]
fn musical_isomorphisms() {
  for dim in 1..=4 {
    let metric = probe_metric(dim);
    for grade in 0..=dim {
      let v = probe_element(dim, grade, 1, Variance::Contravariant);
      let w = probe_element(dim, grade, 2, Variance::Contravariant);
      assert_relative_eq!(
        v.musical(&metric).musical(&metric).components(),
        v.components(),
        epsilon = 1e-12
      );
      assert_relative_eq!(
        pairing(&v.musical(&metric), &w),
        inner(&v, &w, &metric),
        epsilon = 1e-12
      );
    }
  }
}

/// Writing a tensor out on the full basis of $V^(times.circle d)$ and taking
/// the Euclidean dot product there is the packed inner product, scaled by
/// $product_i k_i !$.
///
/// One law over both families, and it is what says the dense embedding and
/// `Factor::induced_form` agree on the normalization. The alternating basis is
/// orthonormal while the symmetric one has $norm(x^alpha)^2 = alpha!$, and both
/// come out right only because the embedding is unnormalized in the same sense
/// the Gramian is: a mismatch shows up here as a factorial.
#[test]
fn the_dense_embedding_matches_the_gramian_up_to_the_factorials() {
  for dim in 1..=3 {
    let metric = Metric::euclidean(dim);
    for degree in 0..=3 {
      for symmetry in [
        multialgebra::Symmetry::Alternating,
        multialgebra::Symmetry::Symmetric,
      ] {
        let slots = multialgebra::tensor::uniform_slots(
          [multialgebra::Factor::new(symmetry, degree)],
          Variance::Covariant,
          dim,
        );
        let len = multialgebra::tensor::tensor_dim(&slots);
        if len == 0 {
          continue;
        }
        let build = |seed: usize| {
          Tensor::new(
            slots.clone(),
            Vector::from_fn(len, |i, _| ((seed + 5 * i) % 7) as f64 - 3.0),
          )
        };
        let (a, b) = (build(2), build(3));
        let scale = multiindex::factorial(degree) as f64;

        assert_relative_eq!(
          a.to_free().components().dot(b.to_free().components()),
          scale * inner(&a, &b, &metric),
          epsilon = 1e-10
        );
      }
    }
  }
}

/// The three dualities of the exterior algebra, and what each of them needs.
///
/// The wedge pairing is metric-free: it asks only for a top grade to land in.
/// The Hodge star is what turns it into the inner product,
/// $angle.l alpha, star beta angle.r_wedge = inner(alpha, beta) vol$, and that
/// step is exactly where the metric enters. Stating the two together is what
/// keeps them from being conflated.
#[test]
fn the_star_is_what_turns_the_wedge_pairing_into_the_inner_product() {
  for dim in 1..=4 {
    for q in 0..=dim {
      let metric = Metric::pseudo_euclidean(dim - q, q);
      let volume = metric.det_sqrt();
      for grade in 0..=dim {
        let alpha = probe_element(dim, grade, 3, Variance::Covariant);
        let beta = probe_element(dim, grade, 4, Variance::Covariant);

        assert_relative_eq!(
          wedge_pairing(&alpha, &beta.star(&metric, Sign::Pos)),
          inner(&alpha, &beta, &metric) * volume,
          epsilon = 1e-10
        );
      }
    }
  }
}

/// A free slot has no Hodge star: there is a top degree only after a quotient
/// by the sign character, and the free power has none.
#[test]
#[should_panic(expected = "no top degree")]
fn a_free_slot_has_no_hodge_star() {
  use multialgebra::{Factor, Symmetry, tensor::uniform_slots};

  let dim = 3;
  let slots = uniform_slots([Factor::new(Symmetry::Free, 2)], Variance::Covariant, dim);
  let tensor = Tensor::zero(slots);
  let _ = tensor.star(&Metric::euclidean(dim), Sign::Pos);
}

/// A metric is a $"Sym"^2$ element: the two readings round-trip, in both
/// variances and at every dimension including the empty one.
///
/// The probe is deliberately not symmetric-looking, distinct off-diagonal
/// entries and a distinct diagonal, because a wrong symmetric multiplicity is
/// exactly the error a probe with equal entries hides.
#[test]
fn a_metric_round_trips_through_its_sym2_tensor() {
  for dim in 0..=4 {
    for metric in [probe_metric(dim), probe_metric(dim).dual()] {
      let tensor = metric.tensor();
      assert_eq!(tensor.slots().len(), 1);
      assert_eq!(tensor.slots()[0].variance, metric.variance());
      assert_eq!(tensor.slots()[0].dim.index(), dim);
      // Packed: the upper triangle, not the full square.
      assert_eq!(tensor.components().len(), dim * (dim + 1) / 2);

      let back = Metric::from_tensor(&tensor);
      assert_eq!(back.variance(), metric.variance());
      assert_relative_eq!(back.matrix(), metric.matrix(), epsilon = 1e-12);
    }
  }
}

/// The multiplicity is real: the packed components are not the matrix entries,
/// differing by $alpha!$ on the diagonal. Without this the round-trip law above
/// would also pass on an implementation that ignored the convention entirely.
#[test]
fn the_sym2_components_carry_the_multiplicity() {
  for dim in 1..=4 {
    let metric = probe_metric(dim);
    let tensor = metric.tensor();

    // The diagonal of the free (dense) form is twice the packed component: both
    // orderings of a repeated symbol land on it and are summed.
    let free = tensor.to_free();
    for i in 0..dim {
      let packed = tensor.components()[tensor.flat_index(&[multiindex::MultiIndex::Mono(
        multiindex::MonoIndex::new(multiindex::Repetition::Allowed, [i, i]),
      )])];
      assert_relative_eq!(metric.matrix()[(i, i)], 2.0 * packed, epsilon = 1e-12);
      assert_relative_eq!(
        free.components()[free.flat_index(&[multiindex::MultiIndex::Word(multiindex::Word::new(
          dim,
          [i, i]
        ))])],
        metric.matrix()[(i, i)],
        epsilon = 1e-12
      );
    }
  }
}

/// The metric pullback is `Tensor::pullback` on the $"Sym"^2$ reading: one
/// functor, not a hand-rolled $J^top g J$ beside it.
#[test]
fn the_metric_pullback_is_the_tensor_pullback() {
  for dim in 1..=4 {
    let metric = probe_metric(dim);
    let jacobian = probe_matrix(dim, dim, 3);

    let by_matrix = metric.pullback(&jacobian);
    let by_functor = Metric::from_tensor(&metric.tensor().pullback(&jacobian));

    assert_relative_eq!(by_matrix.matrix(), by_functor.matrix(), epsilon = 1e-12);
  }
}

/// The metric evaluated on two vectors is the duality pairing of its $"Sym"^2$
/// reading against their symmetric product, $g(v, w) = angle.l g, v dot.circle w
/// angle.r$: the multiplicity cancels between the two sides exactly when the
/// convention is uniform. Checked with $v != w$, where a stray factor of two
/// survives.
#[test]
fn the_metric_is_the_pairing_against_a_symmetric_product() {
  for dim in 1..=4 {
    let metric = probe_metric(dim);
    let v = Vector::from_fn(dim, |i, _| ((3 * i + 1) % 5) as f64 - 2.0);
    let w = Vector::from_fn(dim, |i, _| ((7 * i + 2) % 5) as f64 - 1.0);

    // Degree-one symmetric slots, so the product lands in Sym^2 rather than
    // Lambda^2: at degree one the two families agree as spaces and differ only
    // in which quotient the product then takes.
    let as_slot = |x: &Vector| {
      Tensor::new(
        Slots::from_iter([Slot::new(
          Factor::symmetric(1),
          Variance::Contravariant,
          dim,
        )]),
        x.clone(),
      )
    };
    let product = as_slot(&v).product(&as_slot(&w));

    assert_relative_eq!(
      pairing(&metric.tensor(), &product),
      metric.inner(&v, &w),
      epsilon = 1e-12
    );
  }
}

/// The Gram matrix of a tensor's own basis measures each slot by its own
/// variance, so a mixed tensor draws on g and g⁻¹ at once and no single
/// variance describes the result. That is why it is a bare matrix and not a
/// `Metric`: a constructor supplying one would be guessing.
///
/// On a uniform tensor it agrees with the named per-family metric, and the two
/// sides of that comparison are matrices, so nothing claims a variance.
#[test]
fn the_tensor_gramian_measures_each_slot_by_its_own_variance() {
  for dim in 1..=3 {
    let g = probe_metric(dim);
    for grade in 0..=dim {
      let form = Tensor::one_alternating(grade, Variance::Covariant, dim);
      let vect = Tensor::one_alternating(grade, Variance::Contravariant, dim);

      assert_relative_eq!(
        &tensor_gramian(&form, &g),
        multiform_metric(&g, grade).matrix(),
        epsilon = 1e-12
      );
      assert_relative_eq!(
        &tensor_gramian(&vect, &g),
        multivector_metric(&g, grade).matrix(),
        epsilon = 1e-12
      );

      // The mixed shape: one covariant slot and one contravariant, the shape an
      // endomorphism has. Its Gramian is the Kronecker product of the two, so
      // both g and g^-1 appear and neither variance is the answer.
      let mixed = Slots::from_iter([
        Slot::new(Factor::alternating(grade), Variance::Covariant, dim),
        Slot::new(Factor::alternating(grade), Variance::Contravariant, dim),
      ]);
      let expected = multiform_metric(&g, grade)
        .matrix()
        .kronecker(multivector_metric(&g, grade).matrix());
      assert_relative_eq!(&tensor_gramian(&mixed, &g), &expected, epsilon = 1e-12);

      // And the two factors genuinely differ, or the law says nothing.
      if grade > 0 && dim > 1 {
        assert!(
          (multiform_metric(&g, grade).matrix() - multivector_metric(&g, grade).matrix()).amax()
            > 1e-9,
          "g and g^-1 must differ, or a single variance would do"
        );
      }
    }
  }
}
