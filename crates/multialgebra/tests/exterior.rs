//! The exterior-algebra laws, on a single alternating slot.

use approx::assert_relative_eq;
use multialgebra::tensor::Slots;
use multialgebra::tensor::{pairing, wedge_pairing};
use multialgebra::{
  Factor, Matrix, Slot, Symmetry, Tensor, Variance, Vector, exterior_bases, exterior_dim,
};
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

/// $iota_v$ is an antiderivation of degree $-1$.
#[test]
fn interior_product_antiderivation() {
  for dim in 2..=4 {
    let vector = Tensor::line(
      Vector::from_fn(dim, |i, _| (i + 1) as f64),
      Variance::Contravariant,
    );
    for grade_a in 1..dim {
      for grade_b in 1..=(dim - grade_a) {
        let alpha = probe_element(dim, grade_a, 5, Variance::Covariant);
        let beta = probe_element(dim, grade_b, 6, Variance::Covariant);

        let lhs = alpha.wedge(&beta).interior_product(&vector);
        let rhs = alpha.interior_product(&vector).wedge(&beta)
          + Sign::from_parity(grade_a).as_f64() * alpha.wedge(&beta.interior_product(&vector));
        assert_relative_eq!(lhs.components(), rhs.components(), epsilon = 1e-12);
      }
    }
  }
}

/// $iota_v compose iota_v = 0$.
#[test]
fn interior_product_squares_to_zero() {
  for dim in 2..=4 {
    let vector = Tensor::line(
      Vector::from_fn(dim, |i, _| (2 * i + 1) as f64),
      Variance::Contravariant,
    );
    for grade in 2..=dim {
      let element = probe_element(dim, grade, 7, Variance::Covariant);
      let twice = element.interior_product(&vector).interior_product(&vector);
      assert_relative_eq!(twice.components().norm(), 0.0);
    }
  }
}

/// Pullback and pushforward are adjoint under the duality pairing.
///
/// Over **both** families, which is what pins the two halves of the symmetric
/// multiplicity against each other: the pullback matrix is the transpose of the
/// functor conjugated by $alpha!$, and the pairing carries $alpha!$, and the law
/// holds exactly because the one cancels the other. Get either alone wrong and
/// this fails; on $Lambda^k$ both weights are $1$ and it says nothing about them.
#[test]
fn pullback_pushforward_adjoint() {
  for symmetry in [Symmetry::Alternating, Symmetry::Symmetric] {
    for (n, m) in [(2, 2), (3, 3), (3, 2), (4, 3)] {
      let a = probe_matrix(n, m, 4);
      for k in 0..=n.min(m) {
        let probe = |dim: usize, variance, seed: usize| {
          let slot = Slot::new(Factor::new(symmetry, k), variance, dim);
          Tensor::new(
            Slots::from_iter([slot]),
            Vector::from_fn(slot.multidim(), |i, _| ((seed + 5 * i) % 7) as f64 - 3.0),
          )
        };
        let form = probe(n, Variance::Covariant, 1);
        let vector = probe(m, Variance::Contravariant, 2);
        assert_relative_eq!(
          pairing(&form.pullback(&a), &vector),
          pairing(&form, &vector.pushforward(&a)),
          epsilon = 1e-12
        );
      }
    }
  }
}

/// The wedge is graded-antisymmetric.
#[test]
fn wedge_antisymmetry() {
  for dim in 2..=4 {
    for grade_a in 0..=dim {
      for grade_b in 0..=(dim - grade_a) {
        let a = probe_element(dim, grade_a, 8, Variance::Covariant);
        let b = probe_element(dim, grade_b, 9, Variance::Covariant);
        let sign = Sign::from_parity(grade_a * grade_b).as_f64();
        assert_relative_eq!(
          a.wedge(&b).components(),
          &(sign * b.wedge(&a)).components(),
          epsilon = 1e-12
        );
      }
    }
  }
}

/// The trivial ends: a grade off $[0, n]$ names the zero space, and
/// contracting out of grade zero lands there rather than trapping.
#[test]
fn the_trivial_ends_are_total() {
  for dim in 1..=4 {
    for grade in [-2, -1, dim as i64 + 1, dim as i64 + 2] {
      let grade = multialgebra::ExteriorGrade::new(grade);
      assert_eq!(exterior_dim(dim, grade), 0);
      assert_eq!(exterior_bases(dim, grade).count(), 0);
      let zero = Tensor::zero(Tensor::one_alternating(grade, Variance::Covariant, dim));
      assert_eq!(zero.components().len(), 0);
    }
    // A degree-zero *slot*, not the empty tensor product: contracting out of it
    // must land in the trivial space rather than underflow the slot list.
    let scalar = Tensor::multiform(Vector::from_element(1, 2.5), dim, 0);
    let vector = Tensor::line(Vector::from_element(dim, 1.5), Variance::Contravariant);
    let contracted = scalar.interior_product(&vector);
    assert_eq!(contracted.grade(), multialgebra::ExteriorGrade::new(-1));
    assert_eq!(contracted.components().len(), 0);

    // The empty tensor product is also the scalars, and is a different shape:
    // isomorphic, distinct in arity, with no slot to contract.
    assert_eq!(Tensor::scalar(2.5, dim).slots().len(), 0);
    assert_eq!(scalar.slots().len(), 1);
  }
}

/// An endomorphism $A: V -> V$ as a tensor of $V^* times.circle V$, with the
/// argument slot first.
fn endomorphism(matrix: &Matrix) -> Tensor {
  let n = matrix.nrows();
  assert_eq!(n, matrix.ncols());
  let slots = multialgebra::tensor::Slots::from(
    [
      multialgebra::Slot::new(multialgebra::Factor::alternating(1), Variance::Covariant, n),
      multialgebra::Slot::new(
        multialgebra::Factor::alternating(1),
        Variance::Contravariant,
        n,
      ),
    ]
    .as_slice(),
  );
  // Last slot fastest: the argument index is the outer one.
  Tensor::new(
    slots,
    Vector::from_fn(n * n, |flat, _| matrix[(flat % n, flat / n)]),
  )
}

/// The trace of an endomorphism is the matrix trace, and it takes no metric.
///
/// The operation the uniform-variance design could not express: reaching it
/// there meant raising an index through $g^(-1)$ first, so the answer depended
/// on a metric the concept does not need.
#[test]
fn the_trace_of_an_endomorphism_is_metric_free() {
  for dim in 1..=4 {
    let matrix = probe_matrix(dim, dim, 7);
    let tensor = endomorphism(&matrix);
    assert_eq!(tensor.variance(), None, "an endomorphism is mixed");

    let trace = tensor.trace(0, 1);
    assert_eq!(trace.slots().len(), 0, "both slots are consumed");
    assert_relative_eq!(trace.components()[0], matrix.trace(), epsilon = 1e-12);
  }
}

/// Composing two endomorphisms is a contraction: the tensor product traced over
/// the first's argument against the second's value.
///
/// Matrix multiplication as one instance of the general operation, which is
/// what folding maps into the algebra buys.
#[test]
fn composition_is_a_contraction() {
  for dim in 1..=4 {
    let (a, b) = (probe_matrix(dim, dim, 3), probe_matrix(dim, dim, 5));
    let composed = endomorphism(&a).contract_with(&endomorphism(&b), &[(0, 1)]);

    // The surviving slots are a's value and b's argument, in that order.
    assert_eq!(composed.slots().len(), 2);
    let product = &a * &b;
    for value in 0..dim {
      for argument in 0..dim {
        assert_relative_eq!(
          composed.components()[value * dim + argument],
          product[(value, argument)],
          epsilon = 1e-12
        );
      }
    }
  }
}

/// The duality pairing is the tensor product traced over every slot: the same
/// operation, run to completion.
#[test]
fn the_pairing_is_a_full_contraction() {
  for dim in 1..=4 {
    for grade in 0..=dim {
      let form = probe_element(dim, grade, 2, Variance::Covariant);
      let vector = probe_element(dim, grade, 3, Variance::Contravariant);

      let traced = form.contract_with(&vector, &[(0, 0)]);
      assert_eq!(traced.slots().len(), 0);
      assert_relative_eq!(
        traced.components()[0],
        pairing(&form, &vector),
        epsilon = 1e-12
      );
    }
  }
}

/// Multi-contraction is repeated tracing, so the order the pairs are given in
/// does not change the result.
#[test]
fn multi_contraction_is_order_independent() {
  for dim in 2..=3 {
    let (a, b) = (probe_matrix(dim, dim, 3), probe_matrix(dim, dim, 5));
    let (left, right) = (
      endomorphism(&a).tensor(&endomorphism(&b)),
      endomorphism(&b).tensor(&endomorphism(&a)),
    );

    let forward = left.contract_with(&right, &[(0, 1), (2, 3)]);
    let reversed = left.contract_with(&right, &[(2, 3), (0, 1)]);
    assert_relative_eq!(forward.components(), reversed.components(), epsilon = 1e-12);
    assert_eq!(forward.slots(), reversed.slots());
  }
}

/// The dense form of an alternating slot is antisymmetric, vanishes on a
/// repeated index, and reproduces the packed component at an increasing one.
///
/// The shape claim as well: one axis per unit of degree, C-order.
#[test]
fn the_dense_embedding_is_antisymmetric_on_an_alternating_slot() {
  let dim = 4;
  let form = probe_element(dim, 2, 5, Variance::Covariant);
  let dense = form.to_free();
  assert_eq!(dense.slots().len(), 1);
  assert_eq!(dense.slots()[0].degree(), 2);
  assert!(dense.slots()[0].symmetry().is_free());
  assert_eq!(dense.components().len(), dim * dim);

  let at = |i: usize, j: usize| dense.components()[i * dim + j];
  for i in 0..dim {
    assert_relative_eq!(at(i, i), 0.0, epsilon = 1e-12);
    for j in 0..dim {
      assert_relative_eq!(at(i, j), -at(j, i), epsilon = 1e-12);
    }
  }
  for (rank, blade) in exterior_bases(dim, 2).enumerate() {
    let (i, j) = (blade.iter().next().unwrap(), blade.iter().nth(1).unwrap());
    assert_relative_eq!(at(i, j), form.components()[rank], epsilon = 1e-12);
  }
}

/// A symmetric slot's dense form is symmetric, and a repeated index carries the
/// multiplicity rather than the bare coefficient.
#[test]
fn the_dense_embedding_carries_the_symmetric_multiplicity() {
  let dim = 3;
  let slots = multialgebra::tensor::uniform_slots(
    [multialgebra::Factor::symmetric(2)],
    Variance::Covariant,
    dim,
  );
  let len = multialgebra::tensor::tensor_dim(&slots);
  let tensor = Tensor::new(slots, Vector::from_fn(len, |i, _| (i + 1) as f64));
  let dense = tensor.to_free();

  let at = |i: usize, j: usize| dense.components()[i * dim + j];
  for i in 0..dim {
    for j in 0..dim {
      assert_relative_eq!(at(i, j), at(j, i), epsilon = 1e-12);
    }
  }
  // x^(2 e_0) occupies one word, so its two orderings coincide and sum.
  let square = multiindex::MonoIndex::from_word(multiindex::Repetition::Allowed, [0, 0]).unwrap();
  assert_relative_eq!(
    at(0, 0),
    2.0 * tensor.components()[square.1.rank()],
    epsilon = 1e-12
  );
}

/// The wedge pairing is graded-symmetric,
/// $angle.l beta, alpha angle.r = (-1)^(k(n-k)) angle.l alpha, beta angle.r$,
/// and nondegenerate.
///
/// Nondegeneracy is the content: it is what makes $Lambda^(n-k)$ the dual of
/// $Lambda^k$ with no inner product chosen, which is Poincare duality on the
/// algebra. Checked as the pairing matrix against the basis being invertible.
#[test]
fn the_wedge_pairing_is_graded_symmetric_and_nondegenerate() {
  for dim in 1..=4 {
    for grade in 0..=dim {
      let complement = dim - grade;
      let alpha = probe_element(dim, grade, 3, Variance::Covariant);
      let beta = probe_element(dim, complement, 5, Variance::Covariant);
      let sign = Sign::from_parity(grade * complement).as_f64();

      assert_relative_eq!(
        wedge_pairing(&beta, &alpha),
        sign * wedge_pairing(&alpha, &beta),
        epsilon = 1e-12
      );

      let (rows, cols) = (exterior_dim(dim, grade), exterior_dim(dim, complement));
      let matrix = Matrix::from_fn(rows, cols, |i, j| {
        let ei = Tensor::from_blade_signed(
          dim,
          Sign::Pos,
          exterior_bases(dim, grade).nth(i).unwrap(),
          Variance::Covariant,
        );
        let ej = Tensor::from_blade_signed(
          dim,
          Sign::Pos,
          exterior_bases(dim, complement).nth(j).unwrap(),
          Variance::Covariant,
        );
        wedge_pairing(&ei, &ej)
      });
      assert_eq!(rows, cols);
      assert!(
        matrix.determinant().abs() > 1e-12,
        "dim {dim} grade {grade}: the wedge pairing is degenerate"
      );
    }
  }
}

/// The full tensor power $V^(times.circle d)$ needs no symmetry of its own: it is
/// $d$ slots of degree one, since $Lambda^1 = "Sym"^1 = V$ and the tensor
/// product of $d$ copies of $V$ is what a $d$-slot tensor is.
///
/// Its basis is exactly the cartesian (radix) multi-index, and the strides are
/// the radix, so
/// [`cartesian::cartesian2linear`](multiindex::cartesian::cartesian2linear)
/// and [`Tensor::flat_index`] are one map. That is why a third `Symmetry` would
/// add arity bookkeeping rather than mathematics.
#[test]
fn the_full_tensor_power_is_slots_of_degree_one() {
  use multiindex::{MonoIndex, Repetition, cartesian};

  for dim in 1usize..=3 {
    for degree in 0usize..=3 {
      let slots = multialgebra::tensor::uniform_slots(
        std::iter::repeat_n(multialgebra::Factor::alternating(1), degree),
        Variance::Covariant,
        dim,
      );
      let tensor = Tensor::zero(slots);

      assert_eq!(
        tensor.components().len(),
        dim.pow(degree as u32),
        "dim {dim} degree {degree}: the tensor power has n^d components"
      );

      // Slot ranks are the symbols themselves at degree one, so a basis element
      // is a word, and the flat index is that word read as a radix number.
      for word in cartesian::grid(dim, degree) {
        let basis: Vec<multiindex::MultiIndex> = word
          .iter()
          .rev()
          .map(|&symbol| MonoIndex::single(Repetition::Forbidden, symbol).into())
          .collect();
        assert_eq!(
          tensor.flat_index(&basis),
          cartesian::cartesian2linear(&word, dim),
          "dim {dim} degree {degree}: strides are not the radix"
        );
      }
    }
  }
}

/// The free power is the tensor power: dimension `n^k`, no symmetry, and the
/// two quotients embed into it.
///
/// Λ^k and Sym^k are subspaces of V^⊗k, so forgetting the symmetry must not
/// change what the object *is*: the free form of a form has the same pairing
/// with the free form of a vector, up to the factorials the unnormalized
/// convention carries.
#[test]
fn the_free_power_is_the_unquotiented_one() {
  use multialgebra::{Factor, Symmetry, tensor::uniform_slots};

  for dim in 1usize..=3 {
    for degree in 0usize..=3 {
      let free = Factor::new(Symmetry::Free, degree);
      assert_eq!(
        free.multidim(dim),
        dim.pow(degree as u32),
        "dim {dim} degree {degree}: the free power has n^k components"
      );
      // No quotient means the two quotients are no larger.
      assert!(Factor::alternating(degree).multidim(dim) <= free.multidim(dim));
      assert!(Factor::symmetric(degree).multidim(dim) <= free.multidim(dim));

      let slots = uniform_slots([free], Variance::Covariant, dim);
      let tensor = Tensor::zero(slots);
      assert!(tensor.slots()[0].symmetry().is_free());
      assert!(!tensor.is_alternating() && !tensor.is_symmetric() || degree == 0);
    }
  }
}

/// Forgetting the symmetry is functorial: `to_free` commutes with pullback.
///
/// The embedding is a map of representations, not a serialization, so a map
/// acting on the quotient and then forgetting is the same as forgetting and
/// then acting. That is what makes it belong inside the algebra.
///
/// Swept over **both** families and over rectangular maps, which is what the law
/// actually claims. Restricted to $Lambda^k$ it is much weaker than it looks:
/// every $alpha!$ is $1$ there, so the pullback matrix is the bare transpose of
/// the functor and the multiplicity the symmetric basis carries never enters.
#[test]
fn forgetting_the_symmetry_commutes_with_pullback() {
  for symmetry in [Symmetry::Alternating, Symmetry::Symmetric] {
    for dim in 1..=3 {
      for degree in 0..=dim {
        for cols in 1..=dim {
          let slot = Slot::new(Factor::new(symmetry, degree), Variance::Covariant, dim);
          let form = Tensor::new(
            Slots::from_iter([slot]),
            Vector::from_fn(slot.multidim(), |i, _| ((3 * i + 2) % 7) as f64 - 3.0),
          );
          let map = probe_matrix(dim, cols, 4);

          assert_relative_eq!(
            form.pullback(&map).to_free().components(),
            form.to_free().pullback(&map).components(),
            epsilon = 1e-9
          );
        }
      }
    }
  }
}

/// Slots may be over different spaces, which is what a rectangular map is.
///
/// A linear map $A: V -> W$ is $V^* times.circle W$: one covariant slot over
/// the domain and one contravariant slot over the codomain. With a dimension
/// per tensor that shape was unrepresentable, so a map had to live outside the
/// algebra as a bare matrix.
#[test]
fn slots_may_be_over_different_spaces() {
  use multialgebra::{Factor, Slot, tensor::Slots};

  for (domain, codomain) in [(2, 3), (3, 2), (4, 4)] {
    let matrix = probe_matrix(codomain, domain, 7);
    let slots = Slots::from(
      [
        Slot::covariant(Factor::alternating(1), domain),
        Slot::contravariant(Factor::alternating(1), codomain),
      ]
      .as_slice(),
    );
    // Last slot fastest, so the codomain index runs inside the domain one.
    let map = Tensor::new(
      slots,
      Vector::from_fn(domain * codomain, |flat, _| {
        matrix[(flat % codomain, flat / codomain)]
      }),
    );

    assert_eq!(map.slots()[0].dim.index(), domain);
    assert_eq!(map.slots()[1].dim.index(), codomain);
    assert_eq!(map.uniform_dim().is_some(), domain == codomain);

    // The extents are per slot, exactly as the axes of a dense array.
    assert_eq!(map.slots()[0].multidim(), domain);
    assert_eq!(map.slots()[1].multidim(), codomain);
    assert_eq!(map.components().len(), domain * codomain);
  }
}

/// A tensor product may span spaces, and its extents are the two slots'.
#[test]
fn the_tensor_product_spans_spaces() {
  let form = probe_element(3, 2, 3, Variance::Covariant);
  let other = probe_element(4, 1, 5, Variance::Covariant);

  let joined = form.tensor(&other);
  assert_eq!(joined.slots().len(), 2);
  assert_eq!(joined.slots()[0].dim.index(), 3);
  assert_eq!(joined.slots()[1].dim.index(), 4);
  assert_eq!(joined.uniform_dim(), None);
  assert_eq!(
    joined.components().len(),
    form.components().len() * other.components().len()
  );
}

/// Dualizing a slot is an involution that leaves the components alone, and it
/// is what makes the duality pairing defined: a tensor pairs with its own
/// dualization, and that pairing is nondegenerate.
#[test]
fn dualizing_a_slot_is_a_relabelling() {
  for dim in 1..=4 {
    for grade in 0..=dim {
      let form = probe_element(dim, grade, 2, Variance::Covariant);
      let dualized = form.clone().dualize_slot(0);

      assert_eq!(dualized.slots()[0].variance, Variance::Contravariant);
      assert_eq!(dualized.components(), form.components());
      assert_eq!(dualized.clone().dualize_slot(0).slots(), form.slots());

      // Nondegenerate, which is what a relabelling has to buy to be worth
      // having: the pairing is the sum of the squared components.
      assert_relative_eq!(
        pairing(&form, &dualized),
        form.components().dot(form.components()),
        epsilon = 1e-12
      );
    }
  }
}

/// The blade of a frame is the wedge of its columns, and that identification is
/// what makes it functorial: pushing the frame forward along a map and taking
/// the blade is taking the blade and pushing it forward, $Lambda^k (A F) =
/// (Lambda^k A)(Lambda^k F)$, which is Cauchy-Binet read on one column.
///
/// Stated together with the fold over `wedge`, because the two being the same
/// object is the whole content: one is the definition, the other is what the
/// implementation computes.
#[test]
fn the_blade_of_a_frame_is_the_wedge_of_its_columns() {
  for dim in 1..=4 {
    for grade in 0..=dim {
      let frame = probe_matrix(dim, grade, 3);
      let blade = Tensor::blade_of(&frame, Variance::Contravariant);

      assert_eq!(blade.slots()[0].variance, Variance::Contravariant);
      assert_eq!(blade.components().len(), exterior_dim(dim, grade));

      // The fold: the empty wedge is the unit, so grade 0 is covered too.
      let folded =
        frame
          .column_iter()
          .fold(Tensor::one(dim).dualize_slot(0), |acc: Tensor, column| {
            acc.wedge(&Tensor::line(column.into_owned(), Variance::Contravariant))
          });
      assert_relative_eq!(blade.components(), folded.components(), epsilon = 1e-12);

      // Functoriality along a map of the underlying space.
      let map = probe_matrix(dim, dim, 5);
      let pushed_frame = Tensor::blade_of(&(&map * &frame), Variance::Contravariant);
      assert_relative_eq!(
        pushed_frame.components(),
        blade.pushforward(&map).components(),
        epsilon = 1e-10
      );
    }
  }
}

/// A blade vanishes exactly when its frame is dependent: that is the statement
/// that a decomposable blade *is* the subspace its frame spans, and it is the
/// half of the law that can fail --- an implementation returning zero always
/// would pass the vanishing side alone.
#[test]
fn a_blade_vanishes_exactly_on_a_dependent_frame() {
  for dim in 1..=4 {
    for grade in 1..=dim {
      let independent = probe_matrix(dim, grade, 3);
      assert!(
        Tensor::blade_of(&independent, Variance::Contravariant)
          .components()
          .norm()
          > 1e-9
      );

      // Repeat the first column: dependent as soon as there are two of them.
      let mut dependent = independent.clone();
      if grade >= 2 {
        let first = dependent.column(0).into_owned();
        dependent.set_column(1, &first);
        assert_relative_eq!(
          Tensor::blade_of(&dependent, Variance::Contravariant)
            .components()
            .norm(),
          0.0,
          epsilon = 1e-12
        );
      }
    }
  }
}
