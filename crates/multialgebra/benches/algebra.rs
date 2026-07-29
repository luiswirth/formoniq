//! The inner loop of the monotone families, as a regression guard.
//!
//! The free family is inherently dense and is not the concern; what must not
//! drift is the cost of a wedge or a contraction on an alternating slot.

use divan::Bencher;
use multialgebra::{Factor, Tensor, Variance, Vector, tensor::uniform_slots};

fn main() {
  divan::main();
}

const DIMS: &[usize] = &[2, 3, 4];

fn probe(dim: usize, grade: usize, seed: usize) -> Tensor {
  let slots = uniform_slots([Factor::alternating(grade)], Variance::Covariant, dim);
  let len = multialgebra::exterior_dim(dim, grade);
  Tensor::new(
    slots,
    Vector::from_fn(len, |i, _| ((seed + 5 * i) % 7) as f64 - 3.0),
  )
}

#[divan::bench(args = DIMS)]
fn wedge(bencher: Bencher, dim: usize) {
  bencher.bench_local(|| {
    let mut total = 0.0;
    for a in 0..=dim {
      for b in 0..=(dim - a) {
        let (x, y) = (probe(dim, a, 1), probe(dim, b, 2));
        total += divan::black_box(x.wedge(&y)).components().sum();
      }
    }
    total
  });
}

#[divan::bench(args = DIMS)]
fn interior_product(bencher: Bencher, dim: usize) {
  let vector = Tensor::line(Vector::from_element(dim, 1.5), Variance::Contravariant);
  bencher.bench_local(|| {
    let mut total = 0.0;
    for grade in 1..=dim {
      let form = probe(dim, grade, 1);
      total += divan::black_box(form.interior_product(&vector))
        .components()
        .sum();
    }
    total
  });
}

#[divan::bench(args = DIMS)]
fn basis_sweep(bencher: Bencher, dim: usize) {
  bencher.bench_local(|| {
    let mut total = 0;
    for grade in 0..=dim {
      total += divan::black_box(Factor::alternating(grade).basis(dim).count());
    }
    total
  });
}
