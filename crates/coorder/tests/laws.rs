//! The laws of an affine space and of its maps: the action of the displacements
//! on the points, the affine combination they make well defined, and the
//! characterization of an affine map as one that preserves it.

use coorder::{Ambient, CoordSpace, Coords, Matrix, Vector, affine::AffineTransform};

enum Source {}
impl CoordSpace for Source {
  const NAME: &'static str = "source";
}
enum Target {}
impl CoordSpace for Target {
  const NAME: &'static str = "target";
}

fn close(a: &Vector, b: &Vector) {
  assert_eq!(a.len(), b.len());
  assert!((a - b).norm() < 1e-9, "{a:?} != {b:?}");
}

/// A deterministic full-column-rank `nrows x ncols` matrix (ncols <= nrows):
/// unit lower-triangular columns, echelon and hence injective.
fn full_col_rank(nrows: usize, ncols: usize) -> Matrix {
  Matrix::from_fn(nrows, ncols, |i, j| {
    if i == j {
      1.0
    } else if i > j {
      0.5
    } else {
      0.0
    }
  })
}

fn translation<S: CoordSpace>(dim: usize) -> Coords<S> {
  Coords::new(Vector::from_fn(dim, |i, _| 1.0 + i as f64))
}

/// The `k`-th of a family of distinct points of the given dimension.
fn point<S: CoordSpace>(dim: usize, k: usize) -> Coords<S> {
  Coords::new(Vector::from_fn(dim, |i, _| {
    2.0 - 0.3 * i as f64 + 1.7 * k as f64 * (1.0 + i as f64)
  }))
}

fn displacement(dim: usize, k: usize) -> Vector {
  Vector::from_fn(dim, |i, _| 0.4 * (1 + k) as f64 - 0.25 * i as f64)
}

fn probe(image: usize, domain: usize) -> AffineTransform<Source, Target> {
  AffineTransform::new(translation(image), full_col_rank(image, domain))
}

/// Weights summing to one, so an affine combination of `n` points is defined.
fn weights(n: usize) -> Vec<f64> {
  let raw: Vec<f64> = (0..n).map(|i| 1.0 + 0.5 * i as f64).collect();
  let total: f64 = raw.iter().sum();
  raw.iter().map(|w| w / total).collect()
}

/// The displacements act freely and transitively on the points: `p + v` moves
/// `p` by exactly `v`, and `q - p` is the unique displacement taking `p` to `q`.
#[test]
fn points_are_a_torsor_over_the_displacements() {
  for dim in 0..=4 {
    let p: Coords<Source> = point(dim, 0);
    let q: Coords<Source> = point(dim, 1);
    let v = displacement(dim, 0);
    let w = displacement(dim, 1);

    close(&(&(&p + &v) - &p), &v);
    close(&(&p + &(&q - &p)).into_vector(), q.vector());
    close(
      &(&(&p + &v) + &w).into_vector(),
      &(&p + &(&v + &w)).into_vector(),
    );
    close(&(&(&p + &v) - &v).into_vector(), p.vector());
  }
}

/// An affine combination is a point displaced from any one of its own points,
/// $sum_i lambda_i p_i = p_j + sum_i lambda_i (p_i - p_j)$, and the choice of
/// $p_j$ does not matter: that independence is what the weights summing to one
/// buys, and it is why the combination needs no origin.
#[test]
fn affine_combination_is_independent_of_the_base_point() {
  for dim in 0..=4 {
    for npoints in 1..=4 {
      let points: Vec<Coords<Source>> = (0..npoints).map(|k| point(dim, k)).collect();
      let weights = weights(npoints);
      let combination = Coords::affine_combination(weights.iter().copied().zip(points.iter()));

      for base in &points {
        let displacement: Vector = points
          .iter()
          .zip(&weights)
          .map(|(p, &w)| w * (p - base))
          .sum();
        close(&(base + &displacement).into_vector(), combination.vector());
      }
    }
  }
}

/// The barycenter is the affine combination with equal weights.
#[test]
fn barycenter_is_the_uniform_affine_combination() {
  for dim in 0..=4 {
    for npoints in 1..=4 {
      let points: Vec<Coords<Source>> = (0..npoints).map(|k| point(dim, k)).collect();
      let uniform = 1.0 / npoints as f64;
      close(
        Coords::barycenter(points.iter()).vector(),
        Coords::affine_combination(points.iter().map(|p| (uniform, p))).vector(),
      );
    }
  }
}

/// The affine combination of a single point is that point, whatever its one
/// weight must be.
#[test]
fn affine_combination_of_one_point_is_that_point() {
  for dim in 0..=4 {
    let p: Coords<Source> = point(dim, 0);
    close(Coords::affine_combination([(1.0, &p)]).vector(), p.vector());
    close(Coords::barycenter([&p]).vector(), p.vector());
  }
}

/// The characterization of an affine map: it commutes with affine combinations,
/// $T(sum_i lambda_i p_i) = sum_i lambda_i T(p_i)$. Linear maps satisfy this for
/// all weights, affine ones exactly for those summing to one.
#[test]
fn affine_maps_preserve_affine_combinations() {
  for image in 0..=4 {
    for domain in 0..=image {
      let t = probe(image, domain);
      for npoints in 1..=4 {
        let points: Vec<Coords<Source>> = (0..npoints).map(|k| point(domain, k)).collect();
        let weights = weights(npoints);

        let mapped: Vec<Coords<Target>> = points
          .iter()
          .map(|p| t.apply_forward(p.as_view()))
          .collect();

        close(
          t.apply_forward(
            Coords::affine_combination(weights.iter().copied().zip(points.iter())).as_view(),
          )
          .vector(),
          Coords::affine_combination(weights.iter().copied().zip(mapped.iter())).vector(),
        );
      }
    }
  }
}

/// An affine map is equivariant for the action: it translates a displaced point
/// by the pushforward of the displacement, $T(p + v) = T(p) + A v$.
#[test]
fn affine_maps_are_equivariant_for_the_action() {
  for image in 0..=4 {
    for domain in 0..=image {
      let t = probe(image, domain);
      let p: Coords<Source> = point(domain, 0);
      let v = displacement(domain, 0);
      close(
        t.apply_forward((&p + &v).as_view()).vector(),
        (&t.apply_forward(p.as_view()) + &(&t.linear * &v)).vector(),
      );
    }
  }
}

/// On an injective map, the pseudo-inverse is a left inverse.
#[test]
fn pseudo_inverse_is_a_left_inverse() {
  for image in 0..=4 {
    for domain in 0..=image {
      let t = probe(image, domain);
      let x: Coords<Source> = point(domain, 0);
      let y = t.apply_forward(x.as_view());
      close(
        t.pseudo_inverse().apply_forward(y.as_view()).vector(),
        x.vector(),
      );
    }
  }
}

/// For a square invertible map the pseudo-inverse is two-sided: composing the
/// two gives the identity.
#[test]
fn pseudo_inverse_of_a_bijection_is_two_sided() {
  for dim in 0..=4 {
    let t = probe(dim, dim);
    let inv = t.pseudo_inverse();
    let x: Coords<Source> = point(dim, 0);
    close(
      inv
        .apply_forward(t.apply_forward(x.as_view()).as_view())
        .vector(),
      x.vector(),
    );

    let composed = t.then(&inv);
    let identity = AffineTransform::<Source, Source>::identity(dim);
    close(
      &(&composed.linear * x.vector()),
      &(&identity.linear * x.vector()),
    );
    close(composed.translation.vector(), identity.translation.vector());
  }
}

/// Composition is functorial and the identity is its unit: applying the
/// composite is applying the parts in order. The shared middle space is what the
/// type parameters enforce.
#[test]
fn composition_is_application_in_order() {
  for outer in 1..=4 {
    for mid in 1..=outer {
      for inner in 1..=mid {
        let first: AffineTransform<Source, Ambient> =
          AffineTransform::new(translation(mid), full_col_rank(mid, inner));
        let second: AffineTransform<Ambient, Target> =
          AffineTransform::new(translation(outer), full_col_rank(outer, mid));
        let x: Coords<Source> = point(inner, 0);

        let composed = first.then(&second);
        close(
          composed.apply_forward(x.as_view()).vector(),
          second
            .apply_forward(first.apply_forward(x.as_view()).as_view())
            .vector(),
        );

        let with_identity = first.then(&AffineTransform::<Ambient, Ambient>::identity(mid));
        close(
          with_identity.apply_forward(x.as_view()).vector(),
          first.apply_forward(x.as_view()).vector(),
        );
      }
    }
  }
}

/// Degenerate domain: a map out of $RR^0$ is the constant `b`, and its
/// pseudo-inverse lands in the empty space. Total, no panic.
#[test]
fn zero_dimensional_domain_is_total() {
  let t = probe(3, 0);
  assert_eq!(
    t.apply_forward(Coords::zeros(0).as_view()).vector(),
    translation::<Target>(3).vector()
  );
  let inv = t.pseudo_inverse();
  assert_eq!(inv.dim_domain(), 3);
  assert_eq!(inv.dim_image(), 0);
  assert_eq!(inv.apply_forward(point::<Target>(3, 0).as_view()).dim(), 0);
}
