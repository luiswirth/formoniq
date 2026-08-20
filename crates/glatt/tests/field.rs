//! [`DiffFormClosure`]/[`FieldClosure`]: each constructor evaluates
//! pointwise to the exterior element it promises.

use approx::assert_relative_eq;
use coorder::{Coord, Coords, Vector};
use glatt::field::{CoordField, DiffFormClosure, FieldClosure};

// The pullback naturality law for `CoordField` — the field pulled onto a mesh
// through a cell parametrization — lives in `derham`, the crate that joins
// `glatt` to `simplicial` (see `derham::section`). What is testable here is
// that each constructor evaluates to the exterior element it promises: the
// pointwise law from which that naturality is built.

fn point(entries: &[f64]) -> Coord {
  Coords::new(Vector::from_column_slice(entries))
}

/// A scalar field is grade 0, and evaluates to its closure's value.
#[test]
fn scalar_evaluates_pointwise() {
  for dim in 0..=4 {
    let f = DiffFormClosure::scalar(|x| x.iter().sum(), dim);
    assert_eq!(f.grade(), 0);
    assert_eq!(f.dim(), dim);
    let x = point(&vec![1.5; dim]);
    let value = f.at(&x);
    assert_eq!(value.grade(), 0);
    assert_relative_eq!(value.as_scalar(), 1.5 * dim as f64);
  }
}

/// `coord_component` extracts $x_i$, and `constant_scalar` ignores its
/// argument.
#[test]
fn distinguished_scalar_fields() {
  let dim = 3;
  let x = point(&[2.0, -1.0, 4.0]);
  for i in 0..dim {
    let f = DiffFormClosure::coord_component(i, dim);
    assert_relative_eq!(f.at(&x).as_scalar(), x[i]);
  }
  let c = DiffFormClosure::constant_scalar(7.0, dim);
  assert_relative_eq!(c.at(&x).as_scalar(), 7.0);
}

/// The radial field is the Euclidean distance to its center.
#[test]
fn radial_scalar_is_distance() {
  let dim = 2;
  let center = point(&[1.0, 1.0]);
  let f = DiffFormClosure::radial_scalar(center.clone(), dim);
  let x = point(&[4.0, 5.0]);
  assert_relative_eq!(f.at(&x).as_scalar(), 5.0); // 3-4-5 triangle
  assert_relative_eq!(f.at(&center).as_scalar(), 0.0);
}

/// A grade-1 field carries its coefficient vector verbatim, at either
/// variance: `one_form` as a covector, `vector_field` as a vector.
#[test]
fn grade_one_fields_carry_their_coefficients() {
  let dim = 3;
  let x = point(&[0.0, 0.0, 0.0]);
  let coeffs = Vector::from_column_slice(&[1.0, 2.0, 3.0]);

  let omega = DiffFormClosure::one_form(
    {
      let c = coeffs.clone();
      move |_| c.clone()
    },
    dim,
  );
  assert_eq!(omega.grade(), 1);
  assert_eq!(omega.at(&x).components(), &coeffs);

  let v = FieldClosure::vector_field(
    {
      let c = coeffs.clone();
      move |_| c.clone()
    },
    dim,
  );
  assert_eq!(v.grade(), 1);
  assert_eq!(v.at(&x).components(), &coeffs);
}
