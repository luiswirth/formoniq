//! Manufactured data shared by the Hodge-Laplace examples.
//!
//! Included with `#[path = "util/mod.rs"] mod util;`; it is not itself an
//! example binary (only files directly under `examples/` are).

use {
  common::{
    combo::{Combination, Sign},
    coord::Coord,
  },
  continuum::field::DiffFormClosure,
  exterior::ExteriorElement,
};

/// The manufactured Hodge-Laplace eigenform on the flat box $[0, pi]^n$,
/// satisfying the natural (absolute) boundary conditions of the mixed weak form.
///
/// $u = f dif x^I$ with $I = {1, ..., k}$ the colex-first grade-$k$ blade and
/// $ f = product_(i in I) sin x_i product_(i in.not I) cos x_i, $
/// so each coordinate carries a Dirichlet factor ($sin$) where it is tangential
/// to $dif x^I$ and a Neumann factor ($cos$) where it is normal. This is exactly
/// what the absolute boundary conditions $iota_n u = 0$, $iota_n dif u = 0$
/// demand face by face.
///
/// On flat space the Hodge Laplacian acts diagonally on Cartesian components and
/// each factor is a unit-frequency eigenfunction, so $Delta u = n u$ exactly —
/// for every dimension and every grade, with no per-grade hand derivation. The
/// source problem's exact solution is therefore an eigenform: its load is
/// $Delta u = n u$, and on the contractible box the only harmonic form is the
/// grade-0 constant, to which $u$ is $L^2$-orthogonal.
pub struct BoxEigenform {
  pub dim: usize,
  pub grade: usize,
}

impl BoxEigenform {
  pub fn new(dim: usize, grade: usize) -> Self {
    assert!(grade <= dim);
    Self { dim, grade }
  }

  /// The eigenvalue $lambda = n$ of $Delta u = lambda u$.
  pub fn eigenvalue(&self) -> f64 {
    self.dim as f64
  }

  /// The exact solution $u = f dif x^I$.
  pub fn solution(&self) -> DiffFormClosure {
    let (dim, grade) = (self.dim, self.grade);
    let blade = Combination::from_increasing(0..grade);
    DiffFormClosure::new(
      move |p: &Coord| bump(p, grade) * ExteriorElement::from_blade_signed(dim, Sign::Pos, blade),
      dim,
      grade,
    )
  }

  /// The exact source $Delta u = lambda u$.
  pub fn load(&self) -> DiffFormClosure {
    let (dim, grade) = (self.dim, self.grade);
    let lambda = self.eigenvalue();
    let blade = Combination::from_increasing(0..grade);
    DiffFormClosure::new(
      move |p: &Coord| {
        (lambda * bump(p, grade)) * ExteriorElement::from_blade_signed(dim, Sign::Pos, blade)
      },
      dim,
      grade,
    )
  }

  /// The exterior derivative $dif u = sum_j (diff_j f) dif x^j and dif x^I$, or
  /// `None` at top grade, where $dif u$ vanishes identically.
  pub fn dif_solution(&self) -> Option<DiffFormClosure> {
    if self.grade == self.dim {
      return None;
    }
    let (dim, grade) = (self.dim, self.grade);
    let blade = Combination::from_increasing(0..grade);
    Some(DiffFormClosure::new(
      move |p: &Coord| {
        let e_blade = ExteriorElement::from_blade_signed(dim, Sign::Pos, blade);
        // Terms with $j in I$ drop out: the wedge of a repeated index is zero.
        (0..dim)
          .map(|j| {
            let e_j =
              ExteriorElement::from_blade_signed(dim, Sign::Pos, Combination::from_increasing([j]));
            bump_partial(p, grade, j) * e_j.wedge(&e_blade)
          })
          .sum()
      },
      dim,
      grade + 1,
    ))
  }
}

/// $f(x)$: $sin x_i$ where $i$ is tangential to $dif x^I$ ($i < k$), $cos x_i$
/// where normal.
fn bump(p: &Coord, grade: usize) -> f64 {
  p.iter()
    .enumerate()
    .map(|(i, &x)| if i < grade { x.sin() } else { x.cos() })
    .product()
}

/// $diff_j f$, the derivative in the $j$-th coordinate.
fn bump_partial(p: &Coord, grade: usize, j: usize) -> f64 {
  p.iter()
    .enumerate()
    .map(|(i, &x)| {
      if i == j {
        if i < grade {
          x.cos()
        } else {
          -x.sin()
        }
      } else if i < grade {
        x.sin()
      } else {
        x.cos()
      }
    })
    .product()
}
