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

/// Which boundary condition the manufactured eigenform meets: the two dual
/// Hodge-Laplace problems on the box.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum BoundaryCondition {
  /// Natural / absolute: $iota_n u = 0$, $iota_n dif u = 0$. The full Whitney
  /// complex; harmonic space $H^k (K)$ (box: grade $0$).
  Absolute,
  /// Essential / relative: $"tr" u = 0$, $"tr" dif u = 0$. The relative Whitney
  /// complex; harmonic space $H^k (K, diff K)$ (box: top grade).
  Relative,
}

/// The manufactured Hodge-Laplace eigenform on the flat box $[0, pi]^n$,
/// satisfying either the absolute or the relative boundary conditions of the
/// mixed weak form.
///
/// $u = f dif x^I$ with $I = {1, ..., k}$ the colex-first grade-$k$ blade and,
/// for the absolute problem,
/// $ f = product_(i in I) sin x_i product_(i in.not I) cos x_i, $
/// so each coordinate carries a $sin$ factor where it is tangential to $dif x^I$
/// and a $cos$ factor where it is normal --- exactly what $iota_n u = 0$,
/// $iota_n dif u = 0$ demand face by face. The relative problem is the Hodge
/// dual: it swaps $sin arrow.l.r cos$, so $f = product_(i in I) cos x_i
/// product_(i in.not I) sin x_i$ and $"tr" u = 0$ holds face by face. At grade
/// $0$ this recovers the textbook scalar cases --- absolute is the Neumann
/// eigenfunction $product cos x_i$, relative the Dirichlet $product sin x_i$.
///
/// On flat space the Hodge Laplacian acts diagonally on Cartesian components and
/// each factor is a unit-frequency eigenfunction, so $Delta u = n u$ exactly —
/// for every dimension, grade and boundary condition, with no per-case hand
/// derivation. The source problem's exact solution is therefore an eigenform:
/// its load is $Delta u = n u$, and $u$ is $L^2$-orthogonal to the (dual)
/// harmonic space.
pub struct BoxEigenform {
  pub dim: usize,
  pub grade: usize,
  pub bc: BoundaryCondition,
}

impl BoxEigenform {
  pub fn new(dim: usize, grade: usize, bc: BoundaryCondition) -> Self {
    assert!(grade <= dim);
    Self { dim, grade, bc }
  }

  /// The eigenvalue $lambda = n$ of $Delta u = lambda u$.
  pub fn eigenvalue(&self) -> f64 {
    self.dim as f64
  }

  /// The exact solution $u = f dif x^I$.
  pub fn solution(&self) -> DiffFormClosure {
    let (dim, grade, relative) = (self.dim, self.grade, self.relative());
    let blade = Combination::from_increasing(0..grade);
    DiffFormClosure::new(
      move |p: &Coord| {
        bump(p, grade, relative) * ExteriorElement::from_blade_signed(dim, Sign::Pos, blade)
      },
      dim,
      grade,
    )
  }

  /// The exact source $Delta u = lambda u$.
  pub fn load(&self) -> DiffFormClosure {
    let (dim, grade, relative) = (self.dim, self.grade, self.relative());
    let lambda = self.eigenvalue();
    let blade = Combination::from_increasing(0..grade);
    DiffFormClosure::new(
      move |p: &Coord| {
        (lambda * bump(p, grade, relative))
          * ExteriorElement::from_blade_signed(dim, Sign::Pos, blade)
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
    let (dim, grade, relative) = (self.dim, self.grade, self.relative());
    let blade = Combination::from_increasing(0..grade);
    Some(DiffFormClosure::new(
      move |p: &Coord| {
        let e_blade = ExteriorElement::from_blade_signed(dim, Sign::Pos, blade);
        // Terms with $j in I$ drop out: the wedge of a repeated index is zero.
        (0..dim)
          .map(|j| {
            let e_j =
              ExteriorElement::from_blade_signed(dim, Sign::Pos, Combination::from_increasing([j]));
            bump_partial(p, grade, j, relative) * e_j.wedge(&e_blade)
          })
          .sum()
      },
      dim,
      grade + 1,
    ))
  }

  fn relative(&self) -> bool {
    self.bc == BoundaryCondition::Relative
  }
}

/// Whether coordinate $i$ carries a $sin$ factor: for the absolute BC that is
/// the tangential indices ($i < k$), for the relative BC the normal ones. The
/// two problems are the $sin arrow.l.r cos$ swap.
fn is_sin(i: usize, grade: usize, relative: bool) -> bool {
  (i < grade) != relative
}

/// $f(x)$: a $sin x_i$ or $cos x_i$ factor per coordinate, per [`is_sin`].
fn bump(p: &Coord, grade: usize, relative: bool) -> f64 {
  p.iter()
    .enumerate()
    .map(|(i, &x)| {
      if is_sin(i, grade, relative) {
        x.sin()
      } else {
        x.cos()
      }
    })
    .product()
}

/// $diff_j f$, the derivative in the $j$-th coordinate.
fn bump_partial(p: &Coord, grade: usize, j: usize, relative: bool) -> f64 {
  p.iter()
    .enumerate()
    .map(|(i, &x)| match (i == j, is_sin(i, grade, relative)) {
      (true, true) => x.cos(),
      (true, false) => -x.sin(),
      (false, true) => x.sin(),
      (false, false) => x.cos(),
    })
    .product()
}
