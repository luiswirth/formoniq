//! Shared support for the Hodge-Laplace examples: the manufactured box
//! eigenform and the [`report`] helpers that give the three tables one look.
//!
//! Included with `#[path = "util/mod.rs"] mod util;`; it is not itself an
//! example binary (only files directly under `examples/` are). Each including
//! binary compiles the whole module but uses only a subset, so unused items are
//! expected here rather than a sign of dead code.
#![allow(dead_code)]

use {
  chartan::field::DiffFormClosure,
  coorder::Coord,
  exterior::ExteriorElement,
  multiindex::{Combination, Sign},
};

/// The algebraic (as opposed to asymptotic) convergence rate between two
/// successive errors: $-log_2 ("next" \/ "prev")$.
pub fn algebraic_convergence_rate(next: f64, prev: f64) -> f64 {
  let quot: f64 = next / prev;
  -quot.log2()
}

/// Uniform table formatting for the Hodge-Laplace examples, so the source, the
/// spectrum and the sphere tables read as one.
///
/// Every table opens with the same `| r | ncells |` frame, and every value it
/// prints comes from one of the three formatters below, in one convention:
/// errors in scientific notation to three significant figures (they decay over
/// many orders), eigenvalues in fixed-point to three decimals (they are $O(1)$),
/// convergence rates to two decimals, and one marker — [`NA`] — for every value
/// that does not exist at a row: the first refinement (no coarser level to
/// compare against), the rate of an identically-zero harmonic mode, the missing
/// $dif u$ at top grade. A dash says "no meaningful value here", which is more
/// honest than a printed `inf`, `NaN` or `0.00e0` standing in for one.
///
/// Each formatter returns the bare value; the call site pads it to its column
/// width, so the header and the data rows share one set of width specifiers.
pub mod report {
  /// The absent-value marker: a value that does not exist at this row, as
  /// opposed to one that is small.
  pub const NA: &str = "—";

  /// An error (or any decaying quantity) in scientific notation, or [`NA`] when
  /// it does not exist at this row.
  pub fn err(x: Option<f64>) -> String {
    x.map_or_else(|| NA.to_string(), |x| format!("{x:.2e}"))
  }

  /// A convergence rate, or [`NA`] when there is no coarser level to compare
  /// against (a non-finite rate) or the quantity is an identically-zero harmonic
  /// mode.
  pub fn rate(r: Option<f64>) -> String {
    match r {
      Some(r) if r.is_finite() => format!("{r:.2}"),
      _ => NA.to_string(),
    }
  }

  /// An eigenvalue in fixed-point, the negative zero of a harmonic mode's
  /// residual normalized away.
  pub fn eigval(lambda: f64) -> String {
    let s = format!("{lambda:.3}");
    if s == "-0.000" {
      "0.000".to_string()
    } else {
      s
    }
  }
}

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

impl BoundaryCondition {
  /// The lower-case name used in the table headings.
  pub fn label(self) -> &'static str {
    match self {
      Self::Absolute => "absolute",
      Self::Relative => "relative",
    }
  }
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
