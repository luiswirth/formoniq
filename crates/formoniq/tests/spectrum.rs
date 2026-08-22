//! The Hodge-Laplace spectrum of the flat torus, against the closed form the
//! Fourier lattice gives it.
//!
//! $T^d = RR^d \/ L ZZ^d$ is flat and closed, so a constant coframe
//! trivializes the bundle and $Delta$ acts on each of the $binom(d, k)$
//! components as the scalar Laplacian. The spectrum is therefore the lattice
//! one,
//!
//! $ lambda_m = (2 pi abs(m) \/ L)^2, quad m in ZZ^d, $
//!
//! each with multiplicity $binom(d, k)$ times the number of lattice points on
//! the shell. At $m = 0$ that is the harmonic space, of dimension
//! $b_k (T^d) = binom(d, k)$, a purely topological prediction the discrete
//! problem reproduces exactly at every resolution; the shells above it are
//! approximated, and converge.
//!
//! This is the sharpest available statement about the discrete operator as an
//! operator: an assembly that is wrong in a way no algebraic law can see
//! still has to produce these numbers, with these multiplicities, and no
//! others in between.

#[path = "../examples/util/mod.rs"]
mod util;

use {
  formoniq::{problems::elliptic, whitney_complex::WhitneyComplex},
  multiindex::binomial,
  util::Manifold,
};

use std::f64::consts::PI;

/// The discrete spectrum reproduces the lattice spectrum of $T^d$: the
/// harmonic sector exactly, the first nonzero shell in the limit.
///
/// The gap between $0$ and the first shell, uniform in $h$, is the discrete
/// Poincaré--Friedrichs inequality: $norm(omega) <= C norm(dif omega)$ off
/// the kernel, with $C = lambda_1^(-1\/2)$ bounded independently of the
/// mesh. Coercivity of the method is that gap and nothing else, so it is
/// tested here rather than separately.
///
/// The torus has side $L = pi$, so the first shell sits at
/// $lambda = (2 pi \/ pi)^2 = 4$ with the $2 d$ lattice points $plus.minus
/// e_i$ on it, hence multiplicity $2 d binom(d, k)$. Nothing lies between it
/// and $0$, which is what makes the harmonic count a statement rather than a
/// threshold.
#[test]
fn the_flat_torus_carries_the_spectrum_of_its_lattice() {
  const SHELL: f64 = 4.0;

  for dim in 1..=2usize {
    for grade in 0..=dim {
      let harmonic_dim = binomial(dim, grade);
      let multiplicity = 2 * dim * harmonic_dim;
      let neigen = harmonic_dim + multiplicity + 1;
      let case = format!("T^{dim}, grade {grade}");

      let (mut topology, mut lengths, mut ordering) = Manifold::Torus.coarse_mesh(dim, PI);
      let mut deviations = Vec::new();
      for level in 0..3 {
        if level > 0 {
          let sub = topology.refine_with(&ordering, 2);
          lengths = lengths.refine(&sub, &topology);
          ordering = sub.ordering().clone();
          topology = sub.into_complex();
        }

        let whitney = WhitneyComplex::new(&topology, &lengths);
        let (eigenvals, _, _) = elliptic::solve_evp(&whitney, grade, neigen).unwrap();
        // The coarsest level can carry fewer degrees of freedom than the
        // shell needs, and then the solver returns fewer pairs than asked.
        if eigenvals.len() < neigen {
          continue;
        }

        let (harmonic, rest) = eigenvals.as_slice().split_at(harmonic_dim);
        let (shell, above) = rest.split_at(multiplicity);

        for (i, &lambda) in harmonic.iter().enumerate() {
          assert!(
            lambda.abs() < 1e-6,
            "{case}, level {level}: harmonic {i} is {lambda}"
          );
        }
        // The shell is separated from what follows it, so the count above is
        // a multiplicity and not the first `harmonic_dim` of a continuum.
        let spread = shell.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(
          above[0] > 1.2 * spread,
          "{case}, level {level}: the shell is not separated from {}",
          above[0]
        );

        deviations.push(
          shell
            .iter()
            .map(|lambda| (lambda - SHELL).abs())
            .fold(f64::NEG_INFINITY, f64::max),
        );
      }

      for (level, pair) in deviations.windows(2).enumerate() {
        assert!(
          pair[1] < pair[0],
          "{case}: the shell moved away from {SHELL} between levels {level} and {}",
          level + 1
        );
      }
      let finest = deviations.last().unwrap();
      assert!(
        *finest < 0.05 * SHELL,
        "{case}: the shell sits {finest} away from {SHELL}"
      );
    }
  }
}
