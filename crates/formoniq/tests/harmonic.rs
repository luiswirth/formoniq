//! The discrete Hodge isomorphism: a harmonic representative is closed,
//! weakly coclosed, its integral reading is dual to the cycles, its
//! orthonormal reading is $H^T M H = I$, and it is $L^2$-minimal in its
//! class. Checked on both the full and the relative complex.

use derham::Cochain;
use formoniq::harmonic::harmonics;
use formoniq::linalg::bilinear_form_sparse;
use formoniq::whitney_complex::{HilbertComplex, WhitneyComplex};
use regge::lengths::mesh::MeshLengthsSq;
use regge::mesher::cartesian::CartesianGrid;
use regge::mesher::quotient::FlatQuotient;
use simplicial::linalg::Matrix;
use simplicial::topology::chain::pairing;
use simplicial::topology::complex::Complex;
use simplicial::{Dim, linalg::Vector};

/// A cube (trivial cohomology in positive grade, and the relative complex
/// dual to it) and a torus (every $b_k = binom(d, k)$ nonzero): the second is
/// what keeps these laws from being statements about the empty basis.
fn fixtures() -> Vec<(Complex, MeshLengthsSq)> {
  let mut meshes: Vec<_> = (1..=3)
    .map(Dim::from)
    .map(|dim| {
      let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      let lengths = coords.to_edge_lengths_sq(&topology);
      (topology, lengths)
    })
    .collect();
  meshes.extend(
    (1..=2)
      .map(Dim::from)
      .map(|dim| FlatQuotient::unit_torus(dim, 3).triangulate()),
  );
  meshes
}

/// The laws every harmonic basis obeys, checked on one complex and grade.
fn assert_harmonic<C: HilbertComplex>(complex: &C, grade: Dim) {
  let harmonics = harmonics(complex, grade).expect("Riemannian, so the projection is well posed");
  let mass = complex.mass(grade);
  let dif_prev = complex.dif(grade - 1);
  let dif = complex.dif(grade);

  // The dimension of the harmonic space is a topological invariant, taken
  // from cohomology rather than from an eigenvalue tolerance.
  assert_eq!(harmonics.integral.ncols(), complex.harmonic_dim(grade));
  assert_eq!(harmonics.orthonormal.ncols(), complex.harmonic_dim(grade));

  for h in harmonics.integral.column_iter() {
    let h = h.into_owned();
    let scale = h.norm().max(1.0);

    // Closed: $dif h = dif z = 0$, since $z$ is a cocycle and $dif$ kills the
    // coboundary that was subtracted.
    assert!((&dif * &h).norm() <= 1e-8 * scale, "grade={grade}");

    // Coclosed weakly: $h perp_(M_k) "im" dif^(k-1)$, the defining property
    // of the least-squares residual.
    let coclosed = dif_prev.transpose() * (&mass * &h);
    assert!(coclosed.norm() <= 1e-8 * scale, "grade={grade}");
  }

  // The integral reading is dual to the cycles: $integral_(z_j) h^i =
  // delta^i_j$, so a basis form wraps its own hole and no other. This is a
  // law with content, since the unnormalized basis fails it.
  let cycles = complex.integral_cycles(grade);
  assert_eq!(cycles.len(), complex.harmonic_dim(grade));
  for (i, h) in harmonics.integral.column_iter().enumerate() {
    let form = Cochain::new(grade, h.into_owned());
    for (j, cycle) in cycles.iter().enumerate() {
      let period = pairing(&form, &cycle.extend_scalars(|&c| c as f64));
      let expected = f64::from(u8::from(i == j));
      assert!((period - expected).abs() <= 1e-8, "grade={grade}");
    }
  }

  // The orthonormal reading is what the mixed saddle point assumes:
  // $H^T M_k H = I$.
  let gram = Matrix::from_fn(
    harmonics.orthonormal.ncols(),
    harmonics.orthonormal.ncols(),
    |i, j| {
      bilinear_form_sparse(
        &mass,
        &harmonics.orthonormal.column(i).into_owned(),
        &harmonics.orthonormal.column(j).into_owned(),
      )
    },
  );
  for i in 0..gram.nrows() {
    for j in 0..gram.ncols() {
      assert!((gram[(i, j)] - f64::from(u8::from(i == j))).abs() <= 1e-8);
    }
  }
}

/// The harmonic representative is closed, weakly coclosed, and there are
/// $b_k$ of them: the discrete Hodge isomorphism, over dimensions and grades
/// and over both boundary conditions.
#[test]
fn harmonics_are_harmonic() {
  for (topology, lengths) in fixtures() {
    let whitney = WhitneyComplex::new(&topology, &lengths);
    let relative = whitney.relative();
    for grade in topology.dim().range_inclusive() {
      assert_harmonic(&whitney, grade);
      assert_harmonic(&relative, grade);
    }
  }
}

/// The harmonic representative is the $L^2$-minimal element of its class:
/// $norm(h)_M <= norm(z - D q)_M$ for every $q$, sampled here over the
/// coordinate directions and their sum. This is the claim that makes the
/// representative canonical within the class, and the one the projection is
/// solving for.
#[test]
fn harmonic_representative_is_l2_minimal() {
  for (topology, lengths) in fixtures() {
    let whitney = WhitneyComplex::new(&topology, &lengths);
    for grade in topology.dim().range_inclusive() {
      let mass = whitney.mass(grade);
      let dif_prev = whitney.dif(grade - 1);
      let harmonics = harmonics(&whitney, grade).unwrap();

      for (cocycle, h) in whitney
        .integral_cocycles(grade)
        .iter()
        .zip(harmonics.integral.column_iter())
      {
        let z = cocycle.extend_scalars(|&c| c as f64).coeffs().clone();
        let norm_h = bilinear_form_sparse(&mass, &h.into_owned(), &h.into_owned());

        let ndofs_prev = whitney.ndofs(grade - 1);
        let candidates = (0..ndofs_prev)
          .map(|i| Vector::from_fn(ndofs_prev, |r, _| f64::from(u8::from(r == i))))
          .chain(std::iter::once(Vector::from_element(ndofs_prev, 1.0)));
        for q in candidates {
          let competitor = &z - &dif_prev * q;
          let norm_competitor = bilinear_form_sparse(&mass, &competitor, &competitor);
          assert!(norm_h <= norm_competitor + 1e-9, "grade={grade}");
        }
      }
    }
  }
}
