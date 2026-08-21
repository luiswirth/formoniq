//! Laws for [`realize::advect`]: the per-cell flow generator preserves
//! $sum_i lambda_i = 1$ (its columns sum to zero), and the uniform
//! barycentric sampler draws points genuinely on the simplex, non-negative
//! and summing to one, with every unused slot left at zero.

use derham::{Cochain, interpolate::interpolant::WhitneyInterpolant};
use realize::advect::{flow_generator, uniform_bary};
use regge::coord::mesh::MeshCoords;
use simplicial::{Sign, topology::complex::Complex};

/// The generator preserves $sum_i lambda_i = 1$: its columns sum to zero, so
/// $bb(1)^T dot(lambda) = 0$ and a particle never leaves the affine hull its
/// weights live on. This is the law the whole pass rests on, if it failed,
/// the flow would carry points off the manifold, and it holds in every
/// dimension the ambient admits.
#[test]
fn generator_columns_sum_to_zero() {
  for dim in 1..=3 {
    let topology = Complex::unit(dim);
    let coords = MeshCoords::unit(dim);
    let cochain = Cochain::constant(1.0, topology.skeleton(1));
    let interpolant = WhitneyInterpolant::new(cochain, &topology);
    for cell in topology.cells().handle_iter() {
      let metric = coords.cell_metric(cell);
      let generator = flow_generator(&interpolant, cell.idx(), &metric, Sign::Pos);
      for column in 0..generator.ncols() {
        let sum: f64 = generator.column(column).sum();
        assert!(
          sum.abs() < 1e-10,
          "dim {dim}: column {column} sums to {sum}"
        );
      }
    }
  }
}

/// Uniform barycentric draws are barycentric: non-negative and summing to one,
/// at every dimension the ambient reaches, with the unused slots left at zero.
#[test]
fn uniform_bary_is_barycentric() {
  for dim in 0..=3 {
    for index in 0..64 {
      let bary = uniform_bary(dim, index);
      let sum: f32 = bary.iter().sum();
      assert!((sum - 1.0).abs() < 1e-5, "dim {dim}: weights sum to {sum}");
      assert!(bary.iter().all(|&w| w >= 0.0), "dim {dim}: negative weight");
      assert!(bary[dim + 1..].iter().all(|&w| w == 0.0));
    }
  }
}
