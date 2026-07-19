//! The three maps from $L^2 Lambda^k$ into the Whitney space, and the error
//! between them.
//!
//! A differential form on the manifold reaches the discrete space by one of
//! three routes, and they are genuinely different maps:
//!
//! - $W: C^k -> cal(W) Lambda^k$, the Whitney *interpolation*
//!   ([`WhitneyInterpolant`]): the reconstruction of a form from a cochain, the
//!   right inverse of $R$.
//! - $R: L^2 Lambda^k -> C^k$, the *de Rham map*
//!   ([`derham_map`](derham::project::derham_map)): integration over the
//!   simplices. Canonical and metric-free, and a cochain map --
//!   $R compose dif = dif compose R$ -- which is exactly why the discrete
//!   complex inherits the cohomology of the continuous one. It needs the
//!   traces to exist, so it is not defined on all of $L^2 Lambda^k$.
//! - $P_h: L^2 Lambda^k -> cal(W) Lambda^k$, the $L^2$ *projection*
//!   ([`l2_projection`]): the best approximation in the energy norm, defined on
//!   all of $L^2 Lambda^k$ but *not* commuting with $dif$, and requiring a
//!   global mass solve rather than local integration.
//!
//! $R$ is the one the theory is built on; $P_h$ is the one that is optimal in
//! norm. Neither dominates the other, and the discrete complex is exact only
//! through $R$.

use {
  crate::linalg::faer::FaerLu,
  derham::{cochain::Cochain, interpolate::interpolant::WhitneyInterpolant, section::Section},
  exterior::{multiform_gramian, Covariant},
  simplicial::{
    atlas::{MeshPoint, SimplexQuadRule},
    geometry::{cell_volume, metric::Geometry},
    linalg::{CsrMatrix, Vector},
    topology::complex::Complex,
  },
};

use crate::{assemble::assemble_galvec, operators::SourceElVec, whitney_complex::WhitneyComplex};

/// The $L^2 Lambda^k$ error $norm(omega - W c)_(L^2)$ between an exact form on
/// the manifold and the Whitney reconstruction of a cochain.
///
/// Intrinsic: the pointwise difference is measured in the reference frame of
/// each cell by the induced inner product $Lambda^k g^(-1)$ of that cell's
/// metric. On a curved (embedded) mesh this is the only correct thing to do --
/// the flat ambient Gramian would measure the wrong norm.
pub fn fe_l2_error<F: Section<Covariant>>(
  fe_cochain: &Cochain,
  exact: &F,
  topology: &Complex,
  geometry: &impl Geometry,
) -> f64 {
  let dim = topology.dim();
  let grade = fe_cochain.grade();
  let qr = SimplexQuadRule::degree(dim, 3);
  let fe_whitney = WhitneyInterpolant::new(fe_cochain.clone(), topology);

  let error_sq: f64 = topology
    .cells()
    .handle_iter()
    .map(|cell| {
      let metric = geometry.cell_metric(cell);
      let inner = multiform_gramian(&metric, grade);
      let error_pointwise =
        |point: &MeshPoint| inner.norm_sq((exact.at(point) - fe_whitney.at(point)).coeffs());
      qr.integrate_cell(cell.idx(), &error_pointwise, cell_volume(&metric))
    })
    .sum();

  error_sq.sqrt()
}

/// The $L^2$ projection $P_h omega$ of a form onto the Whitney space: the
/// solution of the mass system
///
/// $M c = b, quad b_sigma = integral_M inner(omega, W_sigma) vol$
///
/// i.e. the best approximation in the $L^2 Lambda^k$ norm, characterized by
/// Galerkin orthogonality $inner(omega - P_h omega, v) = 0$ for all
/// $v in cal(W) Lambda^k$.
///
/// Unlike the de Rham map this is defined for any $L^2$ form, but it does not
/// commute with $dif$ and it costs a global solve. See the module docs.
pub fn l2_projection<F: Sync + Section<Covariant>, G: Geometry + Sync>(
  field: &F,
  whitney: WhitneyComplex<G>,
  qr: Option<SimplexQuadRule>,
) -> Cochain {
  let grade = field.grade();
  let mass = CsrMatrix::from(&whitney.mass(grade));
  let load: Vector = assemble_galvec(
    whitney.topology(),
    whitney.geometry(),
    SourceElVec::new(field, qr),
  );
  // The mass is s.p.d. only on a Riemannian geometry; on an indefinite one it
  // is symmetric non-degenerate, so LU covers every signature uniformly.
  Cochain::new(grade, FaerLu::new(mass).solve(&load))
}

#[cfg(test)]
mod test {
  use super::*;

  use derham::section::CoordFieldExt;
  use glatt::field::DiffFormClosure;
  use simplicial::gen::cartesian::CartesianMeshInfo;

  use approx::assert_relative_eq;

  /// $P_h compose W = id$: the $L^2$ projection is the identity on the Whitney
  /// space, since a discrete form is its own best approximation.
  ///
  /// The sharpest available check that the Hodge mass matrix and the source
  /// load are the same bilinear form seen from two sides.
  #[test]
  fn l2_projection_reproduces_whitney_forms() {
    for dim in 1..=3 {
      let (topology, coords) = CartesianMeshInfo::new_unit(dim, 2).compute_coord_complex();
      let lengths = coords.to_edge_lengths_sq(&topology);
      let whitney = WhitneyComplex::new(&topology, &lengths);

      for grade in 0..=dim {
        let ndofs = topology.nsimplices(grade);
        let cochain = Cochain::new(
          grade,
          Vector::from_iterator(ndofs, (0..ndofs).map(|i| ((i % 7) as f64) - 3.0)),
        );

        let field = WhitneyInterpolant::new(cochain.clone(), &topology);
        let qr = SimplexQuadRule::degree(dim, 3);
        let projected = l2_projection(&field, whitney, Some(qr));

        assert_relative_eq!(projected.coeffs(), cochain.coeffs(), epsilon = 1e-9);
      }
    }
  }

  /// A form that lies in the Whitney space is reproduced exactly by the
  /// projection, and hence has zero $L^2$ error against it: $W$ and $P_h$
  /// agree wherever both are exact.
  #[test]
  fn l2_error_vanishes_on_the_discrete_space() {
    for dim in 1..=3 {
      let (topology, coords) = CartesianMeshInfo::new_unit(dim, 2).compute_coord_complex();
      let lengths = coords.to_edge_lengths_sq(&topology);
      let whitney = WhitneyComplex::new(&topology, &lengths);

      // A globally affine 0-form lies in the Whitney 0-form space.
      let exact = DiffFormClosure::coordinate_component(0, dim);
      let exact = exact.pullback_on(&topology, &coords);
      let projected = l2_projection(&exact, whitney, Some(SimplexQuadRule::degree(dim, 3)));

      let error = fe_l2_error(&projected, &exact, &topology, &lengths);
      assert!(error < 1e-9, "dim={dim} error={error}");
    }
  }
}
