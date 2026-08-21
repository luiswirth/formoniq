//! Laws for [`studio::solve`]: a request survives the encoding it must cross
//! the worker boundary as, producing the same outcome it would have run in
//! place, and the outcome round-trips too, rebuilding the same scene against
//! the mesh the caller kept.

use formoniq_studio::{
  gallery::{MeshSource, Study},
  solve::{SolveOutcome, SolveRequest, decode, encode},
};

/// A request survives the encoding, and the outcome it produces is the same
/// one it would have produced in place. This is the whole contract the worker
/// rests on: what crosses the boundary is the build, unchanged.
#[test]
fn a_request_solves_the_same_after_a_round_trip() {
  let mesh = MeshSource::Sphere { subdivisions: 1 }.build().unwrap();
  let request = SolveRequest::new(
    &mesh,
    Study::Eigenmodes {
      grade: simplicial::Dim::ZERO,
      nmodes: 4,
    },
  );

  let direct = request.run();
  let crossed: SolveRequest = decode(&encode(&request));
  let indirect = crossed.run();

  assert_eq!(direct.fields.len(), indirect.fields.len());
  assert_eq!(direct.line_fields.len(), indirect.line_fields.len());
  for (a, b) in direct.fields.iter().zip(&indirect.fields) {
    assert_eq!(a.name, b.name);
    assert_eq!(a.grade, b.grade);
    assert_eq!(a.cochain.coeffs(), b.cochain.coeffs());
  }
}

/// The outcome round-trips too: it is what comes back from the worker,
/// and rebuilds the same scene against the mesh the caller kept.
#[test]
fn an_outcome_survives_the_return_trip() {
  let mesh = MeshSource::Sphere { subdivisions: 1 }.build().unwrap();
  let request = SolveRequest::new(&mesh, Study::WhitneyBasis);
  let outcome = request.run();
  let returned: SolveOutcome = decode(&encode(&outcome));

  let scene = returned.into_scene(&mesh);
  assert_eq!(scene.topology.nsimplices(0), mesh.0.nsimplices(0));
  assert_eq!(scene.fields.len(), outcome.fields.len());
  assert_eq!(scene.line_fields.len(), outcome.line_fields.len());
}
