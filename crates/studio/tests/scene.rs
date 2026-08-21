//! Laws for [`studio::scene::Scene`]: the volume medium is offered exactly
//! on a solid, transport conserves at the extremal grades on a closed mesh
//! and both evolutions run at every grade, the smoothest closed field is a
//! genuine cocycle, a top-grade bump is a positive density and displaces
//! rigidly per cell while a grade-0 field displaces continuously, the sign
//! gauge is fixed, the harmonic top form reduces to a constant density, a
//! trajectory's frame interpolates and both the heat and wave flows behave
//! as they should, the reduced-grade offers and the standing-wave
//! displacement toggle are total, the Whitney-basis galleries have exactly
//! one field per subsimplex and support only their own DOF cells, the Hodge
//! decomposition splits orthogonally, and every Whitney-basis field bakes.

use derham::{Cochain, interpolate::interpolant::WhitneyInterpolant, section::SectionOps};
use formoniq_studio::{
  demos,
  gallery::MeshSource,
  scene::{
    FieldTime, Scene, ambient_blade_flux, ambient_bump, canonical_sign, hodge_decompose,
    hodge_probe_input, smoothest_closed_space, solenoidal_flux,
  },
  ui::Selection,
};
use nalgebra as na;
use realize::reduce::{corner_values, nodal_heights, surface_corner_heights};
use regge::coord::mesh::MeshCoords;
use simplicial::{
  Dim,
  topology::{complex::Complex, handle::SimplexIdx},
};

/// The medium is offered exactly where there is an interior to march, and
/// that is an intrinsic-dimension question rather than a grade one: at
/// $n <= 2$ the boundary primitive already draws the whole manifold, and at
/// $n >= 3$ it cannot. Swept over both, and over every grade at each, since
/// the answer must not depend on which field is selected.
#[test]
fn the_medium_is_offered_exactly_on_a_solid() {
  for dim in 1..=3 {
    let scene = Scene::whitney_basis(dim);
    let mut asked = 0;
    for selection in scene.selections() {
      assert_eq!(
        scene.offers(selection).volume,
        dim >= 3,
        "dimension {dim} offered the wrong medium"
      );
      asked += 1;
    }
    assert!(asked > 0, "dimension {dim} produced no field to ask about");
  }
}

/// Grade-0 transport conserves $L^2$ exactly on a closed mesh, which is
/// what the solenoidal velocity is for.
///
/// The antisymmetry defect needs the facet terms of neighboring cells to
/// cancel, and at grade 0 that asks two things: $inner(omega, eta)$
/// single-valued, which the continuous shape functions give, and the flux
/// $iota_v vol$ single-valued, which reading the velocity off an
/// $(n-1)$-cochain gives. Exact, not approximate, and a per-cell star
/// without the coherent orientation breaks it, since the field then reverses
/// across every facet where colex disagrees with the manifold.
#[test]
fn grade_zero_transport_conserves_on_a_closed_mesh() {
  use formoniq::problems::advection::{Transport, assemble_transport, solve_transport};

  let (topology, coords) = regge::mesher::sphere::mesh_sphere_surface(2);
  let metric = coords.to_edge_lengths_sq(&topology);
  let velocity = WhitneyInterpolant::new(solenoidal_flux(&topology, &coords, &metric), &topology)
    .hodge_star(&topology, &metric, topology.orientation().unwrap())
    .musical(&topology, &metric);

  let transport = Transport {
    grade: Dim::ZERO,
    velocity: &velocity,
    quad_degree: 2,
  };
  let (mass, _) = assemble_transport(&topology, &metric, &transport);
  let initial = ambient_bump(&topology, &coords, Dim::ZERO);
  let frames = solve_transport(&topology, &metric, &transport, 40, 0.01, &initial);

  let l2 = |c: &Cochain| formoniq::linalg::quadratic_form_sparse(&mass, c.coeffs()).sqrt();
  let drift = (l2(frames.last().unwrap()) - l2(&frames[0])) / l2(&frames[0]);
  assert!(
    drift.abs() < 1e-10,
    "grade-0 transport drifted by {drift:e}"
  );
}

/// The transport velocity is the smoothest closed field the mesh admits, and
/// which of the two branches supplies it is the topology's answer, not a
/// threshold's: a torus has harmonic $1$-forms and uses one, a sphere has
/// none and falls to the lowest exact mode, which is the rigid rotation.
///
/// Either way the flux is a cocycle, which is what conservation at the ends
/// of the grade range rests on.
///
/// The magnitude is checked against the reference field and not against zero,
/// because the failure this guards is a near-vanishing one: a projection
/// onto a space the reference is orthogonal to returns roundoff, which is
/// nonzero, and closed to the same roundoff, so both laws pass on nothing.
#[test]
fn the_velocity_is_the_smoothest_closed_field() {
  let donut = formoniq_studio::gallery::QuotientSurface::Donut.build(20);
  let sphere = regge::mesher::sphere::mesh_sphere_surface(2);

  for (name, topology, coords, harmonics) in [
    ("donut", donut.0, donut.1, 2),
    ("sphere", sphere.0, sphere.1, 0),
  ] {
    let metric = coords.to_edge_lengths_sq(&topology);
    let space = smoothest_closed_space(&topology, &metric).expect("a closed field exists");
    assert_eq!(
      space.ncols().min(3),
      if harmonics > 0 { harmonics } else { 3 },
      "{name}: the bottom eigenspace has the wrong dimension"
    );

    let flux = solenoidal_flux(&topology, &coords, &metric);
    let scale = flux.coeffs().amax();
    let reference = ambient_blade_flux(&topology, &coords).coeffs().amax();
    assert!(
      scale > 1e-6 * reference,
      "{name}: the velocity vanished, {scale:e} against a reference of {reference:e}"
    );
    assert!(
      flux.dif(&topology).coeffs().amax() <= 1e-10 * scale,
      "{name}: the flux is not divergence-free"
    );
  }
}

/// The top-grade bump is a density, so it is positive everywhere on a
/// submanifold too.
///
/// Read through a fixed ambient $n$-blade instead, its pullback is the
/// projection of the tangent multivector onto that blade, which changes sign
/// over a curved surface and vanishes along a whole curve. The sphere is the
/// case that catches it. A full-dimensional mesh cannot.
#[test]
fn the_top_grade_bump_is_a_positive_density() {
  let (topology, coords) = regge::mesher::sphere::mesh_sphere_surface(2);
  let top = ambient_bump(&topology, &coords, topology.dim());

  assert!(
    top.coeffs().iter().all(|c| *c > 0.0),
    "the density changes sign"
  );
}

/// Advection transports: the trajectory moves the bump without losing or
/// gaining it. On the sphere the velocity is a rotation, which is Killing
/// there, so the field is carried around rather than diffused, and the
/// central scheme neither damps nor blows up.
#[test]
fn advection_carries_the_bump_without_losing_it() {
  let (topology, coords) = regge::mesher::sphere::mesh_sphere_surface(2);
  let scene = Scene::advection(topology, coords, Dim::ZERO, 16, 3.0);

  let FieldTime::Trajectory { frames, .. } = &scene.fields[0].time else {
    panic!("advection must produce a sampled trajectory");
  };
  let norm = |c: &Cochain| c.coeffs().norm();
  let (first, last) = (&frames[0], frames.last().unwrap());

  assert!(
    (last.coeffs() - first.coeffs()).norm() > 0.1 * norm(first),
    "the bump did not move"
  );
  let ratio = norm(last) / norm(first);
  assert!(
    (0.5..2.0).contains(&ratio),
    "transport lost or gained the bump: ratio {ratio}"
  );
}

/// A top-grade field displaces its cells rigidly: the height is constant
/// within each cell, which is what makes the fill tear along the field's own
/// discontinuity instead of smoothing across it. Constant to zero spread,
/// not approximately, the Whitney top form is genuinely $P_0$.
#[test]
fn top_grade_displacement_is_constant_within_each_cell() {
  let (topology, coords) = regge::mesher::sphere::mesh_sphere_surface(1);
  let scene = Scene::eigenmodes(&topology, &coords, topology.dim(), 2);
  let baked = realize::bake::BakedMesh::new(&scene.topology, &scene.coords);
  let top = scene.fields.first().expect("a top-grade scalar field");
  let heights = surface_corner_heights(
    &scene.topology,
    &scene.coords,
    &top.cochain,
    baked.fill_triangles(),
  );
  for corner in heights.chunks(3) {
    assert_eq!(
      corner[0], corner[1],
      "a cell's corners must share one rigid height"
    );
    assert_eq!(corner[0], corner[2]);
  }
}

/// A grade-0 field stays continuous: the corners a shared mesh vertex
/// contributes to agree, so the surface displaces as one sheet and does not
/// tear. The other half of the dispatch, and the reason it is a dispatch
/// rather than a switch to rigid displacement everywhere.
#[test]
fn grade_zero_displacement_agrees_at_a_shared_vertex() {
  let (topology, coords) = regge::mesher::sphere::mesh_sphere_surface(1);
  let scene = Scene::eigenmodes(&topology, &coords, Dim::ZERO, 2);
  let baked = realize::bake::BakedMesh::new(&scene.topology, &scene.coords);
  let scalar = scene.fields.first().expect("a grade-0 field");
  let heights = surface_corner_heights(
    &scene.topology,
    &scene.coords,
    &scalar.cochain,
    baked.fill_triangles(),
  );
  let mut seen: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
  for (triangle, corner) in baked.fill_triangles().iter().zip(heights.chunks(3)) {
    for (slot, &vertex) in triangle.iter().enumerate() {
      if let Some(previous) = seen.insert(vertex as usize, corner[slot]) {
        assert!((previous - corner[slot]).abs() < 1e-12);
      }
    }
  }
}

/// The sign gauge is pinned, so a solver returning the opposite eigenvector
/// renders the identical picture: the largest-magnitude coefficient comes out
/// positive either way, and the magnitudes are untouched.
#[test]
fn canonical_sign_is_a_gauge_fix() {
  let c = Cochain::new(0, na::DVector::from_vec(vec![0.5, -2.0, 1.0]));
  let flipped = Cochain::new(0, -c.coeffs());
  assert_eq!(
    canonical_sign(c.clone()).coeffs(),
    canonical_sign(flipped).coeffs()
  );
  // The pivot is made positive, and nothing is rescaled.
  let fixed = canonical_sign(c);
  assert_eq!(fixed.coeffs()[1], 2.0);
  assert_eq!(fixed.coeffs()[0], -0.5);
  // The zero cochain is its own canonical form.
  let zero = Cochain::new(0, na::DVector::zeros(3));
  assert_eq!(canonical_sign(zero.clone()).coeffs(), zero.coeffs());
}

/// The harmonic top-grade form on a closed orientable surface is a multiple
/// of the volume form, $h = c dvol$, so its reduction $star h = c$ is
/// constant over the whole manifold. That makes the reduced readout of the
/// $lambda = 0$ grade-2 mode on the sphere a law with an exact answer, and
/// the sharpest available statement that the Hodge star is being taken
/// against one global volume form rather than each cell's own.
///
/// It is precisely the test the colex vertex order fails without a coherent
/// orientation: the density comes out $plus.minus c$ cell by cell, and the
/// nodal average of that collapses toward zero instead of reproducing $c$.
#[test]
fn harmonic_top_form_reduces_to_a_constant_density() {
  use formoniq::{problems::elliptic::solve_evp, whitney_complex::WhitneyComplex};

  let (topology, coords) = regge::mesher::sphere::mesh_sphere_surface(2);
  let lengths = coords.to_edge_lengths_sq(&topology);
  let (eigenvals, _, eigenfuncs) =
    solve_evp(&WhitneyComplex::new(&topology, &lengths), 2, 1).unwrap();
  // $b_2 = 1$ on the sphere: the lowest grade-2 mode is the harmonic one.
  assert!(eigenvals[0].abs() < 1e-8, "expected the harmonic mode");

  let cochain = Cochain::new(2, eigenfuncs.column(0).into_owned());
  let heights = nodal_heights(&topology, &coords, &cochain);
  let mean = heights.iter().sum::<f64>() / heights.len() as f64;
  assert!(mean.abs() > 1e-3, "the density must not cancel to zero");
  for h in heights {
    assert!(
      (h - mean).abs() / mean.abs() < 1e-10,
      "reduced harmonic density {h} is not the constant {mean}"
    );
  }
}

/// A trajectory's frame at an instant is the linear interpolation of its
/// bracketing samples, clamped to the sampled interval: the interpolation is
/// linear in the cochain coefficients, so it is exact at the samples and
/// affine between them, and its duration is $dif t (N - 1)$.
#[test]
fn frame_at_interpolates_between_samples() {
  let frames = vec![
    Cochain::new(0, na::DVector::from_vec(vec![0.0, 0.0])),
    Cochain::new(0, na::DVector::from_vec(vec![2.0, -2.0])),
    Cochain::new(0, na::DVector::from_vec(vec![4.0, -4.0])),
  ];
  let time = FieldTime::Trajectory { dt: 0.5, frames };
  let base = Cochain::new(0, na::DVector::zeros(2));

  assert_eq!(time.duration(), Some(1.0));
  // Exact at the two endpoints of the sampled interval.
  assert_eq!(time.frame_at(&base, 0.0).coeffs()[0], 0.0);
  assert_eq!(time.frame_at(&base, 1.0).coeffs()[0], 4.0);
  // Affine at the quarter point of the first (dt = 0.5) interval: halfway.
  let mid = time.frame_at(&base, 0.25);
  assert!((mid.coeffs()[0] - 1.0).abs() < 1e-12);
  assert!((mid.coeffs()[1] + 1.0).abs() < 1e-12);
  // Clamped past the end rather than extrapolated.
  assert_eq!(time.frame_at(&base, 100.0).coeffs()[0], 4.0);
}

/// The heat flow of a localized bump is one grade-0 trajectory that decays:
/// the parabolic Hodge-Laplacian damps the $L^2$ norm monotonically toward the
/// held boundary, and the field animates (offers displacement) because it is a
/// trajectory, not because it is an eigenmode. `nsteps` steps give `nsteps + 1`
/// sampled frames.
#[test]
fn heat_trajectory_decays_and_animates() {
  let (topology, coords) = MeshSource::Grid {
    dim: 2,
    cells_axis: 6,
  }
  .build()
  .unwrap();
  let scene = Scene::heat(topology, coords, 0, 20, 0.2);

  assert_eq!(scene.fields.len(), 1);
  assert!(scene.line_fields.is_empty());
  let FieldTime::Trajectory { frames, .. } = &scene.fields[0].time else {
    panic!("the heat flow is a trajectory");
  };
  assert_eq!(frames.len(), 21);
  let l2 = |c: &Cochain| c.coeffs().norm();
  assert!(
    l2(frames.last().unwrap()) < l2(&frames[0]),
    "the heat flow damps the bump"
  );
  assert!(scene.offers(Selection::Scalar(0)).displacement);
  assert!(scene.fields[0].time.eigenvalue().is_none());
}

/// The heat flow is total on a closed mesh, where there is no boundary to
/// hold: `solve_heat` runs the free Neumann flow (the relative complex is the
/// identity inclusion there) instead of panicking on an empty boundary
/// subcomplex. The regression
/// for the sphere preset, whose background solve otherwise never completes.
/// Mass is conserved (the constant is the Neumann kernel), so the bump spreads
/// rather than decaying to zero, the peak drops while the total does not.
#[test]
fn heat_flow_is_total_on_a_closed_mesh() {
  use regge::mesher::sphere::mesh_sphere_surface;
  let (topology, coords) = mesh_sphere_surface(2);
  let scene = Scene::heat(topology, coords, 0, 10, 0.2);

  let FieldTime::Trajectory { frames, .. } = &scene.fields[0].time else {
    panic!("the heat flow is a trajectory");
  };
  assert_eq!(frames.len(), 11);
  let peak = |c: &Cochain| c.coeffs().amax();
  assert!(
    peak(frames.last().unwrap()) < peak(&frames[0]),
    "the closed-mesh heat flow spreads the bump"
  );
}

/// The wave equation of the same bump is one grade-0 trajectory that does not
/// decay: the symplectic integrator conserves energy, so the $L^2$ norm stays
/// bounded near its initial value rather than damping away. Like heat, it
/// animates without being an eigenmode.
#[test]
fn wave_trajectory_is_a_bounded_animating_trajectory() {
  let (topology, coords) = MeshSource::Grid {
    dim: 2,
    cells_axis: 6,
  }
  .build()
  .unwrap();
  let scene = Scene::wave(topology, coords, 0, 30, 4.0);

  let FieldTime::Trajectory { frames, .. } = &scene.fields[0].time else {
    panic!("the wave equation is a trajectory");
  };
  assert_eq!(frames.len(), 31);
  let l2 = |c: &Cochain| c.coeffs().norm();
  let initial = l2(&frames[0]);
  assert!(
    frames.iter().all(|f| l2(f) <= 4.0 * initial),
    "the conservative wave flow stays bounded"
  );
  assert!(scene.offers(Selection::Scalar(0)).displacement);
}

/// Both evolutions are posed at every grade of the de Rham complex, not just
/// at 0: the parabolic flow damps the $L^2$ norm and the symplectic wave flow
/// keeps it bounded, whatever grade the bump is a form of. Swept over every
/// grade of a surface, the extremal ones included, the top grade is where a
/// grade-0-only construction (a scalar bump, a `ndofs(0)` source) would have
/// gone wrong silently.
#[test]
fn both_evolutions_run_at_every_grade() {
  let (topology, coords) = MeshSource::Grid {
    dim: 2,
    cells_axis: 4,
  }
  .build()
  .unwrap();
  let l2 = |c: &Cochain| c.coeffs().norm();

  // A trajectory files under the mark its reduced grade earns, so on a
  // surface grade 1 is a line field and 0 and 2 are densities; the single
  // field is read from whichever list it landed in.
  let only_field = |scene: &Scene| -> FieldTime {
    let times: Vec<FieldTime> = scene
      .fields
      .iter()
      .map(|f| f.time.clone())
      .chain(scene.line_fields.iter().map(|f| f.time.clone()))
      .collect();
    assert_eq!(times.len(), 1, "exactly one trajectory field");
    times.into_iter().next().unwrap()
  };

  for grade in topology.dim().range_inclusive() {
    let heat = Scene::heat(topology.clone(), coords.clone(), grade, 10, 0.2);
    let FieldTime::Trajectory { frames, .. } = &only_field(&heat) else {
      panic!("the heat flow is a trajectory at grade {grade}");
    };
    assert_eq!(frames.len(), 11);
    assert_eq!(frames[0].grade(), grade);
    assert!(
      l2(frames.last().unwrap()) <= l2(&frames[0]) + 1e-12,
      "the heat flow does not grow at grade {grade}"
    );

    let wave = Scene::wave(topology.clone(), coords.clone(), grade, 20, 2.0);
    let FieldTime::Trajectory { frames, .. } = &only_field(&wave) else {
      panic!("the wave equation is a trajectory at grade {grade}");
    };
    assert_eq!(frames[0].grade(), grade);
    let initial = l2(&frames[0]);
    assert!(
      frames.iter().all(|f| l2(f) <= 4.0 * initial + 1e-12),
      "the conservative wave flow stays bounded at grade {grade}"
    );
  }
}

/// What a field offers is its reduced grade's answer, and the answer is total
/// on the range the ambient reaches: every field of the reference cell in
/// every dimension is asked, and each gets the reading its reduction earns,
/// a density the surface it paints (nothing of its own), a line field its
/// three marks. Nothing is dropped and nothing is asked twice.
#[test]
fn offers_follow_the_reduced_grade_in_every_dimension() {
  for dim in 1..=3 {
    let scene = Scene::whitney_basis(dim);
    for index in 0..scene.fields.len() {
      let offers = scene.offers(Selection::Scalar(index));
      assert!(!offers.marks, "dim {dim}: a density has no mark of its own");
    }
    for index in 0..scene.line_fields.len() {
      let offers = scene.offers(Selection::Line(index));
      assert!(offers.marks, "dim {dim}: a line field offers its marks");
      assert!(
        !offers.displacement,
        "dim {dim}: a line field's curves are static -- there is no wave to ride"
      );
    }
  }
}

/// Displacement is offered exactly when there is a standing wave to toggle,
/// which is what an eigenvalue is: the same distinction `FieldDisplay::build`
/// makes when it hands a field with none an amplitude of zero. A raw Whitney
/// basis function is that field, and a grade-0 eigenmode of the same cell
/// complex is its counterpart, so the two together are the rule, not one
/// example of it.
#[test]
fn displacement_is_offered_exactly_to_a_standing_wave() {
  let basis = Scene::whitney_basis(2);
  for index in 0..basis.fields.len() {
    assert!(
      !basis.offers(Selection::Scalar(index)).displacement,
      "a Whitney basis function is no eigenmode: its amplitude is already zero"
    );
    assert!(!basis.offers(Selection::Scalar(index)).any());
  }

  let (topology, coords) = MeshSource::Grid {
    dim: 2,
    cells_axis: 4,
  }
  .build()
  .unwrap();
  let scene = Scene::eigenmodes(&topology, &coords, simplicial::Dim::ZERO, 4);
  assert!(
    !scene.fields.is_empty(),
    "the grade-0 eigensolve produced modes"
  );
  for index in 0..scene.fields.len() {
    assert!(
      scene.offers(Selection::Scalar(index)).displacement,
      "a grade-0 eigenmode has a wave to ride"
    );
  }
}

/// The reference triangle's LSF gallery: one field per subsimplex of every
/// grade, split into scalar densities (grades 0 and 2, the latter through
/// the top-form Hodge star) and the grade-1 line field.
#[test]
fn whitney_basis_reference_triangle_has_one_field_per_subsimplex() {
  let scene = Scene::whitney_basis(2);
  assert_eq!(scene.fields.len(), 3 + 1); // 3 vertices, 1 face
  assert_eq!(scene.line_fields.len(), 3); // 3 edges
}

/// The triforce mesh's GSF gallery: same reduction, but every DOF simplex
/// is now a global simplex of a 4-cell, 6-vertex, 9-edge mesh instead of a
/// subsimplex of a single reference cell.
#[test]
fn whitney_basis_mesh_has_one_field_per_mesh_simplex() {
  let (topology, coords) = demos::triforce();
  assert_eq!(topology.nsimplices(0), 6);
  assert_eq!(topology.nsimplices(1), 9);
  assert_eq!(topology.nsimplices(2), 4);

  let scene = Scene::whitney_basis_mesh(topology, coords);
  assert_eq!(scene.fields.len(), 6 + 4); // vertices, and faces via ⋆
  assert_eq!(scene.line_fields.len(), 9); // edges
}

/// The Hodge decomposition is a theorem, not a golden number. On a genus-1
/// surface ($b_1 = 2$) the probe 1-form splits into three shells that sum
/// back to it exactly, are pairwise $L^2$-orthogonal, and carry a genuinely
/// nonzero harmonic component, the part the two handle cycles pair with.
/// The flat unit square is the contractible base case: the same solve,
/// harmonic shell identically zero because there is no grade-1 homology to
/// project onto. The harmonic dimension is read off the complex and must
/// equal the surface's first Betti number in each case.
///
/// Deliberately not Bob or another gallery mesh: those run to several
/// thousand vertices, and the harmonic solve at that size dominates the
/// entire workspace test suite's runtime. The generated donut is the genus-1
/// case at a few dozen vertices, and being generated it needs no asset at
/// all, what it stands for is $b_1 = 2$, which its construction guarantees
/// rather than a file's contents happening to have it.
#[test]
fn hodge_decomposition_splits_orthogonally() {
  use formoniq::whitney_complex::{HilbertComplex, WhitneyComplex};
  use formoniq_studio::gallery::{QUOTIENT_CELLS, QuotientSurface};

  let torus = || {
    MeshSource::Quotient {
      surface: QuotientSurface::Donut,
      cells_axis: QUOTIENT_CELLS.default,
    }
    .build()
    .unwrap()
  };

  for (label, build, betti_1) in [
    (
      "Donut",
      &torus as &dyn Fn() -> (Complex, MeshCoords),
      2usize,
    ),
    (
      "Grid",
      &(|| {
        MeshSource::Grid {
          dim: 2,
          cells_axis: 4,
        }
        .build()
        .unwrap()
      }),
      0usize,
    ),
  ] {
    let (topology, coords) = build();
    assert!(
      topology.nsimplices(1) > 0,
      "{label}: mesh built empty (unfetched asset?)"
    );

    let input = hodge_probe_input(&topology, &coords);
    let parts = hodge_decompose(&topology, &coords, &input).expect("the decomposition solves");

    let metric = coords.to_edge_lengths_sq(&topology);
    let complex = WhitneyComplex::new(&topology, &metric);
    assert_eq!(
      complex.harmonic_dim(1),
      betti_1,
      "{label}: harmonic dimension is the first Betti number"
    );

    let mass = complex.mass(1);
    let ip = |a: &Cochain, b: &Cochain| (&mass * b.coeffs()).dot(a.coeffs());

    // The three shells reconstruct the input exactly (LU residual aside).
    let sum = parts.exact.coeffs() + parts.coexact.coeffs() + parts.harmonic.coeffs();
    let residual = (&sum - input.coeffs()).norm();
    assert!(
      residual < 1e-8,
      "{label}: shells do not sum to input ({residual})"
    );

    // Pairwise orthogonal in the $L^2 Lambda^1$ inner product.
    let scale = ip(&input, &input).sqrt().max(1e-12);
    for (a, b, name) in [
      (&parts.exact, &parts.coexact, "exact·coexact"),
      (&parts.exact, &parts.harmonic, "exact·harmonic"),
      (&parts.coexact, &parts.harmonic, "coexact·harmonic"),
    ] {
      let cross = ip(a, b).abs() / (scale * scale);
      assert!(cross < 1e-6, "{label}: {name} not orthogonal ({cross})");
    }

    // The harmonic shell is nonzero exactly when the surface has grade-1
    // homology to carry it.
    let harmonic_frac = ip(&parts.harmonic, &parts.harmonic).sqrt() / scale;
    if betti_1 > 0 {
      assert!(
        harmonic_frac > 1e-6,
        "{label}: harmonic shell vanished ({harmonic_frac})"
      );
    } else {
      assert!(
        harmonic_frac < 1e-9,
        "{label}: spurious harmonic shell ({harmonic_frac})"
      );
    }
  }
}

/// The three worked examples are all grade-1 line fields (no scalar
/// density), named for the picker, and the edge-by-vertex-pair lookup in
/// [`formoniq_studio::gallery::CochainSpec::resolve`] found every edge of the
/// triforce's coefficient table without panicking, which is the
/// actual thing under test.
#[test]
fn triforce_cochains_are_three_named_line_fields() {
  let (topology, coords) = demos::triforce();
  let scene = Scene::cochains(topology, coords, &demos::triforce_examples());
  assert!(scene.fields.is_empty());
  let names: Vec<_> = scene.line_fields.iter().map(|f| f.name.as_str()).collect();
  assert_eq!(names, ["constant field", "pure curl", "pure div"]);
}

/// The top-grade Whitney form stars to a constant density: on the flat
/// reference triangle its pointwise Hodge star is the same nonzero scalar at
/// every corner, so the surface renders as a flat color rather than blank.
#[test]
fn grade_top_whitney_basis_stars_to_a_constant_nonzero_density() {
  let scene = Scene::whitney_basis(2);
  let density = scene.fields.last().unwrap();
  let baked = realize::bake::BakedMesh::new(&scene.topology, &scene.coords);
  let colors = corner_values(
    &scene.topology,
    &scene.coords,
    &density.cochain,
    baked.fill_triangles().iter().copied(),
  );
  assert!(!colors.is_empty());
  assert!(colors.iter().all(|&v| v.abs() > 1e-9));
  assert!(colors.iter().all(|&v| (v - colors[0]).abs() < 1e-9));
}

/// The surface tint respects a basis function's support: it is exactly the
/// cells its DOF simplex bounds. Read per corner on the corner's own
/// simplex, every cell the form vanishes on reads exactly zero at all three
/// corners, so a basis function does not bleed into cells that do not contain
/// its DOF. A per-vertex tint could not state this: the DOF's
/// endpoints carry a nonzero nodal value into every incident cell.
#[test]
fn whitney_basis_support_is_exactly_its_dof_cells() {
  let (topology, coords) = demos::triforce();
  let scene = Scene::whitney_basis_mesh(topology, coords);
  let baked = realize::bake::BakedMesh::new(&scene.topology, &scene.coords);
  let n = scene.topology.dim();

  let basis = scene
    .fields
    .iter()
    .map(|f| (f.name.as_str(), &f.cochain))
    .chain(
      scene
        .line_fields
        .iter()
        .map(|f| (f.name.as_str(), &f.cochain)),
    );
  for (name, cochain) in basis {
    // The DOF simplex, as its vertex set, from the single nonzero cochain
    // entry (a Whitney basis function is dual to exactly one simplex).
    let idof = cochain
      .coeffs()
      .iter()
      .position(|&c| c.abs() > 0.5)
      .expect("a basis DOF");
    let dof_vertices = &scene
      .topology
      .skeleton_raw(cochain.grade())
      .simplex_by_kidx(idof)
      .vertices;

    let colors = corner_values(
      &scene.topology,
      &scene.coords,
      cochain,
      baked.fill_triangles().iter().copied(),
    );
    for (cc, corners) in baked.cell_corners.iter().zip(colors.chunks_exact(3)) {
      let cell = SimplexIdx::new(n, cc.cell).handle(&scene.topology);
      let supported = dof_vertices
        .iter()
        .all(|v| cell.simplex().vertices.contains(v));
      assert!(
        supported || corners.iter().all(|&v| v.abs() < 1e-9),
        "field {name} tints a cell outside the support of its DOF {dof_vertices:?}",
      );
    }
  }
}

/// Every field of every Whitney basis gallery bakes, at every dimension the
/// ambient reaches: the scene's grade reduction and the bake's dimension
/// reduction compose without a hole, and each field samples to one colormap
/// value per rendered corner, one surface displacement height per corner and
/// one segment height per mesh vertex.
#[test]
fn every_whitney_basis_field_bakes() {
  use realize::bake::{BakedMesh, PrimBatch};
  for dim in 1..=3 {
    let scene = Scene::whitney_basis(dim);
    assert!(!scene.fields.is_empty());
    let baked = BakedMesh::new(&scene.topology, &scene.coords);
    assert_eq!(baked.positions.len(), scene.coords.nvertices());
    let ncorners = match &baked.cells {
      PrimBatch::Triangles(triangles) => 3 * triangles.len(),
      _ => 0,
    };
    assert_eq!(baked.cell_corners.len(), ncorners / 3);
    let cochains = scene
      .fields
      .iter()
      .map(|f| &f.cochain)
      .chain(scene.line_fields.iter().map(|f| &f.cochain));
    for cochain in cochains {
      let colors = corner_values(
        &scene.topology,
        &scene.coords,
        cochain,
        baked.fill_triangles().iter().copied(),
      );
      assert_eq!(colors.len(), ncorners);
      let surface_heights = surface_corner_heights(
        &scene.topology,
        &scene.coords,
        cochain,
        baked.fill_triangles(),
      );
      assert_eq!(surface_heights.len(), ncorners);
      let heights = nodal_heights(&scene.topology, &scene.coords, cochain);
      assert_eq!(heights.len(), baked.positions.len());
    }
  }
}
