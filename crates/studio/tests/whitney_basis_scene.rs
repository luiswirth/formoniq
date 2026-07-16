use formoniq_studio::scene::Scene;

/// On the reference triangle ($n = 2$) the dispatch is total: grade 0 and the
/// top grade 2 (starred to a density) are scalar fields, grade 1 is a line
/// field. The grade-0 basis lands first, so the three scalar fields ahead of
/// the single grade-2 density are the barycentric coordinates themselves,
/// $W_i(v_j) = delta_(i j)$.
#[test]
fn grade0_whitney_basis_is_the_standard_basis_at_vertices() {
  let scene = Scene::whitney_basis(2);
  // 3 grade-0 scalar fields, then 1 grade-2 density.
  assert_eq!(scene.fields.len(), 4);
  for (i, field) in scene.fields.iter().take(3).enumerate() {
    for (j, &v) in field.values().iter().enumerate() {
      let expected = if i == j { 1.0 } else { 0.0 };
      assert!((v - expected).abs() < 1e-12);
    }
  }
}

/// The top-grade Whitney form stars to a constant density: on the flat
/// reference triangle its pointwise Hodge star is the same nonzero scalar at
/// every vertex, so the surface renders as a flat color rather than blank.
#[test]
fn grade2_whitney_basis_stars_to_a_constant_nonzero_density() {
  let scene = Scene::whitney_basis(2);
  let density = scene.fields.last().unwrap();
  let values = density.values();
  assert!(values.iter().all(|&v| v.abs() > 1e-9));
  let first = values[0];
  assert!(values.iter().all(|&v| (v - first).abs() < 1e-9));
}

/// Grade-1 basis functions reduce to nonzero tangent line fields lying in the
/// reference triangle's own plane -- the standard embedding's ambient frame
/// coincides with the cell's local frame, so no out-of-plane direction
/// component should appear.
#[test]
fn grade1_whitney_basis_is_an_in_plane_line_field() {
  let scene = Scene::whitney_basis(2);
  assert_eq!(scene.line_fields.len(), 3);
  for field in &scene.line_fields {
    assert!(field.bounds().1 > 1e-6);
    for direction in &field.direction {
      assert!(direction.z.abs() < 1e-12);
    }
  }
}

/// The dispatch is total in every dimension via min(k, n-k): on the
/// tetrahedron ($n = 3$) grade 0 and grade 3 are densities, grade 1 and grade 2
/// (starred to grade 1) are line fields -- none are dropped.
#[test]
fn tetrahedron_dispatches_every_grade_via_reduced_grade() {
  let scene = Scene::whitney_basis(3);
  // Scalar: grade 0 (4 vertices) + grade 3 (1 cell). Line: grade 1 (6 edges)
  // + grade 2 (4 faces, each starred to a grade-1 line field).
  assert_eq!(scene.fields.len(), 5);
  assert_eq!(scene.line_fields.len(), 10);
}
