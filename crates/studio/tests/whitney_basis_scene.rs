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
  // A grade-0 field's cochain is the nodal 0-cochain itself, one value per
  // vertex; the reduced-grade sampling and its discontinuity only bite above
  // grade 0.
  for (i, field) in scene.fields.iter().take(3).enumerate() {
    for (j, &v) in field.cochain.coeffs().iter().enumerate() {
      let expected = if i == j { 1.0 } else { 0.0 };
      assert!((v - expected).abs() < 1e-12);
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
