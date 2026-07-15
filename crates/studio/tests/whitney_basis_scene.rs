use formoniq_studio::scene::Scene;

/// The grade-0 Whitney basis functions are the barycentric coordinates
/// themselves: as vertex-valued cochains, $W_i(v_j) = delta_(i j)$.
#[test]
fn grade0_whitney_basis_is_the_standard_basis_at_vertices() {
  let scene = Scene::whitney_basis(2, 8);
  assert_eq!(scene.fields.len(), 3);
  for (i, field) in scene.fields.iter().enumerate() {
    for (j, &v) in field.values().iter().enumerate() {
      let expected = if i == j { 1.0 } else { 0.0 };
      assert!((v - expected).abs() < 1e-12);
    }
  }
}

/// Grade-1 basis functions sharp to nonzero tangent vectors lying in the
/// reference triangle's own plane -- the standard embedding's ambient frame
/// coincides with the cell's local frame, so no out-of-plane component
/// should appear.
#[test]
fn grade1_whitney_basis_sharps_to_in_plane_vectors() {
  let scene = Scene::whitney_basis(2, 8);
  assert_eq!(scene.vector_fields.len(), 3);
  for field in &scene.vector_fields {
    assert!(field.max_magnitude() > 1e-6);
    for sample in &field.samples {
      assert!(sample.vector.z.abs() < 1e-12);
    }
  }
}

/// Grades above 1 have no glyph yet and are intentionally skipped, not
/// silently misrendered.
#[test]
fn higher_grades_are_skipped_not_special_cased() {
  let scene = Scene::whitney_basis(3, 4);
  // Tetrahedron: grade 0 (4 vertices) + grade 1 (6 edges) rendered;
  // grade 2 (4 faces) and grade 3 (1 cell) skipped for now.
  assert_eq!(scene.fields.len(), 4);
  assert_eq!(scene.vector_fields.len(), 6);
}
