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

/// The dispatch is total in every dimension via min(k, n-k), and the $n$ is the
/// **render surface's**, because a mark is the mark of the manifold it is drawn
/// on. A solid is seen through its boundary, so for the tetrahedron the
/// reduction is taken against $dim diff K = 2$, not against 3:
///
/// - grade 0 traces to a 0-form on $diff K$: reduced grade 0, a density;
/// - grade 1 traces to a 1-form on $diff K$: reduced grade $min(1, 1) = 1$, a
///   line field -- the arrows, and the only grade that gets them here;
/// - grade 2 traces to $diff K$'s *top* form: reduced grade $min(2, 0) = 0$, a
///   density. In the volume it would reduce to grade 1, but a flux 2-form has
///   no direction on the surface it is shown through, only an areal density;
/// - grade 3 does not trace at all ($C^3(diff K) = 0$), so it keeps the
///   parent's reduction $min(3, 0) = 0$ and stays a density, read by sampling
///   the cells behind the boundary until a volume mark owns it.
///
/// None are dropped, which is the totality claim; what changed is that the
/// grade-2 basis is now filed where it is actually drawable.
#[test]
fn tetrahedron_dispatches_every_grade_against_the_render_surface() {
  let scene = Scene::whitney_basis(3);
  // Scalar: grade 0 (4 vertices) + grade 2 (4 faces) + grade 3 (1 cell).
  assert_eq!(scene.fields.len(), 9);
  // Line: grade 1 (6 edges) alone.
  assert_eq!(scene.line_fields.len(), 6);
}
