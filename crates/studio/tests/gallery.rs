//! Laws for [`studio::gallery`]: the particles are opt-in everywhere and no
//! preset switches them on, both quotient surfaces build with the topology
//! their gluing says and differ in orientability, the opening preset and
//! the Hodge preset resolve to the fields they mean by name rather than a
//! brittle index, every shipped mesh loads and builds nonempty, and the
//! shipped mesh names are unique and resolve.

use formoniq_studio::{
  gallery::{
    BuiltinMesh, MeshSource, QUOTIENT_CELLS, QuotientSurface, Study, presets, start_preset,
  },
  ui::{Marks, Selection},
};

/// The particles are opt-in, everywhere. Their cost does not scale with the
/// mesh: the population is a fixed count, so there is no mesh on which
/// assuming them is cheap, and a weak GPU should not spend it unasked. The
/// glyphs stay on, so a line field still has a mark.
#[test]
fn the_particles_are_opt_in() {
  let marks = Marks::default();
  assert!(!marks.particles);
  assert!(marks.glyphs, "a line field must still carry a mark");
  // No preset overrides that: a preset is in no position to decide the
  // reader can afford them.
  for preset in presets() {
    if let Some(marks) = preset.marks {
      assert!(
        !marks.particles,
        "{} switches the particles on",
        preset.name
      );
    }
  }
}

/// Both quotient surfaces build, in $RR^3$, with the topology their gluing
/// says, and the Möbius band is the gallery's non-orientable mesh,
/// which the donut is not.
///
/// That contrast is the reason the band is offered at all: the reduced-grade
/// reduction takes a coherent orientation wherever the Hodge star fires, so
/// the band is the mesh on which that path is exercised rather than assumed.
/// A picker entry that could not be non-orientable would not test it.
#[test]
fn the_quotient_surfaces_build_and_differ_in_orientability() {
  let cases = [
    (QuotientSurface::Donut, vec![1, 2, 1], false, true),
    (QuotientSurface::Moebius, vec![1, 1, 0], true, false),
  ];
  for (surface, betti, has_boundary, orientable) in cases {
    let source = MeshSource::Quotient {
      surface,
      cells_axis: QUOTIENT_CELLS.default,
    };
    let (topology, coords) = source.build().expect("a generated mesh always builds");

    assert_eq!(coords.dim(), 3, "{}: the fixed ambient", surface.name());
    assert_eq!(topology.dim(), 2, "{}: a surface", surface.name());
    assert_eq!(topology.betti_numbers(), betti, "{}", surface.name());
    assert_eq!(topology.has_boundary(), has_boundary, "{}", surface.name());
    assert_eq!(
      topology.orientation().is_some(),
      orientable,
      "{}",
      surface.name()
    );
    // The label and the CLI name reach the picker and the command line from
    // the one enum, so neither can name a surface the other cannot.
    assert_eq!(source.label(), surface.label());
    assert_eq!(QuotientSurface::from_name(surface.name()), Some(surface));
  }
}

/// The opening preset resolves to the field it means. Its selection is an
/// index into the scene's line fields, so it is exactly the kind of thing
/// that goes quietly wrong when the study's cochains are reordered, this
/// pins it to the field's name instead, which is what the preset is really
/// choosing.
#[test]
fn the_start_preset_opens_on_the_curl_field() {
  let preset = start_preset();
  let mesh = preset.mesh.build().expect("the starting mesh builds");
  let scene = preset.study.build(&mesh);
  let Some(Selection::Line(index)) = preset.selection else {
    panic!("the start preset opens on a line field");
  };
  assert_eq!(scene.line_fields[index].name, "pure curl");
  // And it opens on the default marks rather than choosing for the reader.
  assert!(preset.marks.is_none());
}

/// Every shipped asset loads. This is what the generated table buys and what
/// it risks: the picker now offers whatever is in `assets/meshes`, so a file
/// dropped in with an unreadable body, or an extension whose reader cannot
/// actually parse it, becomes a broken entry in the UI rather than a compile
/// error. Building each one here is the check that the directory and the
/// readers agree.
///
/// A closed surface is not asserted, a shipped mesh need not be closed,
/// but a mesh with no cells is either an unfetched LFS pointer or a file that
/// is not a mesh at all, and neither belongs in the picker.
#[test]
fn every_shipped_mesh_loads() {
  assert!(
    BuiltinMesh::all().len() > 0,
    "the asset directory must ship at least one mesh"
  );
  for builtin in BuiltinMesh::all() {
    let (topology, coords) = builtin
      .build()
      .unwrap_or_else(|e| panic!("{}: {e}", builtin.name()));
    assert!(
      topology.nsimplices(topology.dim()) > 0,
      "{}: built empty (unfetched LFS asset?)",
      builtin.name()
    );
    assert_eq!(
      coords.nvertices(),
      topology.nsimplices(0),
      "{}: coordinates and vertices disagree",
      builtin.name()
    );
  }
}

/// The names the CLI accepts are the picker's, and they are unique, the
/// file stems, so two assets differing only by extension would collide and
/// `from_name` would silently resolve to whichever sorted first.
#[test]
fn shipped_mesh_names_are_unique_and_resolve() {
  let names: Vec<&str> = BuiltinMesh::all().map(|m| m.name()).collect();
  let unique: std::collections::HashSet<&&str> = names.iter().collect();
  assert_eq!(
    unique.len(),
    names.len(),
    "duplicate mesh names in {names:?}"
  );
  for builtin in BuiltinMesh::all() {
    assert_eq!(BuiltinMesh::from_name(builtin.name()), Some(builtin));
  }
}

/// The other preset that names a field explicitly. Its `Line(3)` is an index
/// into the scene its study builds, so it breaks silently when the shells are
/// reordered, pinned here to the shell's name instead.
///
/// Checked on the generated donut rather than on its own mesh (Bob): what the
/// index depends on is the shell ordering of `Study::HodgeDecomposition` and
/// the surface's first Betti number, and the donut has the same $b_1 = 2$ at
/// a few dozen vertices instead of ~3000. It is a faithful stand-in for what
/// is being tested, not a weaker one, and it keeps Bob's harmonic solve out
/// of the test suite.
#[test]
fn the_hodge_preset_opens_on_the_harmonic_shell() {
  let hodge = presets()
    .into_iter()
    .find(|p| matches!(p.study, Study::HodgeDecomposition))
    .expect("a Hodge decomposition preset");
  let mesh = MeshSource::Quotient {
    surface: QuotientSurface::Donut,
    cells_axis: QUOTIENT_CELLS.default,
  }
  .build()
  .expect("a generated mesh always builds");
  let scene = hodge.study.build(&mesh);
  let Some(Selection::Line(index)) = hodge.selection else {
    panic!("the Hodge preset opens on a line field");
  };
  assert!(
    index < scene.line_fields.len(),
    "Hodge preset opens on line {index} of {}",
    scene.line_fields.len()
  );
  assert!(
    scene.line_fields[index].name.contains("harmonic"),
    "the Hodge preset opens on the harmonic shell, got {:?}",
    scene.line_fields[index].name
  );
}
