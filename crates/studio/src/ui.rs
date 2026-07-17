//! The gallery control panel: a pure function of a `PanelModel` snapshot,
//! returning the changes the user requested this frame as a `PanelResponse`.
//! The windowed wrapper (`app.rs`) builds the model, applies the response, and
//! owns everything stateful (the gallery, the scene, the file dialog) that the
//! panel itself never touches.

use exterior::ExteriorGrade;
use manifold::Dim;

use crate::gallery::{
  BuiltinMesh, MeshSource, Preset, Study, DEFAULT_NMODES, GRID_CELLS_DEFAULT, GRID_CELLS_MAX,
  REFERENCE_CELL_DIM, REFERENCE_CELL_DIM_MAX, SPHERE_SUBDIVISIONS, SPHERE_SUBDIVISIONS_MAX,
};
use crate::scene::FieldOffers;

/// How much of the scene's light survives to the display.
///
/// A ladder, not a product: each rung includes the one below it, and the fourth
/// cell of the two-checkbox version -- bloom without the curve -- is not a
/// picture anyone wants. The glow would be added and then clipped, so the cores
/// go flat white and only the fringes survive. Offering it would be a knob whose
/// only use is to be wrong.
///
/// The reason the choice exists at all is that there is no right answer. The
/// scene target is unbounded and the display is not, so keeping the range above
/// 1 *must* spend range below it -- see `display_transform` in the preamble.
/// Whether the dynamic range or the palette matters more is a question about
/// what is being looked at, which is exactly the kind the code cannot settle
/// from the object.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) enum Post {
  /// The display's range, hard. What an 8-bit target did implicitly, so this is
  /// not an approximation of the renderer before HDR but exactly it.
  Clamp,
  /// The whole range, compressed by the filmic curve.
  Tonemap,
  /// The curve, plus the light that spills past what can be shown.
  #[default]
  Bloom,
}

impl Post {
  pub(crate) fn label(self) -> &'static str {
    match self {
      Self::Clamp => "Clamp",
      Self::Tonemap => "Tone map",
      Self::Bloom => "Bloom",
    }
  }

  /// Every rung above [`Post::Clamp`] maps the full range.
  pub(crate) fn tone_maps(self) -> bool {
    self != Self::Clamp
  }

  pub(crate) fn blooms(self) -> bool {
    self == Self::Bloom
  }

  pub(crate) const ALL: [Self; 3] = [Self::Clamp, Self::Tonemap, Self::Bloom];
}

/// What of the mesh itself is drawn: [`crate::display::MeshDisplay`]'s items.
///
/// The seam is `display.rs`'s own: the mesh is the object every scene has, and
/// the field is read *on* it. So these are independent of which field is
/// selected and unchanged by switching one -- and there is no availability rule
/// to write, because a scene without geometry is not a scene.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct MeshView {
  /// The filled cells. The surface is the mesh's own geometry -- a field only
  /// paints it -- which is why it is here rather than among the field's marks.
  ///
  /// It is also what writes depth, so with it off the marks and the wireframe
  /// stop occluding and the far side of the mesh shows through. That is the
  /// picture a reader turning it off is asking for, not an artifact to guard
  /// against.
  pub(crate) surface: bool,
  /// The drawn 1-skeleton: an overlay on a surface, the cells themselves on a
  /// curve. One item under one name, since the bake already unified the two --
  /// so toggling it off on a 1-manifold leaves an empty picture, which is the
  /// correct total answer and not a case to exclude.
  pub(crate) wireframe: bool,
}

impl Default for MeshView {
  fn default() -> Self {
    Self {
      surface: true,
      wireframe: true,
    }
  }
}

/// How the selected field is read on that mesh: [`crate::display::FieldDisplay`]'s
/// items and the deformation it drives. Which of these apply is the reduced
/// grade's answer, and [`crate::scene::Scene::offers`] is where it is asked.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct FieldView {
  /// The standing wave's displacement of the mesh along its normal.
  ///
  /// Not a mark: it is no item of the draw list but a deformation the mesh's
  /// own items ride -- the field's material writing into the mesh's geometry,
  /// which is exactly what `wave_amplitude` already is. Hence "off" is an
  /// amplitude of zero, the same zero a field with no eigenvalue is given, and
  /// it costs no branch below the display.
  pub(crate) displacement: bool,
  pub(crate) marks: Marks,
}

impl Default for FieldView {
  fn default() -> Self {
    Self {
      displacement: true,
      marks: Marks::default(),
    }
  }
}

/// Which of a line field's marks are drawn.
///
/// The three are not a choice between renderings of one thing; they answer
/// different questions about the same reduced grade-1 field. The streamlines are
/// its *geometry* -- the integral curves, evenly spaced, standing still, the
/// whole picture legible in one frame and in a still. The particles are its
/// *dynamics* -- where the flow carries a point, and how fast, legible only in
/// motion. Both are traced through the same atlas and the same transitions, one
/// on the CPU and one on the GPU, so neither is the other's approximation.
///
/// The glyphs are the third question and the blunt one: not what the field does
/// over a distance, but what it *is* at a point -- read off the Whitney
/// interpolant with no integration at all, at points the atlas places rather
/// than a tracer's seeding chooses. That is what makes them worth having beside
/// curves that are strictly more informative about the flow: a curve is a claim
/// about the field's orbits, and a glyph is a claim about the field, and only
/// the second can be checked against the cochain by eye.
///
/// Hence toggles rather than a mode: which of them a reader wants is a question
/// about what they are looking for, not something the code can decide from the
/// object.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Marks {
  pub(crate) glyphs: bool,
  pub(crate) streamlines: bool,
  pub(crate) particles: bool,
}

impl Default for Marks {
  fn default() -> Self {
    Self {
      // Off by default: the ribbons and the particles already cover the field,
      // and a lattice of arrows over them is a second reading of the same
      // reduction rather than a first look at it.
      glyphs: false,
      streamlines: true,
      particles: true,
    }
  }
}

/// Which field of a scene is on display: its reduced grade decides the mark
/// ([`crate::scene::Scene`]'s own rule), and this is that choice's UI-facing
/// form -- a scalar field colors the surface with its own value; a line field
/// colors the surface with its nodal magnitude and draws its traced streamlines
/// on top. `PartialEq` so `egui::Ui::radio_value` can bind directly to it.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Selection {
  Scalar(usize),
  Line(usize),
}

/// One mode of the currently shown scene, as the picker needs it: the field's
/// [`Selection`], its original grade (before the reduction to a render mark),
/// its eigenvalue (for the degeneracy layout), its DOF label (for the basis
/// grid) and its full name (for the hover). The render mark the selection
/// resolves to is decided elsewhere by the reduced grade; here a mode is just
/// a selectable cell.
pub(crate) struct Entry<'a> {
  pub(crate) selection: Selection,
  pub(crate) grade: ExteriorGrade,
  pub(crate) eigenvalue: Option<f64>,
  pub(crate) dof_label: Option<&'a str>,
  pub(crate) name: &'a str,
}

/// One degeneracy shell of an eigenmode list: a maximal run of consecutive
/// modes whose eigenvalues agree up to the clustering tolerance -- one
/// degenerate eigenspace. A row of the orbital pyramid.
struct Shell {
  /// A representative eigenvalue of the shell (its first member's), labelling
  /// the row.
  eigenvalue: f64,
  /// Indices into the entry list this shell was grouped from.
  members: Vec<usize>,
}

/// The relative gap above which two consecutive eigenvalues are taken to lie in
/// *different* degeneracy shells. Within a shell the discrete eigenvalues of a
/// degenerate eigenspace differ only by the mesh's small symmetry-breaking
/// error, far below this; between distinct shells they jump by an order-one
/// fraction, far above it.
const SHELL_REL_GAP: f64 = 0.3;

/// An absolute tolerance, as a fraction of the spectrum's scale, added to the
/// relative one so a cluster of (near-)zero modes -- a harmonic space, e.g. the
/// constant 0-mode or a flat torus's two 1-cocycles -- stays together instead
/// of splitting on numerical noise, where the relative gap alone has no scale.
const SHELL_ABS_FRAC: f64 = 1e-6;

/// Rounds `x` to `decimals` fractional digits, collapsing IEEE negative zero
/// to positive zero at that precision. A harmonic mode's eigenvalue is a
/// numerical zero, not exactly `0.0` (e.g. `-1e-9`), and formatting it
/// directly prints a spurious minus sign (`-0`) for a quantity that is
/// mathematically zero.
fn round_for_display(x: f64, decimals: i32) -> f64 {
  let scale = 10f64.powi(decimals);
  let rounded = (x * scale).round() / scale;
  if rounded == 0.0 {
    0.0
  } else {
    rounded
  }
}

/// Groups a list of eigenmodes into its degeneracy shells by clustering
/// consecutive near-equal eigenvalues.
///
/// The modes arrive sorted by eigenvalue; a run whose successive gaps stay
/// within [`SHELL_ABS_FRAC`]$dot lambda_max + $[`SHELL_REL_GAP`]$dot lambda$ is
/// one degenerate eigenspace -- a row of the pyramid. This reads the
/// organization straight off the spectrum, with no geometry: on $S^2$ the
/// near-equal clusters are exactly the $(2l+1)$ spherical-harmonic shells,
/// while on a generic mesh with no symmetry the eigenvalues are simple, every
/// gap exceeds the tolerance, and each row collapses to a single member ordered
/// by eigenvalue.
///
/// `None` if any field carries no eigenvalue (not an eigenmode scene, e.g. the
/// raw Whitney basis), where the caller keeps a flat list instead.
fn degeneracy_shells(eigenvalues: impl IntoIterator<Item = Option<f64>>) -> Option<Vec<Shell>> {
  let lambdas: Vec<f64> = eigenvalues.into_iter().collect::<Option<Vec<f64>>>()?;
  let scale = lambdas.iter().map(|l| l.abs()).fold(0.0, f64::max).max(1.0);
  let atol = SHELL_ABS_FRAC * scale;
  let mut shells: Vec<Shell> = Vec::new();
  for (idx, &lambda) in lambdas.iter().enumerate() {
    let same_shell =
      idx > 0 && lambda - lambdas[idx - 1] <= atol + SHELL_REL_GAP * lambdas[idx - 1].abs();
    if same_shell {
      shells.last_mut().unwrap().members.push(idx);
    } else {
      shells.push(Shell {
        eigenvalue: lambda,
        members: vec![idx],
      });
    }
  }
  Some(shells)
}

/// Renders the modes of the currently shown scene as a picker. Eigenmodes
/// (the harmonics) lay out as the orbital pyramid by degeneracy shell; raw
/// Whitney basis functions (LSFs and GSFs alike) lay out as a grade tab row
/// over a DOF dropdown instead, since they carry a DOF label but no
/// eigenvalue; anything carrying neither -- not produced today, but the
/// totality this dispatch is answering to -- falls back to one flat list.
fn render_modes(ui: &mut egui::Ui, entries: &[Entry], selection: &mut Selection, n: Dim) {
  if let Some(shells) = degeneracy_shells(entries.iter().map(|e| e.eigenvalue)) {
    pyramid(ui, &shells, entries, selection);
  } else if entries.iter().all(|e| e.dof_label.is_some()) {
    dof_picker(ui, entries, selection, n);
  } else {
    for entry in entries {
      let selected = *selection == entry.selection;
      if ui.selectable_label(selected, entry.name).clicked() {
        *selection = entry.selection;
      }
    }
  }
}

/// Picks one Whitney basis function: a grade tab row (as the eigenmode grade
/// tabs, but a view over one already-solved scene rather than a re-solve),
/// then a dropdown over that grade's DOFs alone. A mesh's DOF count is
/// unbounded (a reference cell has a handful, a built-in surface thousands),
/// and every grade shown flat at once -- the previous layout -- grew the
/// inspector to the mesh's simplex count; collapsed behind a dropdown, the
/// panel's size no longer depends on the mesh. The active grade is read off
/// the current selection, so switching tabs jumps to that grade's first DOF
/// and the next frame's tab highlight follows it.
fn dof_picker(ui: &mut egui::Ui, entries: &[Entry], selection: &mut Selection, n: Dim) {
  let current_grade = entries
    .iter()
    .find(|e| e.selection == *selection)
    .map_or(0, |e| e.grade);

  let mut active_grade = current_grade;
  ui.horizontal_wrapped(|ui| {
    for grade in 0..=n {
      if entries.iter().any(|e| e.grade == grade)
        && ui
          .selectable_label(current_grade == grade, grade_mark_label(grade, n))
          .clicked()
      {
        active_grade = grade;
      }
    }
  });

  let members: Vec<usize> = entries
    .iter()
    .enumerate()
    .filter(|(_, e)| e.grade == active_grade)
    .map(|(i, _)| i)
    .collect();
  let Some(&first_member) = members.first() else {
    return;
  };
  if active_grade != current_grade {
    *selection = entries[first_member].selection;
  }

  let selected_idx = members
    .iter()
    .copied()
    .find(|&i| entries[i].selection == *selection)
    .unwrap_or(first_member);
  let selected_entry = &entries[selected_idx];
  egui::ComboBox::from_id_salt("whitney-dof")
    .selected_text(selected_entry.dof_label.unwrap_or(selected_entry.name))
    .show_ui(ui, |ui| {
      for &idx in &members {
        let entry = &entries[idx];
        let selected = *selection == entry.selection;
        let label = entry.dof_label.unwrap_or(entry.name);
        if ui.selectable_label(selected, label).clicked() {
          *selection = entry.selection;
        }
      }
    });
}

/// Lays out one grade's eigenmodes as the orbital pyramid: one centered row per
/// degeneracy shell, rows ordered by ascending eigenvalue and labelled by it,
/// each cell a mode selector labelled by its centered within-shell offset (the
/// magnetic index $m in -l..=l$ on the sphere's $2l+1$-fold grade-0 multiplet).
/// Hovering a cell shows the mode's full name and eigenvalue.
fn pyramid(ui: &mut egui::Ui, shells: &[Shell], entries: &[Entry], selection: &mut Selection) {
  // A fixed cell size lines the columns up into a grid; a fixed-width label
  // gutter on the left holds the shell's eigenvalue and keeps the columns
  // aligned across rows. `vertical_centered` then centers each row within the
  // panel, so the shorter shells sit symmetrically over the widest one -- the
  // pyramid. The widest shell (the sphere's five-member grade-0 $l = 2$
  // multiplet) is what sets the panel's minimum width, and it fills that width
  // exactly: there is no trailing spacer, so no dead padding on the right.
  const CELL: [f32; 2] = [26.0, 20.0];
  const GUTTER: f32 = 34.0;
  ui.vertical_centered(|ui| {
    for shell in shells {
      ui.horizontal(|ui| {
        ui.add_sized(
          [GUTTER, CELL[1]],
          // A whole number for the row label -- distinct shells differ by an
          // order-one gap, so the integer part alone separates them, and the
          // precise eigenvalue lives in the cell hover and the transport bar.
          egui::Label::new(format!("λ~{:.0}", round_for_display(shell.eigenvalue, 0))),
        );
        let n = shell.members.len() as isize;
        for (pos, &idx) in shell.members.iter().enumerate() {
          let m = pos as isize - (n - 1) / 2;
          let label = if m == 0 {
            "0".to_string()
          } else {
            format!("{m:+}")
          };
          let entry = &entries[idx];
          let selected = *selection == entry.selection;
          if ui
            .add_sized(CELL, egui::Button::selectable(selected, label))
            .on_hover_text(entry.name)
            .clicked()
          {
            *selection = entry.selection;
          }
        }
      });
    }
  });
}

/// The render mark a grade-$k$ field is drawn with on an $n$-manifold, named
/// for the UI: its reduced grade $min(k, n-k)$ decides between a scalar density
/// and a tangent line field (discussion #101). Whether the reduction went
/// through a Hodge star ($k$ above the fold) is noted so the top-grade section
/// reads as a density arrived at by $star$, not a bare 0-form.
pub(crate) fn grade_mark_label(grade: ExteriorGrade, n: Dim) -> String {
  let reduced = grade.min(n - grade);
  let mark = match reduced {
    0 => "density",
    1 => "line field",
    _ => "sheet",
  };
  if grade == reduced {
    format!("grade {grade} · {mark}")
  } else {
    format!("grade {grade} · {mark} (*)")
  }
}

/// Everything the panel reads to draw one frame -- a snapshot, not a live
/// borrow, so building it and rendering the panel cannot conflict with the
/// `&mut self` calls the caller makes to apply the response afterward.
pub(crate) struct PanelModel<'a> {
  pub(crate) mesh_source: MeshSource,
  pub(crate) study: Study,
  pub(crate) is_loading: bool,
  pub(crate) loading_label: Option<String>,
  pub(crate) last_grade: ExteriorGrade,
  pub(crate) max_grade: Dim,
  pub(crate) scene_dim: Dim,
  pub(crate) entries: Vec<Entry<'a>>,
  pub(crate) mesh_error: Option<String>,
  pub(crate) selection: Selection,
  pub(crate) mesh_view: MeshView,
  pub(crate) field_view: FieldView,
  /// Which of `field_view`'s settings the selected field actually offers -- the
  /// reduced grade's answer, decided in [`crate::scene::Scene::offers`] so the
  /// panel asks rather than dispatches.
  pub(crate) offers: FieldOffers,
  pub(crate) post: Post,
  pub(crate) orthographic: bool,
  pub(crate) presets: &'a [Preset],
  /// The displayed field's Hodge-Laplace eigenvalue $lambda$, when it is an
  /// eigenmode: the transport bar reports it and its frequency $omega =
  /// sqrt(lambda)$. `None` for a field with no standing wave (a raw Whitney
  /// basis function, an explicit cochain).
  pub(crate) eigenvalue: Option<f64>,
  /// Whether the standing-wave clock is running, driving the play/pause control.
  pub(crate) playing: bool,
  /// The clock's current time $t$, for the transport readout.
  pub(crate) time: f32,
}

/// The changes the user requested this frame, for the caller to apply. Each
/// field defaults to the model's own value when nothing moved, so the caller
/// can compare against the model unconditionally rather than branching on
/// whether a widget fired.
pub(crate) struct PanelResponse {
  pub(crate) requested_mesh: MeshSource,
  pub(crate) requested_study: Study,
  /// The index into `presets` of a preset the user picked this frame, if any.
  /// A preset sets both axes and the opening field at once, so the caller
  /// applies it in preference to the individual `requested_*` fields.
  pub(crate) requested_preset: Option<usize>,
  pub(crate) selection: Selection,
  pub(crate) mesh_view: MeshView,
  pub(crate) field_view: FieldView,
  pub(crate) post: Post,
  pub(crate) orthographic: bool,
  /// Whether the standing wave should be running after this frame -- the
  /// play/pause toggle. Defaults to the model's own state when the control
  /// wasn't touched, like the other fields.
  pub(crate) playing: bool,
  /// Whether "Load OBJ…" was clicked -- the one request the panel cannot
  /// resolve itself, since opening the native file browser is the caller's
  /// (`app.rs`'s) stateful `egui_file_dialog`, not something a pure function
  /// of a snapshot can own.
  pub(crate) load_obj_clicked: bool,
  /// Whether "Export PNG…" was clicked -- opens the save dialog the caller
  /// owns, and on a pick the current frame (this field, this camera, this
  /// instant) is written as a still. Same reason as `load_obj_clicked`: the
  /// dialog is stateful, so it cannot live in this pure function.
  pub(crate) export_png_clicked: bool,
}

/// Builds and tessellates the docked shell: a left sidebar (the preset browser
/// and the two raw axes, mesh and study, for free composition), a right
/// inspector (the study's own knobs -- eigenmode grade tabs, the mode picker or
/// basis grid, or a spinner while it solves -- and the camera-mode toggle), a
/// bottom transport bar (play/pause and the standing wave's readouts), and the
/// field-name viewport overlay. No central panel is drawn, so the wgpu scene
/// shows through the middle.
///
/// A pure function of `model`: it reads no other state and its only side effect
/// is on `ctx`'s own egui pass, so the caller is free to apply the returned
/// changes however it likes.
pub(crate) fn panel(ui: &mut egui::Ui, model: &PanelModel) -> PanelResponse {
  let mut requested_mesh = model.mesh_source.clone();
  let mut requested_study = model.study.clone();
  let mut requested_preset = None;
  let mut selection = model.selection;
  let mut mesh_view = model.mesh_view;
  let mut field_view = model.field_view;
  let mut post = model.post;
  let mut orthographic = model.orthographic;
  let mut playing = model.playing;
  let mut load_obj_clicked = false;
  let mut export_png_clicked = false;

  // Left sidebar: the browser. The curated presets on top -- each a point in
  // the mesh × study product, chosen as a whole -- then the raw two axes below.
  egui::Panel::left("browser")
    .default_size(180.0)
    .show(ui, |ui| {
      ui.add_space(4.0);
      ui.heading("Browser");
      ui.separator();

      ui.label("Presets");
      for (i, preset) in model.presets.iter().enumerate() {
        if ui.selectable_label(false, preset.name).clicked() {
          requested_preset = Some(i);
        }
      }
      ui.separator();

      // The study axis. Eigenmodes and the Whitney basis are the two generic
      // studies; an explicit cochain list has no generic form to pick, so it
      // shows as the selected study only when a preset installed it.
      ui.label("Study");
      ui.horizontal_wrapped(|ui| {
        let on_eigenmodes = matches!(model.study, Study::Eigenmodes { .. });
        if ui.selectable_label(on_eigenmodes, "Eigenmodes").clicked() && !on_eigenmodes {
          requested_study = Study::Eigenmodes {
            grade: model.last_grade,
            nmodes: DEFAULT_NMODES,
          };
        }
        let on_whitney = matches!(model.study, Study::WhitneyBasis);
        if ui.selectable_label(on_whitney, "Whitney basis").clicked() {
          requested_study = Study::WhitneyBasis;
        }
        let on_hodge = matches!(model.study, Study::HodgeDecomposition);
        if ui
          .selectable_label(on_hodge, "Hodge decomposition")
          .clicked()
        {
          requested_study = Study::HodgeDecomposition;
        }
        if matches!(model.study, Study::Cochains(_)) {
          let _ = ui.selectable_label(true, "Cochains");
        }
      });
      ui.separator();

      // The mesh axis: every study runs on every mesh, so the picker is always
      // shown. A generated family resets to its default refinement when first
      // chosen; re-picking the current family keeps the slider's value.
      //
      // "Mesh source", not "Mesh": this is which object to build, while the
      // inspector's "Mesh" is what of it to draw. Two questions, and the longer
      // name is the one that says which this is.
      ui.label("Mesh source");
      egui::ComboBox::from_id_salt("mesh-source")
        .selected_text(requested_mesh.label())
        .show_ui(ui, |ui| {
          let is_sphere = matches!(requested_mesh, MeshSource::Sphere { .. });
          if ui.selectable_label(is_sphere, "Sphere").clicked() && !is_sphere {
            requested_mesh = MeshSource::Sphere {
              subdivisions: SPHERE_SUBDIVISIONS,
            };
          }
          let is_grid = matches!(requested_mesh, MeshSource::Grid { .. });
          if ui.selectable_label(is_grid, "Grid").clicked() && !is_grid {
            requested_mesh = MeshSource::Grid {
              cells_axis: GRID_CELLS_DEFAULT,
            };
          }
          let is_ref = matches!(requested_mesh, MeshSource::ReferenceCell { .. });
          if ui.selectable_label(is_ref, "Reference cell").clicked() && !is_ref {
            requested_mesh = MeshSource::ReferenceCell {
              dim: REFERENCE_CELL_DIM,
            };
          }
          let is_triforce = matches!(requested_mesh, MeshSource::Triforce);
          if ui.selectable_label(is_triforce, "Triforce").clicked() {
            requested_mesh = MeshSource::Triforce;
          }
          for builtin in BuiltinMesh::ALL {
            let selected = requested_mesh == MeshSource::Builtin(builtin);
            if ui.selectable_label(selected, builtin.label()).clicked() {
              requested_mesh = MeshSource::Builtin(builtin);
            }
          }
        });
      // A generated family or the reference cell carries a slider; a built-in,
      // the triforce or a loaded mesh has none.
      match &mut requested_mesh {
        MeshSource::Sphere { subdivisions } => {
          ui.add(egui::Slider::new(subdivisions, 0..=SPHERE_SUBDIVISIONS_MAX).text("subdivisions"));
        }
        MeshSource::Grid { cells_axis } => {
          ui.add(egui::Slider::new(cells_axis, 1..=GRID_CELLS_MAX).text("cells/axis"));
        }
        MeshSource::ReferenceCell { dim } => {
          ui.add(egui::Slider::new(dim, 1..=REFERENCE_CELL_DIM_MAX).text("dimension"));
        }
        MeshSource::Triforce
        | MeshSource::Builtin(_)
        | MeshSource::Custom { .. }
        | MeshSource::File(_) => {}
      }
      // Opens the in-egui file browser; the pick itself is retrieved by the
      // caller, which owns the (native-only) `egui_file_dialog` state.
      if ui.button("Load OBJ…").clicked() {
        load_obj_clicked = true;
      }
      if let Some(error) = &model.mesh_error {
        ui.colored_label(egui::Color32::LIGHT_RED, format!("⚠ {error}"));
      }
    });

  // Right inspector: the knobs of what is shown -- the study's own parameters
  // and the camera mode.
  egui::Panel::right("inspector")
    .default_size(250.0)
    .show(ui, |ui| {
      ui.add_space(4.0);
      ui.heading("Inspector");
      ui.separator();

      // One tab per grade of the de Rham complex; every grade is solved and
      // shown, the top grade through its Hodge star just like grade 0.
      if let Study::Eigenmodes { grade, .. } = &model.study {
        let grade = *grade;
        ui.horizontal_wrapped(|ui| {
          for g in 0..=model.max_grade {
            if ui
              .selectable_label(g == grade, grade_mark_label(g, model.max_grade))
              .clicked()
            {
              requested_study = Study::Eigenmodes {
                grade: g,
                nmodes: DEFAULT_NMODES,
              };
            }
          }
        });
        ui.separator();
      }

      if model.is_loading {
        ui.horizontal(|ui| {
          ui.add(egui::Spinner::new().size(20.0));
          if let Some(label) = &model.loading_label {
            ui.label(format!("Solving {label}…"));
          }
        });
      } else {
        match &model.study {
          Study::Eigenmodes { .. } => {
            ui.label("λ shell × order");
          }
          Study::WhitneyBasis => {
            ui.label("grade × DOF");
          }
          Study::HodgeDecomposition => {
            ui.label("ω = dα + δβ + h");
          }
          Study::Cochains(_) => {}
        };
        render_modes(ui, &model.entries, &mut selection, model.scene_dim);
      }

      // What is drawn, in two sections by which object the setting reads: the
      // mesh, and the field read on it. That is `display.rs`'s own seam between
      // `MeshDisplay` and `FieldDisplay`, so these mirror a distinction the code
      // already makes rather than laying a second taxonomy over it.
      ui.separator();
      ui.label("Mesh");
      ui.checkbox(&mut mesh_view.surface, "Surface")
        .on_hover_text("The filled cells. Also what writes depth: with it off, the marks stop occluding and the far side shows through");
      ui.checkbox(&mut mesh_view.wireframe, "Wireframe")
        .on_hover_text("The drawn 1-skeleton: an overlay on a surface, the cells themselves on a curve");

      // The field side is the only one gated, and it asks rather than
      // dispatches: which settings a field offers is its reduced grade's answer
      // (`Scene::offers`). A knob with nothing to toggle is a knob whose only
      // use is to be wrong -- and a section with no knobs is one too, so a
      // density that is no eigenmode shows no section at all.
      if model.offers.any() {
        ui.separator();
        ui.label("Field");
        if model.offers.displacement {
          ui.checkbox(&mut field_view.displacement, "Displacement")
            .on_hover_text("The standing wave, along the surface's normal: the mode's own geometry");
        }
        if model.offers.marks {
          ui.checkbox(&mut field_view.marks.glyphs, "Glyphs")
            .on_hover_text("Arrows on each cell's barycentric lattice: the field at a point");
          ui.checkbox(&mut field_view.marks.streamlines, "Streamlines")
            .on_hover_text("Evenly spaced integral curves: the field's geometry, in one frame");
          ui.checkbox(&mut field_view.marks.particles, "Particles")
            .on_hover_text("Advected points: the field's dynamics, legible in motion");
        }
      }

      // Radio, not two checkboxes: the rungs are cumulative, so the states are
      // three and not four.
      ui.separator();
      ui.label("Light");
      ui.horizontal_wrapped(|ui| {
        for rung in Post::ALL {
          ui.radio_value(&mut post, rung, rung.label());
        }
      });

      ui.separator();
      ui.checkbox(&mut orthographic, "Orthographic")
        .on_hover_text("Parallel projection: no vanishing point, so a flat mesh keeps its scale");

      // Writes exactly what is on screen -- the current field, camera and wave
      // phase -- as a PNG still, at the window's own resolution and the export
      // supersampling. The clip export the transport bar would host is a
      // separate, later concern; this is the plain screenshot.
      if ui
        .button("Export PNG…")
        .on_hover_text("Save the current view as a still")
        .clicked()
      {
        export_png_clicked = true;
      }
    });

  // Bottom transport bar: play/pause and the standing wave's readouts. Only an
  // oscillating mode ($omega > 0$) has a wave to run, so the toggle is disabled
  // for a static or harmonic field, where it would control nothing.
  egui::Panel::bottom("transport").show(ui, |ui| {
    ui.add_space(2.0);
    ui.horizontal(|ui| {
      let omega = model.eigenvalue.map(|l| l.max(0.0).sqrt());
      let has_wave = omega.is_some_and(|w| w > 1e-9);
      let label = if playing { "⏸" } else { "▶" };
      if ui
        .add_enabled(has_wave, egui::Button::new(label))
        .on_hover_text(if playing { "Pause" } else { "Play" })
        .clicked()
      {
        playing = !playing;
      }
      ui.separator();

      match omega {
        Some(omega) if omega > 1e-9 => {
          ui.monospace(format!("t = {:.2} s", model.time));
          ui.separator();
          ui.monospace(format!(
            "λ = {:.4}",
            round_for_display(model.eigenvalue.unwrap(), 4)
          ));
          ui.monospace(format!("ω = {omega:.4}"));
          ui.monospace(format!("T = {:.2} s", std::f64::consts::TAU / omega));
        }
        // A harmonic mode ($lambda = 0$) is a genuine eigenmode with no
        // oscillation: its period is unbounded, not large.
        Some(_) => {
          ui.monospace("λ = 0 · harmonic (no oscillation)");
        }
        None => {
          ui.monospace("static field · no standing wave");
        }
      }
    });
    ui.add_space(2.0);
  });

  PanelResponse {
    requested_mesh,
    requested_study,
    requested_preset,
    selection,
    mesh_view,
    field_view,
    post,
    orthographic,
    playing,
    load_obj_clicked,
    export_png_clicked,
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn shell_sizes(eigenvalues: &[f64]) -> Vec<usize> {
    degeneracy_shells(eigenvalues.iter().map(|&l| Some(l)))
      .unwrap()
      .iter()
      .map(|s| s.members.len())
      .collect()
  }

  /// The measured subdivision-3 icosphere grade-0 spectrum clusters into the
  /// $(2l+1)$ spherical-harmonic shells: the near-equal multiplets group, the
  /// order-one jumps between degrees split.
  #[test]
  fn sphere_spectrum_recovers_2l_plus_1_shells() {
    let spectrum = [0.00, 2.01, 2.01, 2.01, 6.07, 6.07, 6.07, 6.07, 6.07, 12.24];
    assert_eq!(shell_sizes(&spectrum), vec![1, 3, 5, 1]);
  }

  /// A near-zero harmonic space (a flat torus's two 1-cocycles) stays one shell
  /// rather than splitting on numerical noise, since the absolute tolerance
  /// carries a scale the relative gap alone lacks near zero.
  #[test]
  fn near_zero_harmonics_stay_one_shell() {
    let spectrum = [-1e-9, 2e-9, 4.0, 4.0];
    assert_eq!(shell_sizes(&spectrum), vec![2, 2]);
  }

  /// A generic simple spectrum -- no symmetry, no degeneracy -- degenerates the
  /// pyramid to one member per row, ordered by eigenvalue.
  #[test]
  fn simple_spectrum_gives_singletons() {
    let spectrum = [1.0, 2.5, 4.0, 6.0, 9.0];
    assert_eq!(shell_sizes(&spectrum), vec![1, 1, 1, 1, 1]);
  }

  /// A field carrying no eigenvalue (the raw Whitney basis) has no shell
  /// structure, so the caller falls back to a flat list.
  #[test]
  fn missing_eigenvalue_declines_to_shell() {
    assert!(degeneracy_shells([Some(1.0), None, Some(2.0)]).is_none());
  }
}
