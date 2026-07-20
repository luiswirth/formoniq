//! The gallery control panel: a pure function of a `PanelModel` snapshot,
//! returning the changes the user requested this frame as a `PanelResponse`.
//! The windowed wrapper (`app.rs`) builds the model, applies the response, and
//! owns everything stateful (the gallery, the scene, the file dialog) that the
//! panel itself never touches.

use exterior::ExteriorGrade;
use simplicial::{topology::simplex::Simplex, Dim};

use crate::gallery::{
  BuiltinMesh, MeshSource, Preset, Study, DEFAULT_NMODES, DEFAULT_TRAJECTORY_STEPS,
  GRID_CELLS_DEFAULT, GRID_CELLS_MAX, GRID_DIM_DEFAULT, GRID_DIM_MAX, HEAT_FINAL_TIME,
  REFERENCE_CELL_DIM, REFERENCE_CELL_DIM_MAX, SPHERE_SUBDIVISIONS, SPHERE_SUBDIVISIONS_MAX,
  WAVE_FINAL_TIME,
};
use crate::scene::{dof_label, FieldOffers};

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
/// How one $k$-skeleton is drawn: whether it appears, and whether it reflects
/// the field or is the structural geometry ink. The two are independent -- hiding
/// a skeleton and coloring it are separate choices, and coloring is a no-op while
/// it is hidden.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct SkeletonView {
  pub(crate) visible: bool,
  pub(crate) colored: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct MeshView {
  /// The view of each $k$-skeleton, indexed by $k in {0, 1, 2}$: the points, the
  /// edges, the faces. They are peers -- one uniform pair of questions (drawn?
  /// colored?) asked of every skeleton the mesh has, no privileged "surface"
  /// among them. Which skeletons actually exist is the mesh's dimension's answer
  /// ($k <= min(n, 2)$); a skeleton the mesh lacks is simply never offered.
  pub(crate) skeletons: [SkeletonView; 3],
}

impl MeshView {
  /// The view of the $k$-skeleton.
  pub(crate) fn skeleton(&self, k: usize) -> SkeletonView {
    self.skeletons[k]
  }
}

impl Default for MeshView {
  fn default() -> Self {
    // Faces reflect the field (the historical fill); edges and points are the
    // structural geometry ink, so the default view is a colored surface framed
    // by a plain graph.
    Self {
      skeletons: [
        SkeletonView {
          visible: true,
          colored: false,
        },
        SkeletonView {
          visible: true,
          colored: false,
        },
        SkeletonView {
          visible: true,
          colored: true,
        },
      ],
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

/// Below this viewport width, in egui points, the two sidebars cannot dock
/// beside a viewport worth looking at: 180 and 250 points of panel leave a
/// phone with nothing in the middle. It is a threshold on the *window*, not a
/// platform check -- a narrow desktop window gets the same layout, and that is
/// the point. Above it the docked layout is unchanged.
pub(crate) const COMPACT_WIDTH: f32 = 720.0;

/// Whether a sidebar is shown.
///
/// [`Self::Auto`] defers to the layout -- open where there is room to dock it,
/// closed where there is not -- and is what a session starts in, so neither a
/// phone nor a desktop needs the default written down for it. A toggle or a
/// drag makes it explicit, and it stays explicit: once a reader has said what
/// they want, the layout does not override them on the next resize.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) enum Visibility {
  #[default]
  Auto,
  Shown,
  Hidden,
}

impl Visibility {
  /// What this resolves to given what the layout would do on its own.
  fn resolve(self, layout_default: bool) -> bool {
    match self {
      Visibility::Auto => layout_default,
      Visibility::Shown => true,
      Visibility::Hidden => false,
    }
  }

  fn of(shown: bool) -> Self {
    if shown {
      Visibility::Shown
    } else {
      Visibility::Hidden
    }
  }
}

/// Which sidebars the reader has open. Both are collapsible at every width:
/// hiding the controls to see the scene is not a thing only a small screen
/// wants.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) struct Sidebars {
  pub(crate) browser: Visibility,
  pub(crate) inspector: Visibility,
}

/// Which of a line field's marks are drawn.
///
/// The two are not a choice between renderings of one thing; they answer
/// different questions about the same reduced grade-1 field. The glyphs are the
/// blunt question: not what the field does over a distance, but what it *is* at
/// a point -- read off the Whitney interpolant with no integration at all, at
/// points the atlas places rather than a tracer's seeding chooses.
///
/// The particles are the other question, the field's *dynamics* -- where the
/// flow carries a point, and how fast, legible only in motion. Both are traced
/// through the same atlas and the same transitions, one evaluated pointwise and
/// one integrated on the GPU, so neither is the other's approximation.
///
/// Hence toggles rather than a mode: which of them a reader wants is a question
/// about what they are looking for, not something the code can decide from the
/// object.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Marks {
  pub(crate) glyphs: bool,
  pub(crate) particles: bool,
}

impl Default for Marks {
  fn default() -> Self {
    Self {
      // The glyphs are the cheap reading and the one that always says
      // something: a lattice evaluated once per bake, then drawn.
      glyphs: true,
      // The particles are the expensive one, and expensive *everywhere*: the
      // population is a fixed count (`PARTICLE_COUNT`), advected every frame
      // with the deposit atlas stepped alongside it, so a four-triangle mesh
      // costs what a hundred-thousand-cell one does. There is no mesh on which
      // they come free and so no reason to assume them, and the cost is paid
      // continuously rather than once. A weak GPU should not have to spend it
      // to find out whether it wanted to.
      //
      // Nothing is lost by asking: the glyphs carry the direction and the fill
      // carries the magnitude, so the field is fully readable without them.
      // What the particles add is the dynamics, which is a second question
      // about the same reduction.
      particles: false,
    }
  }
}

/// Which field of a scene is on display: its reduced grade decides the mark
/// ([`crate::scene::Scene`]'s own rule), and this is that choice's UI-facing
/// form -- a scalar field colors the surface with its own value; a line field
/// colors the surface with its nodal magnitude and draws its glyphs and
/// particles on top. `PartialEq` so `egui::Ui::radio_value` can bind directly to
/// it.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Selection {
  Scalar(usize),
  Line(usize),
}

/// One mode of the currently shown scene, as the picker needs it: the field's
/// [`Selection`], its original grade (before the reduction to a render mark),
/// its eigenvalue (for the degeneracy layout), its DOF simplex (for the basis
/// grid) and its full name (for the hover). The render mark the selection
/// resolves to is decided elsewhere by the reduced grade; here a mode is just
/// a selectable cell.
pub(crate) struct Entry<'a> {
  pub(crate) selection: Selection,
  pub(crate) grade: ExteriorGrade,
  pub(crate) eigenvalue: Option<f64>,
  pub(crate) dof: Option<&'a Simplex>,
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
/// The DOF picker's text for one entry: its DOF simplex formatted as a label,
/// falling back to the field's name where there is no DOF -- not reached from
/// the picker (which opens only when every entry has one), but total rather than
/// a panic.
fn dof_text(entry: &Entry) -> String {
  entry
    .dof
    .map(dof_label)
    .unwrap_or_else(|| entry.name.to_string())
}

fn render_modes(ui: &mut egui::Ui, entries: &[Entry], selection: &mut Selection, n: Dim) {
  if let Some(shells) = degeneracy_shells(entries.iter().map(|e| e.eigenvalue)) {
    pyramid(ui, &shells, entries, selection);
  } else if entries.iter().all(|e| e.dof.is_some()) {
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
    .selected_text(dof_text(selected_entry))
    .show_ui(ui, |ui| {
      for &idx in &members {
        let entry = &entries[idx];
        let selected = *selection == entry.selection;
        if ui.selectable_label(selected, dof_text(entry)).clicked() {
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
/// A $k$-skeleton's label, with the render primitive it draws to: the reader
/// sees both the intrinsic object and its picture. Faces, edges, points are the
/// $k <= 2$ the ambient reaches.
pub(crate) fn skeleton_label(k: usize) -> String {
  let primitive = match k {
    0 => "points",
    1 => "edges",
    _ => "faces",
  };
  format!("{k}-skeleton · {primitive}")
}

fn skeleton_hover(k: usize) -> &'static str {
  match k {
    0 => "The 0-skeleton: a disc at every vertex, the graph's nodes",
    1 => "The 1-skeleton: every edge, the graph's links",
    _ => "The 2-skeleton: the filled faces. Also what writes depth -- with it off, the far side shows through",
  }
}

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
  /// See [`PanelResponse::sidebars`].
  pub(crate) sidebars: Sidebars,
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
  /// The displayed field's trajectory position `(solve_time, duration)`, when it
  /// is a sampled solve (heat, wave). `None` for a static field or an eigenmode;
  /// at most one of this and [`Self::eigenvalue`] is `Some`. Drives the transport
  /// readout and enables play/pause for a trajectory.
  pub(crate) trajectory: Option<(f64, f64)>,
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
  /// Which sidebars are open after this frame.
  pub(crate) sidebars: Sidebars,
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
  /// of a snapshot can own. Native only, with the button that raises it.
  #[cfg(not(target_arch = "wasm32"))]
  pub(crate) load_obj_clicked: bool,
  /// Whether "Export PNG…" was clicked -- opens the save dialog the caller
  /// owns, and on a pick the current frame (this field, this camera, this
  /// instant) is written as a still. Same reason as `load_obj_clicked`: the
  /// dialog is stateful, so it cannot live in this pure function. Native only.
  #[cfg(not(target_arch = "wasm32"))]
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
  // The one layout decision, and it is a function of the viewport rather than a
  // mode anything stores: below this width the two sidebars cannot dock beside
  // a viewport worth looking at, so they stop being docked. Their *content* is
  // untouched -- the panel taxonomy still mirrors the two objects on screen
  // (see this crate's CLAUDE.md); what changes is only whether a panel sits
  // beside the scene or over it.
  let compact = ui.available_width() < COMPACT_WIDTH;
  // What the layout does when the reader has not said: dock both where there is
  // room, open neither where there is not, so a narrow viewport shows the scene
  // first. Either way both are collapsible, and a toggle overrides this.
  let layout_default = !compact;
  let mut browser_open = model.sidebars.browser.resolve(layout_default);
  let mut inspector_open = model.sidebars.inspector.resolve(layout_default);
  // What the panels were actually drawn with this frame. Anything that differs
  // by the end of it -- a toggle, a panel dragged shut -- is the reader
  // speaking, and is what makes the visibility explicit.
  let drawn = (browser_open, inspector_open);
  let side_width = if compact {
    (ui.available_width() * 0.85).min(280.0)
  } else {
    180.0
  };

  let mut requested_mesh = model.mesh_source.clone();
  let mut requested_study = model.study.clone();
  let mut requested_preset = None;
  let mut selection = model.selection;
  let mut mesh_view = model.mesh_view;
  let mut field_view = model.field_view;
  let mut post = model.post;
  let mut orthographic = model.orthographic;
  let mut playing = model.playing;
  #[cfg(not(target_arch = "wasm32"))]
  let mut load_obj_clicked = false;
  #[cfg(not(target_arch = "wasm32"))]
  let mut export_png_clicked = false;

  // Left sidebar: the browser. The curated presets on top -- each a point in
  // the mesh × study product, chosen as a whole -- then the raw two axes below.
  egui::Panel::left("browser")
    .default_size(side_width)
    .show_collapsible(ui, &mut browser_open, |ui| {
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
        let on_heat = matches!(model.study, Study::Heat { .. });
        if ui.selectable_label(on_heat, "Heat").clicked() && !on_heat {
          requested_study = Study::Heat {
            nsteps: DEFAULT_TRAJECTORY_STEPS,
            final_time: HEAT_FINAL_TIME,
          };
        }
        let on_wave = matches!(model.study, Study::Wave { .. });
        if ui.selectable_label(on_wave, "Wave").clicked() && !on_wave {
          requested_study = Study::Wave {
            nsteps: DEFAULT_TRAJECTORY_STEPS,
            final_time: WAVE_FINAL_TIME,
          };
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
              dim: GRID_DIM_DEFAULT,
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
          for builtin in BuiltinMesh::all() {
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
        MeshSource::Grid { dim, cells_axis } => {
          ui.add(egui::Slider::new(dim, 1..=GRID_DIM_MAX).text("dimension"));
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
      // caller, which owns the (native-only) `egui_file_dialog` state. Absent on
      // the web, which has no local filesystem to browse.
      #[cfg(not(target_arch = "wasm32"))]
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
    .default_size(if compact { side_width } else { 250.0 })
    .show_collapsible(ui, &mut inspector_open, |ui| {
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
          Study::Heat { .. } => {
            ui.label("∂ₜu = −Δu");
          }
          Study::Wave { .. } => {
            ui.label("∂ₜₜu = −Δu");
          }
          Study::Cochains(_) => {}
        };
        render_modes(ui, &model.entries, &mut selection, model.scene_dim);
      }

      // What is drawn, in two sections by which object the setting reads: the
      // mesh, and the field read on it. That is `display.rs`'s own seam between
      // `MeshDisplay` and `FieldDisplay`, so these mirror a distinction the code
      // already makes rather than laying a second taxonomy over it.
      // The skeletons, as peers: one row per k-skeleton the mesh has
      // (k <= min(n, 2)), faces at the top, each with the same two questions --
      // drawn, and colored by the field or left as structural geometry.
      ui.separator();
      ui.label("Skeletons");
      for k in (0..=model.scene_dim.min(2)).rev() {
        let sk = &mut mesh_view.skeletons[k];
        ui.horizontal(|ui| {
          ui.checkbox(&mut sk.visible, skeleton_label(k))
            .on_hover_text(skeleton_hover(k));
          ui.add_enabled(sk.visible, egui::Checkbox::new(&mut sk.colored, "color"))
            .on_hover_text("Reflect the selected field on this skeleton as a heatmap, instead of the structural geometry ink");
        });
      }

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
      // separate, later concern; this is the plain screenshot. Native only:
      // there is no local filesystem to save to on the web.
      #[cfg(not(target_arch = "wasm32"))]
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
      // The sidebar toggles, at every width. Lit while the panel is open, so
      // the control shows the state rather than only changing it.
      if ui
        .selectable_label(browser_open, "\u{2630}")
        .on_hover_text("Browser")
        .clicked()
      {
        browser_open = !browser_open;
        // Nothing to dock beside: a second panel would bury the scene the
        // first one is already covering most of.
        if compact && browser_open {
          inspector_open = false;
        }
      }
      if ui
        .selectable_label(inspector_open, "\u{2699}")
        .on_hover_text("Inspector")
        .clicked()
      {
        inspector_open = !inspector_open;
        if compact && inspector_open {
          browser_open = false;
        }
      }
      ui.separator();
      let omega = model.eigenvalue.map(|l| l.max(0.0).sqrt());
      // A field runs a clock when it oscillates (an eigenmode) or when it is a
      // sampled trajectory; only a static or harmonic field has nothing to play.
      let has_clock = omega.is_some_and(|w| w > 1e-9) || model.trajectory.is_some();
      let label = if playing { "⏸" } else { "▶" };
      if ui
        .add_enabled(has_clock, egui::Button::new(label))
        .on_hover_text(if playing { "Pause" } else { "Play" })
        .clicked()
      {
        playing = !playing;
      }
      ui.separator();

      if let Some((solve_time, duration)) = model.trajectory {
        ui.monospace(format!("t = {solve_time:.3} / {duration:.3}"));
      } else {
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
      }
    });
    ui.add_space(2.0);
  });

  // Only a change from what was drawn is the reader speaking; anything
  // untouched stays on whatever it was, `Auto` included, so resizing the window
  // still moves a sidebar the reader has never touched.
  let sidebars = Sidebars {
    browser: if browser_open == drawn.0 {
      model.sidebars.browser
    } else {
      Visibility::of(browser_open)
    },
    inspector: if inspector_open == drawn.1 {
      model.sidebars.inspector
    } else {
      Visibility::of(inspector_open)
    },
  };

  PanelResponse {
    sidebars,
    requested_mesh,
    requested_study,
    requested_preset,
    selection,
    mesh_view,
    field_view,
    post,
    orthographic,
    playing,
    #[cfg(not(target_arch = "wasm32"))]
    load_obj_clicked,
    #[cfg(not(target_arch = "wasm32"))]
    export_png_clicked,
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// The width threshold means what it claims: a phone falls below it, a
  /// desktop window above, and the two docked sidebars genuinely do not fit
  /// beside a viewport at phone width -- which is the condition the compact
  /// layout exists to escape.
  #[test]
  fn the_threshold_separates_a_phone_from_a_desktop() {
    let phone = 390.0 / UI_ZOOM_FOR_TEST;
    assert!(
      phone < COMPACT_WIDTH,
      "a phone ({phone} points) must land in the compact layout"
    );
    assert!(
      180.0 + 250.0 > phone,
      "the docked sidebars must genuinely exceed a phone's width"
    );
    const { assert!(1280.0 / UI_ZOOM_FOR_TEST > COMPACT_WIDTH) };
  }

  /// A compact sidebar never covers the whole screen: it is an overlay, not a
  /// takeover, so the scene stays partly visible even with one open.
  #[test]
  fn a_compact_sidebar_leaves_some_scene_showing() {
    for width in [320.0_f32, 390.0, 540.0, 700.0] {
      let side = (width * 0.85).min(280.0);
      assert!(
        side < width,
        "width {width}: sidebar {side} covers everything"
      );
    }
  }

  /// The layout decides only what the reader has not. `Auto` follows the
  /// width -- docked where there is room, closed where there is not -- and an
  /// explicit choice outranks it at any width, which is what makes the panels
  /// collapsible on a desktop rather than only on a phone.
  #[test]
  fn visibility_defers_to_the_layout_only_until_the_reader_speaks() {
    let wide = true;
    let narrow = false;
    assert!(Visibility::Auto.resolve(wide));
    assert!(!Visibility::Auto.resolve(narrow));
    // Explicit wins either way.
    assert!(Visibility::Shown.resolve(narrow));
    assert!(!Visibility::Hidden.resolve(wide));
  }

  /// A toggle sticks. The first version of this compared the post-frame state
  /// against the *resolved* value rather than the value the panels were drawn
  /// with, so opening a closed panel was immediately undone and the button did
  /// nothing at all. The round trip is the test: resolve, toggle, write back,
  /// resolve again.
  #[test]
  fn toggling_a_sidebar_survives_the_frame() {
    for layout_default in [true, false] {
      let stored = Visibility::Auto;
      let drawn = stored.resolve(layout_default);

      // The reader clicks it.
      let toggled = !drawn;
      let written = if toggled == drawn {
        stored
      } else {
        Visibility::of(toggled)
      };

      assert_ne!(written, Visibility::Auto, "a click must become explicit");
      assert_eq!(
        written.resolve(layout_default),
        toggled,
        "the next frame must draw what the click asked for"
      );
    }
  }

  /// An untouched sidebar stays on `Auto`, so resizing the window still moves
  /// it -- the state records a decision, not every frame's outcome.
  #[test]
  fn an_untouched_sidebar_keeps_following_the_layout() {
    let stored = Visibility::Auto;
    let drawn = stored.resolve(true);
    let written = if drawn == stored.resolve(true) {
      stored
    } else {
      Visibility::of(drawn)
    };
    assert_eq!(written, Visibility::Auto);
    // And it follows the width when that changes.
    assert!(!written.resolve(false));
  }

  /// Mirrors `app.rs`'s `UI_ZOOM`: the panel widths above are egui points, and
  /// the zoom is what turns a device's pixels into them.
  const UI_ZOOM_FOR_TEST: f32 = 1.25;

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
