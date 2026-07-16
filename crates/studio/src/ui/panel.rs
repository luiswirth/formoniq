//! The gallery control panel: a pure function of a `PanelModel` snapshot,
//! returning the changes the user requested this frame as a `PanelResponse`.
//! The windowed wrapper (`app.rs`) builds the model, applies the response, and
//! owns everything stateful (the gallery, the scene, the file dialog) that the
//! panel itself never touches.

use exterior::ExteriorGrade;
use manifold::Dim;

use crate::gallery::{
  BuiltinMesh, MeshSource, View, GRID_CELLS_DEFAULT, GRID_CELLS_MAX, SPHERE_SUBDIVISIONS,
  SPHERE_SUBDIVISIONS_MAX,
};

/// Which field of a scene is on display: its reduced grade decides the mark
/// ([`crate::scene::Scene`]'s own rule), and this is that choice's UI-facing
/// form -- a scalar field colors the surface with its own value; a line field
/// colors the surface with its nodal magnitude and draws line-integral
/// convolution on top. `PartialEq` so `egui::Ui::radio_value` can bind
/// directly to it.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Selection {
  Scalar(usize),
  Line(usize),
}

impl Selection {
  pub(crate) fn is_line(self) -> bool {
    matches!(self, Selection::Line(_))
  }
}

/// How a line field is drawn: the default is evenly-spaced [`Streamlines`], the
/// integral curves of the field traced on the manifold; [`Lic`] is the older
/// dense line-integral-convolution texture, kept as an alternative for
/// high-frequency fields a discrete curve set would undersample. Only a line
/// field's mark is selectable; a scalar field ignores this.
///
/// [`Streamlines`]: LineMark::Streamlines
/// [`Lic`]: LineMark::Lic
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum LineMark {
  Streamlines,
  Lic,
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
/// Whitney basis functions (LSFs and GSFs alike) lay out as a grid by grade
/// instead, since they carry a DOF label but no eigenvalue; anything carrying
/// neither -- not produced today, but the totality this dispatch is answering
/// to -- falls back to one flat list.
fn render_modes(ui: &mut egui::Ui, entries: &[Entry], selection: &mut Selection, n: Dim) {
  if let Some(shells) = degeneracy_shells(entries.iter().map(|e| e.eigenvalue)) {
    pyramid(ui, &shells, entries, selection);
  } else if entries.iter().all(|e| e.dof_label.is_some()) {
    grade_grid(ui, entries, selection, n);
  } else {
    for entry in entries {
      let selected = *selection == entry.selection;
      if ui.selectable_label(selected, entry.name).clicked() {
        *selection = entry.selection;
      }
    }
  }
}

/// Lays out a Whitney basis gallery (LSFs or GSFs) as one row per grade,
/// ordered $0..=n$ and labelled by [`grade_mark_label`], each row a wrapped
/// flow of DOF cells -- unlike the eigenmode pyramid there is no natural width
/// to center on, since a mesh's edge count need not match its vertex or face
/// count. Hovering a cell shows the basis function's full name.
fn grade_grid(ui: &mut egui::Ui, entries: &[Entry], selection: &mut Selection, n: Dim) {
  const CELL: [f32; 2] = [30.0, 22.0];
  for grade in 0..=n {
    let members: Vec<usize> = entries
      .iter()
      .enumerate()
      .filter(|(_, e)| e.grade == grade)
      .map(|(i, _)| i)
      .collect();
    if members.is_empty() {
      continue;
    }
    ui.label(grade_mark_label(grade, n));
    ui.horizontal_wrapped(|ui| {
      for idx in members {
        let entry = &entries[idx];
        let selected = *selection == entry.selection;
        let label = entry.dof_label.unwrap_or(entry.name);
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
}

/// Lays out one grade's eigenmodes as the orbital pyramid: one centered row per
/// degeneracy shell, rows ordered by ascending eigenvalue and labelled by it,
/// each cell a mode selector labelled by its centered within-shell offset (the
/// magnetic index $m in -l..=l$ on the sphere's $2l+1$-fold grade-0 multiplet).
/// Hovering a cell shows the mode's full name and eigenvalue.
fn pyramid(ui: &mut egui::Ui, shells: &[Shell], entries: &[Entry], selection: &mut Selection) {
  // A fixed cell size makes the columns line up into a grid; the matching
  // gutters left and right keep each row's cells centered under `vertical_centered`.
  const CELL: [f32; 2] = [30.0, 22.0];
  const GUTTER: f32 = 56.0;
  ui.vertical_centered(|ui| {
    for shell in shells {
      ui.horizontal(|ui| {
        ui.add_sized(
          [GUTTER, CELL[1]],
          egui::Label::new(format!("λ = {:.1}", shell.eigenvalue)),
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
        ui.add_space(GUTTER);
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
    format!("grade {grade} · {mark} (⋆)")
  }
}

/// Everything the panel reads to draw one frame -- a snapshot, not a live
/// borrow, so building it and rendering the panel cannot conflict with the
/// `&mut self` calls the caller makes to apply the response afterward.
pub(crate) struct PanelModel<'a> {
  pub(crate) shown_view: View,
  pub(crate) is_loading: bool,
  pub(crate) loading_label: Option<String>,
  pub(crate) last_mesh_grade: ExteriorGrade,
  pub(crate) max_grade: Dim,
  pub(crate) scene_dim: Dim,
  pub(crate) entries: Vec<Entry<'a>>,
  pub(crate) mesh_source: MeshSource,
  pub(crate) mesh_error: Option<String>,
  pub(crate) selection: Selection,
  pub(crate) line_mark: LineMark,
  pub(crate) top_down: bool,
}

/// The changes the user requested this frame, for the caller to apply. Each
/// field defaults to the model's own value when nothing moved, so the caller
/// can compare against the model unconditionally rather than branching on
/// whether a widget fired.
pub(crate) struct PanelResponse {
  pub(crate) requested_view: View,
  pub(crate) requested_mesh_source: MeshSource,
  pub(crate) selection: Selection,
  pub(crate) line_mark: LineMark,
  pub(crate) top_down: bool,
  /// Whether "Load OBJ…" was clicked -- the one request the panel cannot
  /// resolve itself, since opening the native file browser is the caller's
  /// (`app.rs`'s) stateful `egui_file_dialog`, not something a pure function
  /// of a snapshot can own.
  pub(crate) load_obj_clicked: bool,
}

/// Builds and tessellates the control panel: the family picker, the mesh's
/// per-grade tabs, that grade's orbital-pyramid mode picker (or a spinner
/// while it solves), and the camera-mode toggle. A pure function of `model`:
/// it reads no other state and its only side effect is on `ctx`'s own egui
/// pass, so the caller is free to apply the returned changes however it likes.
pub(crate) fn panel(ctx: &egui::Context, model: &PanelModel) -> PanelResponse {
  let mut requested_view = model.shown_view;
  let mut requested_mesh_source = model.mesh_source.clone();
  let mut selection = model.selection;
  let mut line_mark = model.line_mark;
  let mut top_down = model.top_down;
  let mut load_obj_clicked = false;

  egui::Window::new("Gallery").show(ctx, |ui| {
    let on_eigenmodes = matches!(model.shown_view, View::MeshGrade(_));
    ui.horizontal(|ui| {
      if ui.selectable_label(on_eigenmodes, "Eigenmodes").clicked() {
        requested_view = View::MeshGrade(model.last_mesh_grade);
      }
      if ui
        .selectable_label(
          model.shown_view == View::WhitneyBasis,
          "LSF (reference cell)",
        )
        .clicked()
      {
        requested_view = View::WhitneyBasis;
      }
      if ui
        .selectable_label(
          model.shown_view == View::WhitneyBasisMesh,
          "GSF (triforce mesh)",
        )
        .clicked()
      {
        requested_view = View::WhitneyBasisMesh;
      }
      if ui
        .selectable_label(
          model.shown_view == View::WhitneyExamplesMesh,
          "Examples (triforce mesh)",
        )
        .clicked()
      {
        requested_view = View::WhitneyExamplesMesh;
      }
    });
    ui.separator();

    if let View::MeshGrade(grade) = model.shown_view {
      // Mesh source: the surface the eigenmodes live on is a chosen input,
      // not a fixed sphere -- a generated family, a built-in gallery mesh,
      // or a loaded OBJ. Picking any remeshes and re-solves the current
      // grade (`requested_mesh_source`, applied after the closure).
      ui.horizontal(|ui| {
        egui::ComboBox::from_id_salt("mesh-source")
          .selected_text(requested_mesh_source.label())
          .show_ui(ui, |ui| {
            // A generated family resets to its default refinement when
            // first chosen; re-picking the current family keeps the
            // slider's value.
            let is_sphere = matches!(requested_mesh_source, MeshSource::Sphere { .. });
            if ui.selectable_label(is_sphere, "Sphere").clicked() && !is_sphere {
              requested_mesh_source = MeshSource::Sphere {
                subdivisions: SPHERE_SUBDIVISIONS,
              };
            }
            let is_grid = matches!(requested_mesh_source, MeshSource::Grid { .. });
            if ui.selectable_label(is_grid, "Grid").clicked() && !is_grid {
              requested_mesh_source = MeshSource::Grid {
                cells_axis: GRID_CELLS_DEFAULT,
              };
            }
            for builtin in BuiltinMesh::ALL {
              let selected = requested_mesh_source == MeshSource::Builtin(builtin);
              if ui.selectable_label(selected, builtin.label()).clicked() {
                requested_mesh_source = MeshSource::Builtin(builtin);
              }
            }
          });
        // Opens the in-egui file browser; the pick itself is retrieved by the
        // caller, which owns the (native-only) `egui_file_dialog` state.
        if ui.button("Load OBJ…").clicked() {
          load_obj_clicked = true;
        }
      });
      // A generated family carries a refinement slider; a built-in or
      // loaded mesh has none.
      match &mut requested_mesh_source {
        MeshSource::Sphere { subdivisions } => {
          ui.add(egui::Slider::new(subdivisions, 0..=SPHERE_SUBDIVISIONS_MAX).text("subdivisions"));
        }
        MeshSource::Grid { cells_axis } => {
          ui.add(egui::Slider::new(cells_axis, 1..=GRID_CELLS_MAX).text("cells/axis"));
        }
        MeshSource::Builtin(_) | MeshSource::Custom { .. } => {}
      }
      if let Some(error) = &model.mesh_error {
        ui.colored_label(egui::Color32::LIGHT_RED, format!("⚠ {error}"));
      }
      ui.separator();

      // One tab per grade of the de Rham complex; every grade is solved and
      // shown, the top grade through its Hodge star just like grade 0.
      ui.horizontal(|ui| {
        for g in 0..=model.max_grade {
          if ui
            .selectable_label(g == grade, grade_mark_label(g, model.max_grade))
            .clicked()
          {
            requested_view = View::MeshGrade(g);
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
      match model.shown_view {
        View::MeshGrade(_) => {
          ui.label("rows: degeneracy shell (λ) · cells: order");
        }
        View::WhitneyBasis | View::WhitneyBasisMesh => {
          ui.label("rows: grade · cells: DOF simplex");
        }
        View::WhitneyExamplesMesh => {}
      };
      render_modes(ui, &model.entries, &mut selection, model.scene_dim);
    }

    // The mark is only meaningful for a line field, and it applies to the
    // one on screen, so it follows the displayed selection rather than the
    // one this frame's picker may just have moved to.
    if selection.is_line() {
      ui.separator();
      ui.horizontal(|ui| {
        ui.label("line mark:");
        ui.radio_value(&mut line_mark, LineMark::Streamlines, "streamlines");
        ui.radio_value(&mut line_mark, LineMark::Lic, "LIC");
      });
    }

    ui.separator();
    ui.checkbox(&mut top_down, "Top-down (orthographic, drag to pan)");
  });

  PanelResponse {
    requested_view,
    requested_mesh_source,
    selection,
    line_mark,
    top_down,
    load_obj_clicked,
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
