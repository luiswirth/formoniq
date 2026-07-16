# studio platform plan

The content and UI layer on top of the refactor in `PLAN.md`: generalize the
gallery from a fixed enum of views into a compositional platform -- any mesh,
any study, any field -- with a curated list of presets that are *configurations*
of the general setup, never code paths. Sequenced after `PLAN.md` steps 1-3
(model/renderer split, dimension-dispatched bake); the export CLI (step 4) is
what the transport bar's export button invokes.

## The platform model

Today's `View` enum conflates the two axes it should compose: `WhitneyBasis`
is the Whitney study pinned to the reference triangle, `WhitneyBasisMesh` the
same study pinned to the triforce, `WhitneyExamplesMesh` a fixed cochain list
pinned to the triforce, and only `MeshGrade` composes with the mesh picker.
Dissolve it into a product:

```
shown = MeshSource x Study        (the gallery memoizes per pair)
```

Every study runs on every mesh. The gallery's cache, background `PendingLoad`,
and placeholder machinery generalize unchanged -- they key on the pair instead
of the enum. Totality is the requirement, not a nice-to-have: the Whitney study
on a 3000-vertex OBJ and the eigenmode study on the 7-vertex Csaszar torus both
build and display, or the composition claim is false.

### MeshSource, extended

- `Sphere { subdivisions }`, `Grid { cells_axis }`, `Builtin(..)`,
  `Custom { name }` as today.
- `ReferenceCell { dim }`: the standard cell as a one-cell mesh, `dim` a
  slider over $1..=3$. Absorbs the LSF view -- "local shape functions" is not
  a study of its own, it is the Whitney study on the mesh whose only cell is
  the reference cell.
- `Triforce` joins the builtins. Absorbs the other half of the LSF/GSF split.
- `Circle { segments }`: the first 1-manifold, exercising the `PrimBatch`
  segment path of the new bake. Eigenmodes on it are the Fourier modes:
  degenerate $sin$/$cos$ pairs, $lambda_k approx k^2$, a perfect pyramid.
- Later, behind the same `Custom` door: gmsh import (`manifold::io::gmsh`
  already parses it) next to OBJ.

### Study

The second axis: what is computed on the mesh. Parameters live in the variant;
presets fill them with concrete values, the inspector edits them.

```rust
enum Study {
  /// Hodge-Laplace spectrum of one grade. The harmonic forms are its zero
  /// shell, not a separate solver.
  Eigenmodes { grade: ExteriorGrade, nmodes: usize },
  /// One field per DOF simplex of every grade: the one-hot cochains.
  WhitneyBasis,
  /// A named list of explicit cochains (the triforce worked examples, a
  /// loaded cochain file later).
  Cochains(Vec<NamedCochain>),
  /// One cochain split into exact + coexact + harmonic, shown as three
  /// switchable fields plus the original.
  HodgeDecomposition { input: CochainSpec },
  /// Precomputed trajectories (see "time" below).
  Heat  { init: CochainSpec, duration: f64, nframes: usize },
  Wave  { init: CochainSpec, duration: f64, nframes: usize },
  Maxwell { init: CochainSpec, duration: f64, nframes: usize },
  /// A source problem against a continuum exact solution: solve, pull the
  /// exact solution back through ddf, and expose u_h, the interpolant
  /// $Pi u$, and the error field.
  Convergence { exact: ExactSolution },
}
```

`CochainSpec` is where "concrete cochain" stays data: `OneHot(simplex)`,
`ByEdges(..)` (the triforce tables), `Bump { center, width }` (dynamics initial
conditions), `Interpolated(CoordField)` (through the `ddf` pullback bridge).
A preset constructs one; the UI offers the generic ones.

### Presets are configurations

A preset is a value, never a code path:

```rust
struct Preset {
  name: &'static str,
  mesh: MeshSource,
  study: Study,
  selection: Option<Selection>, // the field it opens on
}
```

Selecting one sets the two axes and the selection; everything afterward is the
ordinary platform. The curated first-wave list:

- **Spherical harmonics** -- sphere, eigenmodes, all grades reachable by tabs.
- **Fourier modes** -- circle, eigenmodes: the 1D spectrum, degenerate pairs.
- **Harmonic 1-forms** -- Csaszar (and Bob), eigenmodes grade 1: the zero
  shell has dimension $2g$; the Betti readout names it.
- **Whitney basis** -- reference cell, Whitney study, dim slider.
- **Global shape functions** -- triforce, Whitney study.
- **Constant / curl / div** -- triforce, `Cochains` with the worked tables.
- **Hodge decomposition** -- triforce, decomposition of a mixed cochain: the
  three components of one field, switched in place.
- **Heat flow** -- Spot, heat trajectory from a bump.
- **Struck membrane** -- grid, wave trajectory from a bump.
- **Cavity resonance** -- grid, Maxwell: leapfrog $E$/$B$ trajectory, and the
  cavity modes as an eigenmode preset beside it.
- **Poisson convergence** -- sphere, `Convergence` against a known exact
  solution: $u_h$, $Pi u$ and the error field, with the refinement slider as
  the convergence experiment.

## Time

Three temporal kinds of field data, one transport:

- **Static** -- no animation.
- **Standing** `{ omega }` -- the analytic $cos(omega t)$ modulation an
  eigenmode already has; period $2 pi \/ omega$, loops forever.
- **Sampled** `{ dt, frames: Vec<Cochain> }` -- a precomputed trajectory
  (heat, wave, Maxwell leapfrog), scrubbed over $[0, T]$ and looped.

Trajectories are solved once on the background thread (the same `PendingLoad`
path as an eigensolve), stored model-side as cochains, and played by rewriting
only the attribute stream -- exactly the static/per-field split the
`BakedMesh` bake was designed around. Deterministic by construction, so the
export path renders the identical clip. `nframes` bounds memory; the solver
may step finer than it stores.

## Layers

The displayed state becomes two independently bound channels instead of one
selection: a **color layer** (a scalar field tinting the surface) and a
**line layer** (a line field drawn as streamlines or LIC). Picking a line
field binds both channels to it -- today's behavior as the default case.
Maxwell is what forces the split: $E$ (grade 1, lines) and $B$ (grade 2,
density) are two grades of one solve on screen at once, which no
one-field-at-a-time viewer can show. Comparison stories (decomposition
components, $u_h$ vs. error) stay single-viewport in this wave -- switchable
fields, not split views; interactive compare viewports come later, as the
sibling of the export CLI's `--panels`.

## UI: the docked shell

The floating "Gallery" window becomes a docked layout; the panel stays a pure
function of its model, exactly as the refactor's `ui/panel.rs` established.

- **Left sidebar** -- the browser: the curated preset list on top, and below
  it the two raw axes (mesh picker with families/builtins/load, study picker)
  for free composition.
- **Right inspector** -- parameters of what is shown: the study's own knobs
  (grade tabs, mode pyramid, decomposition component, dynamics durations),
  the layer bindings, mark options (streamlines/LIC), camera toggle.
- **Bottom transport bar** -- play/pause, scrub, loop; readouts ($t$, $lambda$,
  $omega$, energy); the export button, which hands the live configuration to
  the `PLAN.md` step-4 export path.
- **Viewport overlays** -- colormap legend with value range, field name, and
  per-study readouts (Betti numbers, energy, error norms).

## Law tests

- **Decomposition**: the three components are pairwise $L^2$-orthogonal and
  sum to the input, up to solver tolerance; the harmonic component's dimension
  matches `topology::homology`.
- **Harmonics**: the zero shell's dimension equals the Betti number, for every
  builtin mesh and every grade $0..=n$.
- **Circle**: eigenvalues cluster in degenerate pairs with
  $lambda_k -> k^2$ under refinement.
- **Trajectories**: wave and Maxwell energy drift stays bounded (the engine's
  own energy functionals); heat decays monotonically.
- **Totality sweep**: every preset builds and yields a nonempty field list;
  every `Study` variant builds on every `MeshSource` family without panic.

## Execution steps

Each step one or more commits, green under the workspace's four commands.
Steps 1-3 need `PLAN.md` steps 1-2 landed; step 4 needs its step 3 (the bake).

1. **Platform model.** `MeshSource x Study`, the `View` enum dissolved,
   presets as data, `ReferenceCell`/`Triforce` as sources. Behavior-preserving
   for everything on screen today. Amend `CLAUDE.md` in the same commit: a
   preset is a configuration of the general platform, never a code path.
2. **Docked shell.** Sidebar/inspector/transport panels and the viewport
   overlays, carrying the existing standing-wave animation into the transport
   bar.
3. **Trajectories.** The `Sampled` temporal kind, scrubbing, and the heat and
   wave studies with their presets.
4. **Circle.** `MeshSource::Circle` on the segment bake; the Fourier preset.
5. **Topology.** The harmonic-shell presentation: Betti readout, the
   harmonic-forms presets on the genus-1 builtins.
6. **Hodge decomposition.** The study, its component switching, its preset,
   and the orthogonality law tests.
7. **Layers + Maxwell.** The color/line channel split, the leapfrog
   trajectory study ($E$ lines over $B$ density), the cavity presets.
8. **Convergence.** The `Convergence` study through the `ddf` pullback:
   $u_h$ / $Pi u$ / error as switchable fields, error-norm readout against
   the refinement slider.

Order rationale: 1 is the load-bearing generalization and everything else is
expressed in its vocabulary; 2 gives the growing content somewhere to live;
3 precedes 7 because Maxwell is a trajectory; 6 precedes nothing but is cheap
after 1; 8 last because it is the only step reaching outside the viewer, into
`continuum` and `ddf`.
