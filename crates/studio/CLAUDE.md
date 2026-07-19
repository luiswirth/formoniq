# formoniq-studio

The visual, interactive counterpart to `formoniq`: a viewer for inspecting PDE
solutions, meshes and simplicial manifolds, cochains, and the differential
geometry underneath them. It is meant to be both an instrument for a
mathematician or engineer and a way to see the abstractions directly.

This file carries what is particular to the viewer. The parent `CLAUDE.md` still
governs — its invariants, conventions and house style bind here unchanged. What
differs is the vantage point, and that is the whole subject below.

## The one inversion

The parent engine is intrinsic-first, extrinsic-only-for-I/O. `studio` is the
consumer of exactly the carve-out invariant 2 draws: the wrapper that "requires
an embedding" for "I/O, visualization or convenience." Visualization cannot be
done without an embedding — there is nothing to put on screen until a point has a
position — so the polarity flips. **`studio` is extrinsic by necessity,
intrinsic wherever it still can be.**

This is not a relaxation of the parent's discipline; it is the honest statement
of the one place that discipline does not reach, and it inverts the parent's
motto rather than weakening it. The corollary bites the same way the parent's
does: the moment a concept can be expressed without an embedding, it does not
belong in `studio`'s baking layer — it belongs downstack, in the engine.

## The two seams

The embedding is not assumed diffusely throughout the viewer. It lives between
two named boundaries, and intrinsic structure is carried as far toward the screen
as it can go before either one commits.

- **`Scene` is the seam in.** It carries the engine's own types — `Complex`,
  `MeshCoords`, `Cochain` — rather than a lossy export format, so the coloring,
  the displacement and the choice of render mark stay decisions of the viewer,
  made on the real object.
- **The bake is the seam out.** It reduces a complex to what a rasterizer draws:
  simplices of dimension $<= 2$ embedded in $RR^3$, with winding and embedding
  made explicit — the two things the core keeps out, because a graphics API and
  an interchange file both need them. Downstream of the bake there are no FEEC
  types, only ambient geometry.

  The bake's vertex table splits by what a datum depends on: the static half is a
  function of the mesh and its embedding alone (position, normal, curvature cap,
  winding), the other is the field on it. Switching fields, or scrubbing a
  trajectory, therefore rewrites only the field stream. A datum that would have
  to be recomputed to change fields is in the wrong half.

  The field half is itself split, because a reduced-grade Whitney form is
  **discontinuous across cells** — only the tangential part of a section is
  chart-independent, so incident cells disagree at a shared vertex and a basis
  function's support ends on cell edges. Its **colormap** value is therefore read
  once *per rendered corner in the corner's own cell* (the fill's corners are
  unshared already, for the deposit atlas), so a cell the form vanishes on stays
  exactly black instead of inheriting a neighbour's value through a shared vertex.
  A per-vertex tint cannot state this and silently bleeds a DOF's magnitude into
  every incident cell.

  The **displacement height** follows the field's own continuity, by the same
  reduction that picks the mark rather than by a second rule. $cal(W) Lambda^0$
  is $P_1$: a vertex has one value, the nodal recovery *is* the field, and the
  surface displaces as one connected sheet. $cal(W) Lambda^n$ is $P_0$: the
  reduced density is constant per cell and discontinuous across it, so there is
  no continuous height to ride and each cell displaces **rigidly**, by its own
  constant. That tears the surface, and the tear is the mark — cells separate by
  exactly the jump across their shared face, so the discontinuity becomes
  visible space and the surface re-closes under refinement. Smoothing it instead
  would show one field flat-shaded in color and smooth in shape, two
  contradictory claims in one frame. A nodal average where the field is
  genuinely discontinuous is a recovery, and presenting a recovery as the field
  is the thing to avoid. The shared 1-skeleton cannot tear without being
  duplicated, so the segment marks keep the continuous recovery at every grade.

Between the two the discipline is lived, not hoped for: a curve integrator works
in the barycentric charts of the atlas and crosses cells through the
`Transition`, committing to an ambient position only where it must. Anything new
belongs on that same spine — intrinsic until the bake, extrinsic only after it.

## Fixed ambient, general intrinsic

**Ambient dimension is $3$, by deliberate constant** — not a limit to apologize
for. It is the native space of the GPU, so $RR^2$ is the codimension case,
embedded in the $z = 0$ plane, and $RR^1$ a further one; a lower-dimensional cell
embeds as itself there, exactly as a flat surface does. One ambient space,
always $3$, is a unification, not a special case.

**Intrinsic dimension and form grade stay agnostic**, on the range the ambient
allows. A point set, a curve and a surface are one `MeshCoords`-in-$RR^3$
pipeline across grades, not three renderers — a curve renderer split off from a
surface renderer would be the `if dim == 3` of the parent, reappearing here.

Two reductions carry that, and they are the same move made on the two axes:

- **Grade reduces to a mark.** A $k$-form reduces to its *reduced grade*
  $min(k, n-k)$ through the Hodge star, and the render mark is chosen on that.
  Where that star actually fires ($k > n-k$) it needs a *global* volume form, so
  the reduction takes the cell's coherent orientation alongside the metric — the
  parent's invariant 6, and the one place the viewer needs a topological datum
  the solver never asks for. A mesh with no coherent orientation has no such
  reduction to show, so those fields are refused at `Scene::field` rather than
  drawn with a per-cell sign; a field that reaches a mark is already the proof
  that its orientation exists.

  Where a gauge is genuinely free, prefer making the mark independent of it over
  picking a value for it. The rigid cell displacement $d_K n_K$ is the model:
  the density and the cell normal flip together, so the motion is invariant. The
  gauges that remain are pinned only as far as reproducibility needs — a mode's
  sign is arbitrary, so it is fixed canonically; a solve's is physical and is
  left alone.
- **Intrinsic dimension reduces to a render primitive.** An $n$-manifold reduces
  to the primitive $min(n, 2)$ in the bake: a surface to wound triangles, a curve
  to segments, a point cloud to points, and a solid to the 2-simplices of its
  boundary — all of it an observer in $RR^3$ can see.

Each case distinction is confined to its own reduction — to the mark, and to the
bake — never smeared through the renderer, which sees only which *items* a frame
has, never why. The consequence is that one segment pipeline serves the wireframe
overlay, a line field's traced ribbons and a 1-manifold's own cells: they were
one technique described three times, and what differed between them (ink, width,
taper, whether the mark rides the wave) is material data. What the current
ambient does not yet reach — a reduced grade $>= 2$, a point cloud's mark — is
where these extend, not a branch to route around.

## The extrinsic freedom is the embedding, not the metric

Because an embedding is present, `studio` may use it and the ambient geometry it
induces — normals, ambient distances, global position — wherever that is cleaner
than an intrinsic construction. This is the genuine license the core denies
itself.

Name it precisely, because it is easy to overclaim: the freedom is *ambient*
geometry, not *metric*. A `RiemannianMetric` is not an extrinsic object. The core
uses it freely — invariant 5 forbids only letting it leak into a signature that
does not mathematically need one — and every metric here is the one the embedding
already induces. What `studio` grants itself over the core is the embedding and
the ambient space, and the global geometry read off them; nothing about the
metric changes.

## The layers of the viewer

The two seams say where the embedding enters. These say who may know what, and
they bind the same way the parent's invariants do:

- **The model is GPU-free.** The gallery and the scene are the mathematics and
  the state of the viewer; neither names a device, a buffer or a pipeline. What
  is shown is decided there and baked afterward, never the other way round.
- **The display is the callers' shared reduction.** What turns a scene and a
  selection into a draw list -- the bake, the materials, the framing, the
  object-intrinsic fractions the marks are scaled by -- belongs to neither
  caller. A window and a headless export build it identically and differ only in
  where the frame's time comes from; a material constructed at a caller instead
  is how the two silently come to disagree about what a field looks like. The
  corollary is the CLI's: nothing the code can decide from the object or the
  context is asked of the user, because a knob with one right answer only lets
  the answer be wrong.
- **What is asked divides by the object it reads.** Two objects are on screen —
  the mesh, and the field read on it — and `MeshDisplay`/`FieldDisplay` is
  already that split, so the settings mirror the seam rather than laying a second
  taxonomy over it. The mesh's are always live: a scene without geometry is not a
  scene. The field's are its reduced grade's answer, asked where that rule
  already lives, so a knob appears exactly when it has something to do. Neither
  costs a branch below the model — a setting naming an item drops it from the
  draw list, and one naming a deformation the items ride is a material at zero,
  the shape bloom's "off" already has.
- **The renderer sees baked geometry and explicit time, and nothing else.** No
  FEEC types, no clock, no window, no surface. Time is an argument, so the
  interactive loop passes wall-clock seconds and an exporter passes the instant
  it means to render; the frames are deterministic either way, and the two cannot
  drift because there is one frame graph. Which instants those are is the
  exporter's business and not the renderer's: an oscillating field has a period,
  and a clip that samples it as $t_k = k T \/ N$ closes on itself exactly, where
  one pinned to the playback grid would not. A frame is a *draw list* — batches with
  their materials, in submission order — so the number of things on screen is the
  caller's, never a fixed set the renderer declares.

  **A simulation is stepped to an instant, not evaluated at one**, and that is
  the one honest extension of "time is an argument". A standing wave is a
  function of $t$, so any frame can be asked for directly; a mark that carries
  state — an advected population — has no such closed form, and a caller can only
  say how far to advance it. So the count of steps is an argument beside the
  seconds, and what keeps the two callers from drifting is no longer the
  stateless graph but a *deterministic* one: the state's own randomness must be a
  pure function of the thing and its generation, never of a clock, so that a
  given count means the same picture to a window and to an exporter. A mark that
  cannot promise that does not belong in the frame graph.

- **Radiance is the scene's, the display's range is the target's, and one pass
  crosses between them.** The scene target is float and unbounded because
  additive marks accumulate — clipping at the blend destroys the very quantity
  the mark is made of. Everything that must happen in radiance (filtering,
  spilling light) happens before the crossing; the tone map *is* the crossing,
  and it is last. This is a real ordering, not a preference: anything that maps
  the range earlier leaves the passes after it nothing above 1 to find, and they
  fail silently rather than loudly.

  What the crossing costs is unavoidable and worth stating plainly: $[0, 1]$ is
  already fully spent by the marks that live there, so headroom above 1 must take
  range from below it, and every mark shifts. There is no setting that avoids
  this — only the choice of whether the dynamic range or the palette matters more
  for what is being looked at, which is exactly the kind of question the code
  cannot settle from the object, and therefore one of the few the viewer is asked.
- **State lives on the manifold; the screen is presentation only.** Screen-space
  passes are not suspect in themselves -- bloom, supersampling and the tone map
  are screen-space and correctly so, because they model the observation (the
  lens, the eye), never the field. The line is *persistence*: anything that
  survives a frame is a claim about the object, and the object lives on the
  manifold, where the camera cannot touch it. A pass that reads only this
  frame's image may be screen-space; a texture that accumulates across frames
  must be manifold data, indexed by chart and barycentric coordinate -- the
  deposit atlas is the constructive example, and a screen-space trail or
  history buffer is the violation: it would bake the camera into the state, and
  every orbit or export would smear a history that was never the field's. The
  test is the same cut the radiance/display split makes, extended along time.
- **The UI is a pure function of the model** returning requested changes, not a
  mutator of it.

## The platform is a product; presets are points in it

What the viewer shows is a point in `MeshSource × Study` — any study on any
mesh, the two axes independent and every pair total. The cache, the background
load and the placeholder machinery all key on the pair, not on a fixed
enumeration of views. A `Preset` is a named point in that product together with
the field it opens on: selecting one sets the two axes and the selection, and
everything afterward is the ordinary platform.

A preset is therefore a *configuration*, never a code path — the moment a
curated example would need its own branch to build or display, it has stopped
being a preset and the generalization has a hole. This is the same dissolution
the parent's invariants demand, one level up: the reference cell is the mesh
whose only cell is the standard simplex, so the local shape functions are the
Whitney study on it, not a study of their own; the global shape functions are
that study on the triforce; the spherical harmonics are the eigenmode study on
the sphere. Anything that looks like a special view owes the same reduction into
a mesh and a study.

## Rendering

Render the way an expert in computer graphics would: prefer the visually most
pleasing approach, and treat quality as a requirement rather than a finish. The
mathematics decides *what* is drawn — the reduced grade picks the mark, the
eigenvalue drives the standing wave — and the graphics craft decides *how well*.

Two durable conventions, kept general on purpose:

- **Shaders are checked by the test suite**, not only at pipeline creation, so a
  broken shader fails `cargo test` rather than the running viewer. The check runs
  naga's frontend — the one the *native* build uses — so it catches parse and
  validation errors but *not* what a browser's own WGSL→backend compiler rejects.
  The web target is WebKit/Metal, and it is stricter and buggier than naga and
  Tint: a shader that validates and runs on Chrome can still fail there
  ("Vertex library failed creation" is its generic symptom). The concrete rule
  this cost us: **no pipeline-overridable `override` constants specialized through
  the pipeline `constants` map** — WebKit fails to specialize them. Bake such a
  value into the WGSL as a `const` from the Rust side instead (see
  `render::ssaa_prelude`). WebKit is the strict oracle; when a shader change is
  non-trivial, it is the browser, not `cargo test`, that has the last word.
- **The graphics stack is pinned as a unit.** The types crossing the boundary
  between the UI layer and the renderer must come from the *same* underlying GPU
  crate, not merely semver-compatible versions; bump them together, and let the
  build enforce it.

## Two platforms, one viewer

The viewer runs native and on the web (`wasm32`, WebGPU), from the same code.
The split is a discipline, not a fork: **everything browser-specific is confined
to `web.rs`** -- the `wasm-bindgen` entry point, mounting winit's canvas into the
document, and bridging the *async* GPU bootstrap back into the event loop (device
and surface creation cannot block the browser, so the finished `State` is parked
in a slot the loop drains). Nothing web-flavored is allowed to leak into the
shared viewer code; what remains there are thin `#[cfg]` gates at genuine
platform seams, never web logic interleaved with native.

The web is the constrained side, and the constraints are honest, not worked
around:

- **No filesystem, no subprocess.** OBJ loading, PNG/MP4 export and the CLI are
  native features, gated off the web build -- which has nowhere to read or write.
  A feature that needs local files is native by nature, not a web regression to
  fix.
- **Single-threaded.** The plain `wasm32` target has no background thread, so a
  scene's solve runs synchronously where native offloads it (`PendingLoad`), and
  `faer` builds without its `rayon` thread pool. A heavy eigensolve therefore
  blocks the tab briefly; restoring an off-thread build there is a web-worker
  concern, kept out of the shared path.
- **WebGPU only, by choice.** No WebGL2 fallback -- the viewer targets the modern
  backend and fails legibly where it is absent, rather than constraining the
  render features to the WebGL2 subset.

The native build is untouched by any of this: the per-target dependency and
feature splits (the `faer` thread pool, the clipboard backend, the `getrandom`
web entropy source) restore the exact native set, so a native change never pays
for the web target's existence -- the same way `studio` itself stays off the
core's critical path.

## Anti-goals

- No renderer specialized to a fixed intrinsic dimension or grade where the two
  reductions cover it. Marks chosen by the grade's reduction, primitives by the
  dimension's; no second pipeline for what is one technique at a different ink.
- No web-specific logic outside `web.rs`. The shared viewer stays
  platform-neutral; a `wasm`/native divergence is a thin `#[cfg]` at a real seam,
  never a browser concern smeared through the render or model code.
- No dimension dispatch outside the bake, and no grade dispatch outside the mark.
  A `match` on either anywhere else is the case distinction escaping its
  reduction.
- No embedding leaking in outside the two seams; no ambient assumption above
  dimension 3.
- No claiming metric use as the extrinsic divergence — the divergence is the
  embedding and the ambient space, and saying otherwise misreads invariant 5.
- Nothing transient here, exactly as in the parent: no current state, no in-flight
  passes, no version pins written out, no roadmap phrased as a promise.
