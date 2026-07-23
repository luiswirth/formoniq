# formoniq-realize and formoniq-studio

One file for both crates on the extrinsic side of the engine,
because they are one subject: the vantage point the parent's invariants are read from.
`realize` is where intrinsic data becomes extrinsic
(the two reductions, the bake, the exporters),
`studio` is the renderer over it,
and "The two seams" below is the boundary between them,
so splitting the file would cut the argument in half at its own hinge.

`formoniq-studio` is the visual, interactive counterpart to `formoniq`:
a viewer for inspecting PDE solutions, meshes and simplicial manifolds, cochains,
and the differential geometry underneath them.
It is meant to be both an instrument for a mathematician or engineer
and a way to see the abstractions directly.

This file carries what is particular to the viewer.
The parent `CLAUDE.md` still governs:
its invariants, conventions and house style bind here unchanged.
What differs is the vantage point, and that is the whole subject below.

## The one inversion

The parent engine is intrinsic-first, extrinsic-only-for-I/O.
`studio` is the consumer of exactly the carve-out invariant 2 draws:
the wrapper that "requires an embedding" for "I/O, visualization or convenience."
Visualization cannot be done without an embedding
(there is nothing to put on screen until a point has a position),
so the polarity flips.
**`studio` is extrinsic by necessity, intrinsic wherever it still can be.**

This is not a relaxation of the parent's discipline.
It is the honest statement of the one place that discipline does not reach,
and it inverts the parent's motto rather than weakening it.
The corollary bites the same way the parent's does:
the moment a concept can be expressed without an embedding,
it does not belong in `studio`'s baking layer.
It belongs downstack, in the engine.

## The two seams

The embedding is not assumed diffusely throughout the viewer.
It lives between two named boundaries,
and intrinsic structure is carried as far toward the screen as it can go before either one commits.

**Both seams live in `realize`, and the renderer is above them.**
That is the crate split, and it follows from what the seams already are:
everything between them is a pure data transformation
and nothing in it needs a GPU or a window.
So an exporter and the viewer are *peers* consuming the same reduction,
rather than the exporter living inside the viewer,
which is what makes an external tool and the viewer agree about what a field looks like.
A reduction that only one consumer can reach has been put in the wrong crate.

- **`Scene` is the seam in:**
  It carries the engine's own types (`Complex`, `MeshCoords`, `Cochain`)
  rather than a lossy export format,
  so the coloring, the displacement and the choice of render mark
  stay decisions of the viewer, made on the real object.
- **The bake is the seam out:**
  It reduces a complex to what a rasterizer draws:
  simplices of dimension $<= 2$ embedded in $RR^3$, with winding and embedding made explicit,
  the two things the core keeps out,
  because a graphics API and an interchange file both need them.
  Downstream of the bake there are no FEEC types, only ambient geometry.
  The interchange file is the reason to say "both" rather than "the renderer":
  `.vtu` for ParaView, `.obj`/`.mdd` for a mesh,
  each a leaf that consumes the bake and commits further
  (VTU's points are 3-tuples and its cells stop at the tetrahedron,
  so a mesh above three dimensions is *its* refusal, not the bake's).

  The bake's vertex table splits by what a datum depends on:
  the static half is a function of the mesh and its embedding alone
  (position, normal, curvature cap, winding), the other is the field on it.
  Switching fields, or scrubbing a trajectory, therefore rewrites only the field stream.
  A datum that would have to be recomputed to change fields is in the wrong half.

  The field half is itself split,
  because a reduced-grade Whitney form is **discontinuous across cells**:
  only the tangential part of a section is chart-independent,
  so incident cells disagree at a shared vertex
  and a basis function's support ends on cell edges.
  Its **colormap** value is therefore read once *per rendered corner in the corner's own cell*
  (the fill's corners are unshared already, for the deposit atlas),
  so a cell the form vanishes on stays exactly black
  instead of inheriting a neighbour's value through a shared vertex.
  A per-vertex tint cannot state this and silently bleeds a DOF's magnitude into every incident cell.

  The **displacement height** follows the field's own continuity,
  by the same reduction that picks the mark rather than by a second rule.
  $cal(W) Lambda^0$ is $P_1$:
  a vertex has one value, the nodal recovery *is* the field,
  and the surface displaces as one connected sheet.
  $cal(W) Lambda^n$ is $P_0$:
  the reduced density is constant per cell and discontinuous across it,
  so there is no continuous height to ride
  and each cell displaces **rigidly**, by its own constant.
  That tears the surface, and the tear is the mark:
  cells separate by exactly the jump across their shared face,
  so the discontinuity becomes visible space and the surface re-closes under refinement.
  Smoothing it instead would show one field flat-shaded in color and smooth in shape,
  two contradictory claims in one frame.
  A nodal average where the field is genuinely discontinuous is a recovery,
  and presenting a recovery as the field is the thing to avoid.
  The shared 1-skeleton cannot tear without being duplicated,
  so the segment marks keep the continuous recovery at every grade.

  **A mark is sized by the length its own question is about.**
  Two scales are available and they are not interchangeable:
  the object's *extent* and the mesh's *mean edge length*.
  A quantity that should read the same however finely the object is triangulated
  (how far a standing wave swells, how fast a tracer crosses, how dense the glyph lattice is)
  is a fraction of the extent.
  A mark that draws the mesh's own features
  (the stroke of an edge, the size of a per-cell mark)
  is a fraction of the edge length, or of a length already derived from it.
  Getting this backwards reads correctly at exactly one refinement:
  tie a stroke to the extent and refining the mesh shrinks the cells while the strokes stay put,
  until the wireframe is a solid mass and the arrows are stubs.
  A mark whose every dimension is a proportion of one cell-derived length is self-similar,
  and then there is no resolution at which it can be wrong.

  **A displacement is bounded by scaling it, never by clamping it.**
  The bound is the mesh's *reach*,
  the distance to its own medial axis, below which the normal offset is still an embedding.
  Curvature radius is only half of that bound, the local half.
  The other half is the bottleneck, how far the surface is from a different sheet of itself,
  and it is the half that thin features live in.
  A flat plate has infinite curvature radius and reach $t \/ 2$,
  so a curvature-only ceiling lets its two faces pass through each other.
  Given the bound, the amplitude is one global scalar chosen so no vertex exceeds it.
  A per-vertex clamp is the wrong instrument:
  it binds at a different value at every vertex,
  so it flattens the field in patches
  and seams the surface between clamped and unclamped neighbours,
  that is not a bounded deformation but a different one.
  Scaling is the operation an eigenmode is indifferent to, being defined up to a scalar,
  so it bounds the picture without changing which mode the picture is of.

Between the two the discipline is lived, not hoped for:
a curve integrator works in the barycentric charts of the atlas
and crosses cells through the `Transition`,
committing to an ambient position only where it must.
Anything new belongs on that same spine:
intrinsic until the bake, extrinsic only after it.

## Fixed ambient, general intrinsic

**Ambient dimension is $3$, by deliberate constant,** not a limit to apologize for.
It is the native space of the GPU,
so $RR^2$ is the codimension case, embedded in the $z = 0$ plane, and $RR^1$ a further one.
A lower-dimensional cell embeds as itself there, exactly as a flat surface does.
One ambient space, always $3$, is a unification, not a special case.

**Intrinsic dimension and form grade stay agnostic**, on the range the ambient allows.
A point set, a curve and a surface are one `MeshCoords`-in-$RR^3$ pipeline across grades,
not three renderers:
a curve renderer split off from a surface renderer
would be the `if dim == 3` of the parent, reappearing here.

Two reductions carry that, and they are the same move made on the two axes:

- **Grade reduces to a mark:**
  A $k$-form reduces to its *reduced grade* $min(k, n-k)$ through the Hodge star,
  and the render mark is chosen on that.
  Where that star actually fires ($k > n-k$) it needs a *global* volume form,
  so the reduction takes the cell's coherent orientation alongside the metric,
  the parent's invariant 6,
  and the one place the viewer needs a topological datum the solver never asks for.
  A mesh with no coherent orientation has no such reduction to show,
  so those fields are refused at `Scene::field` rather than drawn with a per-cell sign.
  A field that reaches a mark is already the proof that its orientation exists.

  Where a gauge is genuinely free,
  prefer making the mark independent of it over picking a value for it.
  The rigid cell displacement $d_K n_K$ is the model:
  the density and the cell normal flip together, so the motion is invariant.
  The gauges that remain are pinned only as far as reproducibility needs:
  a mode's sign is arbitrary, so it is fixed canonically.
  A solve's is physical and is left alone.
- **Intrinsic dimension reduces to a render primitive:**
  An $n$-manifold reduces to the primitive $min(n, 2)$ in the bake:
  a surface to wound triangles, a curve to segments, a point cloud to points,
  and a solid to the 2-simplices of its boundary,
  all of it an observer in $RR^3$ can see.

**The two reductions compose, and the order is fixed: dimension first, grade second.**
The object a mark is a mark *of* is the render surface
(the mesh itself below $n = 3$, the boundary $diff M$ for a solid),
so the $n$ in $min(k, n-k)$ is the *surface's*, never the mesh's.
`Surface` is that reduction named once,
and it is a genuine manifold one dimension down,
with its own complex, orientation and metric.
A field reaches it by its **trace** $i^*: C^k (M) -> C^k (diff M)$, a cochain map,
hence a real Whitney form on $diff M$ rather than a resampling or a nodal recovery.

Getting the order backwards is what a dimension-blind mark looks like:
a $2$-form on a solid reduces to grade 1 against the volume (arrows)
but to grade 0 against the boundary (a density),
and only the latter is a claim about anything on screen:
a flux has no direction in the surface carrying it.
An arrow glyph is the sharp case,
because a flat mark needs a plane to lie in and a determined perpendicular,
and a tetrahedron supplies neither.

**The trace is total in grade but vanishes at the top**, since $C^n (diff M) = 0$.
A top-grade density is a *volume* quantity,
and reading it on the boundary is a sampling of the cells behind it, never a trace:
the two must not be conflated,
and a mark that needs the volume says so rather than tracing to zero and drawing nothing.
Volume marks (a camera-facing glyph, a slice) are where this extends.
They are a different mark with a different frame, not this one run on cells.

Each case distinction is confined to its own reduction (to the mark, and to the bake),
never smeared through the renderer,
which sees only which *items* a frame has, never why.
The consequence is that one segment pipeline serves the wireframe overlay,
a line field's traced ribbons and a 1-manifold's own cells:
they were one technique described three times,
and what differed between them (ink, width, taper, whether the mark rides the wave) is material data.
What the current ambient does not yet reach (a reduced grade $>= 2$, a point cloud's mark)
is where these extend, not a branch to route around.

## The extrinsic freedom is the embedding, not the metric

Because an embedding is present,
`studio` may use it and the ambient geometry it induces
(normals, ambient distances, global position)
wherever that is cleaner than an intrinsic construction.
This is the genuine license the core denies itself.

Name it precisely, because it is easy to overclaim:
the freedom is *ambient* geometry, not *metric*.
A `RiemannianMetric` is not an extrinsic object.
The core uses it freely
(invariant 5 forbids only letting it leak into a signature that does not mathematically need one),
and every metric here is the one the embedding already induces.
What `studio` grants itself over the core
is the embedding and the ambient space, and the global geometry read off them.
Nothing about the metric changes.

## The layers of the viewer

The two seams say where the embedding enters.
These say who may know what, and they bind the same way the parent's invariants do:

- **The model is GPU-free:**
  The gallery and the scene are the mathematics and the state of the viewer.
  Neither names a device, a buffer or a pipeline.
  What is shown is decided there and baked afterward, never the other way round.
- **The display is the callers' shared reduction:**
  What turns a scene and a selection into a draw list
  (the bake, the materials, the framing, the object-intrinsic fractions the marks are scaled by)
  belongs to neither caller.
  A window and a headless export build it identically
  and differ only in where the frame's time comes from.
  A material constructed at a caller instead
  is how the two silently come to disagree about what a field looks like.
  The corollary is the CLI's:
  nothing the code can decide from the object or the context is asked of the user,
  because a knob with one right answer only lets the answer be wrong.
- **What is asked divides by the object it reads:**
  Two objects are on screen (the mesh, and the field read on it)
  and `MeshDisplay`/`FieldDisplay` is already that split,
  so the settings mirror the seam rather than laying a second taxonomy over it.
  The mesh's are always live: a scene without geometry is not a scene.
  The field's are its reduced grade's answer, asked where that rule already lives,
  so a knob appears exactly when it has something to do.
  Neither costs a branch below the model:
  a setting naming an item drops it from the draw list,
  and one naming a deformation the items ride is a material at zero,
  the shape bloom's "off" already has.

  What builds the object and what draws it are the two sidebars,
  and they keep the two questions apart:
  the **browser** picks the point in `MeshSource × Study` (which mesh, which computation)
  and the **inspector** edits the parameters of the study picked there
  and the display of the two objects it produced.
  `Study`'s variant parameters (the eigenmode grade and count, a trajectory's sampling)
  are the inspector's, not the browser's,
  because they are knobs *on* the chosen study rather than the choice of it.
  An edit that drives a re-solve commits on release, not mid-drag,
  so the background solve fires once.
  What belongs to **neither** object,
  reading and writing files,
  and the view shell itself
  (which sidebars show, the projection, the light ladder, re-framing the camera),
  is a **menu bar**, the conventional home a reader reaches for these by reflex,
  and the one place a command that is not a property of the mesh or the field is allowed to live.
- **The renderer sees baked geometry and explicit time, and nothing else:**
  No FEEC types, no clock, no window, no surface.
  Time is an argument,
  so the interactive loop passes wall-clock seconds
  and an exporter passes the instant it means to render.
  The frames are deterministic either way,
  and the two cannot drift because there is one frame graph.
  Which instants those are is the exporter's business and not the renderer's:
  an oscillating field has a period,
  and a clip that samples it as $t_k = k T \/ N$ closes on itself exactly,
  where one pinned to the playback grid would not.
  A frame is a *draw list* (batches with their materials, in submission order),
  so the number of things on screen is the caller's,
  never a fixed set the renderer declares.

  **A simulation is stepped to an instant, not evaluated at one**,
  and that is the one honest extension of "time is an argument".
  A standing wave is a function of $t$, so any frame can be asked for directly.
  A mark that carries state (an advected population) has no such closed form,
  and a caller can only say how far to advance it.
  So the count of steps is an argument beside the seconds,
  and what keeps the two callers from drifting
  is no longer the stateless graph but a *deterministic* one:
  the state's own randomness must be a pure function of the thing and its generation,
  never of a clock,
  so that a given count means the same picture to a window and to an exporter.
  A mark that cannot promise that does not belong in the frame graph.

- **Radiance is the scene's, the display's range is the target's, and one pass crosses between them:**
  The scene target is float and unbounded because additive marks accumulate:
  clipping at the blend destroys the very quantity the mark is made of.
  Everything that must happen in radiance (filtering, spilling light) happens before the crossing.
  The tone map *is* the crossing, and it is last.
  This is a real ordering, not a preference:
  anything that maps the range earlier
  leaves the passes after it nothing above 1 to find,
  and they fail silently rather than loudly.

  What the crossing costs is unavoidable and worth stating plainly:
  $[0, 1]$ is already fully spent by the marks that live there,
  so headroom above 1 must take range from below it, and every mark shifts.
  There is no setting that avoids this:
  only the choice of whether the dynamic range or the palette matters more
  for what is being looked at,
  which is exactly the kind of question the code cannot settle from the object,
  and therefore one of the few the viewer is asked.
- **State lives on the manifold, the screen is presentation only:**
  Screen-space passes are not suspect in themselves,
  bloom, supersampling and the tone map are screen-space and correctly so,
  because they model the observation (the lens, the eye), never the field.
  The line is *persistence*:
  anything that survives a frame is a claim about the object,
  and the object lives on the manifold, where the camera cannot touch it.
  A pass that reads only this frame's image may be screen-space.
  A texture that accumulates across frames must be manifold data,
  indexed by chart and barycentric coordinate.
  The deposit atlas is the constructive example,
  and a screen-space trail or history buffer is the violation:
  it would bake the camera into the state,
  and every orbit or export would smear a history that was never the field's.
  The test is the same cut the radiance/display split makes, extended along time.
- **The UI is a pure function of the model,** returning requested changes, not a mutator of it.
- **Layout answers to the window, never to the platform:**
  What a narrow viewport changes
  is whether a sidebar is *docked beside* the scene or *laid over* it,
  never what the panels contain, and never which panel a control belongs to,
  because that taxonomy mirrors the two objects on screen
  and a second one keyed on screen size would cut across it.
  Below the width where both sidebars plus a usable viewport fit, nothing docks by default:
  the scene is what a reader sees first and a sidebar is something they open.
  A narrow desktop window gets exactly what a phone gets:
  there is no mobile build, only a narrow one.

  **The sidebars collapse at every width, and the layout only supplies the default.**
  Wanting the controls out of the way to look at the scene
  is not something only a small screen wants,
  so the toggles are always there.
  What the width decides is what an *untouched* sidebar does.
  A reader's explicit choice outranks it and survives a resize.
  That is a third state, not a boolean:
  the default has to stay derivable,
  or the first frame on a phone shows two panels meeting in the middle
  before anything can correct them.

## The platform is a product, presets are points in it

What the viewer shows is a point in `MeshSource × Study`:
any study on any mesh, the two axes independent and every pair total.
The cache, the background load and the placeholder machinery all key on the pair,
not on a fixed enumeration of views.
A `Preset` is a named point in that product together with the field it opens on:
selecting one sets the two axes and the selection,
and everything afterward is the ordinary platform.

**The shipped meshes are the asset directory, not a list of them.**
`build.rs` enumerates `assets/meshes` and generates the table the picker and the CLI read,
so a mesh is added by dropping the file in:
its extension picks the reader, its stem is its name.
Generated at build time rather than scanned at run time
because the assets are embedded in the binary,
which is what lets the web build (with no filesystem to scan) ship the same set.
A hand-written list of what is in a directory
is a second source of truth for something the directory already knows, and the two drift.

A preset is therefore a *configuration*, never a code path:
the moment a curated example would need its own branch to build or display,
it has stopped being a preset and the generalization has a hole.
This is the same dissolution the parent's invariants demand, one level up:
the reference cell is the mesh whose only cell is the standard simplex,
so the local shape functions are the Whitney study on it, not a study of their own.
The global shape functions are that study on the triforce.
The spherical harmonics are the eigenmode study on the sphere.
Anything that looks like a special view owes the same reduction into a mesh and a study.

## Rendering

Render the way an expert in computer graphics would:
prefer the visually most pleasing approach, and treat quality as a requirement rather than a finish.
The mathematics decides *what* is drawn
(the reduced grade picks the mark, the eigenvalue drives the standing wave)
and the graphics craft decides *how well*.

Three durable conventions, kept general on purpose:

- **A mark drawn on a surface is biased in depth, never displaced in space:**
  A glyph in its cell and a wireframe edge along its simplex
  are coplanar with the fill and must win the depth comparison.
  That is a claim about $z$ alone,
  and the rasterizer's depth bias is what makes it,
  after the mark's screen position is already fixed.
  Translating the mark toward the camera instead is the plausible wrong answer:
  it puts the mark at a depth its surface does not have,
  so the two show parallax and the mark slides across its own face as the camera orbits,
  and the offset is measured in the mark's size rather than the gap to what is in front,
  so on a *closed* surface a far face's marks pass through the near one.
  One open sheet hides both faults
  (there is nothing to slide against and nothing in front to pierce),
  so this is a fault that only a solid's boundary reveals,
  and it is why the convention is written down rather than rediscovered.
- **Shaders are checked by the test suite**, not only at pipeline creation,
  so a broken shader fails `cargo test` rather than the running viewer.
  The check runs naga's frontend (the one the *native* build uses),
  so it catches parse and validation errors
  but *not* what a browser's own WGSL→backend compiler rejects.
  The web target is WebKit/Metal, and it is stricter and buggier than naga and Tint:
  a shader that validates and runs on Chrome can still fail there
  ("Vertex library failed creation" is its generic symptom).
  The concrete rule this cost us,
  **no pipeline-overridable `override` constants specialized through the pipeline `constants` map**:
  WebKit fails to specialize them.
  Bake such a value into the WGSL as a `const` from the Rust side instead
  (see `render::ssaa_prelude`).
  WebKit is the strict oracle.
  When a shader change is non-trivial,
  it is the browser, not `cargo test`, that has the last word.
- **The graphics stack is pinned as a unit:**
  The types crossing the boundary between the UI layer and the renderer
  must come from the *same* underlying GPU crate, not merely semver-compatible versions.
  Bump them together, and let the build enforce it.

## Two platforms, one viewer

The viewer runs native and on the web (`wasm32`, WebGPU), from the same code.
The split is a discipline, not a fork.
**Everything browser-specific is confined to `web.rs`**:
the `wasm-bindgen` entry point, mounting winit's canvas into the document,
and bridging the *async* GPU bootstrap back into the event loop
(device and surface creation cannot block the browser,
so the finished `State` is parked in a slot the loop drains).
Nothing web-flavored is allowed to leak into the shared viewer code.
What remains there are thin `#[cfg]` gates at genuine platform seams,
never web logic interleaved with native.

The web is the constrained side, and the constraints are honest, not worked around:

- **No filesystem, no subprocess:**
  OBJ loading, PNG/MP4 export and the CLI are native features,
  gated off the web build, which has nowhere to read or write.
  A feature that needs local files is native by nature, not a web regression to fix.
- **Single-threaded within a context, so the solve moves to another one:**
  The plain `wasm32` target has no background thread and `faer` builds without its `rayon` pool,
  so a study cannot be solved off the main thread the way native does it.
  It is sent to a *worker* instead:
  the build is a request and an outcome (`solve`), not a closure,
  because a closure cannot cross a `postMessage` boundary and a value can.
  Message passing, not shared memory:
  `SharedArrayBuffer` threads would need cross-origin isolation and a nightly toolchain,
  which a static host cannot give and this does not need.
  The worker loads the same module and calls the same solver,
  so there is one implementation and the boundary is only transport.

  The request carries the mesh itself rather than a descriptor of it.
  A descriptor would cover every mesh the gallery can regenerate and miss the one that matters:
  a mesh the reader loaded,
  which exists nowhere else and is the one whose size nobody has bounded.
- **WebGPU only, by choice:**
  No WebGL2 fallback,
  the viewer targets the modern backend and fails legibly where it is absent,
  rather than constraining the render features to the WebGL2 subset.

The native build is untouched by any of this:
the per-target dependency and feature splits
(the `faer` thread pool, the clipboard backend, the `getrandom` web entropy source)
restore the exact native set,
so a native change never pays for the web target's existence,
the same way `studio` itself stays off the core's critical path.

## Anti-goals

- No renderer specialized to a fixed intrinsic dimension or grade where the two reductions cover it.
  Marks chosen by the grade's reduction, primitives by the dimension's.
  No second pipeline for what is one technique at a different ink.
- No web-specific logic outside `web.rs`.
  The shared viewer stays platform-neutral.
  A `wasm`/native divergence is a thin `#[cfg]` at a real seam,
  never a browser concern smeared through the render or model code.
- No dimension dispatch outside the bake, and no grade dispatch outside the mark.
  A `match` on either anywhere else is the case distinction escaping its reduction.
- No embedding leaking in outside the two seams.
  No ambient assumption above dimension 3.
- No claiming metric use as the extrinsic divergence:
  the divergence is the embedding and the ambient space,
  and saying otherwise misreads invariant 5.
- Nothing transient here, exactly as in the parent:
  no current state, no in-flight passes, no version pins written out,
  no roadmap phrased as a promise.
