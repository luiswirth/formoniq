# formoniq-studio

The visual, interactive counterpart to `formoniq`: a viewer for inspecting PDE
solutions, meshes and simplicial manifolds, cochains, and the differential
geometry underneath them. It is meant to be both an instrument for a
mathematician or engineer and a way to see the abstractions directly, and to run
natively and on the web from one source.

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
- **`TriangleSurface3D` (the bake) is the seam out.** It is deliberately
  dimension-specific and coordinate-full: winding and embedding fixed at 3, the
  two things the core keeps out, because a graphics API and an interchange file
  both need them. Downstream of the bake there are no FEEC types, only ambient
  geometry.

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
Grade is reduced before it is drawn: a $k$-form reduces to its *reduced grade*
$min(k, n-k)$ through the Hodge star, and the render mark is chosen on that. The
dispatch is total — the case distinction lives in the mark, not smeared through
the renderer — and the grades the current ambient does not yet reach are where it
extends, not branches to route around. Where a visualization genuinely forces a
case distinction, confine it to the mark, the way the reduction does.

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

## Rendering

Render the way an expert in computer graphics would: prefer the visually most
pleasing approach, and treat quality as a requirement rather than a finish. The
mathematics decides *what* is drawn — the reduced grade picks the mark, the
eigenvalue drives the standing wave — and the graphics craft decides *how well*.

Two durable conventions, kept general on purpose:

- **Shaders are checked by the test suite**, not only at pipeline creation, so a
  broken shader fails `cargo test` rather than the running viewer.
- **The graphics stack is pinned as a unit.** The types crossing the boundary
  between the UI layer and the renderer must come from the *same* underlying GPU
  crate, not merely semver-compatible versions; bump them together, and let the
  build enforce it.

## Anti-goals

- No renderer specialized to a fixed intrinsic dimension or grade where the
  reduced-grade dispatch covers it. One pipeline, marks chosen by the reduction.
- No embedding leaking in outside the two seams; no ambient assumption above
  dimension 3.
- No claiming metric use as the extrinsic divergence — the divergence is the
  embedding and the ambient space, and saying otherwise misreads invariant 5.
- Nothing transient here, exactly as in the parent: no current state, no in-flight
  passes, no version pins written out, no roadmap phrased as a promise.
