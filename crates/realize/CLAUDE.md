# formoniq-realize

The extrinsic side of the engine:
where intrinsic data becomes extrinsic (the two reductions, the bake, the exporters).

This file carries what is particular to that side.
The parent `CLAUDE.md` still governs:
its invariants, conventions and house style bind here unchanged.
What differs is the vantage point, and that is the whole subject below.

## The one inversion

The parent engine is intrinsic-first, extrinsic-only-for-I/O.
`realize` is the consumer of exactly the carve-out invariant 2 draws:
the wrapper that "requires an embedding" for "I/O, visualization or convenience."
Neither a picture nor an interchange file can be made without an embedding
(there is nothing to draw or to write until a point has a position),
so the polarity flips.
**`realize` is extrinsic by necessity, intrinsic wherever it still can be.**

This is not a relaxation of the parent's discipline.
It is the honest statement of the one place that discipline does not reach,
and it inverts the parent's motto rather than weakening it.
The corollary bites the same way the parent's does:
the moment a concept can be expressed without an embedding,
it does not belong here.
It belongs downstack, in the engine.

## The seam out

The embedding is not assumed diffusely.
It lives behind one named boundary,
and intrinsic structure is carried as far toward it as it can go before it commits.

**The seam is the bake, and every consumer is above it.**
That is what the crate is:
everything up to the bake is a pure data transformation
and nothing in it needs a GPU or a window.
So an exporter and a viewer are *peers* consuming the same reduction,
rather than the exporter living inside the viewer,
which is what makes an external tool and a renderer agree about what a field looks like.
A reduction that only one consumer can reach has been put in the wrong crate.

The bake reduces a complex to what a rasterizer draws:
simplices of dimension $<= 2$ embedded in $RR^3$, with winding and embedding made explicit,
the two things the core keeps out,
because a graphics API and an interchange file both need them.
Downstream of the bake there are no FEEC types, only ambient geometry.
The interchange file is the reason to say "every consumer" rather than "the renderer":
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
instead of inheriting a neighbor's value through a shared vertex.
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
and seams the surface between clamped and unclamped neighbors,
that is not a bounded deformation but a different one.
Scaling is the operation an eigenmode is indifferent to, being defined up to a scalar,
so it bounds the picture without changing which mode the picture is of.

Up to the seam the discipline is lived, not hoped for:
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
not three pipelines:
a curve path split off from a surface path
would be the `if dim == 3` of the parent, reappearing here.

Two reductions carry that, and they are the same move made on the two axes:

- **Grade reduces to a mark:**
  A $k$-form reduces to its *reduced grade* $min(k, n-k)$ through the Hodge star,
  and the render mark is chosen on that.
  Where that star actually fires ($k > n-k$) it needs a *global* volume form,
  so the reduction takes the cell's coherent orientation alongside the metric,
  the parent's invariant 6,
  and the one place the extrinsic side needs a topological datum the solver never asks for.
  A mesh with no coherent orientation has no such reduction to show,
  so those fields are refused up front rather than drawn with a per-cell sign.
  A field that reaches a mark is already the proof that its orientation exists.

  Where a gauge is genuinely free,
  prefer making the mark independent of it over picking a value for it.
  The rigid cell displacement $d_K n_K$ is the model:
  the density and the cell normal flip together, so the motion is invariant.
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
never smeared into a consumer,
which sees only which *items* a frame has, never why.
What the current ambient does not yet reach (a reduced grade $>= 2$, a point cloud's mark)
is where these extend, not a branch to route around.

## The extrinsic freedom is the embedding, not the metric

Because an embedding is present,
`realize` may use it and the ambient geometry it induces
(normals, ambient distances, global position)
wherever that is cleaner than an intrinsic construction.
This is the genuine license the core denies itself.

Name it precisely, because it is easy to overclaim:
the freedom is *ambient* geometry, not *metric*.
A metric is not an extrinsic object.
The core uses it freely
(invariant 5 forbids only letting it leak into a signature that does not mathematically need one),
and every metric here is the one the embedding already induces.
What `realize` grants itself over the core
is the embedding and the ambient space, and the global geometry read off them.
Nothing about the metric changes.

## Anti-goals

- No graphics dependency.
  The crate ends at the bake and its exporters;
  a device, a buffer or a pipeline belongs to a consumer.
- No reduction reachable by only one consumer.
  An interchange file and a renderer read the same field the same way,
  or they will disagree about what it shows.
- No dimension dispatch outside the bake, and no grade dispatch outside the mark.
  A `match` on either anywhere else is the case distinction escaping its reduction.
- No embedding leaking in above the seam.
  No ambient assumption above dimension 3.
- No claiming metric use as the extrinsic divergence:
  the divergence is the embedding and the ambient space,
  and saying otherwise misreads invariant 5.
- Nothing transient here, exactly as in the parent:
  no current state, no in-flight passes, no version pins written out,
  no roadmap phrased as a promise.
