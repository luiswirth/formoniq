# formoniq

A Finite Element Exterior Calculus (FEEC) library in Rust: PDEs formulated with
differential forms, solved on simplicial pseudo-Riemannian manifolds of
arbitrary dimension, intrinsically and coordinate-free.

The mathematics is the design. Differential geometry, algebraic topology,
functional analysis and category theory enter as types, traits and laws, not as
commentary on them. Code should read the way a mathematician would write.

## Design goals

- **Unification over special-casing.** One general principle covers the many
  classical special cases. Gradient/curl/divergence are one exterior derivative;
  Poisson/Maxwell/Hodge-Laplace are one Hodge-Laplace problem; scalar and vector
  FEM are Whitney forms at grade 0 and 1; Riemannian and Lorentzian geometry are
  one pseudo-Riemannian metric of signature $(p, q)$, the Hodge star reading the
  signature off the metric itself. Never re-introduce the special cases.
- **Arbitrary dimension, always.** Nothing is hardcoded to 2D or 3D. Dimension
  is a runtime value `Dim`, grade a runtime value `ExteriorGrade`. If you find
  yourself writing `if dim == 3`, the abstraction is wrong.
- **Total on the degenerate boundary.** Dimensional agnosticism is the interior
  claim; the stronger one is that the range is closed at its extremes. The base
  dimension, the extremal grades, an empty skeleton, a one-element system — these
  are where generic code silently breaks (a block sized wrong, an index one past
  the top, a backend's small-input path) and where the special-case temptation is
  strongest. The abstraction must be total there too: an edge case runs on the
  same code and returns the mathematically trivial answer — the empty result, the
  zero operator, the single harmonic mode — rather than being excluded. A base
  case that holds is a proof of the unification; one that panics is a hidden
  `if dim == ...` the design never admitted to.
- Directions being explored, not commitments: BEM and spectral methods within
  the same exterior-calculus frame; higher-order (trimmed polynomial
  $P^-_r Lambda^k$) elements; curvature, higher-order Regge and isoparametric
  cells.

## Architecture

Crate ladder, each layer adding exactly one thing —
`{ multiindex, gramian, coorder } → exterior → { simplicial, glatt } →
derham → formoniq → studio`, where `multiindex`/`gramian`/`coorder` are
foundational siblings and `simplicial`/`glatt` are siblings one level up:

| crate        | is                                  | key contents |
| ------------ | ----------------------------------- | ------------ |
| `multiindex` | combinatorial index structures      | `Combination`/`Sign` (colex-ranked subsets), `cartesian::` (radix multi-indices) |
| `gramian`    | inner-product / metric structure    | `Gramian` (non-degenerate symmetric, any signature), `Metric` (the pseudo-Riemannian metric tensor, any signature; Riemannian is $q = 0$), `CausalType` |
| `coorder`    | typed affine coordinates            | `Coords<S>` (coordinates tagged by their space), `affine::AffineTransform` |
| `exterior`   | the exterior algebra $Lambda^k$     | `ExteriorElement<V>`, `Variance` (`Covariant`/`Contravariant`), `exterior_power`, wedge, interior product, musicals, Hodge star, `pullback`/`pushforward` of a value along a linear map |
| `simplicial` | the simplicial manifold $M_h$       | `topology::` (`Complex`, `Skeleton`, `SimplexRef`, the `role::` witnesses `Cell`/`Facet`/..., boundary operators), `atlas::` (`Chart`, `MeshPoint`, `Transition`, `Bary`/`Local`, `SimplexQuadRule`), `geometry::` (`MeshLengthsSq` the intrinsic Regge primitive the engine consumes, `MeshCoords` and `CellGramians` the sources that convert into it) and `linalg::` (the dense/sparse nalgebra aliases and `CooMatrixExt` block-matrix builder every crate above it reuses) |
| `glatt`    | the continuum manifold $M$          | `Parametrization` (forward map $phi$, derived nearest-point chart, `sphere`/`ball`/`torus`/`graph`), `field::CoordField<V, S>` (analytic data *on* $M$: `DiffFormClosure`, ...) |
| `derham`     | discrete differential forms         | `Cochain`, `section::Section<V>` (sections over the simplicial manifold) with the `Pullback` bridge (`pullback_on`/`pullback_through`) and `Sampler`, `interpolate::` (`WhitneyForm`, `WhitneyInterpolant`), `project::derham_map` |
| `formoniq`   | the FEM engine                      | `assemble`, `operators` (`ElMatProvider`/`ElVecProvider`), `bc`, `time` (`Tableau`, `LinearIrk` and the explicit symplectic `Leapfrog`: structure-preserving time integration), `linalg::` (the faer bridge and shift-invert eigensolving -- the one crate that actually solves anything), `problems::` (elliptic, dirac, heat, wave, ...) |
| `studio`     | the visualizer                      | `Scene` (the engine↔viewer seam, carrying `Complex`/`MeshCoords`/`Cochain`), `BakedMesh` (the $RR^3$ bake, dimension reduced to a render primitive), reduced-grade render marks (scalar density, glyph/particle line field), a wgpu/winit/egui renderer, native and wasm |

No crate exists solely to hold a shared type alias. `Vector`/`Matrix` (dense
nalgebra) are trivial aliases with no nominal identity to share, so `gramian`,
`coorder`, `exterior` and `glatt` each declare their own directly from
`nalgebra` rather than depending on anything for them; `simplicial` is the
lowest crate that needs *sparse* matrices (its boundary operators), so that is
where `CsrMatrix`/`CooMatrix` and the extension traits built on them
(`CooMatrixExt`) live, reused downward by `derham`, `formoniq` and `studio`
because they already depend on `simplicial` for real reasons. `faer` and the
eigensolver go one further: they are needed only in `formoniq`, the one crate
that solves a linear system, so they live there rather than in a shared base
every leaf would then compile for nothing.

Dependencies flow strictly downward. A lower crate never learns about a higher
one: `exterior` must never hear about meshes, `simplicial` never about forms.
`simplicial` (the simplicial $M_h$) and `glatt` (the smooth $M$ it
approximates) are independent objects, so neither depends on the other; their
one relation — pulling continuum data onto the mesh, and the error that costs —
is the join, and it lives in `derham`, the crate above both.

`studio` sits at the top as the visual counterpart to the engine — the one
consumer of the I/O-and-visualization carve-out invariant 2 draws. Visualization
needs an embedding, so `studio` is extrinsic by necessity where the core is
intrinsic by discipline; it depends downward on `formoniq` and below, nothing
depends on it, and it carries its own `crates/studio/CLAUDE.md` for what that
inversion means. The parent's invariants still bind it — they are only read from
the extrinsic side.

**Concepts float up.** A concept belongs in the lowest crate (or module) that can
express it with the dependencies it already has. If expressing it there would
need a new downward dependency, it belongs one level up instead, in the crate
that joins the two — which is why `derham` exists, where `exterior`, `simplicial`
and `glatt` all meet. Never widen a lower crate's dependencies to make a method
fit.

**The building-block crates are standalone, and published as such.** Concepts
floating up leaves each lower crate a self-contained mathematical object, not
FEEC-internal plumbing: `exterior` is an exterior-algebra library, `simplicial`
a simplicial-topology-and-Regge-geometry one, `multiindex` colex combinatorics,
`glatt` continuum differential geometry — each usable, and released, on its own,
with FEEC only the thing `derham` and `formoniq` build on top. This is a goal to
uphold, not just an emergent property: a lower crate must earn its keep for a
reader who has never heard of FEEC. So its public docs explain it in its own
terms — the FEEC application is never a crate's stated reason for existing, and a
higher concept (assembly, Whitney forms, cochains as discrete forms) never
stands in for what the code means intrinsically. The one exception is a
crate-level architecture doc naming its neighbors to place itself: there the
cross-crate relation *is* the content, and naming it is right.

Composition therefore reaches down from above: a free function in the joining
crate by default, or a thin `...Ext` trait where method syntax carries the math
better — `CoordFieldExt::pullback_through` (and its identity special case
`pullback_on`) and `SectionExt::sampled_on` (a `glatt` field meeting a
`simplicial` mesh, in `derham`), `SimplexRefExt` (geometry methods on a topology
handle, which is how invariant 1 is upheld inside `simplicial`, below crate
granularity).

The rule bites *within* a crate too, not just between crates. `metric` must not
import `coord`: an embedding induces a metric, a metric induces no embedding, so
`MeshCoords::cell_metric` (and `to_edge_lengths_sq`) belongs on the `coord`
side. And the atlas sits
below both — the reference cell, its barycentric coordinates and quadrature over
it need neither a metric nor an embedding, so they must live in neither layer.

## The load-bearing invariants

These are the design, not preferences. Breaking one is a bug even if it compiles
and passes tests.

1. **Topology ⊥ Geometry.** The `Complex` is pure combinatorics: it knows
   incidence, orientation, boundary — nothing metric. Geometry is a *separate*
   input, carried by `MeshLengthsSq` (signed squared edge lengths on the
   1-skeleton) and reaching assembly as the per-cell metric its
   `cell_metric(cell) -> Metric` derives. There is deliberately **no `Geometry`
   trait**: a trait over geometry representations buys only lazy per-cell
   streaming, which is negligible next to element-matrix evaluation, at the cost
   of the totality edge lengths give for free. So the engine speaks one concrete
   intrinsic type, and the other representations convert into it.

2. **Intrinsic first, extrinsic second, and edge lengths are the primitive.**
   The engine consumes `MeshLengthsSq`, never coordinates: it is the Regge
   primitive, the source of truth, and the one representation *total over every
   grade* (see below). The *signed squared* length — positive spacelike, zero
   null, negative timelike, mirroring `norm_sq` — is what keeps Regge geometry
   total over every metric signature: Regge calculus was invented for Lorentzian
   spacetimes, and an unsquared length would lose the causal sign. `MeshCoords`
   (an embedding) and `CellGramians` (raw per-cell metrics) are *sources*, not
   engine currencies: each converts to edge lengths at the boundary of the API
   (`to_edge_lengths_sq`), on equal footing precisely because they reduce to the
   same primitive. `CellGramians` also serves as the materialized cell column
   (`from_lengths`) that refinement pulls back. Anything that *requires* an
   embedding is a wrapper for I/O, visualization or convenience — it must not sit
   in the core path. A feature that only works on embedded meshes is an
   unfinished feature.

   **Geometry is defined on every simplex, the chart only on the cells.** The
   metric of any subsimplex — an edge's length, a facet's area, a hinge's metric
   — is the Gramian of that simplex's own edges (`MeshLengthsSq::simplex_metric`,
   `simplex_volume`), well defined from the shared edge data with no containing
   cell consulted. This is why edge lengths, not `CellGramians`, is the
   primitive: it answers geometry at every grade, which is what the boundary
   trace, DEC-style constructions and higher-dimensional Regge curvature need. A
   *chart*, by contrast, exists only on a top-dimensional simplex (invariant 3):
   pinning a metric accessor to the `Cell` witness would conflate *has a metric*
   (all simplices) with *carries a frame* (cells only).

   A **point of the simplicial manifold** is therefore `MeshPoint` — a `Chart`
   plus barycentric coordinates — never a global coordinate, which on a Regge
   manifold does not exist. A **field** is a `Section<V>`: a section of the
   exterior bundle, evaluated at a `MeshPoint`, valued in the reference frame of
   that chart. The `CoordField<V, S>` of analytic data on the *continuum* (exact
   solutions, sources) is a *different* concept, living in `glatt`, and
   reaches the mesh only through the `Pullback` bridge — pulled through a cell's
   parametrization and the continuum chart, the flat domain being the identity
   special case. Sampling back into ambient coordinates (`Sampler`) is not
   canonical — it extends the value along the pseudo-inverse of the cell
   parametrization — and is confined to I/O.

   **The cells are an atlas** (`simplicial::atlas`), and it is a real one. A
   `Chart` *is* a cell — literally: the name is a type alias of the `Cell`
   role witness, so top-dimensionality holds by construction — a face carries
   no chart, so there is no frame on one in which to express a value. Two charts
   overlap in the face they share, and the `Transition` between them is the
   affine relabelling of barycentric weights: metric-free, exact, and obeying the
   cocycle law. Its differential is the change of frame on the tangent space of
   the overlap, which is why only the *tangential* part of a section is
   chart-independent — and hence why the de Rham map is well defined on a face
   while a pointwise Whitney value is not. Anything claiming chart-independence
   owes a `Transition` argument.

   The chart's own structure — reference vertices, barycentric differentials,
   volume, quadrature — is a function of `Dim` alone (the `ref_*` functions), and
   deliberately so: **every chart of the atlas is the same chart up to the
   labelling of its vertices.** That is exactly why element matrices are computed
   once on the reference cell and reused on every cell of the mesh. What differs
   between charts is the labelling, and the labelling is what a `Transition` is
   made of. Do not bind reference-cell data to a cell.

3. **Coordinate spaces are type-level.** Barycentric weights `Bary`
   ($lambda in RR^(n+1)$), the cartesian `Local` coordinates of a chart
   ($x in RR^n$) and the `Ambient` coordinates of an embedding ($RR^N$) are three
   different spaces, and `coorder::Coords<S>` tags each with the space it
   lives in. The maps between them therefore have to be written down, and the
   wrong composition does not compile. A bare `Vector` is a displacement or raw
   linear algebra, never a point.

4. **Variance is type-level.** `ExteriorElement<Covariant>` (multiforms) and
   `ExteriorElement<Contravariant>` (multivectors) stand on fully equal footing.
   The type parameter is what makes the functorial direction (pullback vs.
   pushforward), the duality pairing, the musical isomorphisms and the choice of
   $g$ vs. $g^(-1)$ correct *by construction*. Never collapse the two, and never
   pick the Gramian by hand — go through `V::gramian` / `multiform_gramian` /
   `multivector_gramian`. Sections inherit this: `Pullback` implements `Section`
   only for `Covariant`, so the type system — not a convention — is what stops a
   multivector field from being pulled back.

5. **Metric-free stays metric-free.** The exterior derivative, the boundary
   operator, the wedge, the interior product, the duality pairing and the de Rham
   map involve *no* metric. Only the Hodge star, the musicals and inner products
   do. Do not let a `Metric` leak into a signature that does not
   mathematically need one.

6. **Zero-cost abstractions.** Generics and monomorphization, not `dyn` and
   runtime dispatch, in anything on the assembly hot path. HPC is a requirement,
   not an afterthought (`rayon`-parallel assembly is already the norm).

**Invariants 3 and 4 are proofs, not conventions** — Lean 4 style: a
precondition that is a property of a value (the space a coordinate lives in, the
variance of a form) becomes a type-level witness, not an assertion repeated at
each call. The type demands the property, the check happens once where the
witness is built, and the wrong composition fails to compile. Reach for this
wherever a "trust me, this is an *X*" comment sits in a signature. The simplex
roles of `topology::role` are the same pattern on a runtime dimension: a
`Roled<R>` (`Cell`, `Facet`, ...) is a `SimplexRef` plus the proof of its
dimension proposition, produced for free by navigation (`cells()`,
`facets()`, `vertices()`, `edges()`) and checked once at the index boundary
(`role()`); `Chart` is a type *alias* of the `Cell` witness — the atlas
operations live in `ChartExt` — and `MeshLengthsSq::cell_metric` consumes it, so
"this simplex is a cell" is a type, never a repeated assertion (the `Cell`
witness marks the *chart*; the metric itself is total over every grade through
`simplex_metric`). A role's one datum is its `RoleDim` — a dimension
pinned absolutely or by codimension — and `Complex::role_skeleton::<R>()` is
the total accessor derived from it: `None` where the complex has no such
dimension, which is how the degenerate boundary stays total (a point has no
facets, not an underflow). Roles are propositions, not a partition — the edge
of a 1-complex is an `Edge` *and* a `Cell`. A proof speaks only for the
complex object it was built from; `SimplexRef::belongs_to` is the identity
check consumed where handles cross between complexes.

## Conventions

**Doc comments carry the math, in Typst notation.** This is the house style;
match it exactly (`exterior/src/lib.rs` holds the canonical examples):

```rust
/// The interior product (contraction)
/// $iota_v: Lambda^k -> Lambda^(k-1)$ with a grade-1 element of the dual
/// variance.
///
/// Metric-free. An antiderivation of degree -1 with $iota_v^2 = 0$: the
/// dual of the wedge. With the all-ones vector it IS the boundary
/// operator, $diff = iota_bb(1)$.
```

Not LaTeX, not unicode soup. State *what the object is* mathematically, the laws
it obeys, and the invariants and contracts the code cannot show. Never narrate
what the next line does.

The one exception is the **crate-level overview**, which is the crate's
`README.md` pulled in verbatim by `#![doc = include_str!("../README.md")]` — one
source of truth for the docs.rs landing page and the GitHub/crates.io README,
never two hand-synced copies. A README is rendered by GitHub and crates.io,
which know neither Typst nor rustdoc's intra-doc links, so the shared text is
plain **Unicode** markdown (the sole place Unicode stands in for Typst), prose,
with no intra-doc links and only the little math notation that genuinely helps.
Type- and module-level navigation stays on the item docs, where the links
resolve and the Typst renders as house style.

**Tests are theorems.** The test suite is a machine-checked statement of the
mathematics, and it is how correctness is actually established here. New math
ships with the law that characterizes it, not with a golden number:

- functoriality: $Lambda^k (A B) = (Lambda^k A)(Lambda^k B)$ (Cauchy-Binet)
- adjointness: $angle.l A^* omega, v angle.r = angle.l omega, A_* v angle.r$
- nilpotency: $iota_v^2 = 0$, $diff compose diff = 0$, $dif compose dif = 0$
- involution: $star star = (-1)^(k(n-k))$
- Whitney's theorem: $R compose W = id$
- Stokes: $R compose dif = dif compose R$
- commuting subcomplex: $dif compose W = W compose dif$

Sweep over all dimensions and grades (`for dim in 0..=4`, `for grade in 0..=dim`)
rather than fixing one case. The examples in `crates/formoniq/examples/` are the
end-to-end check — convergence rates, spectra — but they are run and read by
hand, not asserted by `cargo test`.

**Colexicographic order is the one indexing convention.** Basis blades,
combinations, simplex vertices and the simplices within a skeleton are all
colex-ordered, with `Combination::rank()` as the canonical index — that shared
order is what lines up the coefficients of an `ExteriorElement`, the local faces
of a cell, and global position in a `Skeleton`. Lexicographic order compiles
just as well and silently means something else. A new ordered structure is colex.

**Linalg backends by role.** nalgebra dense (`Matrix`/`Vector`) for element-local
math (Gramians, element matrices, exterior powers); `nalgebra-sparse` (`CooMatrix`)
for globally assembled operators, aliased in `simplicial::linalg`; faer for
solves and eigenproblems (sparse LU and Cholesky, and a self-adjoint dense
eigensolve for the projected subspace), confined to `formoniq::linalg` since
`formoniq` is the only crate that solves anything. The workspace is pure Rust,
with no external solver toolchain.

**Naming reflects the mathematics.** `SimplexRef`, `Cochain`, `MultiForm`,
`CellGramians` — a reader who knows the math should recognize every type
immediately, and one who does not should be able to look it up. Where a word has
a precise meaning, it is used precisely, and two words that mean different things
never stand in for each other:

**Affine, flat, linear are three different claims.** *Affine* is about the
**maps**: the cell charts are $x |-> v_0 + A x$, the barycentric weights are an
affine combination, the transition maps are affine gluings. It is metric-free,
and it is what the atlas is — *piecewise affine*, never "piecewise flat". *Flat*
is about the **curvature**, so it presupposes a metric: a Regge manifold is
piecewise flat, curvature vanishing on cell interiors and concentrating on the
codimension-2 hinges. *Linear* is neither, and is wrong here: $lambda_i$ is
affine, not linear, and "piecewise linear FEM" is the classical abuse. Don't
inherit it — the affine/linear distinction is exactly why barycentric
coordinates are the right chart.

**A chart and a parametrization point opposite ways.** A *chart* maps the
manifold **out to** coordinates; a *parametrization* maps coordinates **in to**
the manifold. They are inverse, and the direction is the whole content of the
words. The `Chart` of a cell is barycentric, intrinsic, and exists on every
geometry. The `SimplexCoords<S>` of a cell is its affine *parametrization*
$hat(K) -> S$ into a coordinate space — never call it a chart. It is generic
over that space (invariant 3): `SimplexCoords<Ambient>` (the default, and the
lowest module — `atlas::simplex_coords` — that defines it) is the extrinsic
realization $hat(K) -> RR^N$, and only *it* presupposes an embedding; the metric
and edge lengths it induces are the `geometry::coord` bridges bolted onto that
instantiation. `SimplexCoords<LocalCartesian>` is the metric-free realization in
a chart's own cartesian frame — the reference cell (`standard`) and a
refinement child in its parent's frame — which is why the affine core lives in
`atlas`, reachable by `topology`, not in `geometry::coord`. A
curvilinear coordinate system on the manifold being approximated (spherical on
$S^2$, polar on a disk) is likewise a parametrization: it is written
$(theta, phi) |-> RR^3$, and the fact that it must carry its own inverse as
separate data is the tell that the inverse is the chart.

**Mesh, simplicial manifold, manifold are three different objects.** The *mesh*
**is** the simplicial complex — one object, two words, never used as though they
were two things. The *simplicial manifold* is that complex realized with a
geometry: the piecewise-affine object, on which `Chart` is a chart, `MeshPoint` a
point and `Section` a field. The *manifold* is the continuous thing the simplicial one
approximates — possibly smooth, possibly given by a parametrization, possibly
identical to the simplicial one. What is exact on the simplicial manifold need
not be exact on the manifold, and a name that blurs the two hides precisely that
gap.

**Rust style.** 2-space indent (`rustfmt.toml`); clean under default clippy
lints, with `clippy::pedantic` applied selectively rather than enforced.
Idiomatic and expressive, concise and self-explanatory. Prefer the iterator
chain that states the intent over the loop that states the mechanics.

## Public artifacts

The README, this file, the doc comments, the issues and the commit messages are
all read by people. Write them for a skeptical senior reader, because that is
who shows up.

- **No superlatives, no marketing register.** "The ultimate library" is not a
  claim, it is a tell. State what the code does and let the reader judge. The
  work is strong enough that overselling it only subtracts credibility.
- **Roadmap is direction, not promise.** Say what is being explored, not what is
  coming.
- **Argue from the mathematics and the design, never from this file.** An issue
  that cites CLAUDE.md as its authority documents a process, not a reason. The
  reason has to stand on its own.
- **Verify before asserting.** Every number, flag and capability gets checked
  against the code first. A confident unverified specific is worse than none.
- **Fetch, don't recall.** Anything with a canonical source — a license text, a
  version pin, a CI action version, an external API — is retrieved, never
  reproduced from memory. Recall yields the plausible, which is indistinguishable
  from the correct on the page, and therefore survives review.
- **Keep the tooling out of the content.** AI assistance here is deliberate and
  disclosed, in commit trailers. That is what transparency looks like; it does
  not mean narrating the assistant inside a README, an issue or a doc comment.
- **Plain prose.** No emoji, no decorative dividers, no headers over three-line
  sections, no bold on every other phrase.

## Anti-goals

- No hacks. If a test fails, the mathematics or the abstraction is wrong — fix
  that, never paper over it with a fudge, a tolerance bump or a special case.
  The rule is about diagnosis and does not stop at the math: find the root cause
  in the build, the tooling and the environment too. A change that removes the
  symptom without explaining it is not a fix.
- No dimension- or grade-specific code paths in the core.
- No classical vector-calculus fallback (no separate grad/curl/div, no
  cross-product-flavored shortcuts).
- No embedding assumptions in the core path (see invariant 2).
- No comments that restate the code. Comments carry invariants, contracts and
  mathematical context only.
- Nothing transient in this file. It carries architecture, invariants,
  conventions and anti-goals — what stays true. Never current state, in-flight
  plans, tooling wire-ups, personal details, or pointers to things that move.
  Those belong in issues, commits and the code itself.

## Workflow

Every commit passes all four. They are the bar, not a suggestion:

```sh
cargo fmt --all                        # rustfmt.toml: 2-space indent
cargo clippy --workspace --all-targets # clean at default lints
cargo test --workspace                 # law tests + integration tests; stay green
cargo doc --workspace --no-deps        # doc comments carry the math: no warnings,
                                       # intra-doc links must resolve
```

CI runs the same four on every push and pull request; a red build is a broken
commit, not a flaky one.

`studio` is excluded from the workspace's `default-members`, so a bare `cargo
test`/`clippy`/`doc` (no `--workspace`) skips the wgpu/winit/egui stack and
covers the core crates alone — the fast inner loop. The `--workspace` forms
above are the full bar and are what a cross-cutting commit must pass; CI splits
the core checks (default members) from a separate `studio` job so a core change
is not gated on the graphics stack.

The examples are the end-to-end check and are run by hand:

```sh
cargo run --release --example source
```

Commit messages: `scope: imperative summary`, e.g. `simplicial: cache boundary
operators lazily`. Keep commits structurally coherent — one idea each where
that's easily reached from what's already staged or in progress; splitting
unrelated changes is not worth contorting history over, so bundling a few into
one commit is fine when separating them would be the more artificial move.

A change to the design is not finished until this file reflects it, in the same
commit. Where CLAUDE.md and the code disagree, one of them is a bug — and it is
usually worth asking which, because an invariant that the code has quietly
outgrown is a design decision nobody made deliberately.

## Origin

v0.1 was a BSc-thesis implementation, focused on the elliptic Hodge-Laplace
problem with the first-order Whitney basis. v0.2 is the rebuild toward the
library described above. Where thesis-era code still contradicts the invariants,
the invariants win.
