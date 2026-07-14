# formoniq

A Finite Element Exterior Calculus (FEEC) library in Rust: PDEs formulated with
differential forms, solved on simplicial Riemannian manifolds of arbitrary
dimension, intrinsically and coordinate-free.

The mathematics is the design. Differential geometry, algebraic topology,
functional analysis and category theory enter as types, traits and laws, not as
commentary on them. Code should read the way a mathematician would write.

## Design goals

- **Unification over special-casing.** One general principle covers the many
  classical special cases. Gradient/curl/divergence are one exterior derivative;
  Poisson/Maxwell/Hodge-Laplace are one Hodge-Laplace problem; scalar and vector
  FEM are Whitney forms at grade 0 and 1. Never re-introduce the special cases.
- **Arbitrary dimension, always.** Nothing is hardcoded to 2D or 3D. Dimension
  is a runtime value `Dim`, grade a runtime value `ExteriorGrade`. If you find
  yourself writing `if dim == 3`, the abstraction is wrong.
- Directions being explored, not commitments: BEM and spectral methods within
  the same exterior-calculus frame; higher-order (trimmed polynomial
  $P^-_r Lambda^k$) elements; curvature, higher-order Regge and isoparametric
  cells.

## Architecture

Strict crate ladder, each layer adding exactly one thing:

| crate      | is                                  | key contents |
| ---------- | ----------------------------------- | ------------ |
| `common`   | shared math substrate               | `Combination`/`Sign` (combinatorics), `Gramian`/`RiemannianMetric`, linalg backends (nalgebra/faer/petsc), `Dim` |
| `exterior` | the exterior algebra $Lambda^k$     | `ExteriorElement<V>`, `Variance` (`Covariant`/`Contravariant`), `exterior_power`, wedge, interior product, musicals, Hodge star, `field::CoordField<V>` (flat, mesh-independent) |
| `manifold` | the simplicial Riemannian manifold  | `topology::` (`Complex`, `Skeleton`, `SimplexRef`, boundary operators), `geometry::` (`Geometry` trait, `MeshCoords`, `MeshLengths`, `CellGramians`) and `point::MeshPoint` (the barycentric chart) |
| `ddf`      | discrete differential forms         | `Cochain`, `field::ExteriorField<V>` (sections over the manifold) with `Pullback`/`Sampler`, `whitney::` (`WhitneyLsf`, `WhitneyForm`), `derham::derham_map` |
| `formoniq` | the FEM engine                      | `assemble`, `operators` (`ElMatProvider`/`ElVecProvider`), `bc`, `problems::` (hodge_laplace, maxwell, heat, wave, ...) |

Dependencies flow strictly downward. A lower crate never learns about a higher
one: `exterior` must never hear about meshes, `manifold` never about forms.

**Concepts float up.** A concept belongs in the lowest crate (or module) that can
express it with the dependencies it already has. If expressing it there would
need a new downward dependency, it belongs one level up instead, in the crate
that joins the two — which is why `ddf` exists. Never widen a lower crate's
dependencies to make a method fit.

Composition therefore reaches down from above: a free function in the joining
crate by default, or a thin `...Ext` trait where method syntax carries the math
better — `CoordSimplexExt` (an `exterior` construction on a `manifold` simplex),
`SimplexRefExt` (geometry methods on a topology handle, which is how invariant 1
is upheld inside `manifold`, below crate granularity).

## The load-bearing invariants

These are the design, not preferences. Breaking one is a bug even if it compiles
and passes tests.

1. **Topology ⊥ Geometry.** The `Complex` is pure combinatorics: it knows
   incidence, orientation, boundary — nothing metric. Geometry is a *separate*
   input, and enters only through the `Geometry` trait
   (`fn cell_metric(&self, cell: SimplexRef) -> RiemannianMetric`).

2. **Intrinsic first, extrinsic second.** Assembly consumes only per-cell metric
   tensors, never coordinates. `MeshCoords` (an embedding) is just one
   `Geometry` implementor, on equal footing with Regge edge lengths
   (`MeshLengths`) and raw per-cell metrics (`CellGramians`). Anything that
   *requires* an embedding is a wrapper for I/O, visualization or convenience —
   it must not sit in the core path. A feature that only works on embedded
   meshes is an unfinished feature.

   A **point of the manifold** is therefore `MeshPoint` — a cell plus
   barycentric coordinates, the chart of the piecewise-flat atlas — never a
   global coordinate, which on a Regge manifold does not exist. A **field** is
   an `ExteriorField<V>`: a section of the exterior bundle, evaluated at a
   `MeshPoint`, valued in the reference frame of that cell's chart. The flat
   `CoordField<V>` of analytic data (exact solutions, sources) is a *different*
   concept, and reaches the mesh only through the pullback adapter. Sampling
   back into ambient coordinates (`Sampler`) is not canonical — it extends the
   value along the chart pseudo-inverse — and is confined to I/O.

3. **Variance is type-level.** `ExteriorElement<Covariant>` (multiforms) and
   `ExteriorElement<Contravariant>` (multivectors) stand on fully equal footing.
   The type parameter is what makes the functorial direction (pullback vs.
   pushforward), the duality pairing, the musical isomorphisms and the choice of
   $g$ vs. $g^(-1)$ correct *by construction*. Never collapse the two, and never
   pick the Gramian by hand — go through `V::gramian` / `multiform_gramian` /
   `multivector_gramian`. Fields inherit this: `Pullback` implements
   `ExteriorField` only for `Covariant`, so the type system — not a convention —
   is what stops a multivector field from being pulled back.

4. **Metric-free stays metric-free.** The exterior derivative, the boundary
   operator, the wedge, the interior product, the duality pairing and the de Rham
   map involve *no* metric. Only the Hodge star, the musicals and inner products
   do. Do not let a `RiemannianMetric` leak into a signature that does not
   mathematically need one.

5. **Zero-cost abstractions.** Generics and monomorphization, not `dyn` and
   runtime dispatch, in anything on the assembly hot path. HPC is a requirement,
   not an afterthought (`rayon`-parallel assembly is already the norm).

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
for globally assembled operators; faer and petsc for solves and eigenproblems.
Go through `common::linalg` rather than reaching for a backend directly.

**Naming reflects the mathematics.** `SimplexRef`, `Cochain`, `MultiForm`,
`CellGramians` — a reader who knows the math should recognize every type
immediately, and one who does not should be able to look it up.

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

The examples are the end-to-end check and are run by hand:

```sh
cargo run --release --example hodge_laplace_source
```

Commit messages: `scope: imperative summary`, e.g. `manifold: cache boundary
operators lazily`. Keep commits structurally coherent — one idea each.

A change to the design is not finished until this file reflects it, in the same
commit. Where CLAUDE.md and the code disagree, one of them is a bug — and it is
usually worth asking which, because an invariant that the code has quietly
outgrown is a design decision nobody made deliberately.

## Origin

v0.1 was a BSc-thesis implementation, focused on the elliptic Hodge-Laplace
problem with the first-order Whitney basis. v0.2 is the rebuild toward the
library described above. Where thesis-era code still contradicts the invariants,
the invariants win.
