# formoniq

A Finite Element Exterior Calculus (FEEC) library in Rust: PDEs formulated with
differential forms, solved on simplicial Riemannian manifolds of arbitrary
dimension, intrinsically and coordinate-free.

This is a mathematical discipline taking software form. The math is not a
motivation for the code, it *is* the code: differential geometry, algebraic
topology, functional analysis, category theory and mathematical physics,
realized as types, traits and laws. Write it the way a mathematician would want
to read it.

## Mission

- The ultimate FEEC / differential-geometry / PDE library.
- **Unification over special-casing.** One general principle covers the many
  classical special cases. Gradient/curl/divergence are one exterior derivative;
  Poisson/Maxwell/Hodge-Laplace are one Hodge-Laplace problem; scalar and vector
  FEM are Whitney forms at grade 0 and 1. Never re-introduce the special cases.
- **Arbitrary dimension, always.** Nothing is hardcoded to 2D or 3D. Dimension
  is a runtime value `Dim`, grade a runtime value `ExteriorGrade`. If you find
  yourself writing `if dim == 3`, the abstraction is wrong.
- Long-term: BEM and spectral methods inside the same exterior-calculus frame;
  higher-order (trimmed polynomial $P^-_r Lambda^k$) elements; 
  curvature, higher order Regge or isoparametric cells.

## Architecture

Strict crate ladder, each layer adding exactly one thing:

| crate      | is                                  | key contents |
| ---------- | ----------------------------------- | ------------ |
| `common`   | shared math substrate               | `Combination`/`Sign` (combinatorics), `Gramian`/`RiemannianMetric`, linalg backends (nalgebra/faer/petsc), `Dim` |
| `exterior` | the exterior algebra $Lambda^k$     | `ExteriorElement<V>`, `Variance` (`Covariant`/`Contravariant`), `exterior_power`, wedge, interior product, musicals, Hodge star |
| `manifold` | the simplicial Riemannian manifold  | `topology::` (`Complex`, `Skeleton`, `SimplexRef`, boundary operators) and `geometry::` (`Geometry` trait, `MeshCoords`, `MeshLengths`, `CellGramians`) |
| `ddf`      | discrete differential forms         | `Cochain`, `whitney::` (`WhitneyLsf`, `WhitneyForm`), `derham::derham_map` |
| `formoniq` | the FEM engine                      | `assemble`, `operators` (`ElMatProvider`/`ElVecProvider`), `bc`, `problems::` (hodge_laplace, maxwell, heat, wave, ...) |

Dependencies flow strictly downward. A lower crate never learns about a higher
one: `exterior` must never hear about meshes, `manifold` never about forms.

**Dependencies flow down; concepts float up.** A concept belongs in the lowest
crate (or module) that can express it with the dependencies it already has. If
expressing it there would need a new downward dependency, it belongs one level
up instead, in the crate that joins the two — which is why `ddf` exists. Never
widen a lower crate's dependencies to make a method fit.

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

3. **Variance is type-level.** `ExteriorElement<Covariant>` (multiforms) and
   `ExteriorElement<Contravariant>` (multivectors) stand on fully equal footing.
   The type parameter is what makes the functorial direction (pullback vs.
   pushforward), the duality pairing, the musical isomorphisms and the choice of
   $g$ vs. $g^(-1)$ correct *by construction*. Never collapse the two, and never
   pick the Gramian by hand — go through `V::gramian` / `multiform_gramian` /
   `multivector_gramian`.

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

## Workflow

Every commit passes all four. They are the bar, not a suggestion:

```sh
cargo fmt --all                        # rustfmt.toml: 2-space indent
cargo clippy --workspace --all-targets # clean at default lints
cargo test --workspace                 # law tests + integration tests; stay green
cargo doc --workspace --no-deps        # doc comments carry the math: no warnings,
                                       # intra-doc links must resolve
```

The examples are the end-to-end check and are run by hand:

```sh
cargo run --release --example hodge_laplace_source
```

Commit messages: `scope: imperative summary`, e.g. `manifold: cache boundary
operators lazily`. Keep commits structurally coherent — one idea each.

## Origin

v0.1 was a BSc-thesis implementation (ETH Zurich, supervised by
Prof. Dr. Ralf Hiptmair), focused on the elliptic Hodge-Laplace problem with the
first-order Whitney basis. v0.2 is the rebuild toward the library described
above. Where thesis-era code still contradicts the invariants, the invariants
win.
