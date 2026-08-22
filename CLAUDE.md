# formoniq

`README.md` says what the library is and does.
This file is the design doc:
the invariants, conventions and house style an agent must uphold, and the reasons behind them.

The mathematics is the design.
It enters as types, traits and laws, not as commentary on them.
Code should read the way a mathematician would write.

## Design goals

- **Unification over special-casing.**
  One general principle covers the many classical special cases.
  Never re-introduce them.
- **Arbitrary dimension, always.**
  Nothing is hardcoded to 2D or 3D.
  Dimension and grade are runtime values, the `Degree` newtype in `multiindex`.
  If you find yourself writing `if dim == 3`, the abstraction is wrong.
  `Degree` totalizes its arithmetic, checks bounds relationally against a supplied top degree,
  and a degree off $[0, n]$ denotes the trivial space instead of trapping.
  Public grade/dim APIs take `impl Into<Degree>`, so the signed logic stays sealed inside.
- **Total on the degenerate boundary.**
  The base dimension, the extremal grades, an empty skeleton, a one-element system:
  an edge case runs on the same code and returns the mathematically trivial answer.
  A base case that panics is a hidden `if dim == ...` the design never admitted to.

## Architecture

The README describes every crate. The layering is:

```
multiindex → multialgebra → metric → { regge, glatt } → derham → formoniq
```

with `coorder` foundational, `simplicial` beside `metric`
(the manifold `regge` adds a geometry to),
and `iterative` off to the side, below the mathematics, modeling no part of FEEC.

Dependencies flow strictly downward; a lower crate never learns about a higher one.
Two boundaries are invariants made structural, checkable in a manifest:
`multialgebra` depends on neither the metric nor a mesh (invariant 5),
`simplicial` on no metric (invariant 1).
A concept belongs in the lowest crate that can express it with the dependencies it already has;
needing a new downward dependency puts it in the joining crate.
What becomes extrinsic (I/O formats, anything embedding-dependent)
stays at the boundary of the crate owning the intrinsic object, never a crate of its own.
The lower crates are standalone mathematical objects,
documented for readers who have never heard of FEEC.
Composition reaches down from above: a free function in the joining crate by default,
a thin `...Ext` trait where method syntax carries the math better.

## The load-bearing invariants

These are the design, not preferences.
Breaking one is a bug even if it compiles and passes tests.

1. **Topology ⊥ Geometry.**
   The `Complex` is pure combinatorics; geometry is a separate input,
   consumed as `MeshLengthsSq`, reaching assembly as the per-cell `cell_metric`.
   There is deliberately no `Geometry` trait:
   the engine speaks one concrete intrinsic type, other representations convert into it.
2. **Intrinsic first, extrinsic second, and edge lengths are the primitive.**
   Signed squared edge lengths are total over every grade and every metric signature;
   coordinates and raw per-cell metrics are sources converting into them at the API boundary.
   Anything requiring an embedding stays out of the core path.
   Geometry is defined on every simplex (the Gramian of its own edges), a chart only on cells.
   A point is a `MeshPoint`, chart plus barycentric coordinates, never a global coordinate.
   The cells form an atlas, and every chart is the same chart up to vertex labelling:
   reference data is a function of dimension alone, element matrices computed once.
   A claim of chart-independence owes a `Transition` argument, applied as an operation.
3. **Coordinate spaces are type-level.**
   Barycentric, local cartesian and ambient coordinates are different spaces,
   tagged by `coorder::Coords`, and the maps between them carry their direction,
   so the wrong composition does not compile.
   A point is not a vector; the only combination of points is the affine one.
4. **Variance is per-slot, and stated rather than derived.**
   It is the one datum with no representational footprint,
   so nothing derives it: construction states it and the operations check it.
   Never choose between $g$ and $g^(-1)$ by hand; go through the measuring operations.
5. **Depend on the weakest structure that determines the concept.**
   The derivative, boundary, wedge and pairings need no metric;
   star, musicals and inner products do.
   Asking for less than determines the answer compiles and silently returns one of several.
6. **Orientation is a gauge inside the complex and a datum outside it.**
   No assembly, solve or homology may depend on a coherent orientation.
   Questions asked about the manifold as a whole
   (global volume form, a star whose result is compared between cells)
   take `Complex::orientation`; holding it is the proof of orientability,
   and code that cannot get one refuses rather than proceeding per cell.
7. **A generator's vertex ordering is data, and the mesh cannot recreate it.**
   `CellOrdering` carries it beside the complex, under the face-consistency law;
   nothing in assembly, solving or homology may consult an ordering.
   It exists so uniform refinement composes.
8. **Zero-cost abstractions.**
   Generics and monomorphization on the assembly hot path,
   rayon-parallel assembly the norm.

A precondition that is a property of a value becomes a type-level witness,
checked once where the witness is built (`coorder`'s spaces, `topology::role`),
never an assertion repeated at each call.

## Conventions

**Doc comments carry the math, in Typst notation** (`multialgebra/src/tensor.rs` canonical):
what the object is, the laws it obeys, the contracts the code cannot show.
Never narrate what the next line does.
A crate overview is its README pulled in verbatim, plain Unicode markdown there,
the sole place Unicode stands in for Typst.

**Tests are theorems**, and correctness is established here by the test suite.
Each field contributes its most famous theorems;
a law is swept over every axis it is stated over and trusted once made to fail.

**$Lambda$ and $"Sym"$ are one construction** under `Symmetry`,
every operation written once over all variants;
the Hodge star is the sole genuine exception.
Never a second implementation of either family.

**The stored basis is multiplicative, hence self-dual only on $Lambda^k$.**
Anything dualizing a symmetric slot goes through `Tensor::reciprocal`,
never components, never a factorial written by hand.
Integer structure constants make the algebra ring-generic;
only the dualizing operations ask for a `RationalAlgebra`.
A law that dualizes is swept over both families.

**Colexicographic order is the one indexing convention**, `Combination::rank()` canonical.
Combinations ($Lambda$), compositions ($"Sym"$) and permutations ($S_n$) are different objects.
The combinatorics is the library's own: the enumeration order is load-bearing,
defined and tested here, never inherited from a dependency,
and a backing width is machine data, parameterized inside `multiindex`, named by nothing above it.

**A `Tensor` is for the geometric space, never for the space of unknowns.**
Cochains, load vectors, element matrices and assembled operators are linear algebra.
And a tensorial computation runs through the operations of `multialgebra` and `metric`,
never on pulled-out components: component code is exact on $Lambda$
and silently off by $alpha!$ on $"Sym"$.

**A Kronecker product is never formed to be applied:** hold factors, apply slotwise.

**A constructor states its hypotheses, a predicate decides them**
(`new` unchecked, `is_valid` public, `new_checked` proving);
never under `cfg(debug_assertions)`.

**One datum, derived not stored.**
Caching is the exception and needs measurement to justify it.

**Linalg backends by role.**
nalgebra dense for element-local math, `nalgebra-sparse` for assembled operators
(aliased in `simplicial::linalg`) and behind `iterative`,
faer only in `formoniq`, for direct solves and eigenproblems.
No external solver toolchain.

**Assembled and matrix-free are peers; scatter vs gather is the other axis**,
the one deciding whether a race exists.

**A real problem's operator is real**, whatever field the unknowns live in.
Complex enters by extending scalars, never through assembly.
Built so, the shifted mass system is complex symmetric and not self-adjoint:
no symmetric Krylov method applies, the direct factorization is what solves it.

**Naming reflects the mathematics.**
Where a word has a precise meaning, it is used precisely,
and two words that mean different things never stand in for each other.

**Affine, flat, linear are three different claims** (maps, curvature, neither):
piecewise affine, piecewise flat, never "piecewise linear".
A chart maps the manifold out to coordinates, a parametrization maps coordinates in,
and the direction is the whole content of the words.
Mesh = simplicial complex = one object;
the simplicial manifold and the manifold it approximates are distinct.

**Rust style.**
Clean at default clippy, idiomatic and concise,
the iterator chain stating intent over the loop stating mechanics.

## Anti-goals

- No hacks.
  If a test fails, the mathematics or the abstraction is wrong; diagnose before fixing.
  A change that removes the symptom without explaining it is not a fix.
- No dimension- or grade-specific code paths in the core.
- No classical vector-calculus fallback.
- No embedding assumptions in the core path.
- No comments that restate the code.
- Nothing transient in this file:
  architecture, invariants, conventions and anti-goals only.

## Workflow

Every commit passes all four:

```sh
cargo fmt --all
cargo clippy --workspace --all-targets
cargo test --workspace
cargo doc --workspace --no-deps
```

CI runs the same four; a red build is a broken commit.
The examples are the end-to-end check and are run by hand.

Commit messages: `scope: imperative summary`,
one idea per commit where easily reached,
bundling fine when separating would be the more artificial move.
A change to the design updates this file in the same commit.
Where CLAUDE.md and the code disagree, one of them is a bug,
and it is usually worth asking which.
