We are doing a simplification pass over the formoniq workspace, crate by crate, bottom-up along the dependency ladder.

What "simplification" means here. Primarily: re-derive each layer and find the abstraction that dissolves the special cases, then let the code shrink as a consequence. Unification through mathematical and software logic. Everything is in scope — structure, framing, abstractions, APIs, and the mathematical modelling itself. If the current shape fights you, restructure it. If two things are secretly one thing, make them one thing. If a doc comment and the code disagree, one of them is a bug; say which. Ordinary tidying along the way is welcome, it just isn't what the pass is for.

The kinds of finding that have paid off so far. A type declared as a newtype over another while reimplementing all of its algorithms. One operation implemented three times in two conflicting conventions, with the compensation hidden in the tests. Tests that exist only to assert two implementations of one thing agree — those are migration scaffolding, and they go with the duplication. A framing claim in one module contradicting the one in another. A stale factual claim in a public README, including a README naming a module the crate does not have. A test hardcoding a layout instead of reading it off the object. A layout convention stated in three places and obeyed in two, where the outlier's own test asserted the wrong one as the convention. A derived datum stored on a hot type, justified by a claim that doesn't hold. A public normalization constant with no test that pins it. A concept from a higher layer living in a lower crate, with a second open-coded copy in the crate it belongs to. Laws that pass because both sides read the same wrong weight — the fix is a law stated against something external. An assertion whose message names a stronger property than the one it checks. A guard at a call site compensating for a container that isn't total. A type named for one of its three uses. Two types that are one datum at two scopes, each spelling out the same per-entry accessors. An extension trait that could only assume the Euclidean case on a crate whose whole point is arbitrary signature.

How to work. Work autonomously. Do all straightforward and low-risk things immediately, without asking. Stop and ask only when you are genuinely unsure or the change is genuinely risky — renumbering a basis or an index convention is explicitly allowed, so say what it renumbers and then do it. Look further up the ladder freely to see how something is used or why it is shaped as it is; a change starting in a low crate often has to land in the crates above it in the same commit to stay honest.

Keep the user in the loop by saying out loud what you are working on. Very concise messages. Not reports.

Two standing rules from the user.
1. An unused public item is not dead code. Every crate below `derham` is a standalone published library and is judged as one. The question is always whether the API is elegant for someone who has never heard of FEEC, never whether `formoniq` happens to call it. The exception is an item that is not merely unused but *wrong* for the crate's own stated generality — that is a finding, not dead code, and it goes.
2. The repo captures the current state, never the story of how it got there. No leftovers in code, docs or comments narrating what something used to be, what a previous design could not do, what a migration licenses, or what changed. Commit messages are where history lives. Write every doc comment as though the current design were the only one there had ever been.

State. The pass has walked the ladder bottom-up and is through `formoniq`. Done, on `main`: `multiindex`, `multialgebra`, `metric`, `coorder`, `simplicial`, `regge`, `glatt`, `derham`, `iterative`, `formoniq`. Remaining, in order: `realize`, `studio`. The scope prefix of a commit says which crate it belongs to, so `git log --oneline` is the record; this file carries only what the log cannot.

What the `formoniq` pass did, so you don't redo it.
- Both `HilbertComplex` implementations spelled out their operators inherently and forwarded each through the trait. The bodies live in the trait impls now, and the norm family, which uses nothing but trait operations, moved there as default methods, so the relative complex has it too.
- `MixedGalMats` and `HodgeBlocks` were one datum in two types; `HodgeBlocks` survives, in its own `hodge` module rather than inside `problems::elliptic`, which every other problem had been reaching into. Its degenerate-grade branches are gone: the complex is total in grade, so the general expressions already produce the empty blocks.
- The three mixed element matrices are $R^top M_k C$ with each side the identity or the coboundary, hence `WhitneyPairElmat`'s three constructors.
- The HX preconditioner's structure was written once per way of inverting its blocks; the structure is stated once and the inverses are the two `AuxiliaryBlocks` impls.
- Every SPD direct solve goes through `DirectInverse`, which verifies faer's Cholesky against a probe. Three production solves had been calling `FaerCholesky` raw, unguarded against the silent-inaccuracy failure that wrapper exists for.
- Deliberately left: `multigrid::Grade0Multigrid` names a grade in its type. The restriction is real, a plain V-cycle is only effective at grade 0, and generalizing it would put a knowingly-poor solver in the public surface. The tower under it is already grade-general.
- Measured and reverted: routing `assemble` through a materialized `FaceIncidence`, the way `matfree` does, costs about 2x. `assemble.rs` carries the reason.

What the `iterative` pass did, so you don't redo it.
- The stationary step $x <- x + B(b - A x)$ was written three times (the standalone solve, the `Stationary` preconditioner, the V-cycle's own `smooth`). It is now `stationary::sweeps`, continuing from the incoming iterate, which is what post-smoothing needs and what made the multigrid copy look like a different operation.
- A zero right-hand side was answered three ways, one of them by clamping the norm to `MIN_POSITIVE`. `trivial_solve` answers it once; the clamp is gone.
- `VCycle` carries one sweep count and `Level` forms its own restriction as the transpose of its prolongation, so the `SelfAdjoint` marker rests on the markers of its parts and on nothing a caller must remember.
- Tests moved out of the flat `mod tests` in `lib.rs` into the modules whose objects they state laws about; shared fixtures are `testutil`, which the `aux_space` tests had been duplicating.
- Deliberately left: `Level` stores `restrict` beside `prolong`. It is derived, but re-transposing a CSR per level per Krylov step is real work, and forming it in the constructor is what makes the adjointness structural.

Where to start on `realize`. It is the first of the two crates outside the core: intrinsic data made extrinsic, the grade and dimension reductions, the render primitives and the file formats. `studio` follows and has its own `CLAUDE.md`. Neither has been surveyed for the pass yet.

Carried notes, upstream of where the pass now is.
- `Metric::new_unchecked` falls back to the checked constructor under `debug_assertions` and `on_slot` builds a `Metric` per slot, so a debug build does a symmetric eigendecomposition per `inner()` call. Deliberate, but it makes the debug test suite much slower than it looks.
- Doc comments still narrating history: `studio/src/scene.rs:515`, `studio/src/render/camera.rs:433`.
- `regge/src/io/gmsh.rs` was never read closely during the `regge` pass.
- The examples under `crates/formoniq/examples/` are run by hand, not by `cargo test`, so a change reaching them has to be run. `source` is the slow one, tens of minutes.
