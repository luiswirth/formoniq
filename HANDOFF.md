We are doing a simplification pass over the formoniq workspace, crate by crate, bottom-up along the dependency ladder.

What "simplification" means here. Primarily: re-derive each layer and find the abstraction that dissolves the special cases, then let the code shrink as a consequence. Unification through mathematical and software logic. Everything is in scope — structure, framing, abstractions, APIs, and the mathematical modelling itself. If the current shape fights you, restructure it. If two things are secretly one thing, make them one thing. If a doc comment and the code disagree, one of them is a bug; say which. Ordinary tidying along the way is welcome, it just isn't what the pass is for.

The kinds of finding that have paid off so far. A type declared as a newtype over another while reimplementing all of its algorithms. One operation implemented three times in two conflicting conventions, with the compensation hidden in the tests. Tests that exist only to assert two implementations of one thing agree — those are migration scaffolding, and they go with the duplication. A framing claim in one module contradicting the one in another. A stale factual claim in a public README, including a README naming a module the crate does not have. A test hardcoding a layout instead of reading it off the object. A layout convention stated in three places and obeyed in two, where the outlier's own test asserted the wrong one as the convention. A derived datum stored on a hot type, justified by a claim that doesn't hold. A public normalization constant with no test that pins it. A concept from a higher layer living in a lower crate, with a second open-coded copy in the crate it belongs to. Laws that pass because both sides read the same wrong weight — the fix is a law stated against something external. An assertion whose message names a stronger property than the one it checks. A guard at a call site compensating for a container that isn't total. A type named for one of its three uses. Two types that are one datum at two scopes, each spelling out the same per-entry accessors. An extension trait that could only assume the Euclidean case on a crate whose whole point is arbitrary signature.

How to work. Work autonomously. Do all straightforward and low-risk things immediately, without asking. Stop and ask only when you are genuinely unsure or the change is genuinely risky — renumbering a basis or an index convention is explicitly allowed, so say what it renumbers and then do it. Look further up the ladder freely to see how something is used or why it is shaped as it is; a change starting in a low crate often has to land in the crates above it in the same commit to stay honest.

Keep the user in the loop by saying out loud what you are working on. Very concise messages. Not reports.

Two standing rules from the user.
1. An unused public item is not dead code. Every crate below `derham` is a standalone published library and is judged as one. The question is always whether the API is elegant for someone who has never heard of FEEC, never whether `formoniq` happens to call it. The exception is an item that is not merely unused but *wrong* for the crate's own stated generality — that is a finding, not dead code, and it goes.
2. The repo captures the current state, never the story of how it got there. No leftovers in code, docs or comments narrating what something used to be, what a previous design could not do, what a migration licenses, or what changed. Commit messages are where history lives. Write every doc comment as though the current design were the only one there had ever been.

State. `multiindex`, `multialgebra`, `metric`, `coorder` and `simplicial` are done, on `main`. `regge` has had one pass (six commits, ending `regge: one reading of a signed squared length`) but was **not finished** — start by completing it, then go up: `glatt`, `derham`, `iterative`, `formoniq`, `realize`, `studio`.

What the `regge` pass already did, so you don't redo it.
- `LengthsSq` in `lengths.rs`: `MeshLengthsSq` and `SimplexLengthsSq` are one column at two scopes, so `nedges`/`length_sq`/`length`/`causal_type`/`iter`/`max_length`/`min_length` are written once. Each type keeps only the names that carry something at its scope (`diameter`, `mesh_width_*`).
- Removed: `simplex_lengths_sq_of` (a second name for `SimplexLengthsSq::from_metric`), `MetricComplex`/`unit_metric_complex` (a tuple alias nothing built), and `SimplexCoordsExt` (`metric_tensor`/`to_lengths_sq` could only assume a Euclidean ambient; `SimplexCoords` carries no inner product, `MeshCoords` does, and `MeshCoords::simplex_metric`/`to_edge_lengths_sq` are the real bridges).
- Three `if topology.dim() == 0` guards removed; two compensated for `Complex::edges` being the partial accessor where `skeleton(1)` is total.
- README: named a `metric` module that is `lengths`, omitted `subcomplex`.

Where to pick `regge` back up. These were seen and not acted on:
- `coord/mesh.rs` and `lengths/simplex.rs` still carry `is_degenerate` with a hardcoded `1e-12` in two places, and `SimplexCoords::is_degenerate` in `simplicial` is a third. No test pins the threshold.
- `MeshCoords::coord_iter_mut` returns a fully spelled-out `na::iter::ColumnIterMut<...>`; the two `iter()` methods that did the same were folded into `LengthsSq`, this one was not.
- `lengths/simplex.rs` has both a free `edge_index(vi, vj)` and `nedges(dim)` from `simplicial`, alongside `SimplexLengthsSq::nedges()`. Worth asking whether the local edge indexing has one home.
- `mesher/quotient.rs` is 694 lines and `mesher/quotient_embed.rs` 423, the two largest files in the crate; neither was read closely.
- `coord/locate.rs` (333 lines) and `io/gmsh.rs` were not read at all.
- `regge` still takes a `Dim` (not `impl Into<Dim>`) in several public signatures; `simplicial` was swept for this and `regge` was not. CLAUDE.md states the convention.

Three things noted by earlier agents and deliberately not acted on, for whoever reaches them. `Metric::new_unchecked` falls back to the checked constructor under `debug_assertions` and `on_slot` builds a `Metric` per slot, so a debug build does a symmetric eigendecomposition per `inner()` call — deliberate, but it makes the debug test suite much slower than it looks. And three doc comments upstream still narrate history: `formoniq/src/time.rs:486`, `studio/src/scene.rs:515`, `studio/src/render/camera.rs:433`.

Start by finishing the `regge` inventory above, then move up the ladder.
