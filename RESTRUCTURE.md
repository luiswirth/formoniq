# Restructuring

A task, not doctrine.
The architecture this asks for is in `CLAUDE.md`, which is where the lasting statements live.
This file is the pending work of making the code match them, and it is deleted once that is done.

## The task

The crate identities in this workspace were worked out deliberately and are now written down:
`CLAUDE.md`'s Architecture section states what each crate *is*, one sentence each,
and sorts them into three tiers
(the mathematical objects, the engine, and the crates that model no part of FEEC).
Those statements are the authority.
The code does not yet fully match them.

Make it match.

Sweep the workspace crate by crate, bottom to top.
For every public module and item, ask one question:
is this part of this crate's one-sentence answer?
If it is not, work out where it belongs and move it.

Do not trust any prior list of candidates,
including one found in a commit message or an issue.
Rederive them, because the sweep is the point,
and a list is only someone's earlier pass.

## What decides a placement

Two tests, and their order matters.

**Depend on the weakest structure that determines the concept.**
This says where a thing *may* live.
Something sitting a crate too high because the lower one happened to lack a dependency
is an accident of a manifest, not a fact about mathematics, and it should come down,
taking the dependency with it if the concept is genuinely *of* the lower object.

**Each lower crate is a coherent standalone library, published as such,
that earns its keep for a reader who has never heard of FEEC.**
This says where a thing *should* live.
Where the two tests disagree, this one wins:
a concept can be expressible in a crate and still have no business being in it.

## What to report

The near-misses are where an identity actually gets decided, so report both halves:
what was moved, and what was considered and deliberately left alone,
with the reason in each case.

If a crate's stated identity turns out to be wrong or too narrow once it is pushed on,
say so and propose the correction,
rather than bending the code around a sentence.
The document and the code disagreeing means one of them is a bug,
and it is worth knowing which.

## Constraints

Dependencies flow strictly downward.
The invariants in `CLAUDE.md` bind, and a move that breaks one is a bug even if it compiles.
The crates that are not core stay unreachable from the core path.

Every move carries its laws with it,
and the tests of a metric-free crate stay metric-free and coordinate-free.

A move is not finished until the crate's `README.md`, its `Cargo.toml` description
and `CLAUDE.md` reflect it, in the same commit.
Structure the commits one idea each,
and every one of them passes `fmt`, `clippy`, `test` and `doc` across the workspace.
