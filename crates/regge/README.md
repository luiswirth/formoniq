# regge

Regge geometry: the geometry carried by a simplicial manifold.

The manifold itself is pure combinatorics and piecewise-affine structure —
incidence, orientation, boundary, homology, charts — none of which is metric,
and it lives in [simplicial](https://crates.io/crates/simplicial). A geometry is
a genuinely second input, and this is where it enters: a metric on each
piecewise-flat cell, and with it lengths, volumes and curvature. The two are
separate because the separation is mathematical, not a matter of taste: the
boundary operator, the exterior derivative and the Betti numbers are metric-free
facts, and no amount of combinatorics derives a metric.

The primitive is `MeshLengthsSq`, one signed squared length per edge. Signed
because Regge calculus was invented for Lorentzian spacetimes and an unsquared
length would lose the causal character: positive spacelike, zero null, negative
timelike, mirroring `norm_sq`. It is the source of truth and the one
representation total over every grade — the metric of any subsimplex, an edge's
length, a facet's area, a hinge's metric, is the Gramian of that simplex's own
edges, well defined with no containing cell consulted.

`MeshCoords` (an embedding) and `CellGramians` (raw per-cell metrics) are
*sources*, not currencies: each converts to edge lengths at the boundary of the
API, on equal footing precisely because they reduce to the same primitive.
Anything that requires an embedding is for I/O, visualization or convenience,
never the core path.

The manifold is piecewise flat: curvature vanishes on cell interiors and
concentrates on the codimension-2 hinges, which is what makes a simplicial
manifold with edge lengths a Regge manifold rather than an approximation of one.

## What it provides

- `metric`:
  `MeshLengthsSq` and the per-simplex metrics it derives — `cell_metric`,
  `simplex_metric`, `simplex_volume` — plus `CellGramians` as a source and as
  the materialized cell column a refinement pulls back.
- `coord`:
  `MeshCoords`, the extrinsic realization, and the bridges that induce a metric
  and edge lengths from it. Point location lives here too, an embedding being
  what makes it meaningful.
- `refine`:
  the geometric half of uniform refinement, keyed off `simplicial`'s
  `Subdivision`.
- `mesher`:
  generators — Cartesian/Kuhn grids, quotient tori, sphere surfaces — each
  producing a complex and the coordinates that go with it.
- `io`:
  mesh formats, which are a topology and its coordinates together.
- `vertex_gaussian_curvature`:
  the angle defect, exact rather than approximate — Gauss-Bonnet holds as an
  identity, with no refinement limit to converge under.
