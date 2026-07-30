# coorder

The affine space as a type: points tagged by the space they live in, and the affine maps between them.

A point is not a vector.
There is no origin, no sum of two points and no scalar multiple of one.
What there is, is the difference of two points, which is a displacement,
a point displaced by a vector, which is a point again,
and the affine combination of points with weights summing to one.
That is the whole structure, and `Coords<S>` carries it.

Coordinates alone do not say which space they belong to:
the same tuple of numbers denotes different points in different coordinate systems.
So `Coords<S>` records the coordinate space `S` as a zero-cost type parameter,
the maps between spaces have to be written down,
and passing a point of one space where another is expected does not compile.
The tag is asserted where a raw coordinate vector enters,
and every operation afterwards preserves it.

## What it provides

- `CoordSpace`:
  the compile-time tag, an uninhabited type used only as a name.
  `Ambient` is the one space defined here: the ambient Rⁿ of an embedding.
- `Coords<S>`, generic over its storage, so an owned point
  and a `CoordsRef<'a, S>` view into the column of a matrix are one type.
  Read-only linear algebra is available by dereferencing to that storage.
- The affine structure:
  subtraction of points, the action of a displacement on a point,
  and `affine_combination`, of which `barycenter` is the uniformly weighted case.
- `affine::AffineTransform<From, To>`:
  the affine map x ↦ Ax + b, equivalently the maps preserving affine combinations.
  It carries its direction, so composition demands a shared middle space
  and `pseudo_inverse` returns the map the other way round.
  The linear part may be rectangular, a tall injective one being inverted on its image.
  Total in the degenerate zero-dimensional cases.

The laws in `tests/laws.rs` are what the crate is:
the displacements act freely and transitively on the points,
an affine combination is independent of which of its points is taken as base,
and an affine map is exactly one that commutes with those combinations.

## Place in the ecosystem

`coorder` is the coordinate layer of
[formoniq](https://github.com/luiswirth/formoniq),
a finite element exterior calculus (FEEC) engine,
where barycentric, chart-local and ambient coordinates
are three different spaces that must not be confused,
and where a cell's parametrization and its chart are one map and its inverse.
The crate knows nothing of meshes, manifolds or metrics.
It is flat affine geometry, usable wherever multiple coordinate frames coexist.

## License

Dual-licensed under either MIT or Apache-2.0, at your option.
