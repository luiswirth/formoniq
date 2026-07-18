# coorder

Typed affine coordinates: points tagged by the space they live in.

The same tuple of numbers denotes different points in different coordinate
systems. `Coords<S>` records the coordinate space `S` as a zero-cost type
parameter, so a value carries the space it belongs to: the maps between spaces
have to be written down, and passing a point of one space where another is
expected does not compile. The tag is checked once, where the value is built.

## What it provides

- `CoordSpace`: the compile-time tag, an uninhabited type used only as a name.
  `Ambient` is the one space defined here: the ambient Rⁿ of an embedding.
- `Coords<S>` and the borrowed `CoordsRef<S>`: a point of `S` as its coordinate
  tuple, dereferencing to plain linear algebra for read-only use. The only
  arithmetic between points is subtraction, which returns an untagged vector:
  the difference of two points is a displacement, not a point.
- Converting a raw vector into a tagged point is unchecked: the one place the
  tag is trusted rather than enforced. Every other path preserves tags.
- `affine::AffineTransform`: the affine map x ↦ Ax + b between coordinate spaces,
  with forward application, least-squares backward solution, and the
  Moore-Penrose pseudo-inverse, total in the degenerate zero-dimensional cases.

## Place in the ecosystem

`coorder` is the coordinate layer of
[formoniq](https://github.com/luiswirth/formoniq), a finite element exterior
calculus (FEEC) engine, where barycentric, chart-local and ambient coordinates
are three different spaces that must not be confused. The crate knows nothing of
meshes or manifolds; it is generic tagging machinery, usable wherever multiple
coordinate frames coexist.

## License

Dual-licensed under either MIT or Apache-2.0, at your option.
