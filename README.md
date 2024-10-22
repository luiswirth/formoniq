# formoniq

This library is being developed as part of the
[Bachelor's thesis](https://github.com/luiswirth/bsc-thesis)
of Luis Wirth at ETH Zurich under the supervision of Prof. Dr. Ralf Hiptmair.

Formoniq is a Rust Implementation of a Finite Element (FEM) Library based on
the principles of Finite Element Extieror Calculus (FEEC) to solve partial
differential equations (PDEs) formulated in terms of differential forms over
simplicial manifolds using an intrinsic, coordinate-free approach.

The focus is on solving elliptic Hodge-Laplace problems with the
piecewiese-linear (first-order) Whitney basis.

## Background

Finite Element Exterior Calculus (FEEC) provides a unified framework that
extends the finite element method using the language of differential geometry
and algebraic topology. By employing differential forms and (co-)chain
complexes, FEEC offers a robust approach for preserving key topological and
structural features in the solution of PDEs. This framework is particularly
well-suited for problems such as the Hodge-Laplace equation and Maxwell’s
equations.

Traditional finite element methods rely on explicit coordinate representations
of the computational domain. However, a coordinate-free formulation aligns more
naturally with the intrinsic nature of differential geometry. By representing
the computational domain as a simplicial manifold with an associated Riemannian
metric, we can define geometric quantities (such as lengths, areas, and volumes)
intrinsically, without explicit coordinates. This metric is an inner product
on the tangent spaces and defines operators like the Hodge star, which are
essential in the formulation of the Hodge-Laplace operator.

Rust was chosen for its strong guarantees in memory safety, performance, and
modern language features, making it ideal for high-performance computing tasks
like finite elements. The Rust ownership model, borrow checker, and type system
act as a proof system to ensure there are no memory bugs, race conditions, or
similar undefined behaviors in any program, while achieving performance levels
comparable to C/C++.
