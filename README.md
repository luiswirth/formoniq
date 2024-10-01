# formoniq

This library is being developed as part of the
[bachelor thesis](https://github.com/luiswirth/bsc-thesis)
of Luis Wirth at ETH Zurich.

Formoniq is a Rust implementation of the Finite Element Method (FEM) for solving
partial differential equations (PDEs) in arbitrarily high spatial dimensions,
using the theory of Exterior Calculus of Differential Forms. This particular
formulation of FEM is known as Finite Element Exterior Calculus (FEEC). It
provides a unified treatment of scalar- and vector-valued PDEs by treating general
differential k-forms as solutions.

Unlike traditional FEM approaches that rely on explicit coordinates, formoniq
adopts an intrinsic, coordinate-free framework for defining geometric properties
on simplicial meshes using the Regge metric. This intrinsic formulation allows for
generalizing finite element spaces like Lagrangian, Raviart-Thomas, and Nédélec
spaces into a single FE space of Whitney forms, extending naturally to higher
dimensions and enabling Hodge duality.

Our primary focus is solving elliptic Hodge-Laplace problems on simplicial
meshes, utilizing linear (first-order) Whitney forms. Rust was chosen for its
performance, safety, and concurrency features, making it an ideal tool for
high-performance computing tasks like finite elements.

# Features

- **Coordinate-Free Simplicial Complex Data Structure**: Topological information (incidence and adjacency) with a metric tensor for defining geometry on simplices.
- **Finite Element Spaces and Basis Functions**: Discretization using Whitney forms naturally defined on simplices.
- **Weak Formulation of Hodge-Laplace Operator**: Derived in an intrinsic, coordinate-free setting with primal and mixed formulations.
- **Element Matrices and Assembly**: Parallel assembly using Rust’s concurrency features, supporting high-dimensionality.
- **Testing and Validation**: Accuracy and performance testing across dimensions (e.g., 2D, 3D).

# Visualizations

The following plots and animations show finite element solutions of various PDEs computed using formoniq.

<img src="https://github.com/luiswirth/formoniq/assets/37505890/450e2cd0-ffeb-48ef-8b0a-64de5d75b557" width="33%">
<img src="https://github.com/user-attachments/assets/bcc88a07-907d-4eb4-b18c-3235a6f5b787" width="66%">

# Extensions and Future Work

- Higher-order Whitney forms
- Maxwell's equations on 4D spacetime (relativistic electrodynamics)
- Parametric finite element spaces for variable coefficient functions
- Hodge decomposition and Betti numbers
- Evolution problems (e.g., Heat and Wave equations)
