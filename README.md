# formoniq

This library is being developed as part of the
[bachelor thesis](https://github.com/luiswirth/bsc-thesis)
of Luis Wirth at ETH Zurich.

Formoniq is a Rust implementation of the Finite Element Method (FEM) for solving
partial differential equations (PDEs) in arbitrarily high spatial dimensions,
using the theory of Exterior Calculus of Differential Forms.

This particular formulation of FEM is known as Finite Element Exterior Calculus
(FEEC). It allows for a unified treatment of scalar- and vector-valued PDEs by
treating general differential k-forms as solutions.

This framework unifies multiple Finite Element Spaces (Lagrangian,
Raviart-Thomas, and Nédélec) into one: the FE space of Whitney forms, by
identifying scalar and vector fields through Hodge duality. Additionally, it
extends these spaces from 3D to higher dimensions.

Our primary focus is solving elliptic Hodge-Laplace problems on simplicial
meshes with first-order Whitney forms.

# Example

The following animations shows finite element solutions of various PDEs computed using formoniq.

<img src="https://github.com/luiswirth/formoniq/assets/37505890/450e2cd0-ffeb-48ef-8b0a-64de5d75b557" width="33%">
<img src="https://github.com/luiswirth/formoniq/assets/37505890/1f7116a2-5a44-43a4-a6a5-e5b645e0f9d2" width="66%">


