# Built-in gallery meshes

Curated surface meshes the studio ships as a built-in eigenmode gallery, chosen
to span topology: genus 0 and genus 1. All are closed and orientable. The
procedural sphere and grid the studio also offers cover another genus-0 surface
and a surface with boundary. Genus below is measured from the mesh
($chi = 2 - 2g$), not assumed.

All three are from Keenan Crane's 3D Model Repository
(https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/) and are released
under the CC0 1.0 Universal Public Domain Dedication. CC0 requires no
attribution; this note records provenance anyway.

| file        | model              | genus | vertices | notes |
| ----------- | ------------------ | ----- | -------- | ----- |
| spot.obj    | Spot (cow)         | 0     | 2930     | `spot_triangulated` |
| bob.obj     | Bob                | 1     | 3087     | `bob_isotropic`; a genus-1 blob |
| blub.obj    | Blub (fish)        | 0     | 7106     | `blub_triangulated`; the heaviest solve |

Tracked with git LFS (see `.gitattributes`); run `git lfs pull` to fetch the
actual mesh data.

## Test fixtures

Not part of the picker gallery above -- meshes kept small on purpose, embedded
for a specific unit test rather than curated for viewing.

| file        | is                                  | notes |
| ----------- | ----------------------------------- | ----- |
| torus0.msh  | genus-1 surface (Gmsh 4.1)          | 127 vertices, 254 triangles; the boundary surface Gmsh meshes by default for an OpenCASCADE `Torus(1)` primitive, coarsest of a refinement family. $b_1 = 2$ -- used by `scene::tests::hodge_decomposition_splits_orthogonally` as the fast genus-bearing case in place of Bob, whose harmonic solve at ~3k vertices dominates the whole workspace test suite's runtime. |
