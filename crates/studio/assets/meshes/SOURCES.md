# Built-in meshes

Surface meshes the studio ships, chosen to span topology: genus 0 and genus 1.
All are closed and orientable. The procedural sphere and grid the studio also
offers cover another genus-0 surface and a surface with boundary. Genus below is
measured from the mesh ($chi = 2 - 2g$), not assumed.

**This directory is the mesh list.** `build.rs` enumerates it and generates the
table the picker and the CLI read, so a mesh dropped in here is selectable with
no code to change, and there is nothing that can fall out of step with what
ships. A file's extension chooses its reader (`.obj`, `.msh`), and its stem is
the name the CLI takes and the label the picker shows, capitalized -- so
renaming a mesh means renaming its file. Anything with another extension (this
note) is not a mesh and is skipped.

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
actual mesh data. An unfetched asset is the LFS pointer text, which the readers
report as an empty or malformed mesh rather than silently mis-loading.

## Others

| file        | is                         | notes |
| ----------- | -------------------------- | ----- |
| torus0.msh  | genus-1 surface (Gmsh 4.1) | 127 vertices, 254 triangles; the boundary surface Gmsh meshes by default for an OpenCASCADE `Torus(1)` primitive, coarsest of a refinement family. Selectable like any other, and small enough that every solve on it is instant -- which is also why it stands in for Bob wherever a test needs $b_1 = 2$ without a multi-thousand-vertex harmonic solve (`scene::tests::hodge_decomposition_splits_orthogonally`, `gallery::tests::the_hodge_preset_opens_on_the_harmonic_shell`). |
