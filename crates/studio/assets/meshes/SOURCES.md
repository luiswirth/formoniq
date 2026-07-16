# Built-in gallery meshes

Curated surface meshes the studio ships as a built-in eigenmode gallery, chosen
to span topology: genus 0, genus 1, and (via csaszar) a minimal triangulation.
All are closed and orientable. The procedural sphere and grid the studio also
offers cover another genus-0 surface and a surface with boundary. Genus below is
measured from the mesh ($chi = 2 - 2g$), not assumed.

All four are from Keenan Crane's 3D Model Repository
(https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/) and are released
under the CC0 1.0 Universal Public Domain Dedication. CC0 requires no
attribution; this note records provenance anyway.

| file        | model              | genus | vertices | notes |
| ----------- | ------------------ | ----- | -------- | ----- |
| spot.obj    | Spot (cow)         | 0     | 2930     | `spot_triangulated` |
| bob.obj     | Bob                | 1     | 3087     | `bob_isotropic`; a genus-1 blob |
| blub.obj    | Blub (fish)        | 0     | 7106     | `blub_triangulated`; the heaviest solve |
| csaszar.obj | Császár polyhedron | 1     | 7        | oriented; a diagonal-free 7-vertex triangulation of the torus (Császár 1949) |

Tracked with git LFS (see `.gitattributes`); run `git lfs pull` to fetch the
actual mesh data.
