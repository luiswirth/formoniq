# studio refactor plan

Restructure `formoniq-studio` from a monolithic `lib.rs` into a layered
viewer, generalize the render geometry beyond triangle surfaces, and add a
headless (PNG/MP4) render path that shares one frame graph with the
interactive viewer.

## Current state

- `src/lib.rs` (~3000 lines) interleaves six concerns: the gallery model
  (`MeshSource`, `View`, `Gallery`, `PendingLoad`), egui panel widgets
  (`degeneracy_shells`, `pyramid`, `grade_grid`), GPU resource factories,
  uniform structs, a `State` that hand-builds six pipelines, the frame graph,
  and the winit `App`.
- The seam out is dimension-bound: `render/mesh.rs::surface_geometry` asserts
  `topology.dim() == 2`. The intrinsic layer (`scene.rs`, `streamline.rs`) is
  already dimension-agnostic; only the bake and renderer are triangle-only.
- Two parallel bakes exist: `TriangleSurface3D` (io) and `surface_geometry`
  (render), sharing helpers but not a type.
- Two segment renderers exist: the wireframe pass and the streamline ribbon
  pass are the same technique (instanced screen-facing quads, paired
  per-instance endpoint buffers) duplicated in Rust and WGSL.
- Rendering is welded to the window: `State` owns the surface, egui, and an
  `Instant`; there is no offscreen path. `std::time::Instant` and
  `pollster::block_on` in `resumed` both panic on wasm32, so the wasm target
  is currently broken despite its scaffolding.
- Boilerplate causes: no uniform-binding helper, no pass abstraction. Every
  uniform is buffer + layout + bind group + `_pad` fields by hand.

## Target architecture

```
src/
  lib.rs          crate doc, module decls, run()  (thin)
  app.rs          windowed wrapper: winit, surface, egui, input
  gallery.rs      MeshSource, BuiltinMesh, View, Gallery, PendingLoad  (GPU-free)
  scene.rs        Scene, ScalarField, LineField, reduced_form  (seam in)
  demos.rs        Scene constructors: whitney galleries, eigenmodes, triforce
  bake.rs         BakedMesh, orient/normals, centroid/extent, R^3 embed
  streamline.rs   intrinsic tracer (unchanged)
  export.rs       offscreen target, readback, PNG encode, ffmpeg pipe, panels
  ui/panel.rs     pure panel function -> PanelResponse
  render/
    context.rs    GpuContext { device, queue } -- with or without a surface
    renderer.rs   Renderer: pipelines, intermediates, frame graph
    item.rs       RenderItem, DrawList, GPU batch buffers
    uniform.rs    UniformBinding<T: Pod>
    fill.rs       triangle fill pass
    segments.rs   one segment pass (wireframe, streamlines, 1-manifold cells)
    lic.rs        G-buffer + LIC passes, noise texture
    downsample.rs SSAA resolve
    camera.rs     (unchanged)
    *.wgsl
  io/             obj, blender (unchanged shape)
```

Deleted: `ui/mod.rs` stub, `examples/_check_spectrum.rs`, `mesh3d.rs`
(contents absorbed by `bake.rs` and `demos.rs`), `TriangleSurface3D` as a
public seam (io keeps whatever thin intermediate it needs).

### The bake: `BakedMesh`

The GPU rasterizes simplices of dimension <= 2 embedded in $RR^3$; the bake
states exactly that. Complexes are pure manifolds, so the bake of one complex
is one *cell batch* at primitive dimension $min(d, 2)$ plus *derived
overlays* -- the submesh pattern: one shared vertex table, several index
streams.

```rust
pub struct BakedMesh {
  positions: ...,           // static per-vertex: position, normal,
                            // max_displacement (f32, R^3)
  attributes: ...,          // per-field: value, direction -- swappable
                            // without rebaking
  cells: PrimBatch,         // Triangles | Segments | Points, by min(d, 2)
  wireframe: Vec<[u32; 2]>, // derived 1-skeleton overlay
}
```

- Dimension dispatch lives only here: 2-cells bake to wound triangles,
  1-cells to segments, 0-cells to points, 3-cells to their boundary
  2-skeleton (slicing and volume marks are future extensions of the bake,
  not branches around it). This mirrors the grade side: grade reduces to a
  mark via $min(k, n-k)$; intrinsic dimension reduces to a render primitive
  via the bake.
- One `BakedMesh` per complex, never merged across complexes. Cross-mesh
  merging is batching -- a draw-call optimization, not an abstraction -- and
  it destroys per-object identity (visibility, bounds, colormap, picking).
- The static/per-field split means switching eigenmodes rewrites only the
  attribute stream, not positions, normals, curvature, or winding.

### The draw list

The renderer consumes `Vec<RenderItem>`; an item is a batch plus pipeline
kind plus material parameters. A surface, its wireframe overlay, and its
traced streamline ribbons are three items (streamlines are already a
renderable that is not a complex). Multiple manifolds in one scene are just
more items.

### The renderer/window split

`Renderer` owns pipelines and transient targets (SSAA color, depth,
G-buffer; resized on demand) and records the frame graph into any caller
`TextureView`:

```rust
Renderer::new(&GpuContext, target_format) -> Renderer
Renderer::render(&mut self, target: &TextureView, size, items: &DrawList,
                 camera: &Camera, time: f32)
```

- **Time is an input, not state.** The windowed loop passes wall-clock time;
  the exporter passes $t_k = k / "fps"$. Deterministic frames, and the wasm
  `Instant` problem disappears from the render layer.
- **egui stays outside.** The windowed wrapper (`app.rs`) owns surface,
  winit, egui, input, and composites the panel after the renderer's output.
- **Format and size are constructor/call parameters**, so the swapchain
  format and an offscreen `Rgba8UnormSrgb` export target use the same
  pipelines code.

One frame-graph implementation, two callers; they cannot drift.

### Headless export

`export.rs`: offscreen texture with `COPY_SRC`, row-alignment-padded buffer
readback, PNG via the `image` crate. MP4 by piping raw frames to an `ffmpeg`
subprocess when present (clear error otherwise) -- no vendored encoder. An
eigenmode is periodic with period $2 pi \/ omega$, so exporting exactly one
period yields a seamlessly looping clip.

CLI in `main.rs`:

```
studio                            # interactive (default)
studio export out.png --view ... --field ... --size WxH --supersample N
studio export out.mp4 ... --frames N --fps N
studio export out.png --panels a,b,c    # shared camera, composited CPU-side
```

reusing `View`/`Gallery` scene construction and `Selection` verbatim.
Export resolution is independent of any window.

### Boilerplate removal

- `UniformBinding<T: Pod>`: owns buffer, layout, bind group; `write(&queue,
  value)`. Replaces every hand-rolled uniform triple.
- Each pass is a struct: `new(device, ...)`, optional `resize`, and
  `record(&mut encoder, ...)`. `Renderer::render` becomes a legible sequence
  of pass calls.
- WGSL: shared preamble (camera/wave structs, displacement, colormaps)
  concatenated with each shader body at `include_str!` time; the naga shader
  test validates the same concatenation. Cross-language constants
  (`SSAA_SCALE`) become WGSL `override` pipeline constants set from Rust.

## Execution steps

Each step is one or more commits, each green under `cargo fmt --all`,
`cargo clippy --workspace --all-targets`, `cargo test --workspace`,
`cargo doc --workspace --no-deps`.

1. **Relocation.** Extract `gallery.rs`, `ui/panel.rs`, `app.rs`, `demos.rs`
   from `lib.rs`/`scene.rs`/`mesh3d.rs`. Pure movement, no behavior change.
   Delete the `ui/mod.rs` stub and `examples/_check_spectrum.rs`. The panel
   becomes `fn panel(ctx, &PanelModel) -> PanelResponse` (pure function of
   the model, returns requested changes), replacing the copy-locals-and-diff
   dance in `run_ui`.
2. **Renderer split.** Introduce `UniformBinding` and the pass structs;
   dissolve `State` into `Renderer` (target-view-agnostic,
   format-parameterized, time as input) plus the windowed wrapper.
   Deduplicate WGSL through the shared preamble and `override` constants.
3. **BakedMesh + draw list.** Dimension-dispatched bake with the shared
   vertex table; one segment pipeline replacing the wireframe/streamline
   twins; delete the `dim == 2` assert and the duplicate bake. Law tests:
   bake sweep over $d in 1..=3$; winding consistency (every shared edge
   traversed in opposite directions by its two triangles);
   `Scene::whitney_basis(d)` sweep. Update `CLAUDE.md` in the same commit
   (see below).
4. **Export.** `export.rs` + CLI: PNG, then ffmpeg pipe, then `--panels`.
   Headless smoke test: render a known view, assert readback is non-uniform
   (skip when no adapter is available).
5. **Quality pass.**
   - Colormaps: diverging (zero-centered) for signed fields, perceptually
     uniform sequential (viridis fit) for magnitudes; replace the sine
     rainbow.
   - Camera: zoom clamp and znear/zfar as fractions of scene extent, not
     absolute constants (a loaded OBJ has arbitrary scale).
   - wasm: `web-time` and async init without `block_on`, or remove the wasm
     scaffolding until it is real.
   - `scene.rs::field`: reduced grade >= 2 files into no list (no mark yet)
     instead of `todo!` panicking the viewer.
   - `io/blender.rs`: return `io::Result` instead of unwrapping.

Order rationale: 2 must precede 4 (export needs the split); 3 precedes 4 so
exports exercise the final geometry path; 5 last because everything before
it is structure-preserving.

## CLAUDE.md amendments (with step 3)

- Restate the seam out as the concept, not a struct name: simplices of
  dimension <= 2 embedded in $RR^3$, winding and embedding fixed; no FEEC
  types downstream of the bake.
- Add the dimension-reduction symmetry: grade reduces to a mark, intrinsic
  dimension reduces to a render primitive; both case distinctions confined
  to their reduction. One segment pipeline serves the wireframe, the
  streamlines, and a 1-manifold's cells.
- Add the viewer's internal layering as invariants: the model (gallery,
  scene) is GPU-free; the renderer sees only baked geometry and explicit
  time, never FEEC types or a clock; the UI is a pure function of the model
  returning requests; interactive and headless rendering share one frame
  graph.
