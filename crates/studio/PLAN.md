# studio refactor plan

Restructure `formoniq-studio` from a monolithic `lib.rs` into a layered
viewer, generalize the render geometry beyond triangle surfaces, and add a
headless (PNG/MP4) render path that shares one frame graph with the
interactive viewer.

## Current state

- There is no offscreen path: `app.rs` owns the surface, egui and an `Instant`,
  and only it drives the renderer. `std::time::Instant` and
  `pollster::block_on` in `resumed` both panic on wasm32, so the wasm target is
  currently broken despite its scaffolding.
- Colormap, camera clamps and a handful of error paths are still the thesis-era
  ones; see step 5.

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
  display.rs      Scene + Selection -> DrawList, shared by both callers
  export.rs       offscreen target, readback, PNG encode, ffmpeg pipe
  ui/panel.rs     pure panel function -> PanelResponse
  render/
    context.rs    GpuContext { device, queue } -- with or without a surface
    renderer.rs   Renderer: pipelines, intermediates, frame graph
    item.rs       RenderItem, DrawList, GPU batch buffers
    uniform.rs    UniformBinding<T: Pod>, the uniform values, the bundle
    fill.rs       triangle fill pass
    segments.rs   one segment pass (wireframe, streamlines, 1-manifold cells)
    downsample.rs SSAA resolve
    camera.rs     (unchanged)
    preamble.wgsl shared types/functions, prepended to every body
    *.wgsl
  io/             obj, blender, surface (the interchange intermediate)
```

Deleted: `ui/mod.rs` stub, `examples/_check_spectrum.rs`, `mesh3d.rs`
(contents absorbed by `bake.rs` and `demos.rs`), `TriangleSurface3D` as a
public seam (io keeps it as its own thin intermediate).

### The bake: `BakedMesh`

The GPU rasterizes simplices of dimension <= 2 embedded in $RR^3$; the bake
states exactly that. Complexes are pure manifolds, so the bake of one complex
is one *cell batch* at primitive dimension $min(d, 2)$ plus *derived
overlays* -- the submesh pattern: one shared vertex table, several index
streams.

```rust
pub struct BakedMesh {
  positions: Vec<BakedVertex>, // static per-vertex: position, normal,
                               // max_displacement (f32, R^3)
  cells: PrimBatch,            // Triangles | Segments | Points, by min(d, 2)
  wireframe: Vec<[u32; 2]>,    // derived 1-skeleton overlay
}

fn attributes(values: &[f64]) -> Vec<f32>; // per-field, swappable without rebaking
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
  attribute stream, not positions, normals, curvature, or winding. The bake is
  therefore a function of the mesh alone, and the attribute stream a separate
  function of the field -- rather than one struct carrying both, where the
  split would be a convention instead of a signature.

### The draw list

The renderer consumes `Vec<RenderItem>`; an item is a batch plus pipeline
kind plus material parameters. A surface, its wireframe overlay, and its
traced streamline ribbons are three items (streamlines are already a
renderable that is not a complex). Multiple manifolds in one scene are just
more items.

### The renderer/window split

`Renderer` owns pipelines and transient targets (SSAA color, depth; resized on
demand) and records the frame graph into any caller
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
subprocess when present (clear error otherwise) -- no vendored encoder.

An eigenmode is periodic with period $T = 2 pi \/ omega$, and the clip *is* one
period: frame $k$ is rendered at $t_k = k T \/ N$, so $N$ divides $T$ by
construction and the wrap from the last frame to the first is exact. Sampling
$t_k = k \/ "fps"$ instead would divide the period by a number it has no reason
to divide evenly, leaving up to half a frame of phase at the seam. `--frames`
therefore chooses how densely the period is sampled and `--fps` only the
playback rate; the default $N$ is the one that makes playback wall-clock. A
field that does not oscillate -- not an eigenmode, or a harmonic mode -- has no
period and is one still.

CLI in `main.rs`:

```
studio                            # interactive (default)
studio export out.png --view ... --field ... --size WxH
studio export out.mp4 ... --fps N [--frames N]
```

reusing `View`/`Gallery` scene construction and `Selection` verbatim, through
the shared `display.rs`. Export resolution is independent of any window.

The CLI carries no flag for what the code can decide itself. Supersampling is
the case in point: the window is frame-rate bound and an export is not, so the
factor follows from the context and a `--supersample` would only let a caller
ask for a worse image. It is a `Renderer::new` parameter (construction data,
like the target format, since it is baked into every pipeline as the WGSL
`SSAA_SCALE` override) with two call sites, not a knob. `--frames` stays because
it is the one number the mathematics does not always supply.

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
   Drop the LIC mark and its G-buffer: it and the streamlines are two
   renderings of the same reduced grade-1 field, and the streamlines are the
   intrinsic one -- traced through the atlas rather than advected in pixel
   space, hence view-independent where the LIC never was. Dropping it takes
   the only branch out of the frame graph, along with `LineMark`, the nodal
   `direction` the G-buffer alone consumed, and the `Vertex` attribute
   carrying it.
3. **BakedMesh + draw list.** Dimension-dispatched bake with the shared
   vertex table; one segment pipeline replacing the wireframe/streamline
   twins; delete the `dim == 2` assert and the duplicate bake. Law tests:
   bake sweep over $d in 1..=3$; winding consistency (every shared edge
   traversed in opposite directions by its two triangles);
   `Scene::whitney_basis(d)` sweep. Update `CLAUDE.md` in the same commit
   (see below).

   The segment merge is one pipeline, not two off one shader: what remained
   after the endpoint types unified was blend and depth, and both dissolve.
   Alpha-blending with $alpha = 1$ *is* `REPLACE`, so the wireframe loses
   nothing; and the wireframe never needed to write depth, since nothing in the
   scene is drawn after it, while the translucent ribbons must not. Ink, width,
   end taper and the node fade become the material; a mark that does not ride the
   standing wave carries a zero normal, which makes the displacement the identity
   on it rather than a branch.

   `PrimBatch::Points` bakes but carries no mark yet -- the same "a future mark,
   not a special case to route around" the reduced grade $>= 2$ gets in step 5.
   The curvature cap likewise has no meaning for $d >= 3$'s boundary surface (the
   volume's own estimators are not the boundary's), so displacement there is
   uncapped rather than clamped by a quantity that means something else.
4. **Export.** `export.rs` + CLI: PNG, then the ffmpeg pipe. Preceded by
   lifting the display layer (the `Scene`-to-`DrawList` reduction) out of
   `app.rs`, which held it only because the window was its sole caller: it is
   what the two callers must share for a material not to drift between them.
   Headless smoke test: render a known view, assert readback is non-uniform
   (skip when no adapter is available). Pointed at a grade-1 view, whose field
   reduces to the streamline mark -- grade 0 is scalars only and would leave the
   segment pipeline uncovered.
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
