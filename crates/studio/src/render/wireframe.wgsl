// Wireframe: the mesh's 1-skeleton as billboard quads of constant world-space
// thickness (see `billboard_perp` in the preamble). The two endpoints arrive as
// two separate per-instance vertex buffers (`Vertex::desc_endpoint_a`/`_b` in
// `mesh.rs`) bound side by side, since a `step_mode: Vertex` buffer only ever
// exposes the current vertex to the shader, never a neighbor -- both ends of
// the same edge have to be visible to one invocation to compute the
// perpendicular.

@group(0) @binding(0) var<uniform> camera: CameraUniform;
// Same standing-wave displacement as the fill, so the wireframe tracks the
// displaced surface instead of the flat rest mesh. A line field sets the wave
// amplitude to zero, so its wireframe stays on the undisplaced surface.
@group(1) @binding(0) var<uniform> wave: Wave;
@group(2) @binding(0) var<uniform> width: SegmentWidth;

struct EndpointA {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) value: f32,
    @location(3) max_displacement: f32,
};
struct EndpointB {
    @location(4) position: vec3<f32>,
    @location(5) normal: vec3<f32>,
    @location(6) value: f32,
    @location(7) max_displacement: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(a: EndpointA, b: EndpointB, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let world_a = wave_displace(wave, a.position, a.normal, a.value, a.max_displacement);
    let world_b = wave_displace(wave, b.position, b.normal, b.value, b.max_displacement);
    let perp = billboard_perp(world_a, world_b, camera.eye.xyz);
    let corner = billboard_corner(world_a, world_b, perp, width.half_width_world, vertex_index);

    var out: VertexOutput;
    out.clip_position = depth_biased(camera.view_proj * vec4<f32>(corner, 1.0));
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0); // black edges
}
