pub mod camera;
pub mod mesh;

#[cfg(test)]
mod tests {
  /// Every WGSL source in this module parses and validates against naga's own
  /// frontend -- the one wgpu itself uses to build a pipeline -- so a broken
  /// shader fails `cargo test` instead of the pipeline-creation panic at
  /// startup a syntax or type error would otherwise cause.
  #[test]
  fn shaders_parse_and_validate() {
    let sources: &[(&str, &str)] = &[
      ("shader.wgsl", include_str!("shader.wgsl")),
      ("wireframe.wgsl", include_str!("wireframe.wgsl")),
      ("gbuffer.wgsl", include_str!("gbuffer.wgsl")),
      ("lic.wgsl", include_str!("lic.wgsl")),
      ("downsample.wgsl", include_str!("downsample.wgsl")),
    ];
    for (name, source) in sources {
      let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("{name} failed to parse: {e}"));
      naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
      )
      .validate(&module)
      .unwrap_or_else(|e| panic!("{name} failed to validate: {e}"));
    }
  }
}
