{
  description = "formoniq";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = {
    nixpkgs,
    flake-utils,
    rust-overlay,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        overlays = [
          (import rust-overlay)
          (self: super: {
            rust-toolchain = self.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
          })
        ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };
      in
        with pkgs; {
          devShell = mkShell {
            buildInputs = [
              pkg-config
              rust-toolchain
              rust-analyzer

              bacon
              cargo-edit
              cargo-flamegraph
              linuxPackages_latest.perf

              (python3.withPackages (ps: with ps; [
                python-lsp-server

                numpy
                scipy
                matplotlib

                meshio
              ]))
            ];
          };
        }
    );
}
