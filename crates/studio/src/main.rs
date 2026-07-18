//! Binary entry point.
//!
//! The `studio` binary is native only: it is the CLI viewer and headless
//! exporter, and its whole implementation lives in [`cli`]. The web build ships
//! as a `cdylib` entered through `formoniq_studio::web`, so on `wasm32` this
//! binary compiles to an empty `main`.

#[cfg(not(target_arch = "wasm32"))]
mod cli;

fn main() {
  #[cfg(not(target_arch = "wasm32"))]
  cli::main();
}
