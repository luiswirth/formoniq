//! The `studio` binary: the interactive viewer by default, a headless render
//! with `export`.
//!
//! The two are not two frontends. Both build the same scene through the same
//! gallery construction and the same display layer; the export subcommand adds
//! only a destination and a frame clock. What is *not* here is a flag for
//! anything the code can decide for itself -- the wave amplitude, the mark
//! widths and the supersampling are properties of the object and of the
//! context, and asking the caller for them would be asking for information they
//! do not have.

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use formoniq_studio::export::{export, ExportSpec};
use formoniq_studio::gallery::{
  BuiltinMesh, CochainSpec, MeshSource, NamedCochain, Study, DEFAULT_NMODES,
  DEFAULT_TRAJECTORY_STEPS, HEAT_FINAL_TIME, REFERENCE_CELL_DIM, WAVE_FINAL_TIME,
};

#[derive(Parser)]
#[command(about = "A viewer for FEEC solutions on simplicial manifolds")]
struct Cli {
  #[command(subcommand)]
  command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
  /// Render a scene to a PNG still or an MP4 clip, with no window.
  Export {
    /// Destination. `.mp4` writes a clip through `ffmpeg`; anything else is a
    /// PNG still.
    out: PathBuf,
    /// Which study to run: `grade<k>` for a Hodge-Laplace eigenmode grade,
    /// `whitney` for the Whitney basis, or `hodge` for the grade-1 Hodge
    /// decomposition.
    #[arg(long, value_parser = parse_study, default_value = "grade0")]
    study: Study,
    /// Which mesh to run the study on: `sphere`, `grid`, `refcell`,
    /// `triforce`, a built-in by name, or a directory holding a mesh saved by
    /// the engine's own `topology.cbor`/`coords.cbor` pair.
    #[arg(long, value_parser = parse_mesh, default_value = "sphere")]
    mesh: MeshSource,
    /// A cochain saved by the engine (`Cochain::save`) to show, in addition
    /// to `--study`'s own. Repeatable; each is named for its file stem.
    /// Given at least one, the scene shows exactly these cochains rather than
    /// `--study`'s built-in fields -- the general path for anything the
    /// gallery did not itself solve (a solve's output, a paper figure).
    #[arg(long = "cochain")]
    cochains: Vec<PathBuf>,
    /// Which field of the scene, in the picker's order. Defaults to the first,
    /// as the viewer opens on.
    #[arg(long)]
    field: Option<usize>,
    /// Export resolution, `WxH`. Independent of any window.
    #[arg(long, value_parser = parse_size, default_value = "1920x1080")]
    size: (u32, u32),
    /// How densely to sample the standing wave's period (mp4). The clip is
    /// exactly one period at any count -- that is what makes it loop seamlessly
    /// -- so this only chooses how many frames that period is cut into.
    /// Defaults to the count that makes playback at `--fps` run at wall-clock
    /// speed. A field that is not an eigenmode has no period and renders one
    /// still.
    #[arg(long)]
    frames: Option<u32>,
    #[arg(long, default_value_t = 60)]
    fps: u32,
  },
}

fn parse_study(s: &str) -> Result<Study, String> {
  match s {
    "whitney" => Ok(Study::WhitneyBasis),
    "hodge" | "decomposition" => Ok(Study::HodgeDecomposition),
    "triforce-examples" => Ok(Study::Cochains(formoniq_studio::demos::triforce_examples())),
    "heat" => Ok(Study::Heat {
      nsteps: DEFAULT_TRAJECTORY_STEPS,
      final_time: HEAT_FINAL_TIME,
    }),
    "wave" => Ok(Study::Wave {
      nsteps: DEFAULT_TRAJECTORY_STEPS,
      final_time: WAVE_FINAL_TIME,
    }),
    _ => s
      .strip_prefix("grade")
      .and_then(|g| g.parse().ok())
      .map(|grade| Study::Eigenmodes {
        grade,
        nmodes: DEFAULT_NMODES,
      })
      .ok_or_else(|| {
        format!("expected `grade<k>`, `whitney`, `hodge`, `heat` or `wave`, got `{s}`")
      }),
  }
}

/// The mesh names an export accepts: the generated families, the reference cell
/// and triforce, and every embedded built-in by its own name. A mesh the user
/// loaded into the viewer is not among them -- `MeshSource::Custom` is not
/// regenerable from its descriptor, so there is nothing here to name it by.
fn parse_mesh(s: &str) -> Result<MeshSource, String> {
  match s {
    "sphere" => Ok(MeshSource::START),
    "grid" => Ok(MeshSource::Grid { cells_axis: 16 }),
    "refcell" | "reference" => Ok(MeshSource::ReferenceCell {
      dim: REFERENCE_CELL_DIM,
    }),
    "triforce" => Ok(MeshSource::Triforce),
    _ if Path::new(s).is_dir() => Ok(MeshSource::File(PathBuf::from(s))),
    _ => BuiltinMesh::from_name(s)
      .map(MeshSource::Builtin)
      .ok_or_else(|| {
        let builtins = BuiltinMesh::ALL.map(BuiltinMesh::name).join("`, `");
        format!(
          "expected `sphere`, `grid`, `refcell`, `triforce`, `{builtins}`, or a mesh directory, \
           got `{s}`"
        )
      }),
  }
}

fn parse_size(s: &str) -> Result<(u32, u32), String> {
  let (w, h) = s
    .split_once(['x', 'X'])
    .ok_or_else(|| format!("expected `WxH`, got `{s}`"))?;
  let parse = |v: &str, axis| {
    v.parse::<u32>()
      .ok()
      .filter(|&v| v > 0)
      .ok_or_else(|| format!("{axis} must be a positive integer, got `{v}`"))
  };
  Ok((parse(w, "width")?, parse(h, "height")?))
}

fn main() {
  let cli = Cli::parse();
  match cli.command {
    None => pollster::block_on(formoniq_studio::run()),
    Some(Command::Export {
      out,
      study,
      mesh,
      cochains,
      field,
      size,
      frames,
      fps,
    }) => {
      let _ = env_logger::try_init();
      let study = if cochains.is_empty() {
        study
      } else {
        Study::Cochains(
          cochains
            .into_iter()
            .map(|path| NamedCochain {
              name: path
                .file_stem()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.to_string_lossy().into_owned()),
              spec: CochainSpec::File(path),
            })
            .collect(),
        )
      };
      let spec = ExportSpec {
        study,
        mesh_source: mesh,
        field,
        size,
        frames,
        fps,
      };
      if let Err(error) = export(&spec, &out) {
        eprintln!("export failed: {error}");
        std::process::exit(1);
      }
      println!("wrote {}", out.display());
    }
  }
}
