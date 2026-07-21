//! The native CLI: the interactive viewer by default, a headless render with
//! `export`. Native only -- the web build enters through the `cdylib` and
//! `formoniq_studio::web`, never this binary.

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use formoniq_studio::export::{ExportSpec, export};
use formoniq_studio::gallery::{
  BuiltinMesh, CochainSpec, DEFAULT_NMODES, DEFAULT_TRAJECTORY_STEPS, GRID_DIM_DEFAULT,
  GRID_DIM_MAX, HEAT_FINAL_TIME, MeshSource, NamedCochain, QUOTIENT_CELLS_DEFAULT, QuotientSurface,
  REFERENCE_CELL_DIM, REFERENCE_CELL_DIM_MAX, Study, WAVE_FINAL_TIME,
};
use simplicial::Dim;

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
    /// Which mesh to run the study on: `sphere`, `grid[dim]`, `refcell[dim]`
    /// (the optional trailing digit is the intrinsic dimension, 1..=3 -- e.g.
    /// `grid3` is a cube of tetrahedra, `refcell1` a single edge), `triforce`,
    /// `donut` or `moebius` (the two flat quotients that fit in RR^3, the
    /// second non-orientable), a built-in by name, or a directory holding a mesh
    /// saved by the engine's own `topology.cbor`/`coords.cbor` pair.
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
    _ => parse_graded_study(s).ok_or_else(|| {
      format!("expected `grade<k>`, `whitney`, `hodge`, `heat[<k>]` or `wave[<k>]`, got `{s}`")
    }),
  }
}

/// The studies posed at a single grade, named `<study><k>` with the grade
/// suffix optional and defaulting to 0 (`heat`, `heat2`, `wave1`). The
/// eigenproblem's own name *is* its grade (`grade<k>`), so it carries no
/// default.
fn parse_graded_study(s: &str) -> Option<Study> {
  let graded = |prefix: &str| {
    s.strip_prefix(prefix).and_then(|g| {
      if g.is_empty() {
        Some(0)
      } else {
        g.parse().ok()
      }
    })
  };
  if let Some(grade) = s.strip_prefix("grade").and_then(|g| g.parse().ok()) {
    return Some(Study::Eigenmodes {
      grade,
      nmodes: DEFAULT_NMODES,
    });
  }
  if let Some(grade) = graded("heat") {
    return Some(Study::Heat {
      grade,
      nsteps: DEFAULT_TRAJECTORY_STEPS,
      final_time: HEAT_FINAL_TIME,
    });
  }
  graded("wave").map(|grade| Study::Wave {
    grade,
    nsteps: DEFAULT_TRAJECTORY_STEPS,
    final_time: WAVE_FINAL_TIME,
  })
}

/// The mesh names an export accepts: the generated families, the reference cell
/// and triforce, and every embedded built-in by its own name. A mesh the user
/// loaded into the viewer is not among them -- `MeshSource::Custom` is not
/// regenerable from its descriptor, so there is nothing here to name it by.
fn parse_mesh(s: &str) -> Result<MeshSource, String> {
  // `grid` and `refcell` take an optional intrinsic dimension as a trailing
  // digit -- `grid3` is a cube of tetrahedra, `refcell1` a single edge -- so one
  // spelling reaches every dimension the fixed ambient RR^3 embeds, no separate
  // name per dimension.
  let dimensioned = |stem: &str, default: Dim, max: Dim| -> Option<Result<Dim, String>> {
    let rest = s.strip_prefix(stem)?;
    if rest.is_empty() {
      return Some(Ok(default));
    }
    Some(match rest.parse::<Dim>() {
      Ok(d) if (1..=max).contains(&d) => Ok(d),
      _ => Err(format!(
        "`{stem}` dimension must be 1..={max}, got `{rest}`"
      )),
    })
  };

  if let Some(dim) = dimensioned("grid", GRID_DIM_DEFAULT, GRID_DIM_MAX) {
    return dim.map(|dim| MeshSource::Grid {
      dim,
      cells_axis: 16,
    });
  }
  if let Some(dim) = dimensioned("refcell", REFERENCE_CELL_DIM, REFERENCE_CELL_DIM_MAX)
    .or_else(|| dimensioned("reference", REFERENCE_CELL_DIM, REFERENCE_CELL_DIM_MAX))
  {
    return dim.map(|dim| MeshSource::ReferenceCell { dim });
  }

  match s {
    "sphere" => Ok(MeshSource::START),
    "triforce" => Ok(MeshSource::Triforce),
    _ if QuotientSurface::from_name(s).is_some() => Ok(MeshSource::Quotient {
      surface: QuotientSurface::from_name(s).unwrap(),
      cells_axis: QUOTIENT_CELLS_DEFAULT,
    }),
    _ if Path::new(s).is_dir() => Ok(MeshSource::File(PathBuf::from(s))),
    _ => BuiltinMesh::from_name(s)
      .map(MeshSource::Builtin)
      .ok_or_else(|| {
        let builtins = BuiltinMesh::all()
          .map(BuiltinMesh::name)
          .collect::<Vec<_>>()
          .join("`, `");
        format!(
          "expected `sphere`, `grid[dim]`, `refcell[dim]`, `triforce`, `donut`, `moebius`, \
           `{builtins}`, or a mesh directory, got `{s}`"
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

pub fn main() {
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
