use crate::cochain::Cochain;

use manifold::Dim;

use std::{
  fs::File,
  io::{self, BufRead, BufReader, BufWriter},
  path::Path,
};

pub fn read_cochain_from_file(path: impl AsRef<Path>, dim: Dim) -> io::Result<Cochain> {
  let file = File::open(path)?;
  let reader = BufReader::new(file);

  let mut coeffs = Vec::new();

  for line in reader.lines() {
    let line = line?;
    if let Ok(coeff) = line.trim().parse::<f64>() {
      coeffs.push(coeff);
    }
  }

  Ok(Cochain::new(dim, coeffs.into()))
}
pub fn save_cochain_to_file(cochain: &Cochain, path: impl AsRef<Path>) -> std::io::Result<()> {
  let file = File::create(path)?;
  let writer = BufWriter::new(file);
  write_cochain(writer, cochain)
}

pub fn write_cochain<W: std::io::Write>(mut writer: W, cochain: &Cochain) -> std::io::Result<()> {
  for coeff in cochain.coeffs().iter() {
    write!(writer, "{coeff:.6} ")?;
    writeln!(writer)?;
  }
  Ok(())
}
