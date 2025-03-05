use crate::cochain::Cochain;

use std::{fs::File, io::BufWriter, path::Path};

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
