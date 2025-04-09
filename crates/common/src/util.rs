pub trait CumsumExt {
  fn cumsum(self) -> impl Iterator<Item = usize>;
}
impl<I: IntoIterator<Item = usize>> CumsumExt for I {
  fn cumsum(self) -> impl Iterator<Item = usize> {
    self.into_iter().scan(0, |acc, x| {
      *acc += x;
      Some(*acc)
    })
  }
}

pub trait IterAllEqExt<T> {
  fn unique_eq(self) -> Option<T>;
}
impl<T: PartialEq, I: IntoIterator<Item = T>> IterAllEqExt<T> for I {
  fn unique_eq(self) -> Option<T> {
    let mut iter = self.into_iter();
    let first = iter.next()?;
    iter.all(|elem| elem == first).then_some(first)
  }
}

pub fn algebraic_convergence_rate(next: f64, prev: f64) -> f64 {
  let quot: f64 = next / prev;
  -quot.log2()
}

pub fn indicies_to_flags(indicies: &[usize], len: usize) -> Vec<bool> {
  let mut flags = vec![false; len];
  indicies.iter().for_each(|&i| flags[i] = true);
  flags
}

pub fn flags_to_indicies(flags: &[bool]) -> Vec<usize> {
  flags
    .iter()
    .enumerate()
    .filter_map(|(i, &flag)| flag.then_some(i))
    .collect()
}

pub fn sparse_to_dense_data<T>(sparse: Vec<(usize, T)>, len: usize) -> Vec<Option<T>> {
  let mut dense = Vec::from_iter((0..len).map(|_| None));
  sparse.into_iter().for_each(|(i, t)| dense[i] = Some(t));
  dense
}

pub fn dense_to_sparse_data<T>(dense: Vec<Option<T>>) -> Vec<(usize, T)> {
  dense
    .into_iter()
    .enumerate()
    .filter_map(|(i, o)| o.map(|v| (i, v)))
    .collect()
}

pub trait VecInto<S> {
  fn vec_into(self) -> Vec<S>;
}

impl<T, S> VecInto<S> for Vec<T>
where
  T: Into<S>,
{
  fn vec_into(self) -> Vec<S> {
    self.into_iter().map(|v| v.into()).collect()
  }
}

pub fn phase_to_rgb(phase: f64) -> [f32; 3] {
  let hue = (phase / std::f64::consts::PI + 1.0) / 2.0;
  hsv_to_rgb(hue as f32, 1.0, 1.0)
}

pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
  let i = (h * 6.0).floor() as i32;
  let f = h * 6.0 - i as f32;
  let p = v * (1.0 - s);
  let q = v * (1.0 - f * s);
  let t = v * (1.0 - (1.0 - f) * s);
  match i % 6 {
    0 => [v, t, p],
    1 => [q, v, p],
    2 => [p, v, t],
    3 => [p, q, v],
    4 => [t, p, v],
    5 => [v, p, q],
    _ => [0.0, 0.0, 0.0],
  }
}
