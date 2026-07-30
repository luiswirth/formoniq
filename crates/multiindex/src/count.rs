//! The elementary counting functions, and the table the ranking runs on.

/// Pascal's triangle up to the index ceiling, computed once.
///
/// A rank is a sum of binomials and ranking is the innermost thing the algebra
/// does, so this is a table lookup rather than a division loop. It covers every
/// shifted symbol a [`MonoIndex`](crate::MonoIndex) can hold; beyond that the
/// exact computation still runs, so nothing here is a bound on what may be
/// counted.
const BINOMIAL_TABLE_SIZE: usize = crate::monotone::MAX_SHIFTED_SYMBOLS + 1;
static BINOMIALS: std::sync::LazyLock<[[usize; BINOMIAL_TABLE_SIZE]; BINOMIAL_TABLE_SIZE]> =
  std::sync::LazyLock::new(|| {
    let mut table = [[0usize; BINOMIAL_TABLE_SIZE]; BINOMIAL_TABLE_SIZE];
    for n in 0..BINOMIAL_TABLE_SIZE {
      table[n][0] = 1;
      for k in 1..=n {
        table[n][k] = table[n - 1][k - 1].saturating_add(table[n - 1].get(k).copied().unwrap_or(0));
      }
    }
    table
  });

pub fn binomial(n: usize, k: usize) -> usize {
  if n < BINOMIAL_TABLE_SIZE && k < BINOMIAL_TABLE_SIZE {
    BINOMIALS[n][k]
  } else {
    num_integer::binomial(n, k)
  }
}
pub fn factorial(num: usize) -> usize {
  (1..=num).product()
}
pub fn factorial_f64(num: usize) -> f64 {
  factorial(num) as f64
}
