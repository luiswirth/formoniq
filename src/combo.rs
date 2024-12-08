mod index_set;
mod sign;

pub use index_set::*;
pub use sign::*;

pub mod exterior;
pub mod simplicial;

pub fn binomial(n: usize, k: usize) -> usize {
  num_integer::binomial(n, k)
}
pub fn factorial(num: usize) -> usize {
  (1..=num).product()
}
