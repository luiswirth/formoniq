#!/usr/bin/env sh

set -e

cargo b --example $1 --release
cp ~/.cache/target/release/examples/$1 ./exe
patchelf --set-interpreter /lib64/ld-linux-x86-64.so.2 --set-rpath "" exe
scp exe euler.ethz.ch:formoniq/
ssh euler.ethz.ch "cd formoniq; sbatch run_exe.sh"
rm exe
