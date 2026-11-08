# Matrix multiplication experiment

Branch: `experiment/matrixmultiply`. The scalar baseline is commit `3c86e22`.

For batches of two or more, each projection packs the request vectors into a
`batch × width` matrix, calls `matrixmultiply::sgemm`, then copies the output
rows back to the individual request states. A batch of one uses the original
scalar dot product.

Results on this machine (AMD Ryzen 7 5800H), using `stories15M.bin`, release
build, 16 positions, and the median of five runs:

| Batch | Separate forwards | Scalar batch | `sgemm` batch | Speedup vs separate |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 1.279 s | 1.308 s | 0.352 s | 3.59× |
| 8 | 2.561 s | 2.631 s | 0.356 s | 7.13× |

Run the comparison with:

```sh
cargo run --release --bin bench_forward -- ../stories15M.bin 4 16 5
cargo run --release --bin bench_forward -- ../stories15M.bin 8 16 5
```

These numbers measure model forward work only. The server still has one active
request, so it does not yet call the model with a multi-request batch.
