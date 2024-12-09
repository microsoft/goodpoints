This package contains algorithms from Debiased Distribution Compression by Li et al. 2024 implemented in JAX.

## Demos
See [here](https://github.com/microsoft/goodpoints/tree/main/examples/debias/notebook/demo) for examples of how to use various algorithms provided by this package.

## Debiased distribution compression algorithms
The `dtc.py` module contains the meta debias-then-compress algorithm for debiased compression along with 6 instantiations (functions `skt`, `lskt`, `sr`, `lsr`, `sc`, `lsc`) corresponding to the 6 algorithms in the paper.

### Usage
All debiased compression algorithms have the same input and output formats. 
For instance, Stein kernel thinning implementation `skt` function has signature `skt(kernel, points, out_size)` where
- `kernel` is a function implemented in JAX that broadcasts the two input point sets
- `points` is a `SliceablePoints` instance, and `out_size` is the output size. `SliceablePoints` is a data structure that includes the input points as well as precomputed statistics used for kernel evaluation. 
- This function returns two numpy arrays, `w_thin` and `supp`, where `w_thin` is the weight vector of the coreset of size equal to `points`, and `supp` is a list of indices of the support of the coreset. Hence, the coreset can be recovered as point set `points[supp]` with weight vector `w_thin[supp]`.
An example of `skt` is given in `demo/mixture_skt.ipynb`.

## Details on available algorithms

For debiasing, this package provides
- Stein Thinning (`st.py`) for quadratic-time debiasing of input points.
- Low-rank Debiasing (`lr_debias.py`) for sub-quadratic-time debiasing of input points.

For compression, this package provides
- Kernel Thinning (`kt.py`) for compressing weighted input points into an equal-weighted coreset.
- Recombination Thinning (`recomb_thin.py`) for compressing weighted input points into a simplex-weighted coreset.
- Cholesky Thinning (`chol_thin.py`) for compressing weighted input points into a constant-preserving coreset.

The `rpc.py` module implements a weighted extension of the Randomly pivoted Cholesky algorithm by [Chen et al. 2023](https://arxiv.org/pdf/2207.06503).

For evaluation, `mmd.py` provides a function to compute the MMD between two sets of weighted points, using tiling to speed up the computation while reducing the memory footprint.


The `kernel/` package includes modules to build kernels with automatic kernel parameter tuning. As we often deal with points with additional per-point attributes (such as the score), `sliceable_points.py` provides a JAX data structure to store points with additional attributes with helper functions, which is then input to the kernel.

The `serial/` package provides utilities for serialization and caching of intermediate outputs.

## Dependencies
The main dependency is JAX; the install instructions can be found at [here](https://jax.readthedocs.io/en/latest/quickstart.html).

Additional requirements:
- `tqdm` (progress bars)
- `pymongo` and `h5py` (only if using MongoDB caching)
