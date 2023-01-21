# GoodPoints

### A Python package for generating concise, high-quality summaries of a probability distribution

GoodPoints is a collection of tools for compressing a distribution more effectively than independent sampling:

- Given an initial summary of n input points, **kernel thinning** returns s << n output points with comparable integration error across a reproducing kernel Hilbert space
- **Compress++** reduces the runtime of generic thinning algorithms with minimal loss in accuracy
- **Compress Then Test** accelerates kernel two-sample testing using high-fidelity compression

## Installation
To install the `goodpoints` package, use the following pip command:
```
pip install goodpoints
```
## Getting started
The primary kernel thinning function is `thin` in the `kt` module:
```python
from goodpoints import kt
coreset = kt.thin(X, m, split_kernel, swap_kernel, delta=0.5, seed=123, store_K=False, 
                  verbose=False)
    """Returns kernel thinning coreset of size floor(n/2^m) as row indices into X
    
    Args:
      X: Input sequence of sample points with shape (n, d)
      m: Number of halving rounds
      split_kernel: Kernel function used by KT-SPLIT (typically a square-root kernel, krt);
        split_kernel(y,X) returns array of kernel evaluations between y and each row of X
      swap_kernel: Kernel function used by KT-SWAP (typically the target kernel, k);
        swap_kernel(y,X) returns array of kernel evaluations between y and each row of X
      delta: Run KT-SPLIT with constant failure probabilities delta_i = delta/n
      seed: Random seed to set prior to generation; if None, no seed will be set
      store_K: If False, runs O(nd) space version which does not store kernel
        matrix; if True, stores n x n kernel matrix
      verbose: If False, do not print intermediate time taken in thinning rounds, 
        if True print that info
    """
```
For example uses, please refer to [examples/kt/run_kt_experiment.ipynb](examples/kt/run_kt_experiment.ipynb).

The primary Compress++ function is `compresspp` in the `compress` module:
```python
from goodpoints import compress
coreset = compress.compresspp(X, halve, thin, g)
    """Returns Compress++(g) coreset of size sqrt(n) as row indices into X

    Args: 
        X: Input sequence of sample points with shape (n, d)
        halve: Function that takes in an (n', d) numpy array Y and returns 
          floor(n'/2) distinct row indices into Y, identifying a halved coreset
        thin: Function that takes in an (n', d) numpy array Y and returns
          2^g sqrt(n') row indices into Y, identifying a thinned coreset
        g: Oversampling factor
    """
```
For example uses, please refer to [examples/compress/construct_compresspp_coresets.py](examples/compress/construct_compresspp_coresets.py).

The primary Compress Then Test function is `ctt` in the `ctt` module:
```python
from goodpoints import ctt
test_results = ctt.ctt(X1, X2, g)
    """Compress Then Test two-sample test with sample sequences X1 and X2
    and auxiliary kernel k' = target kernel k.

    Args:
      X1: 2D array of size (n1,d)
      X2: 2D array of size (n2,d)
      g: compression level; integer >= 0
      B: number of permutations (int)
      s: total number of compression bins will be num_bins = min(2*s, n1+n2)
      lam: positive kernel bandwidth 
      kernel: kernel name; valid options include "gauss" for Gaussian kernel
        exp(-||x-y||^2/lam^2)
      null_seed: seed used to initialize random number generator for
        randomness in simulating null
      statistic_seed: seed used to initialize random number generator for
        randomness in computing test statistic
      alpha: test level
      delta: KT-Compress failure probability
      
    Returns: TestResults object.
    """
```
For example uses, please refer to [examples/mmd_test/test.py](examples/mmd_test/test.py).

## Examples

Code in the `examples` directory uses the `goodpoints` package to recreate the experiments of the following research papers.
***
#### [Kernel Thinning](https://arxiv.org/pdf/2105.05842.pdf)
```
@article{dwivedi2021kernel,
  title={Kernel Thinning},
  author={Raaz Dwivedi and Lester Mackey},
  journal={arXiv preprint arXiv:2105.05842},
  year={2021}
}
```
1. The script `examples/kt/submit_jobs_run_kt.py` reproduces the vignette experiments of Kernel Thinning  on a Slurm cluster
by executing `examples/kt/run_kt_experiment.ipynb` with appropriate parameters. For the MCMC examples, it assumes that necessary data was downloaded and pre-processed following the steps listed in `examples/kt/preprocess_mcmc_data.ipynb`, where in the last code block we  report the median heuristic based bandwidth parameteters (along with the code to compute it).
2. After all results have been generated, the notebook `examples/kt/plot_results.ipynb` can be used to reproduce the figures of Kernel Thinning.

#### [Generalized Kernel Thinning](https://arxiv.org/pdf/2110.01593.pdf) 
```
@inproceedings{dwivedi2022generalized,
  title={Generalized Kernel Thinning},
  author={Raaz Dwivedi and Lester Mackey},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```
1. The script `examples/gkt/submit_gkt_jobs.py` reproduces the vignette experiments of Generalized Kernel Thinning on a Slurm cluster by executing `examples/gkt/run_generalized_kt_experiment.ipynb` with appropriate parameters. For the MCMC examples, it assumes that necessary data was downloaded and pre-processed following the steps listed in `examples/kt/preprocess_mcmc_data.ipynb`.
2. Once the coresets are generated, `examples/gkt/compute_test_function_errors.ipynb` can be used to generate integration errors for different test functions.
3. After all results have been generated, the notebook `examples/gkt/plot_gkt_results.ipynb` can be used to reproduce the figures of Generalized Kernel Thinning.

#### [Distribution Compression in Near-linear Time](https://arxiv.org/pdf/2111.07941.pdf)
```
@inproceedings{shetty2022distribution,
  title={Distribution Compression in Near-linear Time},
  author={Abhishek Shetty and Raaz Dwivedi and Lester Mackey},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```
1. The notebook `examples/compress/script_to_deploy_jobs.ipynb` reproduces the experiments of Distribution Compression in Near-linear Time in the following manner: 
    1. It generates various coresets and computes their mmds by executing `examples/compress/construct_{THIN}_coresets.py` for `THIN` in `{compresspp, kt, st, herding}` with appropriate parameters,
        where the flag kt stands for kernel thinning, st stands for standard thinning (choosing every t-th point), and herding refers to kernel herding.
    2. It computes the runtimes of different algorithms by executing `examples/compress/run_time.py`.
    3. For the MCMC examples, it assumes that necessary data was downloaded and pre-processed following the steps listed in `examples/kt/preprocess_mcmc_data.ipynb`. 
    4. The notebook currently deploys these jobs on a slurm cluster, but setting deploy_slurm = False in `examples/compress/script_to_deploy_jobs.ipynb` will submit the jobs as independent python calls on terminal.
2. After all results have been generated, the notebook `examples/compress/plot_compress_results.ipynb` can be used to reproduce the figures of Distribution Compression in Near-linear Time.

#### [Compress Then Test: Powerful Kernel Testing in Near-linear Time](https://arxiv.org/pdf/2301.05974.pdf)
```
@article{domingoenrich2023compress,
  title={Compress Then Test: Powerful Kernel Testing in Near-linear Time},
  author={Carles Domingo-Enrich and Raaz Dwivedi and Lester Mackey},
  journal={arXiv preprint arXiv:2301.05974},
  year={2023}
}
```
See [examples/mmd_test/README.md](examples/mmd_test/README.md).

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
