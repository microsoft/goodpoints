# Compress Then Test Experiments

## Running experiments locally

#### Single-bandwidth tests
In order to run experiments locally, one must run the `test.py`, `postprocessing.py` and `figures.py` scripts in this order. As an example, to get the power plots corresponding to the average of 100 tests in the Gaussian setting with mean difference 0.048 and n=16384, we would run:
```
python ./test.py --name gaussians --task_id 1 --d 10 --n 16384 --n_tests 100 --mean_diff 0.048 --B 39 --seed_0 0 --s_rff 0 --s_permute 16
```
followed by
```
python postprocessing.py --name gaussians --n 16384 --mean_diff 0.048 --B 39 --n_tests 100 --total_n_tests 100 --s_rff 0 --s_permute 16
```
and then by
```
python figures.py --name gaussians --n 16384 --d 10 --n_tests 100 --total_n_tests 100 --mean_diff 0.048 --no_asymp --log_time_scale --B 39 --long_times 0.1 --wilson_intervals --s_rff 0 --s_permute 16
```
By default, the test groups that are being run in the single-bandwidth mode are `ctt` (which stands for CTT/Compress Then Test), `block_wb` (which stands for wild bootstrap block test), `incomplete_wb` (which stands for wild bootstrap incomplete test), `block_asymp` (which stands for asymptotic block test), and `incomplete_asymp` (which stands for asymptotic incomplete test).  
If we want to rerun a certain group of tests (e.g. block WB) and prevent some other group of tests from running (e.g. incomplete asymp.), we should use the command
```
python test.py --name gaussians --task_id 1 --d 10 --n 16384 --n_tests 100 --mean_diff 0.048 --B 39 --seed_0 0 --recompute_block_wb --no_incomplete_asymp
```
Similarly, we should include the flag `--no_incomplete_asymp` when running the `postprocessing.py` and `figures.py` scripts. 
To get the power plots for the EMNIST setting with even probability 0.46, we run
```
python ./test.py --name EMNIST --task_id 1 --d 10 --n 16384 --n_tests 100 --p_even 0.46 --B 39 --seed_0 0 --s_rff 0 --s_permute 16
```
```
python postprocessing.py --name EMNIST --n 16384 --p_even 0.46 --B 39 --n_tests 100 --total_n_tests 100 --s_rff 0 --s_permute 16
```
```
python figures.py --name EMNIST --n 16384 --d 10 --n_tests 100 --total_n_tests 100 --p_even 0.46 --no_asymp --log_time_scale --B 39 --long_times 0.2 --wilson_intervals --s_rff 0 --s_permute 16
```
In order to get size plots, we just need to replace the flag `--mean_diff 0.048` by `--mean_diff 0.0` in the Gaussian setting, and `--p_even 0.46` by `--p_even 0.46` in the EMNIST setting.

#### Aggregated tests
Analogously, one must run the `test.py`, `postprocessing.py` and `figures.py` scripts in this order. As an example, to get the plots corresponding to 100 tests in the Blobs setting with epsilon 1.4 and n=16384, we would run:
```
python test.py --name blobs --task_id 1 --d 2 --n 16384 --epsilon 1.4 --n_tests 100 --aggregated --B 299 --B_2 200 --seed_0 0
```
followed by
```
python postprocessing.py --name blobs --d 2 --n 16384 --epsilon 1.4 --n_tests 100 --aggregated --B 299 --B_2 200 --total_n_tests 100
```
and then by
```
python figures.py --name blobs --d 2 --n 16384 --epsilon 1.4 --n_tests 100 --aggregated --B 299 --B_2 200 --total_n_tests 100 --no_asymp --log_time_scale --long_times 0.1 --wilson_intervals
```
By default, the test groups that are being run in the aggregated mode are `ctt` and `incomplete_wb`. We should use the flags `--no_ctt`/`--no_incomplete_wb` to stop `ctt`/`incomplete_wb` tests from running. We should use the flags `--recompute_ctt`/`--recompute_incomplete_wb` to rerun `ctt`/`incomplete_wb` tests. To get the power plots for the Higgs setting, we run
```
python test.py --name Higgs --task_id 1 --n_components 2 --n 16384 --p_poisoning 0.0 --n_tests 100 --aggregated --B 299 --B_2 200 --seed_0 0 --mixing
```
```
python postprocessing.py --name Higgs --n_components 2 --n 16384 --p_poisoning 0.0 --n_tests 100 --total_n_tests 100 --aggregated --B 299 --B_2 200 --mixing
```
```
python figures.py --name Higgs --n_components 2 --n 16384 --p_poisoning 0.0 --n_tests 100 --total_n_tests 100 --aggregated --B 299 --B_2 200 --no_asymp --log_time_scale --long_times 0.1 --wilson_intervals --mixing
```
In order to get size plots, we just need to replace the flag `--epsilon 1.4` by `--epsilon 1.0` in the Gaussian setting, and `--p_poisoning 0.0` by `--p_poisoning 1.0` in the EMNIST setting.


## Running experiments in a Slurm-managed cluster
The plots shown in the paper were generated in a Slurm-managed cluster. We show the commands that were run below. When running the figures.py script some flags of the form --no_block_asymp must be used to prevent certain test groups from showing; the current flag configuration shows all test groups at once.

#### Gaussian $n=4^9$, size plot
```
sbatch gaussian_test_null.slurm
```
```
python postprocessing.py --name gaussians --n 262144 --mean_diff 0.0 --B 39 --n_tests 1 --total_n_tests 400 --s_rff 0 --s_permute 16
```
```
python figures.py --name gaussians --n 262144 --d 10 --n_tests 1 --total_n_tests 400 --mean_diff 0.0 --no_asymp --log_time_scale --B 39 --long_times 10 --wilson_intervals --no_violations --s_rff 0 --s_permute 16
```

#### Gaussian $n=4^9$, power plot
```
sbatch gaussian_test.slurm
```
```
python postprocessing.py --name gaussians --n 262144 --mean_diff 0.012 --B 39 --n_tests 1 --total_n_tests 400 --s_rff 0 --s_permute 16
```
```
python figures.py --name gaussians --n 262144 --d 10 --n_tests 1 --total_n_tests 400 --mean_diff 0.012 --no_asymp --log_time_scale --B 39 --long_times 10 --wilson_intervals --no_violations --s_rff 0 --s_permute 16
```

#### EMNIST $n=4^9$, size plot
```
sbatch emnist_test_null.slurm
```
```
python postprocessing.py --name EMNIST --n 262144 --p_even 0.5 --B 39 --n_tests 1 --total_n_tests 400 --s_rff 0 --s_permute 16
```
```
python figures.py --name EMNIST --n 262144 --d 49 --n_tests 1 --total_n_tests 400 --p_even 0.5 --no_asymp --log_time_scale --B 39 --long_times 10 --wilson_intervals --no_violations --s_rff 0 --s_permute 16
```

#### EMNIST $n=4^9$, power plot
```
sbatch emnist_test.slurm
```
```
python postprocessing.py --name EMNIST --n 262144 --p_even 0.49 --B 39 --n_tests 1 --total_n_tests 400 --s_rff 0 --s_permute 16
```
```
python figures.py --name EMNIST --n 262144 --d 49 --n_tests 1 --total_n_tests 400 --p_even 0.49 --no_asymp --log_time_scale --B 39 --long_times 10 --wilson_intervals --no_violations --s_rff 0 --s_permute 16
```

#### Blobs $n=4^7$, size plot
```
sbatch blobs_test_null.slurm
```
```
python postprocessing.py --name blobs --d 2 --n 16384 --epsilon 1.0 --n_tests 1 --aggregated --B 299 --B_2 200 --total_n_tests 400
```
```
python figures.py --name blobs --d 2 --n 16384 --epsilon 1.4 --n_tests 1 --aggregated --B 299 --B_2 200 --total_n_tests 400 --no_asymp --log_time_scale --long_times 0.1 --wilson_intervals
```

#### Blobs $n=4^7$, power plot
```
sbatch blobs_test.slurm
```
```
python postprocessing.py --name blobs --d 2 --n 16384 --epsilon 1.4 --n_tests 1 --aggregated --B 299 --B_2 200 --total_n_tests 400
```
```
python figures.py --name blobs --d 2 --n 16384 --epsilon 1.4 --n_tests 1 --aggregated --B 299 --B_2 200 --total_n_tests 400 --no_asymp --log_time_scale --long_times 0.1 --wilson_intervals
```

#### Higgs $n=4^7$, size plot
```
sbatch higgs_test_null.slurm
```
```
python postprocessing.py --name Higgs --n_components 2 --n 16384 --p_poisoning 1.0 --n_tests 1 --aggregated --B 299 --B_2 200 --total_n_tests 400 --mixing
```
```
python figures.py --name Higgs --n_components 2 --n 16384 --p_poisoning 1.0 --n_tests 400 --aggregated --B 299 --B_2 200 --total_n_tests 400 --no_asymp --log_time_scale --long_times 0.1 --wilson_intervals --mixing
```

#### Higgs $n=4^7$, power plot
```
sbatch higgs_test.slurm
```
```
python postprocessing.py --name Higgs --n_components 2 --n 16384 --p_poisoning 0.0 --n_tests 1 --aggregated --B 299 --B_2 200 --total_n_tests 400 --mixing
```
```
python figures.py --name Higgs --n_components 2 --n 16384 --p_poisoning 0.0 --n_tests 1 --aggregated --B 299 --B_2 200 --total_n_tests 400 --no_asymp --log_time_scale --long_times 0.1 --wilson_intervals --mixing
```

All the other plots in the paper are obtained in a similar way to the ones presented, making slight modifications in the slurm files and the flags of the executions of `postprocessing.py` and `figures.py`.