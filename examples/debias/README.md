# Experiments for Debiased Distribution Compression

## Additional dependencies
In addition to the dependencies mentioned in `goodpoints/jax/README.md`, we require the following packages:
- `qpsolvers` for quadratic programming
- `h5py` for compression and storing binary data
- `wandb` for logging experiments
- `hydra-core` for managing configurations
- `scipy` for loading .mat files
- `pandas` for loading .csv files
- `matplotlib` for plotting
- `numpyro` for sampling gold points using HMC for the approximate MCMC experiment (optional if you download the gold points---see below)

## Quick demos
In `notebooks/demo` we provide a few self-contained notebooks that demonstrate how to use the algorithms in this package.

## Reproducing experiments in the paper
We use [Hydra](https://hydra.cc/docs/intro/) to manage configurations. The configuration files are in `src/config`. The general command to run a specific experiment is
```
python -m src.<exp>.run problem.<key>=<value> debias=<alg> compress=<alg> debias.<key>=<value> compress.<key>=<value>'
```
where `<exp>` is the experiment name, `<alg>` is a string for the algorithm name, and `<key>`, `<<value>` are key-value pairs for configuring the problem and the respective algorithms.

### Caching intermediate results
To store the intermediate results, we use a caching mechanism implemented in `goodpoints/jax/serial/`.
By default, we use the dummy cache from `serial/dummy_cache.py` which does not store any intermediate results.
We provide a MongoDB implementation of the cache in `serial/mongo_cache.py`.
To use it, you must first create a MongoDB database.
We recommend using MongoDB Atlas since no additional server setup is needed.
Once you created a MongoDB Atlas account, You can find your local IP by `curl ifconfig.me` and then add it to Atlas.
Next, copy the database URI (you can get by clicking "Get connection string") to `src/config/cache/mongodb.yaml` under the `uri` key and choose a database name (`db_name`) and collection name (`collection_name`) accordingly.

### Correcting for burn-in
First, download the datasets from Riabiz et al. 2021 from [this link](https://dataverse.harvard.edu/dataset.xhtml;jsessionid=715f1c20ad6ae81571b3943e5bfd?persistentId=doi%3A10.7910%2FDVN%2FMDKNWM&version=&q=&fileAccess=&fileTag=&fileSortField=&fileSortOrder=) (or using command `curl -L https://dataverse.harvard.edu/api/access/dataset/:persistentId\?persistentId\=doi:10.7910/DVN/MDKNWM  -v --output riabiz_data.zip`) and unzip them into a folder. Let's call the folder `riabiz_data`.
Next, copy the VK diagnostics meta files from `src/burnin/vk` to `riabiz_data/Goodwin` for each MCMC chain.
For example, for MALA samples, copy `src/burnin/vk/MALA.yaml` to `riabiz_data/Goodwin/MALA/meta.yaml`.

To run a single burn-in correction experiment using Stein Kernel Thinning, with `examples/debias` being the current working directory, run
```
python -m src.burnin.run cache=mongodb problem.num_sample=65536 debias=st compress=kt
```
which uses Stein Thinning (`st`) for debiasing and Kernel Thinning (`kt`) for compression to process 65536 samples.
If MongoDB is not available, you can use the `cache=dummy` option instead.
The default MCMC chain is preconditioned MALA and can be changed.
The full list of parameters can be found in `src/config/burnin.yaml`.

To reproduce all burn-in correction experiments from the paper, we provide the corresponding wandb sweep files in `src/sweep/burnin`. For more information about wandb sweeps, see [here](https://docs.wandb.ai/guides/sweeps).
For example, 
```
wandb sweep --project <project-name> src/sweep/burnin/skt.yaml
```
will lauch a sweep for the burn-in experiment using Stein kernel thinning. This command wil return a sweep ID, so using `wandb agent <sweep-ID>` will start an agent that runs the sweep.
We also provide a script `tmux_spawn.py` to spawn multiple agents to run the sweep in parallel in a tmux session.

### Correcting for approximate MCMC
The observation data for the approximate MCMC experiment in the paper is provided in `src/data/covertype.mat`.
We use `numpyro` to generate the gold points using HMC for evaluation purposes, and this will occur automatically when running the experiment for the first time.
Since we are generating 1 million samples using HMC, it can take many hours.
Alternatively, you can download the gold points from [this link](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/NGEM9H) and change `problem.gold.save_file` in `src/config/approx.yaml`to point to the downloaded file (or via the command line).

To run a single approximate MCMC correction experiment using Stein Kernel Thinning, with `examples/debias` being the current working directory, run
```
python -m src.approx.run cache=mongodb problem.post.num_point=65536 debias=st compress=kt
```
which uses Stein Thinning (`st`) for debiasing and Kernel Thinning (`kt`) for compression to process 65536 samples.
This will automatically generate approximate samples using SGFS; by default, SGFS will take 2^24 steps which can take a few hours. You may want to decrease `problem.mcmc.num_step` parameter.
The full list of parameters can be found in `src/config/approx.yaml`.


### Correcting for tempering
The tempering correction experiment uses the same dataset by Riabiz et al. 2021 as in the burn-in experiment.
Follow the same steps as in [Correcting for burn-in](#correcting-for-burn-in).
We need one additional preprocessing step before running the tempering experiment.
Copy the `src/temper/redup_data.py` to `riabiz_data/Cardiac/Tempered_posterior/seed_1` and run it, which creates a HDF5 file `points_scores_nowarmup.h5` in the same directory.
This step is necessary to add back the de-duplicated points to the dataset, and packing the data in an HDF5 binary format allows faster loading compared to CSV text files.

To run a single tempering correction experiment using Stein Kernel Thinning, with `examples/debias` being the current working directory, run
```
python -m src.temper.run cache=mongodb problem.num_sample=65536 debias=st compress=kt
```
which uses Stein Thinning (`st`) for debiasing and Kernel Thinning (`kt`) for compression to process 65536 samples.
The full list of parameters can be found in `src/config/temper.yaml`.


## Generating the figures in the paper
To generate the figures for each experiment in the paper, use the corresponding notebook in `notebook/plot`.
This assumes that all experiment sweeps have been run and the results are stored in the MongoDB database whose URI is specified in `src/config/cache/mongodb.yaml`.
