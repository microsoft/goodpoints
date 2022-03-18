# file for

import numpy as np
import numpy.random as npr
import numpy.linalg as npl
from scipy.spatial.distance import pdist

from argparse import ArgumentParser
import pickle as pkl
import pathlib
import os
import os.path

# import kernel thinning
from goodpoints import kt # kt.thin is the main thinning function; kt.split and kt.swap are other important functions
from goodpoints.util import isnotebook # Check whether this file is being executed as a script or as a notebook
from goodpoints.util import fprint  # for printing while flushing buffer
from goodpoints.tictoc import tic, toc # for timing blocks of code


# utils for generating samples, evaluating kernels, and mmds
from util_sample import sample, compute_params_p, sample_string
from util_k_mmd import kernel_eval, compute_params_k, compute_power_kernel_params_k
from util_k_mmd import p_kernel, ppn_kernel, pp_kernel, pnpn_kernel, squared_mmd, get_combined_results_filename
from util_parse import init_parser
# for partial functions, to use kernel_eval for kernel
from functools import partial


def run_kernel_thinning_experiment(m, params_p, params_k_split, params_k_swap, rep_ids,
                                   thin_fun=kt.thin, thin_str="", 
                     delta=None, store_K=False,
                      sample_seed=1234567, thin_seed=9876543,
                      compute_mmds = True, compute_fun_diff = True,
                      rerun=False, results_dir="results_new",
                                  compute_last_mmd_only=True):
    """Runs kernel thinning experiment using samples from params_p for repetitions over rep_ids,
    saves coresets to disk, saves and returns mmd evaluations to disk mmd evaluation
    
    Args:
      m: Number of halving rounds (number of sample points n = 2^{2m})
      params_p: Dictionary of distribution parameters recognized by sample()
      params_k_split: Dictionary of kernel parameters recognized by kernel_eval()
      params_k_swap: Dictionary of kernel parameters recognized by kernel_eval()
      rep_ids: Which replication numbers of experiment to run; the replication
        number determines the seeds set for reproducibility
      delta: delta/(4^m) is the failure probability for
        adaptive threshold sequence;
      store_K: If False, runs O(nd) space version which does not store kernel
        matrix; if True, stores n x n kernel matrix
      sample_seed: (Optional) random seed is set to sample_seed + rep
        prior to generating input sample for replication rep
      thin_seed: (Optional) random seed is set to thin_seed + rep
        prior to running thinning for replication rep
      rerun: (Optional) If False and results have been previously saved to
        disk, load results from disk instead of rerunning experiment
      results_dir: (Optional) Directory in which results should be saved
      compute_mmds: (Optional) Whether to compute mmds of coresets (using params_k_swap)
      compute_fun_diff: (Optional) whether to compute (Pf - Pnf); default f = k(0, .), where k is defined via params_k_swap
      compute_last_mmd_only: (Optional) whether to compute mmd for entire range(m+1), or just m; to speed up computation for large m
    """
    # range of m for which mmd is evaluated
    mmd_eval_ms = range(m, m+1) if compute_last_mmd_only else range(m+1)
    
    # Create results directory if necessary
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)

    split_kernel = partial(kernel_eval, params_k=params_k_split)
    swap_kernel = partial(kernel_eval, params_k=params_k_swap)
    
    # Construct results filename template with placeholder for rep value
    d = params_p["d"]
    assert(d==params_k_split["d"])
    assert(d==params_k_swap["d"])
    
    sample_str = sample_string(params_p, sample_seed)
    split_kernel_str = "{}_var{:.3f}_seed{}".format(params_k_split["name"], params_k_split["var"], thin_seed)
    swap_kernel_str =  "{}_var{:.3f}".format(params_k_swap["name"], params_k_swap["var"])
    thresh_str = f"delta{delta}"
    
    
    file_template = os.path.join(results_dir, f"kt{thin_str}-coresets-{sample_str}-split{split_kernel_str}-swap{swap_kernel_str}-d{d}-m{m}-{thresh_str}-rep{{}}.pkl")
    
    # Create array to store MMD evaluations from P, and Sin

    mmds_p = np.zeros((m+1, len(rep_ids)))
    mmds_sin = np.zeros((m+1, len(rep_ids)))
    
    # when Pnmax is changed; name changes only for mmd file names
    if params_p["flip_Pnmax"]:
        mmd_p_sample_str =  sample_str + "_flip_Pnmax_"
    else:
        mmd_p_sample_str = sample_str
        
    mmd_p_file_template = os.path.join(results_dir, 
                                     f"kt{thin_str}-mmd-{mmd_p_sample_str}-split{split_kernel_str}-swap{swap_kernel_str}-d{d}-m{m}-{thresh_str}-rep{{}}.pkl")
    mmd_sin_file_template = os.path.join(results_dir, 
                                     f"kt{thin_str}-mmd-sin-{sample_str}-split{split_kernel_str}-swap{swap_kernel_str}-d{d}-m{m}-{thresh_str}-rep{{}}.pkl")
    

    fun_diff_p = np.zeros((m+1, len(rep_ids)))
    fun_diff_sin = np.zeros((m+1, len(rep_ids)))
    fun_diff_p_file_template = os.path.join(results_dir, 
                                     f"kt{thin_str}-fundiff-{mmd_p_sample_str}-split{split_kernel_str}-swap{swap_kernel_str}-d{d}-m{m}-{thresh_str}-rep{{}}.pkl")
    fun_diff_sin_template = os.path.join(results_dir, 
                                     f"kt{thin_str}-fundiff-sin-{sample_str}-split{split_kernel_str}-swap{swap_kernel_str}-d{d}-m{m}-{thresh_str}-rep{{}}.pkl")
    split_kernel = partial(kernel_eval, params_k=params_k_split)
    swap_kernel = partial(kernel_eval, params_k=params_k_swap)

    # Number of sample points
    n = int(2**(2*m))
    fprint(f"Running kernel thinning {thin_str} experiment with template {file_template}.....")
    tic()
    for r_i, rep in enumerate(rep_ids):
        tic()
        # Include replication number in filename
        filename = file_template.format(rep)
        mmd_p_filename = mmd_p_file_template.format(rep)
        mmd_sin_filename = mmd_sin_file_template.format(rep)
        fun_diff_p_filename = fun_diff_p_file_template.format(rep)
        fun_diff_sin_filename = fun_diff_sin_template.format(rep)
        # Generate matrix of input sample points
        X = sample(n, params_p, seed=sample_seed+rep)
        
        if not rerun and os.path.exists(filename):
            # Return previously saved results
            with open(filename, 'rb') as file:
                coresets = pkl.load(file)
        else:
            # Obtain sequence of thinned coresets
            print(f"Kernel Thinning {thin_str} rep {rep}...", flush=True)
            coresets = thin_fun(X, m, split_kernel, swap_kernel, delta=delta, seed=thin_seed+rep, store_K=store_K)

            # Save coresets to disk
            with open(filename, 'wb') as file:
                pkl.dump(coresets, file, protocol=pkl.HIGHEST_PROTOCOL)
            #toc()
            
        # Evaluate final coreset MMD
        if compute_mmds:
            if not rerun and os.path.exists(mmd_sin_filename):
                # Return previously saved results
                with open(mmd_sin_filename, 'rb') as file:
                    mmds_sin[:, r_i] = pkl.load(file)                
            else:
                print(f"Evaluating KT MMD_Sin for m = {mmd_eval_ms}", flush=True)
                tic()
                for j in mmd_eval_ms:
                    nj = int(2**j)
                    params_p_sin = dict()
                    params_p_sin["d"] = d
                    params_p_sin["name"] =  params_p["name"]+ "_sin"
                    params_p_sin["Pnmax"] = X
                    params_p_sin["saved_samples"] = False
                    mmds_sin[j, r_i] = np.sqrt(squared_mmd(params_k_swap, params_p_sin, X[coresets[:nj]]))
                toc()
                # Save MMD results to disk
                with open(mmd_sin_filename, 'wb') as file:
                    pkl.dump(mmds_sin[:, r_i], file, protocol=pkl.HIGHEST_PROTOCOL)
                
            if not rerun and os.path.exists(mmd_p_filename):
                # Return previously saved results
                with open(mmd_p_filename, 'rb') as file:
                    mmds_p[:, r_i] = pkl.load(file)              
            else:
                print(f"Evaluating KT MMD_P for m = {mmd_eval_ms}", flush=True)
                tic()
                for j in mmd_eval_ms:
                    nj = int(2**j)
                    if params_k_swap["name"] == "gauss":
                        mmds_p[j, r_i] = np.sqrt(
                            squared_mmd(params_k_swap, params_p, X[coresets[:nj]]))
                    else:
                        mmds_p[j, r_i] = mmds_sin[j, r_i]
                        
                toc()
                # Save MMD results to disk
                with open(mmd_p_filename, 'wb') as file:
                    pkl.dump(mmds_p[:, r_i], file, protocol=pkl.HIGHEST_PROTOCOL)
                    
                
        # Evaluate final coreset fun diff
        if compute_fun_diff:
            if not rerun and os.path.exists(fun_diff_sin_filename):
                # Return previously saved results
                with open(fun_diff_sin_filename, 'rb') as file:
                    fun_diff_sin[:, r_i] = pkl.load(file)             
            else:
                print(f"Evaluating KT fun diff results with P_in  for m = {mmd_eval_ms}", flush=True)
                tic()
                for j in mmd_eval_ms:
                    nj = int(2**j)
                    pin_fun = np.mean(kernel_eval(np.zeros((1, d)), X, params_k_swap))
                    pout_fun = np.mean(kernel_eval(np.zeros((1, d)), X[coresets[:nj]], params_k_swap))
                    fun_diff_sin[j, r_i] =  np.abs(pin_fun-pout_fun)
                toc()
                # Save results to disk
                with open(fun_diff_sin_filename, 'wb') as file:
                    pkl.dump(fun_diff_sin[:, r_i], file, protocol=pkl.HIGHEST_PROTOCOL)
                
            if not rerun and os.path.exists(fun_diff_p_filename):
                # Return previously saved results
                with open(fun_diff_p_filename, 'rb') as file:
                    fun_diff_p[:, r_i] = pkl.load(file)            
            else:
                print(f"Evaluating KT fun diff results with P for m = {mmd_eval_ms}", flush=True)
                tic()
                for j in mmd_eval_ms:
                    nj = int(2**j)
                    if params_k_swap["name"] == "gauss":
                        p_fun = p_kernel(np.zeros((1, d)), params_k=params_k_swap, params_p=params_p)[0] # fun is fixed to be k(0, .)
                    
                        pout_fun = np.mean(kernel_eval(np.zeros((1, d)), X[coresets[:nj]], params_k_swap))
                        fun_diff_p[j, r_i] = np.abs(p_fun-pout_fun)
                    else:
                        fun_diff_p[j, r_i] = fun_diff_sin[j, r_i]
                    
                toc()
                # Save results to disk
                with open(fun_diff_p_filename, 'wb') as file:
                    pkl.dump(fun_diff_p[:, r_i], file, protocol=pkl.HIGHEST_PROTOCOL)
                    
            
        toc()
    toc()
    if compute_mmds and compute_fun_diff:
        return(mmds_p, mmds_sin, fun_diff_p, fun_diff_sin)
    if compute_mmds:
        return(mmds_p, mmds_sin)
    else:
        return(mmds_p, mmds_sin, fun_diff_p, fun_diff_sin)
    

def run_standard_thinning_experiment(m, params_p, params_k_mmd, rep_ids, sample_seed=1234567, 
                      rerun=False, results_dir="results_new", compute_mmds=True, compute_fun_diff=True,
                                    compute_last_mmd_only=True):
    """Evaluates MMD of iid Monte Carlo draws, and saves it to disk 
    
    Args:
      m: Number of halving rounds (defines number of sample points via n = 2^{2m})
      params_p: Dictionary of distribution parameters recognized by sample()
      params_k_mmd: Dictionary of kernel parameters for MMD evaluation
      rep_ids: Which replication numbers of experiment to run; the replication
        number determines the seeds set for reproducibility
      sample_seed: (Optional) random seed is set to sample_seed + rep
        prior to generating input sample for replication rep
      rerun: (Optional) If False and results have been previously saved to
        disk, load results from disk instead of rerunning experiment
      results_dir: (Optional) Directory in which results should be saved
      compute_mmds: (Optional) Whether to compute mmds of coresets (using params_k_mmd)
      compute_fun_diff: (Optional) whether to compute (Pf - Pnf); default f = k(0, .), where k is defined via params_k_mmd
      compute_last_mmd_only: (Optional) whether to compute mmd for entire range(m+1), or just m; to speed up computation for large m
    """
    
    # range of m for which mmd is evaluated
    mmd_eval_ms = range(m, m+1) if compute_last_mmd_only else range(m+1)

    # Create results directory if necessary
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Create array to store MMD evaluations
    mmds_p = np.zeros((m+1, len(rep_ids)))
    mmds_sin = np.zeros((m+1, len(rep_ids)))
    fun_diff_p = np.zeros((m+1, len(rep_ids)))
    fun_diff_sin = np.zeros((m+1, len(rep_ids)))

    # Construct results filename template with placeholder for rep value
    d = params_p["d"]
    assert(d == params_k_mmd["d"])
    sample_str = sample_string(params_p, sample_seed)
    kernel_str = "{}_var{:.3f}".format(params_k_mmd["name"], params_k_mmd["var"])
    
    if params_p["flip_Pnmax"]:
        mmd_p_sample_str =  sample_str + "_flip_Pnmax_"
    else:
        mmd_p_sample_str = sample_str
    
    mmd_p_file_template = os.path.join(results_dir, f"mc-mmd-{mmd_p_sample_str}-{kernel_str}-d{d}-m{m}-rep{{}}.pkl")
    mmd_sin_file_template = os.path.join(results_dir, f"mc-mmd-sin-{sample_str}-{kernel_str}-d{d}-m{m}-rep{{}}.pkl")
    fun_diff_p_file_template = os.path.join(results_dir, f"mc-fundiff-{mmd_p_sample_str}-{kernel_str}-d{d}-m{m}-rep{{}}.pkl")
    fun_diff_sin_file_template = os.path.join(results_dir, f"mc-fundiff-sin-{sample_str}-{kernel_str}-d{d}-m{m}-rep{{}}.pkl")
    
    # Number of sample points
    n = int(2**(2*m))
    
    fprint(f"Running standard thinning experiment for m={m} with template {mmd_p_file_template}")
    tic()
    if params_p["saved_samples"]:
        rep_ids = np.zeros(len(rep_ids), dtype=int)
        # don't repeat any standard thinning experiment with MCMC data which has saved_samples = True; load rep=0 results always
        # such hack is useful since the data is fixed, and other rep_ids don't provide any different result
    for r_i, rep in enumerate(rep_ids):
        # Include replication number in filename
        fprint(f"Standard thinning {r_i} (rep={rep})")
        mmd_p_filename = mmd_p_file_template.format(rep)
        mmd_sin_filename = mmd_sin_file_template.format(rep)
        fun_diff_p_filename = fun_diff_p_file_template.format(rep)
        fun_diff_sin_filename = fun_diff_sin_file_template.format(rep)
        
        if compute_mmds:
            if not rerun and os.path.exists(mmd_sin_filename):
                # Return previously saved results
                with open(mmd_sin_filename, 'rb') as file:
                    mmds_sin[:, r_i] = pkl.load(file)
            else:
                X = sample(n, params_p, seed=sample_seed+rep)
                print(f"Evaluating Monte Carlo MMD_Sin for m = {mmd_eval_ms}", flush=True)
                tic()
                for j in mmd_eval_ms:
                    # Target coreset size
                    coreset_size = int(2**j)
                    input_size = int(coreset_size**2)
                    step_size = coreset_size
                    end = input_size

                    # redefining target p as distribution on Sin
                    params_p_sin = dict()
                    params_p_sin["d"] = d
                    params_p_sin["name"] =  params_p["name"]+"_sin"
                    params_p_sin["Pnmax"] = X
                    params_p_sin["saved_samples"] = False
                    mmds_sin[j, r_i] = np.sqrt(squared_mmd(params_k_mmd, params_p_sin, X[(step_size-1):end:step_size]))
                toc()
                # Save MMD results to disk
                with open(mmd_sin_filename, 'wb') as file:
                    pkl.dump(mmds_sin[:, r_i], file, protocol=pkl.HIGHEST_PROTOCOL)
                
            if not rerun and os.path.exists(mmd_p_filename):
                # Return previously saved results
                with open(mmd_p_filename, 'rb') as file:
                    mmds_p[:, r_i] = pkl.load(file)
            else:
                X = sample(n, params_p, seed=sample_seed+rep)
                tic()
                print(f"Evaluating Monte Carlo MMD_P for m = {mmd_eval_ms}", flush=True)
                for j in mmd_eval_ms:
                    if params_k_mmd["name"] == "gauss":
                        # Target coreset size
                        coreset_size = int(2**j)
                        input_size = int(coreset_size**2)
                        step_size = coreset_size
                        end = input_size
                        mmds_p[j, r_i] = np.sqrt(squared_mmd(params_k_mmd, params_p, X[(step_size-1):end:step_size]))
                    else:
                        mmds_p[j, r_i] = mmds_sin[j, r_i]
                toc()
                # Save MMD results to disk
                with open(mmd_p_filename, 'wb') as file:
                    pkl.dump(mmds_p[:, r_i], file, protocol=pkl.HIGHEST_PROTOCOL)

            
        if compute_fun_diff:
            if not rerun and os.path.exists(fun_diff_sin_filename):
                # Return previously saved results
                with open(fun_diff_sin_filename, 'rb') as file:
                    fun_diff_sin[:, r_i] = pkl.load(file)
            else:
                X = sample(n, params_p, seed=sample_seed+rep)
                tic()
                print(f"Evaluating Monte Carlo fun diff with Pin_f for m = {mmd_eval_ms}", flush=True)
                for j in mmd_eval_ms:
                    # Target coreset size
                    coreset_size = int(2**j)
                    input_size = int(coreset_size**2)
                    step_size = coreset_size
                    end = input_size
                    pin_fun = np.mean(kernel_eval(np.zeros((1, d)), X, params_k_mmd))
                    pout_fun = np.mean(kernel_eval(np.zeros((1, d)), X[(step_size-1):end:step_size], params_k_mmd))
                    fun_diff_sin[j, r_i] =  np.abs(pin_fun-pout_fun)
                toc()
                # Save MMD results to disk
                with open(fun_diff_sin_filename, 'wb') as file:
                    pkl.dump(fun_diff_sin[:, r_i], file, protocol=pkl.HIGHEST_PROTOCOL)
                
            if not rerun and os.path.exists(fun_diff_p_filename):
                # Return previously saved results
                with open(fun_diff_p_filename, 'rb') as file:
                    fun_diff_p[:, r_i] = pkl.load(file)
            else:
                X = sample(n, params_p, seed=sample_seed+rep)
                tic()
                print(f"Evaluating Monte Carlo fun diff Pf for m = {mmd_eval_ms}", flush=True)
                for j in mmd_eval_ms:
                    if params_k_mmd["name"] == "gauss":
                        # Target coreset size
                        coreset_size = int(2**j)
                        input_size = int(coreset_size**2)
                        step_size = coreset_size
                        end = input_size
                        p_fun = p_kernel(np.zeros((1, d)), params_k=params_k_mmd, params_p=params_p)[0] # fun is fixed to be k(0, .)
                        pout_fun = np.mean(kernel_eval(np.zeros((1, d)), X[(step_size-1):end:step_size], params_k_mmd))
                        fun_diff_p[j, r_i] = np.abs(p_fun-pout_fun)
                    else:
                        fun_diff_p[j, r_i] = fun_diff_sin[j, r_i]
                        
                toc()
                # Save MMD results to disk
                with open(fun_diff_p_filename, 'wb') as file:
                    pkl.dump(fun_diff_p[:, r_i], file, protocol=pkl.HIGHEST_PROTOCOL)
        
    toc()
    return(mmds_p, mmds_sin, fun_diff_p, fun_diff_sin)

def run_standard_thinning_experiment(m, params_p, params_k_mmd, rep_ids, sample_seed=1234567, 
                      rerun=False, results_dir="results_new", compute_mmds=True, compute_fun_diff=True,
                                    compute_last_mmd_only=True):
    """Evaluates MMD of iid Monte Carlo draws, and saves it to disk 
    
    Args:
      m: Number of halving rounds (defines number of sample points via n = 2^{2m})
      params_p: Dictionary of distribution parameters recognized by sample()
      params_k_mmd: Dictionary of kernel parameters for MMD evaluation
      rep_ids: Which replication numbers of experiment to run; the replication
        number determines the seeds set for reproducibility
      sample_seed: (Optional) random seed is set to sample_seed + rep
        prior to generating input sample for replication rep
      rerun: (Optional) If False and results have been previously saved to
        disk, load results from disk instead of rerunning experiment
      results_dir: (Optional) Directory in which results should be saved
      compute_mmds: (Optional) Whether to compute mmds of coresets (using params_k_mmd)
      compute_fun_diff: (Optional) whether to compute (Pf - Pnf); default f = k(0, .), where k is defined via params_k_mmd
      compute_last_mmd_only: (Optional) whether to compute mmd for entire range(m+1), or just m; to speed up computation for large m
    """
    
    # range of m for which mmd is evaluated
    mmd_eval_ms = range(m, m+1) if compute_last_mmd_only else range(m+1)

    # Create results directory if necessary
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Create array to store MMD evaluations
    mmds_p = np.zeros((m+1, len(rep_ids)))
    mmds_sin = np.zeros((m+1, len(rep_ids)))
    fun_diff_p = np.zeros((m+1, len(rep_ids)))
    fun_diff_sin = np.zeros((m+1, len(rep_ids)))

    # Construct results filename template with placeholder for rep value
    d = params_p["d"]
    assert(d == params_k_mmd["d"])
    sample_str = sample_string(params_p, sample_seed)
    kernel_str = "{}_var{:.3f}".format(params_k_mmd["name"], params_k_mmd["var"])
    
    if params_p["flip_Pnmax"]:
        mmd_p_sample_str =  sample_str + "_flip_Pnmax_"
    else:
        mmd_p_sample_str = sample_str
    
    mmd_p_file_template = os.path.join(results_dir, f"mc-mmd-{mmd_p_sample_str}-{kernel_str}-d{d}-m{m}-rep{{}}.pkl")
    mmd_sin_file_template = os.path.join(results_dir, f"mc-mmd-sin-{sample_str}-{kernel_str}-d{d}-m{m}-rep{{}}.pkl")
    fun_diff_p_file_template = os.path.join(results_dir, f"mc-fundiff-{mmd_p_sample_str}-{kernel_str}-d{d}-m{m}-rep{{}}.pkl")
    fun_diff_sin_file_template = os.path.join(results_dir, f"mc-fundiff-sin-{sample_str}-{kernel_str}-d{d}-m{m}-rep{{}}.pkl")
    
    # Number of sample points
    n = int(2**(2*m))
    
    fprint(f"Running standard thinning experiment for m={m} with template {mmd_p_file_template}")
    tic()
    if params_p["saved_samples"]:
        rep_ids = np.zeros(len(rep_ids), dtype=int)
        # don't repeat any standard thinning experiment with MCMC data which has saved_samples = True; load rep=0 results always
        # such hack is useful since the data is fixed, and other rep_ids don't provide any different result
    for r_i, rep in enumerate(rep_ids):
        # Include replication number in filename
        fprint(f"Standard thinning {r_i} (rep={rep})")
        mmd_p_filename = mmd_p_file_template.format(rep)
        mmd_sin_filename = mmd_sin_file_template.format(rep)
        fun_diff_p_filename = fun_diff_p_file_template.format(rep)
        fun_diff_sin_filename = fun_diff_sin_file_template.format(rep)
        
        if compute_mmds:
            if not rerun and os.path.exists(mmd_sin_filename):
                # Return previously saved results
                with open(mmd_sin_filename, 'rb') as file:
                    mmds_sin[:, r_i] = pkl.load(file)
            else:
                X = sample(n, params_p, seed=sample_seed+rep)
                print(f"Evaluating Monte Carlo MMD_Sin for m = {mmd_eval_ms}", flush=True)
                tic()
                for j in mmd_eval_ms:
                    # Target coreset size
                    coreset_size = int(2**j)
                    input_size = int(coreset_size**2)
                    step_size = coreset_size
                    end = input_size

                    # redefining target p as distribution on Sin
                    params_p_sin = dict()
                    params_p_sin["d"] = d
                    params_p_sin["name"] =  params_p["name"]+"_sin"
                    params_p_sin["Pnmax"] = X
                    params_p_sin["saved_samples"] = False
                    mmds_sin[j, r_i] = np.sqrt(squared_mmd(params_k_mmd, params_p_sin, X[(step_size-1):end:step_size]))
                toc()
                # Save MMD results to disk
                with open(mmd_sin_filename, 'wb') as file:
                    pkl.dump(mmds_sin[:, r_i], file, protocol=pkl.HIGHEST_PROTOCOL)
                
            if not rerun and os.path.exists(mmd_p_filename):
                # Return previously saved results
                with open(mmd_p_filename, 'rb') as file:
                    mmds_p[:, r_i] = pkl.load(file)
            else:
                X = sample(n, params_p, seed=sample_seed+rep)
                tic()
                print(f"Evaluating Monte Carlo MMD_P for m = {mmd_eval_ms}", flush=True)
                for j in mmd_eval_ms:
                    if params_k_mmd["name"] == "gauss":
                        # Target coreset size
                        coreset_size = int(2**j)
                        input_size = int(coreset_size**2)
                        step_size = coreset_size
                        end = input_size
                        mmds_p[j, r_i] = np.sqrt(squared_mmd(params_k_mmd, params_p, X[(step_size-1):end:step_size]))
                    else:
                        mmds_p[j, r_i] = mmds_sin[j, r_i]
                toc()
                # Save MMD results to disk
                with open(mmd_p_filename, 'wb') as file:
                    pkl.dump(mmds_p[:, r_i], file, protocol=pkl.HIGHEST_PROTOCOL)

            
        if compute_fun_diff:
            if not rerun and os.path.exists(fun_diff_sin_filename):
                # Return previously saved results
                with open(fun_diff_sin_filename, 'rb') as file:
                    fun_diff_sin[:, r_i] = pkl.load(file)
            else:
                X = sample(n, params_p, seed=sample_seed+rep)
                tic()
                print(f"Evaluating Monte Carlo fun diff with Pin_f for m = {mmd_eval_ms}", flush=True)
                for j in mmd_eval_ms:
                    # Target coreset size
                    coreset_size = int(2**j)
                    input_size = int(coreset_size**2)
                    step_size = coreset_size
                    end = input_size
                    pin_fun = np.mean(kernel_eval(np.zeros((1, d)), X, params_k_mmd))
                    pout_fun = np.mean(kernel_eval(np.zeros((1, d)), X[(step_size-1):end:step_size], params_k_mmd))
                    fun_diff_sin[j, r_i] =  np.abs(pin_fun-pout_fun)
                toc()
                # Save MMD results to disk
                with open(fun_diff_sin_filename, 'wb') as file:
                    pkl.dump(fun_diff_sin[:, r_i], file, protocol=pkl.HIGHEST_PROTOCOL)
                
            if not rerun and os.path.exists(fun_diff_p_filename):
                # Return previously saved results
                with open(fun_diff_p_filename, 'rb') as file:
                    fun_diff_p[:, r_i] = pkl.load(file)
            else:
                X = sample(n, params_p, seed=sample_seed+rep)
                #toc()
                tic()
                print(f"Evaluating Monte Carlo fun diff Pf for m = {mmd_eval_ms}", flush=True)
                for j in mmd_eval_ms:
                    if params_k_mmd["name"] == "gauss":
                        # Target coreset size
                        coreset_size = int(2**j)
                        input_size = int(coreset_size**2)
                        step_size = coreset_size
                        end = input_size
                        p_fun = p_kernel(np.zeros((1, d)), params_k=params_k_mmd, params_p=params_p)[0] # fun is fixed to be k(0, .)
                        pout_fun = np.mean(kernel_eval(np.zeros((1, d)), X[(step_size-1):end:step_size], params_k_mmd))
                        fun_diff_p[j, r_i] = np.abs(p_fun-pout_fun)
                    else:
                        fun_diff_p[j, r_i] = fun_diff_sin[j, r_i]
                        
                toc()
                # Save MMD results to disk
                with open(fun_diff_p_filename, 'wb') as file:
                    pkl.dump(fun_diff_p[:, r_i], file, protocol=pkl.HIGHEST_PROTOCOL)
        
    toc()
    return(mmds_p, mmds_sin, fun_diff_p, fun_diff_sin)


def run_iid_thinning_experiment(m, params_p, params_k_mmd, rep_ids, sample_seed=1234567, thin_seed= 9876543,
                      rerun=False, results_dir="results_new", compute_mmds=True, compute_fun_diff=True,
                                    compute_last_mmd_only=True):
    """Evaluates MMD of iid Monte Carlo draws, and saves it to disk 
    
    Args:
      m: Number of halving rounds (defines number of sample points via n = 2^{2m})
      params_p: Dictionary of distribution parameters recognized by sample()
      params_k_mmd: Dictionary of kernel parameters for MMD evaluation
      rep_ids: Which replication numbers of experiment to run; the replication
        number determines the seeds set for reproducibility
      sample_seed: (Optional) random seed is set to sample_seed + rep
        prior to generating input sample for replication rep
      rerun: (Optional) If False and results have been previously saved to
        disk, load results from disk instead of rerunning experiment
      results_dir: (Optional) Directory in which results should be saved
      compute_mmds: (Optional) Whether to compute mmds of coresets (using params_k_mmd)
      compute_fun_diff: (Optional) whether to compute (Pf - Pnf); default f = k(0, .), where k is defined via params_k_mmd
      compute_last_mmd_only: (Optional) whether to compute mmd for entire range(m+1), or just m; to speed up computation for large m
    """
    
    # range of m for which mmd is evaluated
    mmd_eval_ms = range(m, m+1) if compute_last_mmd_only else range(m+1)

    # Create results directory if necessary
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Create array to store MMD evaluations
    mmds_p = np.zeros((m+1, len(rep_ids)))
    mmds_sin = np.zeros((m+1, len(rep_ids)))
    fun_diff_p = np.zeros((m+1, len(rep_ids)))
    fun_diff_sin = np.zeros((m+1, len(rep_ids)))

    # Construct results filename template with placeholder for rep value
    d = params_p["d"]
    assert(d == params_k_mmd["d"])
    sample_str = sample_string(params_p, sample_seed)
    kernel_str = "{}_var{:.3f}".format(params_k_mmd["name"], params_k_mmd["var"])
    
    if params_p["flip_Pnmax"]:
        mmd_p_sample_str =  sample_str + "_flip_Pnmax_"
    else:
        mmd_p_sample_str = sample_str
        
    mmd_p_file_template = os.path.join(results_dir, f"mc-iid-mmd-{mmd_p_sample_str}-{kernel_str}-d{d}-m{m}-rep{{}}.pkl")
    mmd_sin_file_template = os.path.join(results_dir, f"mc-iid-mmd-sin-{sample_str}-{kernel_str}-d{d}-m{m}-rep{{}}.pkl")
    fun_diff_p_file_template = os.path.join(results_dir, f"mc-iid-fundiff-{mmd_p_sample_str}-{kernel_str}-d{d}-m{m}-rep{{}}.pkl")
    fun_diff_sin_file_template = os.path.join(results_dir, f"mc-iid-fundiff-sin-{sample_str}-{kernel_str}-d{d}-m{m}-rep{{}}.pkl")
    
    # Number of sample points
    n = int(2**(2*m))
    
    fprint(f"Running iid thinning experiment for m={m} with template {mmd_p_file_template}.....")
    tic()
    for r_i, rep in enumerate(rep_ids):
        # Include replication number in filename
        fprint(f"IID thinning {r_i} (rep={rep})")
        mmd_p_filename = mmd_p_file_template.format(rep)
        mmd_sin_filename = mmd_sin_file_template.format(rep)
        fun_diff_p_filename = fun_diff_p_file_template.format(rep)
        fun_diff_sin_filename = fun_diff_sin_file_template.format(rep)
        
        if compute_mmds:
            

            if not rerun and os.path.exists(mmd_sin_filename):
                # Return previously saved results
                with open(mmd_sin_filename, 'rb') as file:
                    mmds_sin[:, r_i] = pkl.load(file)
            else:
                X = sample(n, params_p, seed=sample_seed+rep)
                print(f"Evaluating Monte Carlo MMD_Sin for m = {mmd_eval_ms}", flush=True)
                tic()
                for j in mmd_eval_ms:
                    # Target coreset size
                    coreset_size = int(2**j)
                    input_size = int(coreset_size**2)
                    thin_idx = npr.default_rng(thin_seed+rep).choice(input_size, 
                                                                    coreset_size, replace=False)

                    # redefining target p as distribution on Sin
                    params_p_sin = dict()
                    params_p_sin["d"] = d
                    params_p_sin["name"] =  params_p["name"]+"_sin"
                    params_p_sin["Pnmax"] = X
                    params_p_sin["saved_samples"] = False
                    mmds_sin[j, r_i] = np.sqrt(squared_mmd(params_k_mmd, params_p_sin, 
                                                           X[thin_idx]))
                toc()
                # Save MMD results to disk
                with open(mmd_sin_filename, 'wb') as file:
                    pkl.dump(mmds_sin[:, r_i], file, protocol=pkl.HIGHEST_PROTOCOL)
            
            if not rerun and os.path.exists(mmd_p_filename):
                # Return previously saved results
                with open(mmd_p_filename, 'rb') as file:
                    mmds_p[:, r_i] = pkl.load(file)
            else:
                X = sample(n, params_p, seed=sample_seed+rep)
                tic()
                print(f"Evaluating Monte Carlo MMD_P for m = {mmd_eval_ms}", flush=True)
                for j in mmd_eval_ms:
                    if params_k_mmd["name"] == "gauss":
                        # Target coreset size
                        coreset_size = int(2**j)
                        input_size = int(coreset_size**2)
                        thin_idx = npr.default_rng(thin_seed+rep).choice(input_size, 
                                                                        coreset_size, replace=False)
                        mmds_p[j, r_i] = np.sqrt(squared_mmd(params_k_mmd, params_p, 
                                                             X[thin_idx]))
                    else:
                        mmds_p[j, r_i] = mmds_sin[j, r_i]
                toc()
                # Save MMD results to disk
                with open(mmd_p_filename, 'wb') as file:
                    pkl.dump(mmds_p[:, r_i], file, protocol=pkl.HIGHEST_PROTOCOL)
                
        if compute_fun_diff:
            

            if not rerun and os.path.exists(fun_diff_sin_filename):
                # Return previously saved results
                with open(fun_diff_sin_filename, 'rb') as file:
                    fun_diff_sin[:, r_i] = pkl.load(file)
            else:
                X = sample(n, params_p, seed=sample_seed+rep)
                tic()
                print(f"Evaluating Monte Carlo fun diff with Pin_f for m = {mmd_eval_ms}", flush=True)
                for j in mmd_eval_ms:
                    # Target coreset size
                    coreset_size = int(2**j)
                    input_size = int(coreset_size**2)
                    thin_idx = npr.default_rng(thin_seed+rep).choice(input_size, 
                                                                    coreset_size, replace=False)
                    pin_fun = np.mean(kernel_eval(np.zeros((1, d)), X, params_k_mmd))
                    pout_fun = np.mean(kernel_eval(np.zeros((1, d)), X[thin_idx], params_k_mmd))
                    fun_diff_sin[j, r_i] =  np.abs(pin_fun-pout_fun)
                toc()
                # Save MMD results to disk
                with open(fun_diff_sin_filename, 'wb') as file:
                    pkl.dump(fun_diff_sin[:, r_i], file, protocol=pkl.HIGHEST_PROTOCOL)
                
            if not rerun and os.path.exists(fun_diff_p_filename):
                # Return previously saved results
                with open(fun_diff_p_filename, 'rb') as file:
                    fun_diff_p[:, r_i] = pkl.load(file)
            else:
                X = sample(n, params_p, seed=sample_seed+rep)
                tic()
                print(f"Evaluating Monte Carlo fun diff Pf for m = {mmd_eval_ms}", flush=True)
                for j in mmd_eval_ms:
                    if params_k_mmd["name"] == "gauss":
                        # Target coreset size
                        coreset_size = int(2**j)
                        input_size = int(coreset_size**2)
                        thin_idx = npr.default_rng(thin_seed+rep).choice(input_size, 
                                                                        coreset_size, replace=False)
                        p_fun = p_kernel(np.zeros((1, d)), params_k=params_k_mmd, params_p=params_p)[0] # fun is fixed to be k(0, .)
                        pout_fun = np.mean(kernel_eval(np.zeros((1, d)), X[thin_idx], params_k_mmd))
                        fun_diff_p[j, r_i] = np.abs(p_fun-pout_fun)
                    else:
                        fun_diff_p[j, r_i] = fun_diff_sin[j, r_i]
                toc()
                # Save MMD results to disk
                with open(fun_diff_p_filename, 'wb') as file:
                    pkl.dump(fun_diff_p[:, r_i], file, protocol=pkl.HIGHEST_PROTOCOL)
    toc()
    return(mmds_p, mmds_sin, fun_diff_p, fun_diff_sin)