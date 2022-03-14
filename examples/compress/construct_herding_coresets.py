import numpy as np
from argparse import ArgumentParser

import pathlib
import os
import os.path
import pickle as pkl
import numpy.random as npr

# goodpoints imports
from goodpoints.herding import herding 
from goodpoints.tictoc import tic, toc # for timing blocks of code
from goodpoints.util import fprint  # for printing while flushing buffer

# utils for getting arguments, generating samples, evaluating kernels, and mmds, getting filenames
from util_parse import get_args_from_terminal
from util_sample import compute_params_p, sample, sample_string
from util_k_mmd import compute_params_k, squared_mmd
from util_filenames import get_file_template

from functools import partial

# When called as a script, call construct_herding_coresets
def main():
    return(construct_herding_coresets(get_args_from_terminal()))

def construct_herding_coresets(args):
    '''

    Generate/load/save herding coresets size 4**args.size/2**args.m from inputs of 
    size 4**args.size and load/save its mmd based on multiple arguments. 
    It does the following tasks:

    1. Generate fresh coreset and save to disk if args.rerun == True OR the coresets do not exist on disk, else load from disk
    2. Compute mmd and save to disk if args.computemmd == True AND (mmd not on disk OR args.recomputemmd OR args.rerun), else load from disk
    3. Return the coreset if args.returncoreset == True
    4. Return the mmd if args.returncoreset == False and args.computemmd == True

    The function takes multiple input arguments as a dictionary that isnormally processed by 
    the parser function. One can directly specify / access the arguments as args.argument where 
    all the arguments are listed below:
    (all arguments are optional, and the default value is specified below)

    resultsfolder: (str; default coresets_folder) folder to save results (relative to where the script is run)
    seed         : (int; default 123456789) seed for experiment
    rerun        : (bool; default False) whether to rerun the experiments
    size         : (int; default 2) sample set of size in log base 4, i.e., input size = 4**size, 
                                    and output size will be 4**size/2**m
    rep0         : (int; default 0) starting experiment id (useful for multiple repetitions)
    repn         : (int; default 1) number of experiment replication (useful for multiple repetitions)
    d            : (int; default 2) dimension of the points
    returncoreset: (bool; default False) whether to return coresets
    verbose      : (bool; default False) whether to print coresets and mmds
    computemmd   : (bool; default False) whether to compute mmd results; if exist on disk load from it, 
                                         else compte and save to disk; return them if returncoreset is False
    recomputemmd : (bool; default False) whether to re-compute mmd results and save on disk (refreshes disk results)
    setting      : (str; default gauss) name of the target distribution P for running the experiments 
                                      (needs to be supported by functions: compute_params_p and sample function in util_sample.py; 
                                       and p_kernel, ppn_kernel and pp_kernel functions in util_k_mmd.py)
    m            : (int; default 2) number of thinning rounds; output size = 4**size / 2**m
    M            : (int; default None) number of mixture for diag mog in d=2, used only when setting = mog
                                       by compute_params_p (and in turn by compute_mog_params_p) in util_sample.py
    filename     : (str; default None) name for MCMC target, used only when setting = mcmc
                                       by compute_params_p (and in turn by compute_mcmc_params_p) in util_sample.py
                                       this setting would require samples to be preloaded in mcmcfolder
    mcmcfolder   : (str; default data) folder to load MCMC data from, and save some 
                                       PPk like objects to save time while computing mmd
    '''
    pathlib.Path(args.resultsfolder).mkdir(parents=True, exist_ok=True)
    ####### seeds ####### 

    seed_sequence = np.random.SeedSequence(entropy = args.seed)
    seed_sequence_children = seed_sequence.spawn(3)
    sample_seeds_set = seed_sequence_children[0].generate_state(1000)

    # compute d, params_p and var_k for the setting
    d, params_p, var_k = compute_params_p(args)
    
    # define the kernels
    params_k_split, params_k_swap, split_kernel, swap_kernel = compute_params_k(d=d, var_k=var_k, 
                                                        use_krt_split=args.krt, name="gauss") 
    ### we will only use swap_kernel for kernel herding
    params_k_herding = params_k_swap
    herding_kernel = swap_kernel
    
    ### other experiments parameters
    reps = range(20) if args is None else np.arange(args.rep0, args.rep0+args.repn)

    # mmd 
    mmds = np.zeros(args.repn)
    
    for i, rep in enumerate(np.arange(args.rep0, args.rep0+args.repn)):
        sample_seed = sample_seeds_set[rep]

        prefix = "Herd"
        
        file_template = get_file_template(args.resultsfolder, prefix, d, args.size, args.m, params_p, params_k_split=None, params_k_swap=params_k_herding,
                         delta=None, 
                         sample_seed=sample_seed, 
                         thin_seed=None, 
                         compress_seed=None,
                         compressalg=None, 
                         g=None,
                          )

        # Include replication number in filename
        tic()
        filename = file_template.format("coresets", rep)
        
        if args.rerun or not os.path.exists(filename):
            fprint(f"Running herding experiment with template {filename}.....")
            print('(re) Generating coreset')
            X = sample(4**(args.size),params_p, seed = sample_seed)
            
            # coreset_size = int(4**(args.size)/2**(args.m))
            coreset = herding(X, args.m, kernel = herding_kernel) 
        
            with open(filename, 'wb') as file:
                pkl.dump(coreset, file, protocol=pkl.HIGHEST_PROTOCOL)
        else:
            print(f"Loading coreset from {filename} (already present)")
            with open(filename, 'rb') as file:
                coreset = pkl.load(file)

        # Include replication number in mmd filenames
        filename = file_template.format('mmd', rep)
        if args.computemmd:
            if not args.rerun and not args.recomputemmd and os.path.exists(filename):                
                print(f"Loading mmd from {filename} (already present)")
                with open(filename, 'rb') as file:
                    mmd = pkl.load(file)
            else:
                print("computing mmd")
                if 'X' not in locals(): X = sample(4**(args.size),params_p, seed = sample_seed)
                if params_p["saved_samples"]:# if MCMC data compute MMD(Sin)
                    params_p_eval = dict()
                    params_p_eval["data_dir"] = params_p["data_dir"]
                    params_p_eval["d"] = d
                    params_p_eval["name"] =  params_p["name"]+ "_sin"
                    params_p_eval["Pnmax"] = X
                    params_p_eval["saved_samples"] = False
                else:
                    params_p_eval = params_p
                mmd = np.sqrt(squared_mmd(params_k=params_k_swap,  params_p=params_p_eval, xn=X[coreset]))
                with open(filename, 'wb') as file:
                    pkl.dump(mmd, file, protocol=pkl.HIGHEST_PROTOCOL)
            mmds[i] = mmd
        toc()
        if args.verbose:
            print(f"CORESET: {coreset}")
            print(f"mmds: {mmds}")
    if args.returncoreset:
        if 'X' not in globals(): X = sample(4**(args.size),params_p, seed = sample_seed)
        return(X, X[coreset])
    else:
        if args.computemmd:
            return(mmds)

    
if __name__ == "__main__":
   main()
    
