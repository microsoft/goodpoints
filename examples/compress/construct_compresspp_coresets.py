import numpy as np
from numpy.lib.arraysetops import unique
import numpy.random as npr
from argparse import ArgumentParser

import pathlib
import os
import os.path
import pickle as pkl

# goodpoints imports
from goodpoints import kt, compress
from goodpoints.herding import herding 
from goodpoints.tictoc import tic, toc # for timing blocks of code
from goodpoints.util import fprint  # for printing while flushing buffer

# utils for getting arguments, generating samples, evaluating kernels, and mmds, getting filenames
from util_parse import get_args_from_terminal
from util_sample import compute_params_p, sample, sample_string
from util_k_mmd import compute_params_k, squared_mmd
from util_filenames import get_file_template

from functools import partial

# When called as a script, call construct_compresspp_coresets
def main():
    return(construct_compresspp_coresets(get_args_from_terminal()))

def construct_compresspp_coresets(args):
    '''

    Generate/load/save compress++ coresets of size 2**args.size from inputs of 
    size 4**args.size and its mmd based on multiple arguments. It does the following tasks:

    1. Generate fresh coreset and save to disk if args.rerun == True OR the coresets do not exist on disk, else load from disk
    2. Compute mmd and save to disk if args.computemmd == True AND (mmd not on disk OR args.recomputemmd OR args.rerun), else load from disk
    3. Return the coreset if args.returncoreset == True
    4. Return the mmd if args.returncoreset == False and args.computemmd == True

    The function takes multiple input arguments as a dictionary that is normally processed by 
    the parser function. One can directly specify / access the arguments as args.argument where 
    all the arguments are listed below:
    (all arguments are optional, and the default value is specified below)

    resultsfolder: (str; default coresets_folder) folder to save results (relative to where the script is run)
    seed         : (int; default 123456789) seed for experiment
    rerun        : (bool; default False) whether to rerun the experiments
    size         : (int; default 2) sample set of size in log base 4, i.e., input size = 4**size, 
                                    and output size will be 2**size
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
    compressalg  : (str; default kt) name of the algorithm to be used as halve/thin in compress++
                                     currently takes value in {kt, herding}, can extend the list by changing code
                                     at "SUPPORT FOR NEW THINNING ALGORITHMS" mark in construct_compresspp_coresets.py  
    symm1        : (bool; default True) whether to symmetrize halve output in compress
    g            : (int; default 0) the oversampling parameter g for compress
    krt          : (bool; default False) whether to use root kernel when using kt in compress++; sqrt kernel
                                         needs to be computed by compute_params_k in util_k_mmd.py
    M            : (int; default None) number of mixture for diag mog in d=2, used only when setting = mog
                                       by compute_params_p (and in turn by compute_mog_params_p) in util_sample.py
    filename     : (str; default None) name for MCMC target, used only when setting = mcmc
                                       by compute_params_p (and in turn by compute_mcmc_params_p) in util_sample.py
                                       this setting would require samples to be preloaded in mcmcfolder
    mcmcfolder   : (str; default data) folder to load MCMC data from, and save some 
                                       PPk like objects to save time while computing mmd
    '''
    
    pathlib.Path(args.resultsfolder).mkdir(parents=True, exist_ok=True)
    assert(args.g <= args.size) # for compress++ to work oversampling parameter should not be larger than log_4 input size
    
    ####### seeds #######
    seed_sequence = np.random.SeedSequence(entropy = args.seed)
    seed_sequence_children = seed_sequence.spawn(3)

    sample_seeds_set = seed_sequence_children[0].generate_state(1000)
    thin_seeds_set = seed_sequence_children[1].generate_state(1000)
    compress_seeds_set = seed_sequence_children[2].generate_state(1000)

    # compute d, params_p and var_k for the setting
    d, params_p, var_k = compute_params_p(args)
    
    # define the kernels
    params_k_split, params_k_swap, split_kernel, swap_kernel = compute_params_k(d=d, var_k=var_k, 
                                                        use_krt_split=args.krt, name="gauss") 
    
    # Specify base failure probability for kernel thinning
    delta = 0.5
    # Each Compress Halve call applied to an input of length l uses KT( l^2 * halve_prob ) 
    halve_prob = delta / ( 4*(4**args.size)*(2**args.g)*( args.g + (2**args.g) * (args.size  - args.g) ) )
    ###halve_prob = 0 if size == g else delta * .5 / (4 * (4**size) * (4 ** g) * (size - g) ) ###
    # Each Compress++ Thin call uses KT( thin_prob )
    thin_prob = delta * args.g / (args.g + ( (2**args.g)*(args.size - args.g) ))
    ###thin_prob = .5 
        
    # mmd array
    mmds = np.zeros(args.repn)
    
    for i, rep in enumerate(np.arange(args.rep0, args.rep0+args.repn)):
        sample_seed = sample_seeds_set[rep]
        thin_seed = thin_seeds_set[rep]
        compress_seed = compress_seeds_set[rep]
        

        halve_rng = npr.default_rng(compress_seed)
        
        #### SUPPORT FOR NEW THINNING ALGORITHMS ####
        ## to add support for new thinning algorithms, add if/else logic here, and pass the appropriate flag in args.compressalg while calling this function
        ## currently this code adds the logic for kt.thin, and herding

        if args.compressalg == "kt":
            if args.symm1:
                halve = compress.symmetrize(lambda x: kt.thin(X = x, m=1, split_kernel = split_kernel, swap_kernel = swap_kernel, seed = halve_rng, unique=True, delta = halve_prob*(len(x)**2)))
            else:
                halve = lambda x: kt.thin(X = x, m=1, split_kernel = split_kernel, swap_kernel = swap_kernel , seed = halve_rng, delta = halve_prob*(len(x)**2))

        if args.compressalg == "herding":
            if args.symm1: 
                halve = compress.symmetrize(partial(herding, m=1, kernel = swap_kernel, unique=True), seed = halve_rng)
            else:
                halve = partial(herding, m=1, kernel = swap_kernel, unique=True)

        thin_rng = npr.default_rng(thin_seed)

        if args.compressalg == "kt":
            thin = partial(kt.thin, m=args.g, split_kernel = split_kernel, swap_kernel = swap_kernel, 
                           seed = thin_rng, delta = thin_prob)
        if args.compressalg == "herding":
            thin = partial(herding, m = args.g, kernel = swap_kernel, unique = True)
            
        prefix = "Compresspp" 
        # change prefix to accommodate for the two variations
        prefix += "-symm1-" if args.symm1 else "" # symm1 means whether we symmetrize the output in stage 1 compress (True) or not (False)

        
        file_template = get_file_template(args.resultsfolder, prefix, d, args.size, args.m, params_p, params_k_split, params_k_swap, 
                          delta=delta, sample_seed=sample_seed, thin_seed=thin_seed, 
                          compress_seed=compress_seed,
                          compressalg=args.compressalg, 
                          g=args.g,
                          )

        # Include replication number in filename
        filename = file_template.format('coresets', rep)

        tic()
        
        # local functions
        def generate_coreset_and_save():
            fprint(f"Running Compress++ experiment with template {filename}.....")
            print('(re) Generating coreset')
            X = sample(4**(args.size),params_p, seed = sample_seed)
            coreset = compress.compresspp(X, halve, thin, args.g)

            with open(filename, 'wb') as file: pkl.dump(coreset, file, protocol=pkl.HIGHEST_PROTOCOL)
            return(X, coreset)
        def fun_compute_mmds():
            print("computing mmd")
            if 'X' not in globals(): X = sample(4**(args.size),params_p, seed = sample_seed)
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
            return(mmd)
        
        if args.rerun or not os.path.exists(filename):
            X, coreset = generate_coreset_and_save()
        else:
            print(f"Loading coreset from {filename} (already present)")
            try:
                with open(filename, 'rb') as file: coreset = pkl.load(file)
            except:
                print(f"Error loading coreset from {filename}")                
                X, coreset = generate_coreset_and_save()

        # Include replication number in mmd filenames
        filename = file_template.format('mmd', rep)
        if args.computemmd:
            if not args.rerun and not args.recomputemmd and os.path.exists(filename):                
                try: 
                    with open(filename, 'rb') as file: mmd = pkl.load(file)
                    print(f"Loading mmd from {filename} (already present)")
                except: 
                    mmd = fun_compute_mmds()
            else:
                mmd = fun_compute_mmds()
                with open(filename, 'wb') as file: pkl.dump(mmd, file, protocol=pkl.HIGHEST_PROTOCOL)
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



    

    