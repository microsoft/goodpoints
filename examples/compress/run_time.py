# from timeit 
import pathlib
import os
import os.path
import numpy as np
import numpy.random as npr

import timeit 
from functools import partial

from goodpoints import kt 
from goodpoints.herding import herding
from goodpoints.compress import compress, compresspp, symmetrize
from goodpoints.tictoc import tic, toc

# utils for getting arguments, generating samples, evaluating kernels, and mmds, getting filenames
from util_parse import get_args_from_terminal
from util_sample import compute_params_p, sample
from util_k_mmd import compute_params_k
from util_filenames import get_file_template



import joblib 

# When called as a script, call run_time
def main():
    return(run_time(get_args_from_terminal()))

# When called as a script
def run_time(args):    
    '''

    Load/save runtimes of various thinning algorithms for thinning an input of size 4**args.size 
    to 4**args.size / 2**args.m when using kt/herding, or 2**args.size when using compresspp based
    mmd based on multiple arguments. It does the following task:

    ---Run thinning algorithm and save the runtime to disk if args.rerun == True OR the run_time does not exist on disk, else load from disk
    ---results are saved in results_dir

    The function takes multiple input arguments as a dictionary that isnormally processed by 
    the parser function. One can directly specify / access the arguments as args.argument where 
    all the arguments are listed below:
    (all arguments are optional, and the default value is specified below)
    resultsfolder: (str; default coresets_folder) folder to save results (relative to where the script is run)
                                                  we specify this to results/run_time while calling this script in plot_compress_results notebook
    seed         : (int; default 123456789) seed for experiment
    rerun        : (bool; default False) whether to rerun the experiments
    size         : (int; default 2) sample set of size in log base 4, i.e., input size = 4**size, 
                                    and output size will be 2**size
    rep0         : (int; default 0) starting experiment id (useful for multiple repetitions)
    repn         : (int; default 1) number of experiment replication (useful for multiple repetitions)
    d            : (int; default 2) dimension of the points
    verbose      : (bool; default False) whether to print runtimes, and various helpful comments in the run of the code
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
                                       PPk like objects to save time while computing 
    thinalg      : (str; default compresspp) name of the thinning algorithm to be used for runtime experiments
                                     currently takes value in {compresspp, kt, herding}, can extend the list by changing code
                                     at "SUPPORT FOR NEW THINNING ALGORITHMS" mark below (in run_time function of run_time.py)                           
    '''
    
    pathlib.Path(args.resultsfolder).mkdir(parents=True, exist_ok=True)
    # intialize seed sequences
    seed_sequence = np.random.SeedSequence(entropy = args.seed)
    seed_sequence_children = seed_sequence.spawn(3)

    sample_seeds_set = seed_sequence_children[0].generate_state(1000)
    thin_seeds_set = seed_sequence_children[1].generate_state(1000)
    compress_seeds_set = seed_sequence_children[2].generate_state(1000)
    
    # save results
    results_dir = "results/run_time"
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    
    # compute parameters
    rt = np.zeros(args.repn)
    delta = 0.5
    
    d, params_p, var_k = compute_params_p(args)
    params_k_split, params_k_swap, split_kernel, swap_kernel = compute_params_k(d=args.d, var_k=var_k, 
                                                                use_krt_split=args.krt, name=args.setting)
    
    cpp_str = f", compress_alg = {args.compressalg}" if args.thinalg == "compresspp" else ""
    print(f"Running runtime experiments reps {np.arange(args.rep0, args.rep0+args.repn)} for d = {args.d}, size = {args.size}, alg = {args.thinalg} {cpp_str}")
    tic()

    for i, rep in enumerate(np.arange(args.rep0, args.rep0+args.repn)):
        # set seeds
        sample_seed = sample_seeds_set[rep]
        thin_seed = thin_seeds_set[rep]
        compress_seed = compress_seeds_set[rep]
        
        thin_rng = npr.default_rng(thin_seed) 
        halve_rng = npr.default_rng(compress_seed)
        
        # create filename
        filename = os.path.join(results_dir, f"{args.thinalg}{{}}_{args.prefix}_{args.setting}P_{params_k_swap['name']}k_d_{args.d}_size_{args.size}_rep_{rep}.pkl")
        
        # generate data
        X = sample(4**(args.size), params_p, seed = sample_seed)

        #### SUPPORT FOR NEW THINNING ALGORITHMS ####
        ## to add support for new thinning algorithms, add if/else logic here, and pass the appropriate flag in args.thinalg/ args.compressalg while calling this function
        ## currently this code adds the logic for kt, herding, compresspp
        if args.thinalg == "kt":
            filename = filename.format("")
            testNTimer = timeit.Timer(partial(kt.thin, X, args.size, split_kernel, swap_kernel, delta, seed= thin_rng))
            
        if args.thinalg == "herding":
            filename = filename.format("")
            testNTimer = timeit.Timer(partial(herding, X, args.size, swap_kernel))

        if args.thinalg == "compresspp":
            assert(args.g <= args.size) # for compress++ to work
            if args.compressalg == "kt":
                filename = filename.format(f"_g{args.g}_kt_symm{str(args.symm1)}")
                thin = partial(kt.thin, m=args.g , split_kernel = split_kernel, swap_kernel = swap_kernel, 
                           seed = thin_rng, delta= (delta*(2**args.g))/np.sqrt(args.size))
                if args.symm1:
                    halve = symmetrize(lambda x: kt.thin(X = x, m=1, split_kernel = split_kernel, swap_kernel = swap_kernel , seed = halve_rng, unique=True, delta = delta*len(x)/args.size))
                else:
                    halve = lambda x: kt.thin(X = x, m=1, split_kernel = split_kernel, swap_kernel = swap_kernel, seed = halve_rng, delta = delta*len(x)/args.size)
                
            if args.compressalg == "herding":
                filename = filename.format(f"_g{args.g}_herding_symm{str(args.symm1)}")
                thin = partial(herding, m = args.g, kernel = swap_kernel, unique = True)
                if args.symm1:
                    halve = symmetrize(partial(herding, m=1, kernel = swap_kernel, unique=True), seed = halve_rng)
                else:
                    halve = partial(herding, m=1, kernel = swap_kernel, unique=True)
            
            testNTimer = timeit.Timer(partial(compresspp, X, halve, thin, args.g))
        
        if args.rerun or not os.path.exists(filename):
            if args.verbose:
                print(f'(Re) Running experiment rep {rep} and saving results to {filename}')
            if args.thinalg == "kt" and args.size > 9:  # don't accidentally run kt for larger sizee
                rt[i] = 0.
            else: # run experiments
                rt[i] = testNTimer.timeit(number=1)
            joblib.dump(rt[i], filename)
        else: # load from file
            if args.verbose:
                print(f'Loading results from {filename}')
            rt[i] = joblib.load(filename)
            
        if args.verbose:
            print(f'Rep {rep} complete, runtime {rt[i]}')

    if args.verbose:
        print(f"saving results to {filename}") 
       
    toc()
    return(rt)

if __name__ == "__main__":
    main()
