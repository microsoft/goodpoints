import numpy as np
from goodpoints import ctt
from itertools import product
import util_classes
import util_sqMMD_estimators
import util_parser
import util_sampling
import util_tests
        
def wild_bootstrap_test(X1,X2,B,alpha,lam,seed,group_results_dict,args):
    """
    Non-asymptotic wild bootstrap tests
    For each block size in args.wb_block_size_list and number of pairs in args.wb_incomplete_list, 
    simultaneously computes and stores the corresponding wild bootstrap tests of level alpha with B permutations
    
    X1: 2D array of size (n_samples_1,d)
    X2: 2D array of size (n_samples_2,d)
    B: int, number of Rademacher variables used
    alpha: float, level of the tests
    lam: bandwidth (float)
    seed: random seed (numpy random number generator)
    group_results_dict: list of dictionaries. Each dictionary corresponds to a test, and contains one 
      util_classes.Group_Results object for each test group, where results are stored   
    args: arguments
    """
    if group_results_dict['block_wb'].compute_group:
        for size in args.wb_block_size_list:
            if group_results_dict['block_wb'].group_tests['bk'+str(size)].compute:
                util_sqMMD_estimators.block_sqMMD_Rademacher(X1,X2,B,lam,size,group_results_dict,seed=seed)
            
        # Get statistic value
        group_results_dict['block_wb'].set_statistic_value()
        # Reorder list of values for each estimator
        group_results_dict['block_wb'].sort_estimator_values()      
        # Compute threshold by looking at the right quantile of sqMMD_list
        group_results_dict['block_wb'].set_threshold(alpha)
        # Check if test statistic is above threshold to compute reject
        group_results_dict['block_wb'].set_reject()
        # Save group results
        group_results_dict['block_wb'].set_group_results()
        group_results_dict['block_wb'].save_results(args)
        # In principle, no need to save the TestResults objects; uncomment the next line if needed
        ## group_results_dict['block_wb'].save_objects(args)
    
    if group_results_dict['incomplete_wb'].compute_group:
        util_sqMMD_estimators.incomplete_sqMMD_Rademacher_subdiagonals(X1,X2,B,lam,group_results_dict,args,seed=seed)
        
        # Get statistic value
        group_results_dict['incomplete_wb'].set_statistic_value()
        # Reorder list of values for each estimator
        group_results_dict['incomplete_wb'].sort_estimator_values()      
        # Compute threshold by looking at the right quantile of sqMMD_list
        group_results_dict['incomplete_wb'].set_threshold(alpha)
        # Check if test statistic is above threshold to compute reject
        group_results_dict['incomplete_wb'].set_reject()
        # Save group results
        group_results_dict['incomplete_wb'].set_group_results()
        group_results_dict['incomplete_wb'].save_results(args)
        # In principle, no need to save the TestResults objects; uncomment the next line if needed
        ## group_results_dict['incomplete_wb'].save_objects(args)
            

def wild_bootstrap_aggregated(X1,X2,B,alpha,lam,seed,group_results_dict,args):
    """
    Aggregated non-asymptotic wild bootstrap test
    For each number of pairs in args.wb_incomplete_list, 
    simultaneously computes and stores the corresponding wild bootstrap tests of level alpha with B permutations
    
    X1: 2D array of size (n_samples_1,d)
    X2: 2D array of size (n_samples_2,d)
    B: int, number of Rademacher variables used
    alpha: float, level of the tests
    lam: vector of positive real-valued kernel bandwidths
    seed: random seed (numpy random number generator)
    group_results_dict: list of dictionaries. Each dictionary corresponds to a test, and contains one 
      util_classes.Group_Results object for each test group, where results are stored   
    args: arguments
    """
    if group_results_dict['incomplete_wb'].compute_group:
        util_sqMMD_estimators.incomplete_sqMMD_Rademacher_subdiagonals(X1,X2,B+args.B_2,lam,group_results_dict,args, seed=seed,aggregated=True)
        
        # Split into estimator_values and estimator_values_2
        group_results_dict['incomplete_wb'].split_tests()
        # Get statistic value
        group_results_dict['incomplete_wb'].set_statistic_value() 
        # Reorder list of values for each estimator
        group_results_dict['incomplete_wb'].sort_estimator_values()
        # Compute hat_u_alpha using the bisection method
        group_results_dict['incomplete_wb'].compute_hat_u_alpha(args.B_3, args.alpha)
        # Get reject
        group_results_dict['incomplete_wb'].set_reject()
        # Obtain computation time means
        group_results_dict['incomplete_wb'].set_total_times(sum_across_bw=False)
        # Get threshold for median bandwidth single test
        group_results_dict['incomplete_wb'].set_threshold(alpha)
        # Get reject for median bandwidth single test
        group_results_dict['incomplete_wb'].set_reject_median()
        # Save group results
        group_results_dict['incomplete_wb'].set_group_results()
        group_results_dict['incomplete_wb'].save_results(args)
        # In principle, no need to save the TestResults objects; uncomment the next line if needed
        ## group_results_dict['incomplete_wb'].save_objects(args)
        

def asymptotic_test(X1,X2,alpha,lam,seed,group_results_dict,args):
    """
    Asymptotic tests
    For each block size in args.wb_block_size_list and number of pairs in args.wb_incomplete_list, 
    simultaneously computes and stores the corresponding wild bootstrap tests of level alpha with B permutations
    
    X1: 2D array of size (n_samples_1,d)
    X2: 2D array of size (n_samples_2,d)
    B: int, number of Rademacher variables used
    alpha: float, level of the tests
    lam: bandwidth (float)
    seed: integer seed for random number generator
    group_results_dict: list of dictionaries. Each dictionary corresponds to a test, and contains one 
      util_classes.Group_Results object for each test group, where results are stored   
    args: arguments
    """
    if group_results_dict['block_asymp'].compute_group:
        # Compute test statistics for asymptotic block tests
        for size in args.asymptotic_block_size_list:
            compute_size = False
            for n_var in args.n_var:
                if group_results_dict['block_asymp'].group_tests['bk'+str(size)+'nv'+str(n_var)].compute:
                    compute_size = True
                if group_results_dict['block_asymp'].group_tests['bk'+str(size)+'nv'+str(n_var)+'b'].compute:
                    compute_size = True
                if group_results_dict['block_asymp'].group_tests['bk'+str(size)+'nv'+str(n_var)+'c'].compute:
                    compute_size = True
                
            if compute_size:     
                # Compute block statistic
                util_sqMMD_estimators.block_sqMMD(X1,X2,lam,alpha,size,group_results_dict,args)

                # Compute rearranged block statistic
                util_sqMMD_estimators.block_sqMMD_reordered(X1,X2,lam,alpha,size,group_results_dict,args,seed=seed)
            
    if group_results_dict['incomplete_asymp'].compute_group:
        print(f'Run asymp. incomplete')
        # Compute test statistics for asymptotic incomplete tests
        util_sqMMD_estimators.incomplete_sqMMD(X1,X2,lam,alpha,group_results_dict,args,seed=seed)
        
    if group_results_dict['block_asymp'].compute_group or group_results_dict['incomplete_asymp'].compute_group:
        # Compute sigma_2_sqd and set thresholds for asymptotic block and incomplete statistics correspondingly
        util_sqMMD_estimators.compute_sigma_2_sqd(X1,X2,lam,alpha,group_results_dict,args)
        
    if group_results_dict['block_asymp'].compute_group:
        # Check if test statistic is above threshold to compute reject
        group_results_dict['block_asymp'].set_reject()
        # Save group results
        group_results_dict['block_asymp'].set_group_results()
        group_results_dict['block_asymp'].save_results(args)
        # In principle, no need to save the TestResults object; uncomment the next line if needed
        ## group_results_dict['block_asymp'].save_objects(args)
        
    if group_results_dict['incomplete_asymp'].compute_group:
        # Check if test statistic is above threshold to compute reject
        group_results_dict['incomplete_asymp'].set_reject()
        # Save group results
        group_results_dict['incomplete_asymp'].set_group_results()
        group_results_dict['incomplete_asymp'].save_results(args)
        # In principle, no need to save the TestResults object; uncomment the next line if needed
        ## group_results_dict['incomplete_asymp'].save_objects(args)
                
        
def ctt_test(X1,X2,B,alpha,lam,seed,group_info,args):
    """
    Compress then Test (CTT)
    Computes and stores CTT tests of level alpha with B permutations
    
    X1: 2D array of size (n_samples_1,d)
    X2: 2D array of size (n_samples_2,d)
    B: int, number of permutations used
    alpha: float, level of the tests
    lam: bandwidth (float)
    seed: random seed (numpy random number generator)
    group_results_dict: list of dictionaries. Each dictionary corresponds to a test, and contains one 
      util_classes.Group_Results object for each test group, where results are stored
    args: arguments
    """
    # Create null and statistic seeds from input seed
    rng = np.random.default_rng(seed)
    ss_test = rng.bit_generator._seed_seq
    child_ss_test = ss_test.spawn(2)
    # Use integer seeds since they will be shared across multiple experimental 
    # settings
    null_seed = child_ss_test[0].generate_state(1)
    statistic_seed = child_ss_test[1].generate_state(1)
    
    if group_info.compute_group:
        print(f'Run CTT')
        for g in args.block_g_list:
            test_name = 't'+str(g)
            #ctt_test_g = group_info.group_tests['t'+str(g)]
            if group_info.compute[test_name]:
                group_info.group_tests[test_name] = ctt.ctt(X1,X2,g,B=B,s=args.s_permute,lam=lam,null_seed=null_seed,
                                                            alpha=alpha,statistic_seed=statistic_seed)
                # In principle, no need to save the TestResults object; uncomment the next line if needed
                ## group_info.group_tests[test_name].save(fname=group_info.fname[test_name])
                
        # Save group results
        group_info.set_group_results()            
        group_info.save_results()
        
def ctt_test_aggregated(X1,X2,B,B_2,B_3,alpha,lam,weights,seed,group_info,args):
    """
    Aggregated Compress then Test (ACTT)
    Computes and stores ACTT tests of level alpha with B permutations
    
    X1: 2D array of size (n_samples_1,d)
    X2: 2D array of size (n_samples_2,d)
    B: int, number of permutations used
    alpha: float, level of the tests
    lam: vector of positive real-valued kernel bandwidths
    seed: random seed (numpy random number generator)
    group_results_dict: list of dictionaries. Each dictionary corresponds to a test, and contains one 
      util_classes.Group_Results object for each test group, where results are stored
    args: arguments
    """
    # Create null and statistic seeds from input seed
    rng = np.random.default_rng(seed)
    ss_test = rng.bit_generator._seed_seq
    child_ss_test = ss_test.spawn(2)
    # Use integer seeds since they will be shared across multiple experimental 
    # settings
    null_seed = child_ss_test[0].generate_state(1)
    statistic_seed = child_ss_test[1].generate_state(1)
    
    if group_info.compute_group:
        print(f'Run ACTT')
        for g in args.block_g_list:
            test_name = 't'+str(g)
            if group_info.compute[test_name]:
                group_info.group_tests[test_name] = ctt.actt(X1,X2,g,B=B,B_2=B_2,B_3=B_3,s=args.s,
                                                             lam=lam,weights=weights,
                                                             null_seed=null_seed,statistic_seed=statistic_seed,
                                                             same_compression=not args.different_compression,alpha=alpha)
                # In principle, no need to save the TestResults object; uncomment the next line if needed
                ## group_info.group_tests[test_name].save(fname=group_info.fname[test_name])
                
        # Save group results
        group_info.set_group_results()            
        group_info.save_results()
        
def rff_test(X1,X2,B,alpha,lam,seed,group_info,args):
    """
    Random Fourier Features (RFF) Test
    Computes and stores RFF tests of level alpha with B permutations
    
    X1: 2D array of size (n_samples_1,d)
    X2: 2D array of size (n_samples_2,d)
    B: int, number of permutations used
    alpha: float, level of the tests
    lam: vector of positive real-valued kernel bandwidths
    seed: random seed (numpy random number generator)
    group_results_dict: list of dictionaries. Each dictionary corresponds to a test, and contains one 
      util_classes.Group_Results object for each test group, where results are stored
    args: arguments
    """
    # Create null and statistic seeds from input seed
    rng = np.random.default_rng(seed)
    ss_test = rng.bit_generator._seed_seq
    child_ss_test = ss_test.spawn(2)
    # Use integer seeds since they will be shared across multiple experimental 
    # settings
    null_seed = child_ss_test[0].generate_state(1)
    statistic_seed = child_ss_test[1].generate_state(1)
    
    if group_info.compute_group:
        print(f'Run RFF')
        for r in args.n_features_list:
            test_name = 'r'+str(r)
            #rff_test_r = group_info.group_tests['r'+str(r)]
            if group_info.compute[test_name]:
                group_info.group_tests[test_name] = ctt.rff(X1,X2,r,B=B,lam=lam,null_seed=null_seed,
                                                            statistic_seed=statistic_seed)
                # In principle, no need to save the TestResults object; uncomment the next line if needed
                ## group_info.group_tests[test_name].save(fname=group_info.fname[test_name])
                
        # Save group results
        group_info.set_group_results()            
        group_info.save_results()
            
        
def ctt_rff_test(X1,X2,B,alpha,lam,seed,group_info,args):
    """
    Low-Rank CTT Test based on Random Fourier Features (LR-CTT-RFF)
    Computes and stores LR-CTT-RFF tests of level alpha with B permutations
    
    X1: 2D array of size (n_samples_1,d)
    X2: 2D array of size (n_samples_2,d)
    B: int, number of permutations used
    alpha: float, level of the tests
    lam: vector of positive real-valued kernel bandwidths
    seed: random seed (numpy random number generator)
    group_results_dict: list of dictionaries. Each dictionary corresponds to a test, and contains one 
      util_classes.Group_Results object for each test group, where results are stored
    args: arguments
    """
    # Create null and statistic seeds from input seed
    rng = np.random.default_rng(seed)
    ss_test = rng.bit_generator._seed_seq
    child_ss_test = ss_test.spawn(2)
    # Use integer seeds since they will be shared across multiple experimental 
    # settings
    null_seed = child_ss_test[0].generate_state(1)
    statistic_seed = child_ss_test[1].generate_state(1)
    
    if group_info.compute_group:
        print(f'Run Low Rank CTT')
        # Consider each compression level
        for g in args.block_g_list_ctt_rff:
            # Consider each RFF feature count
            for r in args.n_features_list_ctt_rff:
                test_name = 'r'+str(g)+'_'+str(r)+'_'+str(args.s_rff)+'_'+str(args.s_permute)
                if group_info.compute[test_name]: 
                    group_info.group_tests[test_name] = ctt.lrctt(X1,X2,g,r,B=B,a=0,s=args.s_permute,lam=lam,
                                                                  use_permutations=True,null_seed=null_seed,
                                                                  statistic_seed=statistic_seed)
                    # In principle, no need to save the TestResults object; uncomment the next line if needed
                    ## group_info.group_tests[test_name].save(fname=group_info.fname[test_name])
                    
        # Save group results
        group_info.set_group_results()            
        group_info.save_results()

def set_args_for_task_id(args, task_id):
    """
    Sets arguments in args for each job
    
    args: arguments
    task_id: job number
    """
    grid = {
        'seed': [i for i in range(args.seed_0,args.seed_0+args.number_of_jobs)]
    }
    gridlist = list(dict(zip(grid.keys(), vals)) for vals in product(*grid.values()))
    assert task_id >= 1 and task_id <= len(gridlist), 'wrong task_id!'
    elem = gridlist[task_id - 1]
    for k, v in elem.items():
        setattr(args, k, v)

def run_single_test():
    """
    Runs a total of args.n_tests non-aggregated tests and stores results into group_results_dict
    """
    # Build list of estimators
    args.estimator_list = args.estimators['block_wb'] + args.estimators['incomplete_wb'] + args.estimators['block_asymp'] + args.estimators['incomplete_asymp'] + args.estimators['ctt'] + args.estimators['rff'] + args.estimators['ctt_rff']
    
    # Build list of test groups
    baseline_test_groups = ['block_wb', 'incomplete_wb', 'block_asymp', 'incomplete_asymp']
    ctt_test_groups = ['ctt', 'rff', 'ctt_rff']
    test_groups = baseline_test_groups + ctt_test_groups
    
    # Get directories to store the results of each test group
    resdir = util_classes.get_group_directories(args, test_groups)
            
    # Get file names to store the results
    groupname, testname = util_classes.get_test_file_names(args, test_groups, resdir, n_tests=args.n_tests, aggregated=False)
    
    fname, file_exists, fname_group, file_exists_group = util_classes.get_fname_and_file_exists(args,test_groups,resdir,groupname,testname)
        
    group_results_dict = dict()
        
    # Initialize base random number generator
    rng = np.random.default_rng(args.seed)
    # Create seed sequence for each test
    seed_seqs = rng.bit_generator._seed_seq.spawn(args.n_tests)
    for t in range(args.n_tests):
        print(f'Test number {t}')
        # From this test's seed sequence, construct two seeds,
        # one for randomness in constructing the test data and 
        # one for randomness in the test itself
        child_seed_seqs = seed_seqs[t].spawn(2)
        data_seed = child_seed_seqs[0]
        data_rng = np.random.default_rng(data_seed)
        # Create integer seed for test randomness
        test_seed = child_seed_seqs[1].generate_state(1)
        
        [X1, X2] = util_sampling.generate_samples(args,data_rng)
        
        for group in baseline_test_groups:
            group_results_dict[group] = util_classes.GroupResults(args.n_tests, args.B, fname_group[t][group], file_exists_group[t][group], 1, lam)
            
        for group in ctt_test_groups:
            group_results_dict[group] = ctt.GroupResults(args.n_tests, args.B, fname_group[t][group], file_exists_group[t][group], 1, lam)
            
        for group in test_groups:
            group_results_dict[group].set_compute_group(no_compute[group], recompute[group])
            group_results_dict[group].set_group_names(args.estimators[group], args.estimator_names[group], args.estimator_labels[group])
            group_results_dict[group].set_compute(no_compute[group], recompute[group], fname[t][group], file_exists[t][group])
            print(f'compute_group for {group}: {group_results_dict[group].compute_group}, file_exists for {group}: {group_results_dict[group].file_exists}')
            
        # Compute wild bootstrap (block and incomplete) tests
        wild_bootstrap_test(X1,X2,args.B,args.alpha,lam,test_seed,group_results_dict,args)
                
        # Compute asymptotic (block and incomplete) tests
        asymptotic_test(X1,X2,args.alpha,lam,test_seed,group_results_dict,args)
        
        # Compute CTT permutation tests
        ctt_test(X1,X2,args.B,args.alpha,lam,test_seed,group_results_dict['ctt'],args)
        
        # Compute random Fourier features permutation tests
        rff_test(X1,X2,args.B,args.alpha,lam,test_seed,group_results_dict['rff'],args)
        
        # Compute CTT random Fourier features permutation tests
        ctt_rff_test(X1,X2,args.B,args.alpha,lam,test_seed,group_results_dict['ctt_rff'],args)

def get_bandwidths(lam, args):
    """
    Given a bandwidth lam (typically the bandwidth given by the median criterion), compute the set of
    bandwidths to be used for aggregated tests (multiples/submultiples of lam)
    """
    bw_vec = np.zeros(args.n_bandwidths)
    for i in range(args.n_bandwidths):
        bw_vec[args.n_bandwidths-1-i] = lam/2**i
    weights_vec = np.ones(args.n_bandwidths)/args.n_bandwidths
    
    return bw_vec, weights_vec
        
def run_aggregated_test():
    """
    Runs a total of args.n_tests aggregated tests and stores results into group_results_dict
    """
    # Build list of estimators
    args.estimator_list = args.estimators['incomplete_wb'] + args.estimators['ctt'] + args.estimators['rff']
    
    # Build list of test groups
    baseline_test_groups = ['incomplete_wb']
    ctt_test_groups = ['ctt']
    test_groups = baseline_test_groups + ctt_test_groups
    
    # Get directories to store the results of each test group
    resdir = util_classes.get_group_directories(args, test_groups, aggregated=True) 
            
    # Get group and test names to store the results 
    groupname, testname = util_classes.get_test_file_names(args, test_groups, resdir, n_tests=args.n_tests, aggregated=True)
    
    # Get file names and whether they exist
    fname, file_exists, fname_group, file_exists_group = util_classes.get_fname_and_file_exists(args,test_groups,resdir,groupname,testname) 
    
    # Compute bandwidths and weights
    args.bw_vec, args.weights_vec = get_bandwidths(lam, args)
    
    # Initialize group_results_dict to be a list of empty objects
    group_results_dict = dict()

    # Initialize base random number generator
    rng = np.random.default_rng(args.seed)
    # Create seed sequence for each test
    seed_seqs = rng.bit_generator._seed_seq.spawn(args.n_tests)
    for t in range(args.n_tests):
        print(f'Test number {t}')
        # From this test's seed sequence, construct two seeds,
        # one for randomness in constructing the test data and 
        # one for randomness in the test itself
        child_seed_seqs = seed_seqs[t].spawn(2)
        data_seed = child_seed_seqs[0]
        data_rng = np.random.default_rng(data_seed)
        # Create integer seed for test randomness
        test_seed = child_seed_seqs[1].generate_state(1)
        
        [X1, X2] = util_sampling.generate_samples(args,data_rng)
        
        for group in baseline_test_groups:
            group_results_dict[group] = util_classes.GroupResults(args.n_tests, args.B, fname_group[t][group], file_exists_group[t][group], args.n_bandwidths, args.bw_vec, B_2 = args.B_2, weights_vec = args.weights_vec)
            
        for group in ctt_test_groups:
            group_results_dict[group] = ctt.GroupResults(args.n_tests, args.B, fname_group[t][group], file_exists_group[t][group], args.n_bandwidths, args.bw_vec, B_2 = args.B_2, weights_vec = args.weights_vec)
            
        for group in test_groups:
            group_results_dict[group].set_compute_group(no_compute[group], recompute[group]) 
            group_results_dict[group].set_group_names(args.estimators[group], args.estimator_names[group], args.estimator_labels[group])
            group_results_dict[group].set_compute(no_compute[group], recompute[group], fname[t][group], file_exists[t][group])
            print(f'compute_group for {group}: {group_results_dict[group].compute_group}, file_exists for {group}: {group_results_dict[group].file_exists}')
        
        # Compute incomplete wild bootstrap tests
        wild_bootstrap_aggregated(X1,X2,args.B,args.alpha,args.bw_vec,test_seed,group_results_dict,args)
        
        # Compute thinned permutation tests
        ctt_test_aggregated(X1,X2,args.B,args.B_2,args.B_3,args.alpha,args.bw_vec,args.weights_vec,test_seed,
                            group_results_dict['ctt'],args)

if __name__ == '__main__':
    
    # Get arguments
    args = util_parser.get_args_test()
    
    if args.name == 'gaussians':
        print(f'args.mean_diff: {args.mean_diff}')
    elif args.name == 'MNIST' or args.name == 'EMNIST':
        print(f'args.p_even: {args.p_even}. args.n: {args.n}')
    
    util_tests.get_attributes_tests(args)
    
    # Store no-compute choices for each test group
    no_compute = dict()
    no_compute['block_wb'] = args.no_block_wb
    no_compute['incomplete_wb'] = args.no_incomplete_wb
    no_compute['block_asymp'] = args.no_block_asymp
    no_compute['incomplete_asymp'] = args.no_incomplete_asymp
    no_compute['ctt'] = args.no_ctt
    no_compute['rff'] = args.no_rff
    no_compute['ctt_rff'] = args.no_ctt_rff
    
    # Store recompute choices for each test group
    recompute = dict()
    recompute['block_wb'] = args.recompute_block_wb
    recompute['incomplete_wb'] = args.recompute_incomplete_wb
    recompute['block_asymp'] = args.recompute_block_asymp
    recompute['incomplete_asymp'] = args.recompute_incomplete_asymp
    recompute['ctt'] = args.recompute_ctt
    recompute['rff'] = args.recompute_rff
    recompute['ctt_rff'] = args.recompute_ctt_rff
    test_groups = ['block_wb', 'incomplete_wb', 'block_asymp', 'incomplete_asymp', 'ctt', 'rff', 'ctt_rff']
    if args.recompute_all:
        for group in test_groups:
            recompute[group] = True
    
    # Reset default values depending on args.name
    if args.name == 'gaussians':
        args.d = 10
    
    if args.name == 'blobs':
        args.d = 2
    
    if args.name == 'MNIST' or args.name == 'EMNIST':
        args.d = 49
        
    if args.name == 'Higgs':
        args.d = args.n_components
        if args.p_poisoning > 0:
            args.mixing = True
        
    if args.name == 'sine':
        args.d = 10
    
    if args.task_id is not None:
        set_args_for_task_id(args, args.task_id)
        
    print(f'args.seed: {args.seed}')
    
    # Compute lam
    rng = np.random.default_rng(10)
    lam_computation_samples = np.minimum(args.n,512)
    [X1, X2] = util_sampling.generate_samples(args, rng)
    lam = util_sqMMD_estimators.median_criterion(X1[:lam_computation_samples,:],X2[:lam_computation_samples,:])
    print(f'lambda: {lam}')
    
    if args.aggregated:
        run_aggregated_test()
    else:
        run_single_test()
        
    print('This is the end')
