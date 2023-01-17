import numpy as np
import os
import glob
import argparse
import pickle
import util_classes
import util_tests


def get_results(args):
    resdir = util_classes.get_group_directories(args, test_groups, aggregated=args.aggregated)
            
    groupname, testname = util_classes.get_test_file_names(args, test_groups, resdir, save=False, aggregated=args.aggregated)
    
    joint_resdir = util_classes.get_joint_group_directories(args, test_groups, aggregated=args.aggregated)
    
    joint_filename = util_classes.get_joint_filename(args, test_groups, aggregated=args.aggregated)
    
    joint_fname = util_classes.get_fname_joint(args, test_groups, joint_resdir, joint_filename)
    
    fnames = dict()
    file_exists = dict()
    joint_group_results_dict = dict()
    
    for group in test_groups:
        if not no_compute[group]:
            name = os.path.join(resdir[group],groupname[group])
            print(f'File to retrieve for {group}: {name}')
            fnames[group] = glob.glob(name)
            assert len(fnames[group]) > 0, 'no files! ({})'.format(name)
            if len(fnames[group]) and not args.interactive:
                file_exists[group] = True
                print(f'{len(fnames[group])} files exist')
            else:
                file_exists[group] = False

            total_n_tests_found = len(fnames[group])
            #print(f'{args.n_tests}x{len(fnames[group])} = {total_n_tests_found} tests found. {args.total_n_tests} needed.')
            print(f'{total_n_tests_found} tests found. {args.total_n_tests} needed.')

            #print(f'args.aggregated: {args.aggregated}. args.n_bandwidths: {args.n_bandwidths}')
            if args.aggregated:
                joint_group_results_dict[group] = util_classes.JointGroupResults(joint_fname[group], args.total_n_tests, args.B, args.n_bandwidths, file_exists[group])
            else:    
                joint_group_results_dict[group] = util_classes.JointGroupResults(joint_fname[group], args.total_n_tests, args.B, 1, file_exists[group])

            joint_group_results_dict[group].set_compute(no_compute[group])

            #print(f'fnames[group]: {fnames[group]}')
            for i, fname in enumerate(fnames[group]):
                if i >= args.total_n_tests:
                    print(f'{i}/{args.total_n_tests} processed, all remaining tests are disregarded.')
                    break
                res = pickle.load(open(fname, 'rb'))

                #print(f'joint_group_results_dict[group].group_names: {joint_group_results_dict[group].group_names}')
                if joint_group_results_dict[group].group_names is None:
                    group_names = res['group_names']
                    full_group_names = res['full_group_names']
                    group_labels = res['group_labels']
                    joint_group_results_dict[group].set_group_names(group_names, full_group_names, group_labels)
                    #print('set_group_names done')
                       
                if joint_group_results_dict[group].bw is None:
                    joint_group_results_dict[group].set_bandwidth(res['bw'])

                joint_group_results_dict[group].update_attributes(res)

            joint_group_results_dict[group].compute_info()

            joint_group_results_dict[group].print_info()

            joint_group_results_dict[group].save()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMD tests results')
    
    #General arguments
    parser.add_argument('--name', default='gaussians', help='experiment name')
    parser.add_argument('--n', type=int, default=262144, help='number of samples')
    parser.add_argument('--d', type=int, default=49, help='dimension')
    parser.add_argument('--B', type=int, default=39, help='number of permutations/Rademacher variables used')
    parser.add_argument('--alpha', type=float, default=0.05, help='level of the test')
    parser.add_argument('--n_tests', type=int, default=1, help='number of tests')
    parser.add_argument('--total_n_tests', type=int, default=200, help='number of tests')
    parser.add_argument('--s', type=int, default=16, help='number of bins for CTT')
    parser.add_argument('--s_rff', type=int, default=16, help='number of bins for Low Rank CTT compression')
    parser.add_argument('--s_permute', type=int, default=16, help='number of bins for Low Rank CTT permutation')
    
    #Argument for gaussians
    parser.add_argument('--mean_diff', type=float, default=0.024, help='mean difference (for gaussians)')
    
    #Arguments for blobs
    parser.add_argument('--grid_size', type=int, default=3, help='dimension of the grid of the distribution (for blobs)')
    parser.add_argument('--epsilon', type=float, default=2, help='covariance eigenvalue (for blobs)')
    
    #Arguments for MNIST and EMNIST
    parser.add_argument('--p_even', type=float, default=0.49, help='joint probability of all even digits (for MNIST and EMNIST)')
    
    #Arguments for Higgs
    parser.add_argument('--mixing', action='store_true', help='if passed use test mixing between classes')
    parser.add_argument('--null', action='store_true', help='if passed use null hypothesis, else use alternative')
    parser.add_argument('--n_components', type=int, default=4, help='number of dimensions to use')
    parser.add_argument('--p_poisoning', type=float, default=0.9, help='poisoning probability of class 1 with class 0')
    
    #Argument for sine
    parser.add_argument('--omega', type=float, default=0.05, help='sine frequency')
    
    #Arguments for aggregated tests
    parser.add_argument('--aggregated', action='store_true', help='if True, compute aggregated test, else compute single test')
    parser.add_argument('--n_bandwidths', type=int, default=5, help='number of bandwidths used in the aggregated test')
    parser.add_argument('--B_2', type=int, default=100, help='number of permutations used for Monte Carlo estimation in agg.')
    parser.add_argument('--B_3', type=int, default=15, help='number of bisection iterations for aggregated test')
    parser.add_argument('--different_compression', action='store_true', help='if passed use different compression calls for different bandwidths')
    
    #Arguments to avoid postprocessing specific test groups
    parser.add_argument('--no_block_wb', action='store_true', help='if True, do not compute block_wb tests')
    parser.add_argument('--no_incomplete_wb', action='store_true', help='if True, do not compute incomplete_wb tests')
    parser.add_argument('--no_block_asymp', action='store_true', help='if True, do not compute block_asymp tests')
    parser.add_argument('--no_incomplete_asymp', action='store_true', help='if True, do not compute incomplete_asymp tests')
    parser.add_argument('--no_ctt', action='store_true', help='if True, do not compute ctt tests')
    parser.add_argument('--no_rff', action='store_true', help='if True, do not compute rff tests')
    parser.add_argument('--no_ctt_rff', action='store_true', help='if True, do not compute ctt_rff tests')
    args = parser.parse_args()
    
    #Reset default values depending on args.name
    if args.name == 'gaussians':
        args.d = 10
        
    if args.name == 'blobs':
        args.d = 2
        
    if args.name == 'Higgs':
        args.d = args.n_components
        if args.p_poisoning > 0:
            args.mixing = True
        
    if args.name == 'sine':
        args.d = 10
    
    args.interactive = False
    
    util_tests.get_attributes_tests(args)
    
    #Store no-compute choices for each test group
    no_compute = dict()
    no_compute['block_wb'] = args.no_block_wb
    no_compute['incomplete_wb'] = args.no_incomplete_wb
    no_compute['block_asymp'] = args.no_block_asymp
    no_compute['incomplete_asymp'] = args.no_incomplete_asymp
    no_compute['ctt'] = args.no_ctt
    no_compute['rff'] = args.no_rff
    no_compute['ctt_rff'] = args.no_ctt_rff
    
    if not args.aggregated:
        #Build list of test groups
        test_groups = ['block_wb', 'incomplete_wb', 'block_asymp', 'incomplete_asymp', 'ctt', 'rff', 'ctt_rff']
        args.estimator_list = test_groups
    else:
        #Build list of test groups
        test_groups = ['incomplete_wb', 'ctt']
        args.estimator_list = test_groups
        
    get_results(args)
    