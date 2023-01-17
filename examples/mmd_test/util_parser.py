import argparse

def get_args_test():
    parser = argparse.ArgumentParser(description='MMD tests')
    
    #General arguments
    parser.add_argument('--name', type=str, default='MNIST', help='experiment name')
    parser.add_argument('--use_grid', action='store_true', help='use grid')
    parser.add_argument('--n', type=int, default=1024, help='number of samples')
    parser.add_argument('--d', type=int, default=10, help='dimension')
    parser.add_argument('--B', type=int, default=39, help='number of permutations/Rademacher variables used')
    parser.add_argument('--seed', type=int, default=38, help='seed')
    parser.add_argument('--seed_0', type=int, default=0, help='when running sweep jobs, first value of the seed')
    parser.add_argument('--alpha', type=float, default=0.05, help='level of the test')
    parser.add_argument('--n_tests', type=int, default=1, help='number of tests')
    parser.add_argument('--krt', type=int, default=0, help='1 if use_krt_split, 0 otherwise')
    parser.add_argument('--interactive', action='store_true', help='interactive, i.e. do not save results')
    parser.add_argument('--task_id', type=int, default=None, help='task id for sweep jobs')
    parser.add_argument('--number_of_jobs', type=int, default=200, help='number of sweep jobs')
    parser.add_argument('--aggregated', action='store_true', help='if passed compute aggregated test, else compute single test')
    parser.add_argument('--no_cython_compress', action='store_true', help='if passed do not use cython compress')
    parser.add_argument('--s', type=int, default=16, help='number of bins for CTT')
    parser.add_argument('--s_rff', type=int, default=16, help='number of bins for Low Rank CTT compression')
    parser.add_argument('--s_permute', type=int, default=16, help='number of bins for Low Rank CTT permutation')
    
    #Argument for gaussians
    parser.add_argument('--mean_diff', type=float, default=0.024, help='difference between means (for gaussians)')
    
    #Argument for blobs
    parser.add_argument('--grid_size', type=int, default=3, help='dimension of the grid of the distribution (for blobs)')
    parser.add_argument('--epsilon', type=float, default=2, help='covariance eigenvalue (for blobs)')
    
    #Argument for MNIST and EMNIST
    parser.add_argument('--p_even', type=float, default=0.49, help='joint probability of all even digits')
    
    #Arguments for Higgs
    parser.add_argument('--mixing', action='store_true', help='if passed use test mixing between classes')
    parser.add_argument('--null', action='store_true', help='if passed use null hypothesis, else use alternative')
    parser.add_argument('--n_components', type=int, default=4, help='number of dimensions to use')
    parser.add_argument('--p_poisoning', type=float, default=0.9, help='if mixing, poisoning probability of class 1 with class 0')
    
    #Argument for sine
    parser.add_argument('--omega', type=float, default=20, help='sine frequency')
    
    #Arguments for aggregated tests
    parser.add_argument('--n_bandwidths', type=int, default=5, help='number of bandwidths used in the aggregated test')
    parser.add_argument('--B_2', type=int, default=100, help='number of permutations used for Monte Carlo estimation in agg.')
    parser.add_argument('--B_3', type=int, default=20, help='number of bisection iterations for aggregated test')
    parser.add_argument('--different_compression', action='store_true', help='if passed use different compression calls for different bandwidths')

    #Arguments to avoid computing specific test groups, overriding default behavior 
    #(which is computing test groups for which no result files exist)
    parser.add_argument('--no_block_wb', action='store_true', help='if passed do not compute block_wb tests')
    parser.add_argument('--no_incomplete_wb', action='store_true', help='if passed do not compute incomplete_wb tests')
    parser.add_argument('--no_block_asymp', action='store_true', help='if passed do not compute block_asymp tests')
    parser.add_argument('--no_incomplete_asymp', action='store_true', help='if passed do not compute incomplete_asymp tests')
    parser.add_argument('--no_ctt', action='store_true', help='if passed do not compute ctt tests')
    parser.add_argument('--no_rff', action='store_true', help='if passed do not compute rff tests')
    parser.add_argument('--no_ctt_rff', action='store_true', help='if passed do not compute ctt_rff tests')

    #Arguments to recompute specific test groups, overriding default behavior
    parser.add_argument('--recompute_all', action='store_true', help='if passed recompute all tests')
    parser.add_argument('--recompute_block_wb', action='store_true', help='if passed recompute block_wb tests')
    parser.add_argument('--recompute_incomplete_wb', action='store_true', help='if passed recompute incomplete_wb tests')
    parser.add_argument('--recompute_block_asymp', action='store_true', help='if passed recompute block_asymp tests')
    parser.add_argument('--recompute_incomplete_asymp', action='store_true', help='if passed recompute incomplete_asymp tests')
    parser.add_argument('--recompute_ctt', action='store_true', help='if passed recompute ctt tests')
    parser.add_argument('--recompute_rff', action='store_true', help='if passed recompute rff tests')
    parser.add_argument('--recompute_ctt_rff', action='store_true', help='if passed recompute ctt_rff tests')

    #set argument values
    args = parser.parse_args()
    
    args.cython_compress = not args.no_cython_compress
    
    return args