from argparse import ArgumentParser

def init_parser():
    '''
    Initialize/parse arguments.
    See the list below that explains each argument.
    '''
    parser = ArgumentParser()
    parser.add_argument('--rep0', '-r0', type=int, default=0,
                        help="starting experiment id")
    parser.add_argument('--P', '-P', type=str, default="gauss",
                        help="target distribution setting") # gauss, mog, mcmc
    parser.add_argument('--kernel', '-kernel', type=str, default="gauss",
                        help="target distribution setting") # gauss, sinc, laplace
    parser.add_argument('--repn', '-rn', type=int, default=1,
                        help="number of experiment replication")
    ### store_K flag needs to change to int in the FUTURE
    parser.add_argument('--store_K', '-sk', type=int, default=0,
                        help="whether to save K matrix, 2-3x faster runtime (turn on when set to non-zero), but larger memory O(n^2)")
    parser.add_argument('--m', '-m', type=int, default=6,
                        help="number of thinning rounds")
    parser.add_argument('--d', '-d', type=int, default=2,
                        help="dimensions")
    parser.add_argument('--M', '-M', type=int, default=None,
                        help="number of mixture for diag mog in d=2")
    parser.add_argument('--filename', '-f', type=str, default=None,
                       help="name for saved (MCMC) samples")
    parser.add_argument('--rerun', '-rr', type=int, default=0,
                        help="whether to rerun coreset (anything other than 0 to rerun)")
    parser.add_argument('--computemmd', '-cm', type=int, default=1,
                        help="whether to compute mmd (anything other than 0 to compute)")
    parser.add_argument('--save_combined_results', '-scr', type=int, default=0,
                        help="whether to save combined results for mmd and fun_diff; should be set to 1 once all experiments are done running")
    parser.add_argument('--nu', '-nu', type=float, default=0.5,
                        help="IMQ/Matern kernel nu parameter or Bspline beta parameter (beta=nu)")
    parser.add_argument('--power', '-pow', type=float, default=0.5,
                        help="Power kernel parameter")
    parser.add_argument('--computepower', '-cp', type=int, default=1,
                        help="Power kernel parameter (anything other than 0 to turn-on)")
    parser.add_argument('--ktplus', '-ktplus', type=int, default=1,
                        help="whether to run KT+ (0 to turn-off)")
    parser.add_argument('--targetkt', '-tkt', type=int, default=1,
                        help="whether to run target KT (0 to turn-off)")
    parser.add_argument('--powerkt', '-pkt', type=int, default=1,
                        help="whether to run power/root KT (0 to turn-off); takes power parameter from power")
    parser.add_argument('--stdthin', '-st', type=int, default=0,
                        help="whether to run standard thinning")
    
    
    return(parser)

def convert_arg_flags(args):
    '''
    function to change flags that were assignted to ints---because
    of issue with argument parse---back to boolean
    
    any such flag is set to true if the int value is non-zero
    '''
    args.computepower = args.computepower!=0 
    args.targetkt = args.targetkt!=0 
    args.powerkt = args.powerkt!=0
    args.ktplus = args.ktplus!=0
    args.rerun = args.rerun!=0
    args.store_K = args.store_K!=0
    args.stdthin = args.stdthin!=0
    args.save_combined_results = args.save_combined_results!=0
    args.computemmd = args.computemmd!=0
    return(args)
