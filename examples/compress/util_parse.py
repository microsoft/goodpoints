from argparse import ArgumentParser

def get_args_from_terminal():
    '''
    Returns a dictionary of arguments known to init_parse passed from terminal
    '''
    parser = init_parser()
    args, opt = parser.parse_known_args()
    return(format_args(args)) # convert the integer flags to boolean

def init_parser():
    '''
    Note we have to use integer type for flags to take input from terminal. 
    format_parser converts these ints back to boolean (set to False if int value
    assigned to flag is 0, else True)
    '''
    parser = ArgumentParser()
    
    ################ seed, rerun, input size, rep_ids, thinning parameters ################
    # results folder; where to save results
    parser.add_argument('--resultsfolder', '-rfolder', type=str, default="coresets_folder",
                            help="folder to save results (relative to where the script is run)")
    
    # experiment seed
    parser.add_argument('--seed','-ssd',type=int,default=123456789, help="seed for experiment") 
    
    # rerun flag, if set to 0 then coresets/results are loaded from the disk if available
    # else coresets are regenerated
    # *** THIS IS A FLAG PARAMETER (will be converted to boolean by format_args) ***
    parser.add_argument('--rerun','-rr',type=int, default = 0, help="whether to rerun the experiments") 
    
    # input size in log base 4
    parser.add_argument('--size','-sz',type=int,default=2,
                            help='sample set of size in log base 4, i.e., input size = 4**size')
    
    # starting repetition id for the experiment
    parser.add_argument('--rep0', '-r0', type=int, default=0,
                            help="starting experiment id")
    
    # number of experiments; we run experiment with rep_id in rep0, rep0+1, ..., rep0+repn-1
    parser.add_argument('--repn', '-rn', type=int, default=1,
                            help="number of experiment replication")
    
    # thinning parameter
    parser.add_argument('--m', '-m', type=int, default=2,
                            help="number of thinning rounds; output size = 4**size / 2**m")
    
    # dimension of the points; d also determines the parameters of Gauss kernel, and Gauss P
    parser.add_argument('--d', '-d', type=int, default=2,
                            help="dimension of the points")

    ################ generic arguments for construct_{XYZ}_coresets.py ################
    # set to non-zero when you want the compress functions to return coresets
    # *** THIS IS A FLAG PARAMETER (will be converted to boolean by format_args) ***
    parser.add_argument('--returncoreset', '-rc', type=int, default=0, 
                        help="whether to return coreset, set to anything other than 0 ")
    
    # set to non-zero when you want the construct functions to print coresets and mmds
    # *** THIS IS A FLAG PARAMETER (will be converted to boolean by format_args) ***
    parser.add_argument('--verbose', '-v', type=int, default=1, 
                help="whether to print coresets and mmds, set to anything other than 0 ") 
    
    # whether to compute mmd or not--load from disk when the file exists 
    # *** THIS IS A FLAG PARAMETER (will be converted to boolean by format_args) ***
    parser.add_argument('--computemmd', '-cmmd', type=int, default=0,
                            help="whether to compute mmd results and save to disk set to anything other than 0 ")
    
    # *** THIS IS A FLAG PARAMETER (will be converted to boolean by format_args) ***
    # whether to recompute mmd--overwrites the file on disk
    parser.add_argument('--recomputemmd', '-rcmmd', type=int, default=0,
                            help="whether to re-compute mmd results (refresh results on disk)\
                            , set to anything other than 0 ")
    
    # what target distribution setting to run, currently this code supports gauss, mog, and mcmc
    parser.add_argument('--setting', '-setting', type=str, default="gauss",
                            help="what P setting to run the experiment")
    
    ################ kt related arguments ################
    
    # flag to decide if krt kernel should be used for kt.split set to any non-zero value to use krt.
    # *** THIS IS A FLAG PARAMETER (will be converted to boolean by format_args) ***
    parser.add_argument('--krt', '-krt', type=int, default=0,
                            help="whether to use krt for kt.split")
    
    
    ################ compress++ specific arguments ################
    # algorithm used inside compress ; 
    # currently one can use "kt" for kt.thin, and "herding" for herding; needs to add logic for 
    parser.add_argument('--compressalg','-ca',type =str , default = "kt", 
                        help="name of the algorithm to be used as halve/thin in compress++") 
    
    # whether to symmetrize in stage 1
    # *** THIS IS A FLAG PARAMETER (will be converted to boolean by format_args) ***
    parser.add_argument('--symm1','-symm1',type =int , default = 1,
                       help="whether to symmetrize halve output in compress") 
    
    # whether to rechalve in stage 2
    # *** THIS IS A FLAG PARAMETER (will be converted to boolean by format_args) ***
    parser.add_argument('--rh2','-rh2',type =int , default = 0,
                       help="whether to symmetrize halve output in compress") 
    
    # the oversampling parameter g for compress
    parser.add_argument('--g','-g',type=int, default = 0,
                       help="the oversampling parameter g for compress (called g here)") 
    

    ################ target distribution related parameters ################
    # number of components for diag_mog target in compute_mog_params functions in util_sample.py
    parser.add_argument('--M', '-M', type=int, default=None,
                            help="number of mixture for diag mog in d=2")
    # filename when dealing with MCMC target in compute_mcmc_params functions in util_sample.py
    parser.add_argument('--filename', '-f', type=str, default=None,
                           help="name for MCMC target") 
    parser.add_argument('--mcmcfolder', '-folder', type=str, 
                        default='/accounts/projects/binyu/raaz.rsk/kernel_thinning/kernel_thinning_plus/data',
                           help="folder to load MCMC data from, and save some \
                            PPk like objects to save time while computing mmd") 
    
    
    ################ runtime experiment specific args ################
    # which thin algorithm is being profiled
    parser.add_argument('--thinalg', '-talg' , type=str, default="compresspp") 
    # what prefix to use for saving results
    parser.add_argument('--prefix', '-prefix' , type=str, default="")
    
    return(parser)

def format_args(args):
    '''
    Converts flags of args from int to boolean.
    '''
    args.rerun = False if args.rerun == 0 else True
    args.computemmd = False if args.computemmd == 0 else True
    args.recomputemmd = False if args.recomputemmd == 0 else True
    args.verbose = False if args.verbose == 0 else True
    args.returncoreset = False if args.returncoreset == 0 else True
    
    args.symm1 = False if args.symm1 == 0 else True
    args.rh2 = False if args.rh2 == 0 else True
    
    args.krt = False if args.krt == 0 else True
    
    return(args)