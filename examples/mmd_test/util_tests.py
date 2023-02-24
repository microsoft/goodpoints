import numpy as np

def get_attributes_tests(args):
    #Build list of test groups
    test_groups = ['block_wb', 'incomplete_wb', 'block_asymp', 'incomplete_asymp', 'incomplete_wb_HSIC', 'ctt', 'rff', 'ctt_rff']
    
    #Define dicts to store estimator, estimator names (used when printing results), and estimator labels (used in plots)
    args.estimators = dict()
    args.estimator_names = dict()
    args.estimator_labels = dict()
    for group in test_groups:
        args.estimators[group] = []
        args.estimator_names[group] = []
        args.estimator_labels[group] = []
        
    #Build list of estimators for block tests (with wild bootstrap)
    if args.n == 262144:
        args.wb_block_size_list = [2,1024,4096,8192,16384,32768,65536,131072,262144] #[2,64,256,1024,2048,4096]
    else:
        args.wb_block_size_list = [2,128,256,512,1024,2048,4096,8192,16384]
    for i in range(len(args.wb_block_size_list)):
        args.wb_block_size_list[i] = np.minimum(args.wb_block_size_list[i],args.n)
    for size in args.wb_block_size_list:
        args.estimators['block_wb'].append('bk'+str(size))
        args.estimator_names['block_wb'].append('WB block'+str(size))
        args.estimator_labels['block_wb'].append('B='+str(size))
        
    #Build list of estimators for incomplete tests (with wild bootstrap)    
    if args.n == 262144:
        args.wb_incomplete_list_multiples = [1,512,2048,4096,8192,16384,32768,65536,131072] #[1,1024,4096,8192,16384,32768]
    else:
        args.wb_incomplete_list_multiples = [1,64,128,256,512,1024,2048,4096,8192,16384]
    for i in range(len(args.wb_incomplete_list_multiples)):
        args.wb_incomplete_list_multiples[i] = np.minimum(args.wb_incomplete_list_multiples[i],(args.n-1)/2)
    args.wb_incomplete_list = ((args.n*np.array(args.wb_incomplete_list_multiples)).astype(int)).tolist()
    for l in args.wb_incomplete_list:
        args.estimators['incomplete_wb'].append('i'+str(l))
        args.estimator_names['incomplete_wb'].append('WB incomplete'+str(l))
    for l in args.wb_incomplete_list_multiples:
        args.estimator_labels['incomplete_wb'].append('l='+str(l)+'*n')
        
    #Build lists of number of samples used for variance computation of asymptotic tests    
    args.n_var = [2048,8192] #[8192,16384,32768] #[1024,4096,8192]
    for i in range(len(args.n_var)):
        args.n_var[i] = np.minimum(args.n_var[i],args.n)
    
    #Build lists of estimators for asymptotic block tests
    if args.n == 262144:
        args.asymptotic_block_size_list = [2,1024,4096,8192,16384,32768,65536,131072,262144] #[2,1024,4096,8192,16384] #[2,1024,4096,8192]
    else:
        args.asymptotic_block_size_list = [2,128,256,512,1024,2048,4096,8192,16384]
    for i in range(len(args.asymptotic_block_size_list)):
        args.asymptotic_block_size_list[i] = np.minimum(args.asymptotic_block_size_list[i],args.n)
    for n_var in args.n_var:
        for size in args.asymptotic_block_size_list:
            args.estimators['block_asymp'].append('bk'+str(size)+'nv'+str(n_var))
            args.estimators['block_asymp'].append('bk'+str(size)+'nv'+str(n_var)+'b')
            args.estimators['block_asymp'].append('bk'+str(size)+'nv'+str(n_var)+'c')
            args.estimator_names['block_asymp'].append('asymp. block'+str(size)+',n_var'+str(n_var))
            args.estimator_names['block_asymp'].append('asymp. block'+str(size)+',n_var'+str(n_var)+'b')
            args.estimator_names['block_asymp'].append('asymp. block'+str(size)+',n_var'+str(n_var)+'c')
            args.estimator_labels['block_asymp'].append('B='+str(size))
            args.estimator_labels['block_asymp'].append('B='+str(size))
            args.estimator_labels['block_asymp'].append('B='+str(size))
         
    #Build lists of estimators for asymptotic incomplete tests
    if args.n == 262144:
        args.asymptotic_incomplete_list_multiples = [1,1024,4096,8192,16384,32768,65536] #[1,1024,4096,8192,16384]
    else:
        args.asymptotic_incomplete_list_multiples = [1,64,128,256,512,1024,2048,4096,8192]
    args.asymptotic_incomplete_list = (args.n*np.array(args.asymptotic_incomplete_list_multiples)).tolist()
    for n_var in args.n_var:
        for l in args.asymptotic_incomplete_list:
            args.estimators['incomplete_asymp'].append('i'+str(l)+'nv'+str(n_var))
            args.estimators['incomplete_asymp'].append('i'+str(l)+'nv'+str(n_var)+'b')
            args.estimator_names['incomplete_asymp'].append('asymp. incomplete'+str(l)+',n_var'+str(n_var))
            args.estimator_names['incomplete_asymp'].append('asymp. incomplete'+str(l)+',n_var'+str(n_var)+'b')
        for l in args.asymptotic_incomplete_list_multiples:
            args.estimator_labels['incomplete_asymp'].append('l='+str(l)+'*n')
            args.estimator_labels['incomplete_asymp'].append('l='+str(l)+'*n')
               
    #Build list of estimators for incomplete HSIC tests (with wild bootstrap)    
    args.wb_incomplete_HSIC_list_multiples = [1,1024,4096,8192,16384,32768]
    for i in range(len(args.wb_incomplete_HSIC_list_multiples)):
        args.wb_incomplete_HSIC_list_multiples[i] = np.minimum(args.wb_incomplete_HSIC_list_multiples[i],int(np.ceil(args.n/8)))
    args.wb_incomplete_HSIC_list = (args.n*np.array(args.wb_incomplete_HSIC_list_multiples)).tolist()
    for l in args.wb_incomplete_HSIC_list:
        args.estimators['incomplete_wb_HSIC'].append('i'+str(l))
        args.estimator_names['incomplete_wb_HSIC'].append('WB incomplete HSIC '+str(l))
    for l in args.wb_incomplete_list_multiples:
        args.estimator_labels['incomplete_wb_HSIC'].append('l='+str(l)+'*n')
        
    #Build list of estimators for CTT tests
    if args.name == 'Higgs':
        args.block_g_list = [0,1,2,3,4]
    else:
        args.block_g_list = [0,1,2,3] ###[0,1,2,3]
    for g in args.block_g_list:
        args.estimators['ctt'].append('t'+str(g))
        args.estimator_names['ctt'].append('thinned_g'+str(g))
        args.estimator_labels['ctt'].append('g='+str(g))
        
    #Build list of estimators for RFF tests
    args.n_features_list = [1,4,16,64,128,256] #[1,4,16,64,128,256,512] #[10,20,40,60,80,100,150,200,300,500]
    for nf in args.n_features_list:
        args.estimators['rff'].append('r'+str(nf))
        args.estimator_names['rff'].append('rff'+str(nf))
        args.estimator_labels['rff'].append('nf='+str(nf))
        
    #Build list of estimators for RFF tests
    #args.s_rff = 16 #number of bins for Low Rank CTT compression
    #args.s_permute = 16 #number of bins for Low Rank CTT permutation
    args.block_g_list_ctt_rff = [0,1,2,3]
    args.n_features_list_ctt_rff = [1,2,4,8,16,32,64,128,256,512,1024] #[16,64,256,1024]
    for g in args.block_g_list_ctt_rff:
        for nf in args.n_features_list_ctt_rff:
            args.estimators['ctt_rff'].append('r'+str(g)+'_'+str(nf)+'_'+str(args.s_rff)+'_'+str(args.s_permute))
            args.estimator_names['ctt_rff'].append('ctt_rff'+str(g)+'_'+str(nf)+'_'+str(args.s_rff)+'_'+str(args.s_permute))
            args.estimator_labels['ctt_rff'].append('g='+str(g)+',nf='+str(nf)+',s_rff='+str(args.s_rff) +',s_p='+str(args.s_permute))