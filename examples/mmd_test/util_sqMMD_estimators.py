import numpy as np
import time
import scipy
from goodpoints import gaussianc
from functools import partial

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Constants %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
SQRT_2 = np.sqrt(2)

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Kernel functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

def median_criterion(X1,X2):
    """
    given samples X1, X2, compute bandwidth according to median criterion.
    
    X1: 2D array of size (n1,dimension)
    X2: 2D array of size (n2,dimension)
    """
    n_samples = X1.shape[0] + X2.shape[0]
    U = np.concatenate((X1,X2), axis=0)
    U_expanded_1 = np.tile(np.expand_dims(U, axis=1),(1,n_samples,1))
    U_expanded_2 = np.tile(np.expand_dims(U, axis=0),(n_samples,1,1))
    distances = np.linalg.norm(U_expanded_1-U_expanded_2,axis=2)
    return SQRT_2*np.median(distances)

def median_criterion_one_sample(X):
    """
    given samples X, compute bandwidth according to median criterion.
    
    X: 2D array of size (n1,dimension)
    """
    n_samples = X.shape[0]
    X_expanded_1 = np.tile(np.expand_dims(X, axis=1),(1,n_samples,1))
    X_expanded_2 = np.tile(np.expand_dims(X, axis=0),(n_samples,1,1))
    distances = np.linalg.norm(X_expanded_1-X_expanded_2,axis=2)
    return SQRT_2*np.median(distances)

def compute_params(lam, use_krt_split=True, name="gauss"):
    """
    return parameters, and functions for split and swap kernel;
    parameters returned should be understood by kernel_eval, p_kernel,
    ppn_kernel, and pp_kernel functions
        
    lam: bandwidth for the kernel (float)
    use_krt_split: Whether to use square root kernel for split
    name: name of the kernel; "gauss" for Gaussian kernel
      other kernels need to be implemented
    """
    if name=="gauss":
        if use_krt_split:
            split_kernel = partial(gaussian_kernel_by_row, lam=lam/SQRT_2)
            #split_matrix_kernel = partial(gaussian_kernel, lam=lam/SQRT_2)
            split_matrix_kernel = partial(gaussian_kernel_same, lam=lam/SQRT_2)
        else:
            split_kernel = partial(gaussian_kernel_by_row, lam=lam)
            #split_matrix_kernel = partial(gaussian_kernel, lam=lam)
            split_matrix_kernel = partial(gaussian_kernel_same, lam=lam)
        swap_kernel = partial(gaussian_kernel_by_row, lam=lam)
        #swap_matrix_kernel = partial(gaussian_kernel, lam=lam)
        swap_matrix_kernel = partial(gaussian_kernel_same, lam=lam)
        return (split_kernel, swap_kernel, split_matrix_kernel, swap_matrix_kernel)
    raise ValueError("Unrecognized kernel name {}".format(name))

def gaussian_kernel_by_row(X1,X2,lam_sqd):
    """
    return 1D array of kernel evaluations, when either X1 is a 1D array of dimension d
    and X2 is a 2D array of dimensions (n_samples,d)
    lam_sqd: squared bandwidth (float)
    """
    n_samples = X2.shape[0]
    K = np.empty(n_samples)
    gaussianc.gaussian_kernel_by_row(X1,X2,lam_sqd,K)
    return np.exp(-sqdist/lam_sqd)

def gaussian_kernel(X1,X2,lam_sqd):
    """
    return 2D array of matrix evaluations of size (n1,n2)
    
    X1: 2D array of size (n1,d)
    X2: 2D array of size (n2,d)
    lam_sqd: squared bandwidth (float)
    """
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    K = np.empty([n1,n2])
    gaussianc.gaussian_kernel(X1,X2,lam_sqd,K)
    return K

def gaussian_kernel_same(X1,lam_sqd):
    """
    return 2D array of matrix evaluations of size (n1,n1)
    
    X1: 2D array of size (n1,d)
    lam_sqd: squared bandwidth (float)
    """
    n1 = X1.shape[0]
    K = np.empty([n1,n1])
    gaussianc.gaussian_kernel_same(X1,lam_sqd,K)
    return K

def sum_gaussian_kernel_linear_eval(X1,X2,lam_sqd):
    """
    returns sum of kernel evaluations of the i-th row of X1 and the i-th row of X2
    
    X1: 2D array of size (n1,d)
    X2: 2D array of size (n2,d)
    lam_sqd: squared bandwidth (float)
    """
    return gaussianc.sum_gaussian_kernel_linear_eval(X1,X2,lam_sqd)

def sum_gaussian_kernel_aggregated(X1,X2,lam_sqd):
    """
    return sum of the kernel evaluations between all rows of X1 and all
    rows of X2, for all bandwidths in lam, output is an array
    
    X1: 2D array of size (n1,d)
    X2: 2D array of size (n2,d)
    lam_sqd: squared bandwidth (array)
    """
    results = np.zeros(len(lam_sqd))
    gaussianc.sum_gaussian_kernel_aggregated(X1,X2,lam_sqd,results)
    return results

def sum_gaussian_kernel_same_aggregated(X1,lam_sqd):
    """
    return sum of the kernel evaluations between all rows of X1 and all
    rows of X1, for all bandwidths in lam, output is an array
    
    X1: 2D array of size (n1,d)
    lam_sqd: squared bandwidth (array)
    """
    results = np.zeros(len(lam_sqd))
    gaussianc.sum_gaussian_kernel_same_aggregated(X1,lam_sqd,results)
    return results
                      
                      
"""
%%%%%%%%%%%%%%%%%%%%%%%% Squared MMD estimators %%%%%%%%%%%%%%%%%%%%%%%%%%
"""
    
                      
def biased_sqMMD(X1,X2,lam,name="gauss"):
    """
    Computation of the biased (complete) squared MMD.
    Returns float
    
    X1: 2D array of size (n1,d)
    X2: 2D array of size (n2,d)
    lam: bandwidth (float)
    name: name of the kernel; implemented for Gaussian kernels
    """
    if name=="gauss":
        return gaussianc.biased_sqMMD_gaussian(X1,X2,lam**2)
    raise ValueError("Unrecognized kernel name {}".format(name))
    
def unbiased_sqMMD(X1,X2,lam,name="gauss"):
    """
    Computation of the unbiased (complete) squared MMD.
    Returns float
    
    X1: 2D array of size (n1,d)
    X2: 2D array of size (n2,d)
    lam: bandwidth (float)
    name: name of the kernel; implemented for Gaussian kernels
    """
    if name=="gauss":
        return gaussianc.unbiased_sqMMD_gaussian(X1,X2,lam**2)
    raise ValueError("Unrecognized kernel name {}".format(name))
    
def block_sqMMD(X1,X2,lam,alpha,block_size,group_results_dict,args,name="gauss"):
    """
    Computation of the block squared MMD: https://proceedings.neurips.cc/paper/2013/file/a49e9411d64ff53eccfdd09ad10a15b3-Paper.pdf
    Returns float
    
    X1: 2D array of size (n1,d)
    X2: 2D array of size (n2,d)
    lam: bandwidth (float)
    alpha: level of the test
    block_size: size of each block
    group_results_dict: object used to store results
    name: name of the kernel; implemented for Gaussian kernels
    """
    if name=="gauss":
        
        # Define the Gaussian inverse CDF and compute the inverse CDF value for 1-alpha 
        invPhi = lambda x: scipy.stats.norm.ppf(x)
        stdev_factor = invPhi(1-alpha)
        
        results = np.empty(2)
        
        start = time.time()
        gaussianc.block_sqMMD_gaussian(X1,X2,lam**2,block_size,results)
        end = time.time()
        
        for n_var in args.n_var:
            group_results_dict['block_asymp'].group_tests['bk'+str(block_size)+'nv'+str(n_var)].statistic_values = results[0]
            group_results_dict['block_asymp'].group_tests['bk'+str(block_size)+'nv'+str(n_var)].total_times = end-start
            group_results_dict['block_asymp'].group_tests['bk'+str(block_size)+'nv'+str(n_var)+'c'].statistic_values = results[0]
            group_results_dict['block_asymp'].group_tests['bk'+str(block_size)+'nv'+str(n_var)+'c'].total_times = end-start
            group_results_dict['block_asymp'].group_tests['bk'+str(block_size)+'nv'+str(n_var)+'c'].threshold_values = np.sqrt(results[1])*stdev_factor
        
    else:
        raise ValueError("Unrecognized kernel name {}".format(name))
    
def block_sqMMD_reordered(X1,X2,lam,alpha,block_size,group_results_dict,args,seed=0,name="gauss"):
    """
    Computation of the block squared MMD: https://proceedings.neurips.cc/paper/2013/file/a49e9411d64ff53eccfdd09ad10a15b3-Paper.pdf
    Returns float
    
    X1: 2D array of size (n1,d)
    X2: 2D array of size (n2,d)
    lam: bandwidth (float)
    alpha: level of the test
    block_size: size of each block
    group_results_dict: object used to store results
    seed: integer seed for random number generator
    name: name of the kernel; implemented for Gaussian kernels
    """
    if name=="gauss":
        
        # Define the Gaussian inverse CDF and compute the inverse CDF value for 1-alpha 
        invPhi = lambda x: scipy.stats.norm.ppf(x)
        stdev_factor = invPhi(1-alpha)
        
        results = np.empty(4)
        epsilon = np.zeros(block_size)
        epsilon = epsilon.astype(int)
        
        start = time.time()
        gaussianc.block_sqMMD_gaussian_reordered(X1,X2,lam**2,block_size,seed,epsilon,results)
        end = time.time()
        
        for n_var in args.n_var:
            group_results_dict['block_asymp'].group_tests['bk'+str(block_size)+'nv'+str(n_var)+'b'].statistic_values = results[0]
            group_results_dict['block_asymp'].group_tests['bk'+str(block_size)+'nv'+str(n_var)+'b'].total_times = end-start
            group_results_dict['block_asymp'].group_tests['bk'+str(block_size)+'nv'+str(n_var)+'b'].threshold_values = np.sqrt(results[3])*stdev_factor
        
    else:
        raise ValueError("Unrecognized kernel name {}".format(name))
    
def block_sqMMD_Rademacher(X1,X2,B,lam,block_size,group_results_dict,seed=None,name="gauss"):
    """
    Computation of the block squared MMD for several Rademacher vectors: https://proceedings.neurips.cc/paper/2013/file/a49e9411d64ff53eccfdd09ad10a15b3-Paper.pdf
    Returns float
    
    X1: 2D array of size (n1,d)
    X2: 2D array of size (n2,d)
    B: number of Rademacher vectors (int)
    lam: bandwidth (float)
    block_size: size of each block
    group_results_dict: object used to store results
    name: name of the kernel; implemented for Gaussian kernels
    """
    if name=="gauss":
        n_samples = X1.shape[0]
        
        test_rng = np.random.default_rng(seed)
        
        epsilon = 2*test_rng.integers(2, size=(n_samples,B))-1
        
        split_values = np.zeros(B+1)
        sqMMD_values = np.zeros(B+1)
        
        time = gaussianc.block_sqMMD_gaussian_Rademacher(X1,X2,lam**2,block_size,epsilon,split_values,sqMMD_values)
        
        # Store results
        group_results_dict['block_wb'].group_tests['bk'+str(block_size)].estimator_values[:] = sqMMD_values
        group_results_dict['block_wb'].group_tests['bk'+str(block_size)].total_times = time 
        
    else:
        raise ValueError("Unrecognized kernel name {}".format(name))
    
def incomplete_sqMMD(X1,X2,lam,alpha,group_results_dict,args,seed=0,name="gauss"):
    """
    Computation of the incomplete squared MMD: https://openreview.net/pdf?id=BkG5SjR5YQ
    Returns dictionaries sqMMD_list, sqMMD_times
    
    X1: 2D array of size (n_samples,d)
    X2: 2D array of size (n_samples,d)
    lam: bandwidth (float)
    alpha: level of the test
    group_results_dict: object used to store results
    seed: integer seed for random number generator
    name: name of the kernel; implemented for Gaussian kernels
    """
    if name=="gauss":
        
        # Define the Gaussian inverse CDF and compute the inverse CDF value for 1-alpha 
        invPhi = lambda x: scipy.stats.norm.ppf(x)
        stdev_factor = invPhi(1-alpha)
        
        max_l = np.max(args.asymptotic_incomplete_list)
        n_estimators = len(args.asymptotic_incomplete_list)
        
        sqMMD_values = np.zeros(n_estimators)
        sqMMD_variances = np.zeros(n_estimators)
        times = np.zeros(n_estimators)
        
        asymptotic_incomplete_list = np.array(args.asymptotic_incomplete_list)
        asymptotic_incomplete_list = asymptotic_incomplete_list.astype(int)
        
        gaussianc.incomplete_sqMMD_gaussian(X1,X2,asymptotic_incomplete_list,max_l,lam**2,
                                            seed,sqMMD_values,sqMMD_variances,times)
        
        # Store results
        for i in range(len(args.asymptotic_incomplete_list)):
            for n_var in args.n_var:
                group_results_dict['incomplete_asymp'].group_tests['i'+str(args.asymptotic_incomplete_list[i]) +'nv'+str(n_var)].statistic_values = sqMMD_values[i]
                group_results_dict['incomplete_asymp'].group_tests['i'+str(args.asymptotic_incomplete_list[i]) +'nv'+str(n_var)+'b'].statistic_values = sqMMD_values[i]
                group_results_dict['incomplete_asymp'].group_tests['i'+str(args.asymptotic_incomplete_list[i]) +'nv'+str(n_var)].total_times = times[i]
                group_results_dict['incomplete_asymp'].group_tests['i'+str(args.asymptotic_incomplete_list[i]) +'nv'+str(n_var)+'b'].total_times = times[i]
                group_results_dict['incomplete_asymp'].group_tests['i'+str(args.asymptotic_incomplete_list[i]) +'nv'+str(n_var)+'b'].threshold_values = np.sqrt(sqMMD_variances[i])*stdev_factor
        
    else:
        raise ValueError("Unrecognized kernel name {}".format(name))
        
def incomplete_sqMMD_Rademacher_subdiagonals(X1,X2,B,lam,group_results_dict,args,seed=None,name="gauss",aggregated=False, estimator_values_2=False):
    """
    Computation of the incomplete squared MMD for several Rademacher vectors: https://openreview.net/pdf?id=BkG5SjR5YQ
    Returns dictionaries sqMMD_list, sqMMD_times
    
    X1: 2D array of size (n_samples,d)
    X2: 2D array of size (n_samples,d)
    B: number of Rademacher vectors (int)
    lam: bandwidth (float)
    group_results_dict: object used to store results
    seed: integer seed for random number generator
    name: name of the kernel; implemented for Gaussian kernels
    """
    if name=="gauss":
        n_samples = X1.shape[0]
        
        test_rng = np.random.default_rng(seed)
        
        max_l = np.max(args.wb_incomplete_list)
        
        n_estimators = len(args.wb_incomplete_list)
        
        epsilon = 2*test_rng.integers(2, size=(n_samples,B))-1
        epsilon = epsilon.astype(int)
        
        incomplete_list = np.empty(n_estimators)
        for i in range(n_estimators):
            incomplete_list[i] = args.wb_incomplete_list[i]
        incomplete_list = incomplete_list.astype(int)
        
        if aggregated:
            sqMMD_matrix = np.zeros((n_estimators,len(lam),B+1))
            sqMMD_vector = np.zeros((len(lam),B+1))
            times = np.zeros(n_estimators)
            h_values = np.zeros(len(lam))

            print(f'Run gaussianc.incomplete_sqMMD_gaussian_Rademacher')
            gaussianc.incomplete_sqMMD_gaussian_Rademacher_aggregated(X1,X2,incomplete_list,B,max_l,lam**2,
                                                                epsilon,sqMMD_matrix,sqMMD_vector,times,h_values)

            # Store results
            for i in range(len(args.wb_incomplete_list)):
                for j, bw in enumerate(lam):
                    group_results_dict['incomplete_wb'].group_tests['i'+ str(args.wb_incomplete_list[i])].all_estimator_values[bw] = sqMMD_matrix[i,j]*(n_samples/args.wb_incomplete_list[i])
                group_results_dict['incomplete_wb'].group_tests['i'+str(args.wb_incomplete_list[i])].times = times[i]
            
        else:
            sqMMD_matrix = np.zeros((n_estimators,B+1))
            sqMMD_vector = np.zeros(B+1)
            times = np.zeros(n_estimators)
        
            print(f'Run gaussianc.incomplete_sqMMD_gaussian_Rademacher')
            gaussianc.incomplete_sqMMD_gaussian_Rademacher_subdiagonals(X1,X2,incomplete_list,B,max_l,lam**2,
                                                                  epsilon,sqMMD_matrix,sqMMD_vector,times)

            # Store results
            for i in range(len(args.wb_incomplete_list)):
                group_results_dict['incomplete_wb'].group_tests['i'+str(args.wb_incomplete_list[i])].estimator_values[:] = sqMMD_matrix[i]*(n_samples/args.wb_incomplete_list[i])
                group_results_dict['incomplete_wb'].group_tests['i'+str(args.wb_incomplete_list[i])].total_times = times[i]
        
    else:
        raise ValueError("Unrecognized kernel name {}".format(name))
    
def compute_sigma_2_sqd(X1,X2,lam,alpha,group_results_dict,args,name="gauss"):
    if name=="gauss":
        n_samples = X1.shape[0]
        
        #Define the Gaussian inverse CDF and compute the inverse CDF value for 1-alpha 
        invPhi = lambda x: scipy.stats.norm.ppf(x)
        stdev_factor = invPhi(1-alpha)
            
        for n_var in args.n_var:

            #Start counting time for variance computation
            start = time.time()

            #Compute E_{z,z'}[(h(z,z'))^2]
            expectation_2 = gaussianc.expectation_h_2(X1[:n_var],X2[:n_var],lam**2)
            end = time.time()

            #sigma_2_sqd = E_{z,z'}[(h(z,z'))^2] - E_z[(E_{z'}h(z,z'))^2], but the second term is zero under the null
            sigma_2_sqd = expectation_2
            print(f'sigma_2_sqd: {sigma_2_sqd}.')

            # Store results
            if group_results_dict['block_asymp'].compute:
                for size in args.asymptotic_block_size_list:
                    variance = 2*sigma_2_sqd/(n_samples*(size-1))
                    group_results_dict['block_asymp'].group_tests['bk'+str(size)+'nv'+str(n_var)].threshold_values = np.sqrt(variance)*stdev_factor
                    group_results_dict['block_asymp'].group_tests['bk'+str(size)+'nv'+str(n_var)].total_times += end-start

            # Store results
            if group_results_dict['incomplete_asymp'].compute:    
                for l in args.asymptotic_incomplete_list:
                    variance = sigma_2_sqd*(1/l + 2/(n_samples*(n_samples-1)))
                    group_results_dict['incomplete_asymp'].group_tests['i'+str(l)+'nv'+str(n_var)].threshold_values = np.sqrt(variance)*stdev_factor
                    group_results_dict['incomplete_asymp'].group_tests['i'+str(l)+'nv'+str(n_var)].total_times += end-start
                    
    else:
        raise ValueError("Unrecognized kernel name {}".format(name))

def lambda_computation_2(X1,X2):
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    n_kernels = 4*(13+7)+1
    log_bw_vec = np.linspace(-7,13,num=n_kernels)
    bw_vec = 2**log_bw_vec
    X1a = X1[:int(n1/2),:]
    X1b = X1[int(n1/2):,:]
    X2a = X2[:int(n1/2),:]
    X2b = X2[int(n1/2):,:]
    max_ratio = 0
    best_bw = -1
    for bw in bw_vec:
        bw_sqd = bw**2
        lsqMMD = (sum_gaussian_kernel_linear_eval(X1a,X1b,bw_sqd) + sum_gaussian_kernel_linear_eval(X2a,X2b,bw_sqd)
                  - sum_gaussian_kernel_linear_eval(X1a,X2a,bw_sqd)
                  - sum_gaussian_kernel_linear_eval(X1b,X2b,bw_sqd))/int(n1/2)
        usqMMD = unbiased_sqMMD(X1[:1096,:],X2[:1096,:],gaussian_kernel,bw)
        var_lsqMMD = (sum_gaussian_kernel_linear_eval(X1a,X1b,bw_sqd) + sum_gaussian_kernel_linear_eval(X2a,X2b,bw_sqd) - sum_gaussian_kernel_linear_eval(X1a,X2a,bw_sqd) - sum_gaussian_kernel_linear_eval(X1b,X2b,bw_sqd))**2/int(n1/2) - lsqMMD**2
        sigma2 = var_lsqMMD*int(n1/2)+1e-30
        ratio = usqMMD/np.sqrt(sigma2)
        print(f'Bandwidth: {bw:.5e}. usqMMD: {usqMMD:.5e}. Sigma2: {sigma2}. Ratio: {ratio:.5e}.')
        if ratio > max_ratio:
            best_bw = bw 
            max_ratio = ratio
    return best_bw
