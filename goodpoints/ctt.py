"""Compress Then Test (CTT), Random Fourier Features (RFF) Test, 
   Low Rank-CTT, and Aggregated CTT

Implementations of the CTT, LR-CTT, and ACTT tests of
  Carles Domingo-Enrich, Raaz Dwivedi, and Lester Mackey.
  Compress Then Test: Powerful Kernel Testing in Near-linear Time.
  https://arxiv.org/pdf/2301.05974.pdf

and the RFF test (with permutation-based null) of
  Ji Zhao and Deyu Meng.
  FastMMD: Ensemble of Circular Discrepancy for Efficient Two-Sample Test.
  https://arxiv.org/pdf/1405.2664.pdf
"""

import warnings
import numpy as np
import time
import pickle
from goodpoints import compress
from goodpoints import gaussianc
from goodpoints import cttc
from goodpoints.cttc import signed_matrix_sum

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Constants %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
SQRT_2 = np.sqrt(2)

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Tests %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

def ctt(X1,X2,g,B=39,s=16,lam=1.,kernel="gauss",null_seed=None,
        statistic_seed=None,alpha=.05,delta=0.5):
    """Compress Then Test two-sample test with sample sequences X1 and X2
    and auxiliary kernel k' = target kernel k.
        
    Note: Assumes that bin_size = max(1, (n1+n2)//(2s)) evenly divides n1 and n2
    
    Args:
      X1: 2D array of size (n1,d)
      X2: 2D array of size (n2,d)
      g: compression level; integer >= 0
      B: number of permutations (int)
      s: total number of compression bins will be num_bins = min(2*s, n1+n2);
        X1 will be divided into num_bins * n1 / (n1 + n2) compression bins;
        X2 will be divided into num_bins * n2 / (n1 + n2) compression bins
      lam: positive kernel bandwidth 
      kernel: kernel name; valid options include "gauss" for Gaussian kernel
        exp(-||x-y||^2/lam^2)
      null_seed: seed used to initialize random number generator for
        randomness in simulating null; can be any valid input to 
        numpy.random.default_rng
      statistic_seed: seed used to initialize random number generator for
        randomness in computing test statistic; can be any valid input to 
        numpy.random.default_rng
      alpha: test level
      delta: KT-Compress failure probability
      
    Returns: TestResults object.
    """
    # Measure runtime
    time_0 = time.time()
    
    if kernel != "gauss":
        raise ValueError(f"Unsupported kernel name {kernel}")
    
    # Store squared bandwidth in an array
    lam_sqd = np.array([lam**2])
    
    # Number of sample poins
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    
    # Number of KT-Compress bins per dataset
    num_bins_total = min(2*s, n1+n2)
    bin_size = (n1+n2) // num_bins_total
    num_bins1 = n1 // bin_size
    num_bins2 = num_bins_total - num_bins1

    # Prepare integer random number generator seeds for each dataset
    stat_rng = np.random.default_rng(statistic_seed)
    stat_ss = stat_rng.bit_generator._seed_seq.spawn(2)
    stat_seed1 = stat_ss[0].generate_state(1)
    stat_seed2 = stat_ss[1].generate_state(1)
        
    # Identify coreset for each compression bin
    hatX1_indices = compress.compress_gsn_kt(
        X1, g = g, num_bins = num_bins1, lam_sqd = lam_sqd, delta = delta, 
        seed = stat_seed1)
    hatX2_indices = compress.compress_gsn_kt(
        X2, g = g, num_bins = num_bins2, lam_sqd = lam_sqd, delta = delta, 
        seed = stat_seed2)

    # Compute sum of Gaussian kernel evaluations between each pair
    # of coresets
    avg_matrix = np.empty((num_bins_total,num_bins_total))
    gaussianc.sum_gaussian_kernel_by_bin(X1[hatX1_indices], X2[hatX2_indices],
                                         lam_sqd, avg_matrix)
    # Normalize each sum by the number of kernel evaluations
    avg_matrix /= (bin_size**2)

    # Initialize generator for generating permutations
    test_rng = np.random.default_rng(null_seed)
    
    # Compute permuted squared MMD values
    estimator_values = np.empty(B+1)
    for b in range(B+1):
        # First select permuted order
        if b == B:
            # Store original order in final permutation bin
            perm_signs = np.arange(num_bins_total, dtype=int)
        else:
            # Generate random permutation of num_bins_total indices
            perm_signs = test_rng.permutation(num_bins_total)
        # Then assign a +/-1 sign to each datapoint indicating if
        # bin was assigned a permutation index < num_bins1
        perm_signs = (perm_signs < num_bins1)*2-1

        # Finally compute the unnormalized squared MMD as a signed matrix sum
        # sum_{i,j} sign(i) sign(j) avg_matrix[i,j] 
        estimator_values[b] = (
            signed_matrix_sum(avg_matrix, perm_signs) )

    # Normalize squared MMDs
    estimator_values /= (num_bins1*num_bins2)
    
    # Measure runtime
    time_1 = time.time()
    total_time = time_1 - time_0
    
    # Return test results
    return TestResults('ctt', estimator_values, total_time, alpha=alpha)

def kernel_features(X1,X2,r,lam=1.,kernel="gauss",feat_type="rff",
                    seed=None,bin_size=1):
    """Returns approximate kernel features (like random Fourier features).
    
    For the concatenation of X1 and X2, computes associated features 
    for approximating the specified kernel, partitions the rows into
    consecutive bins of size bin_size, and returns the average 
    feature vector for each bin.
    
    Note: assumes bin_size divides n1+n2 evenly
    
    Args:
      X1: 2D array of size (n1,d)
      X2: 2D array of size (n2,d)
      r: number of features to return; positive integer
      lam: bandwidth of Gaussian kernel exp(-||x-y||^2/lam^2)
      kernel: kernel name; use "gauss" for Gaussian kernel
        exp(-||x-y||^2/lam^2)
      feat_type: type of features to extract; use "rff" for random Fourier 
        features 
      seed: seed used to initialize random number generator; 
        can be any valid input to numpy.random.default_rng
      bin_size: size of each bin used for averaging
    """
    if kernel != "gauss":
        raise ValueError(f"Unsupported kernel name {kernel}")
    if feat_type != "rff":
        raise ValueError(f"Unsupported feature type {feat_type}")
        
    # Generate random Fourier feature directions W and offsets b
    d = X1.shape[1]
    rng = np.random.default_rng(seed)
    W = rng.standard_normal(size=(r,d))*SQRT_2/lam
    b = 2*np.pi*rng.uniform(size=r)

    # Compute RFF features averaged by bin for the concatenation of (X1,X2)
    # and store in features
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    features = np.zeros(((n1+n2)//bin_size,r))
    cttc.rff(X1,X2,W,b,features)
    return features

def rff(X1,X2,r,B=39,lam=1.,kernel="gauss",
        null_seed=None,statistic_seed=None,alpha=.05):
    """Random Fourier Features two-sample test with sample sequences X1 and X2.
    
    Runs RFF two-sample test on the sample sequences
    X1 and X2 and returns the results as a TestResults object.
    
    Args:
      X1: 2D array of size (n1,d)
      X2: 2D array of size (n2,d)
      r: number of random Fourier features; positive integer
      B: number of permutations (int)
      lam: positive kernel bandwidth (float)
      kernel: kernel name; valid options include "gauss" for Gaussian kernel
        exp(-||x-y||^2/lam^2)
      null_seed: seed used to initialize random number generator for
        randomness in simulating null; can be any valid input to 
        numpy.random.default_rng
      statistic_seed: seed used to initialize random number generator for
        randomness in computing test statistic; can be any valid input to 
        numpy.random.default_rng
      alpha: test level

    Returns: TestResults object.
    """
    # Measure runtime
    time_0 = time.time()
    
    if kernel != "gauss":
        raise ValueError(f"Unsupported kernel name {kernel}")
        
    # Store squared bandwidth in an array
    lam_sqd = np.array([lam**2])
    
    # Number of sample points
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    n_both = n1 + n2
    
    #
    # RFF phase
    #

    # Prepare random number generator
    stat_rng = np.random.default_rng(statistic_seed)

    # Get random Fourier features  
    # for the concatenated datasets [hatX1; hatX2]
    features = kernel_features(X1,X2,r,lam=lam,kernel=kernel,
                               feat_type="rff",seed=stat_rng,bin_size=1)

    #
    # Permutation phase
    #

    # Initialize generator for generating permutations
    test_rng = np.random.default_rng(null_seed)

    # Compute permuted squared MMD estimates
    estimator_values = np.empty(B+1)
    for b in range(B+1):
        # First select permuted order
        if b == B:
            # Store original order in final permutation bin
            perm_signs = np.arange(n_both, dtype=int)
        else:
            # Generate random permutation of num_bins_total indices
            perm_signs = test_rng.permutation(n_both)
        # Then assign a +/-1 sign to each datapoint indicating if
        # bin was assigned a permutation index < perm_bins1
        perm_signs = (perm_signs < n1)*2-1

        # Compute the unnormalized squared MMD as a signed vector squared norm
        # ||sum_{i} sign(i) features[i,:]||^2 
        estimator_values[b] = (
            cttc.signed_vector_sqd_norm(features, perm_signs))

    # Normalize squared MMDs
    estimator_values /= (n1*n2)
    
    # Measure runtime
    time_1 = time.time()
    total_time = time_1 - time_0
    
    # Return test results
    return TestResults('rff', estimator_values, total_time, alpha=alpha)

def lrctt(X1,X2,g,r,B=39,a=0,s=16,lam=1.,use_permutations=False,
          kernel="gauss",feat_type="rff",null_seed=None,statistic_seed=None,
          alpha=0.05,delta=0.5):
    """Low-Rank Compress Then Test two-sample test with sample sequences 
    X1 and X2 and auxiliary kernel k' = target kernel k.
    
    Note: Assumes that perm_bin_size = max(1, (n1+n2)//(2*s)) 
      evenly divides n1 and n2
    
    Args:
      X1: 2D array of size (n1,d)
      X2: 2D array of size (n2,d)
      g: compression level; integer >= 0
      r: number of approximate kernel features; positive integer      
      B: number of permutations (int)
      lam: positive kernel bandwidth (float)
      a: coreset size factor; positive integer or zero; if zero, 
        a is set to r / (4^g 2^{indicator[r > 4^{g+1}]}).
      s: total number of permutation bins will be 
        perm_bins = min(2*s, n1+n2);
        X1 will contribute perm_bins * n1 / (n1 + n2) permutation bins;
        X2 will be divided into perm_bins * n2 / (n1 + n2) permutation bins
      use_permutations: if True, simulate null with permutations; otherwise
        use Rademacher wild bootstrap (in this case n1 must equal n2)
      kernel: kernel name; valid options include "gauss" for Gaussian kernel
        exp(-||x-y||^2/lam^2)
      feat_type: name of feature approximation; a valid input to features()
      null_seed: seed used to initialize random number generator for
        randomness in simulating null; can be any valid input to 
        numpy.random.default_rng
      statistic_seed: seed used to initialize random number generator for
        randomness in computing test statistic; can be any valid input to 
        numpy.random.default_rng
      alpha: test level
      delta: KT-Compress failure probability

    Returns: TestResults object.
    """
    # Measure runtime
    time_0 = time.time()
    
    if kernel != "gauss":
        raise ValueError(f"Unsupported kernel name {kernel}")
    if feat_type != "rff":
        raise ValueError(f"Unsupported feature type {feat_type}")
    
    # Store squared bandwidth in an array
    lam_sqd = np.array([lam**2])
    
    # Number of sample points
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    
    # Number of permutation bins per dataset
    n_both = n1 + n2
    perm_bins = min(2*s, n_both)
    perm_bins1 = n1 * perm_bins // n_both
    perm_bins2 = perm_bins - perm_bins1

    four_to_g = 4**g
                    
    #
    # Compression phase
    #

    if a == 0:
        # Specify (r,g)-dependent value of a
        if r <= 4*four_to_g:
            # For small values of r, use RFF without compression
            # Since comp_bins1 = 4^g n1 a^2 / r^2, we are effectively setting 
            # a = r / 4^g
            comp_bins1 = n1//four_to_g
            comp_bins2 = n2//four_to_g
        else:
            # For larger values of r, use minimal compression
            # Since comp_bins1 = 4^g n1 a^2 / r^2, we are effectively setting 
            # a = r / (2 * 4^g)
            comp_bins1 = n1//(four_to_g*4)
            comp_bins2 = n2//(four_to_g*4)
    else:
        # Store squared coreset size factor
        a_sqd = a**2
        # Compute number of compression bins per dataset
        r_sqd = r**2
        comp_bins1 = four_to_g * n1 * a_sqd // r_sqd
        comp_bins2 = four_to_g * n2 * a_sqd // r_sqd

    # Skip test if there are fewer compression bins 
    # than requested permutation bins
    if comp_bins1 < perm_bins1 or comp_bins2 < perm_bins2:
        warnings.warn(f'Warning: comp_bins1 = {comp_bins1} < '
                      f'perm_bins1 = {perm_bins1} or '
                      f'comp_bins2 = {comp_bins2} < '
                      f'perm_bins2 = {perm_bins2}; '
                      f'terminating test')
        return None

    # Skip test if requested compression output size
    # is larger than the input size
    out_thresh1 = comp_bins1 * four_to_g
    out_thresh2 = comp_bins2 * four_to_g
    if out_thresh1 > n1 or out_thresh2 > n2:
        warnings.warn(f'Warning: requested compression output size '
                      f'exceeds input size: '
                      f'comp_bins1 * 4^g = {out_thresh1} > '
                      f'n1 = {n1} or ' 
                      f'comp_bins2 * 4^g = {out_thresh2} > '
                      f'n2 = {n2}; terminating test')
        return None

    # Prepare integer random number generator seeds for each dataset
    stat_rng = np.random.default_rng(statistic_seed)
    stat_ss = stat_rng.bit_generator._seed_seq.spawn(2)
    stat_seed1 = stat_ss[0].generate_state(1)
    stat_seed2 = stat_ss[1].generate_state(1)

    # Identify coreset for each compression bin
    if out_thresh1 == n1:
        # Skip compression step
        hatX1 = X1
    else:
        hatX1_indices = compress.compress_gsn_kt(
            X1, g = g, num_bins = comp_bins1, 
            lam_sqd = lam_sqd, delta = delta, 
            seed = stat_seed1)
        hatX1 = X1[hatX1_indices]
    if out_thresh2 == n2:
        # Skip compression step
        hatX2 = X2
    else:
        hatX2_indices = compress.compress_gsn_kt(
            X2, g = g, num_bins = comp_bins2, 
            lam_sqd = lam_sqd, delta = delta, 
            seed = stat_seed2)
        hatX2 = X2[hatX2_indices]

    #
    # RFF phase
    #

    # Get average random Fourier features per permutation bin 
    # for the concatenated datasets [hatX1; hatX2]
    perm_bin_size = hatX1.shape[0] // perm_bins1
    features = kernel_features(
        hatX1,hatX2,r,lam=lam,kernel=kernel,
        feat_type=feat_type,seed=stat_rng,bin_size=perm_bin_size)

    #
    # Permutation phase
    #

    # Initialize generator for generating permutations
    test_rng = np.random.default_rng(null_seed)

    # Compute RFF squared MMD estimate for original data ordering
    # and for each permuted / signed version
    estimator_values = np.empty(B+1)
    
    # If number of features is larger than number of permutation bins
    # form perm_bins by perm_bins linear kernel matrix explicitly
    form_kernel_matrix = (r >= perm_bins) 
    if form_kernel_matrix:
        # Compute the linear kernel matrix features * features^t
        K = np.empty((perm_bins, perm_bins))
        cttc.linear_kernel_same(features, K)

    for b in range(B+1):
        # First select permuted order
        if b == B:
            # Store original order in final permutation bin
            perm_signs = np.arange(perm_bins, dtype=int)
        else:
            # Generate random permutation of num_bins_total indices
            perm_signs = test_rng.permutation(perm_bins)
        # Then assign a +/-1 sign to each datapoint indicating if
        # bin was assigned a permutation index < perm_bins1
        perm_signs = (perm_signs < perm_bins1)*2-1

        if form_kernel_matrix:
            # Compute the unnormalized squared MMD as a signed sum
            # sum_{i,j} sign(i) sign(j) K[i,j] 
            estimator_values[b] = (
                cttc.signed_matrix_sum(K, perm_signs))
        else:
            # Compute the unnormalized squared MMD as a signed vector squared norm
            # ||sum_{i} sign(i) features[i,:]||^2 
            estimator_values[b] = (
                cttc.signed_vector_sqd_norm(features, perm_signs))

    # Normalize squared MMDs
    estimator_values /= (perm_bins1*perm_bins2)
    
    # Measure runtime
    time_1 = time.time()
    total_time = time_1 - time_0

    # Return test results
    return TestResults(f'lrctt-{feat_type}', estimator_values, total_time, alpha=alpha)
    
def actt(X1,X2,g,B=299,B_2=200,B_3=20,s=16,lam=np.array([1.]),weights=np.array([1.]),kernel="gauss",null_seed=None,
         statistic_seed=None,same_compression=True,alpha=0.05):
    """Aggregated Compress Then Test two-sample test with sample sequences 
    X1 and X2 and auxiliary kernel k' = target kernel k.
    
    Note: Assumes that bin_size = max(1, (n1+n2)//(2s)) evenly divides n1 and n2
    
    Args:
      X1: 2D array of size (n1,d)
      X2: 2D array of size (n2,d)
      g: compression level; integer >= 0
      B: number of permutations (int)
      lam: vector of positive real-valued kernel bandwidths 
      s: total number of compression bins will be num_bins = min(2*s, n1+n2);
        X1 will be divided into num_bins * n1 / (n1 + n2) compression bins;
        X2 will be divided into num_bins * n2 / (n1 + n2) compression bins
      kernel: kernel name; valid options include "gauss" for Gaussian kernel
        exp(-||x-y||^2/lam^2)
      null_seed: seed used to initialize random number generator for
        randomness in simulating null; can be any valid input to 
        numpy.random.default_rng
      statistic_seed: seed used to initialize random number generator for
        randomness in computing test statistic; can be any valid input to 
        numpy.random.default_rng
      same_compression: if True, use the sum of kernels to do compression,
        else compress with each kernel separately
        
    Returns: AggregatedTestResults object.
    """
    # Measure runtime
    time_0 = time.time()
    
    if kernel != "gauss":
        raise ValueError(f"Unsupported kernel name {kernel}")
        
    # KT-Compress failure probability
    delta = 0.5
    
    # Store array of squared bandwidths
    lam_sqd = lam**2
    L = len(lam)
    
    # Number of sample poins
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    
    # Number of KT-Compress bins per dataset
    num_bins_total = min(2*s, n1+n2)
    bin_size = (n1+n2) // num_bins_total
    num_bins1 = n1 // bin_size
    num_bins2 = num_bins_total - num_bins1

    # Prepare integer random number generator seeds
    stat_rng = np.random.default_rng(statistic_seed)
    if same_compression:
        # Generate one seed per dataset
        stat_ss = stat_rng.bit_generator._seed_seq.spawn(2)
    else:
        # Generate one seed per bandwidth and dataset
        stat_ss = stat_rng.bit_generator._seed_seq.spawn(2*L)
                        
    # Initialize matrix for storing sum of Gaussian kernel evaluations
    # between each pair of coresets for each bandwidth
    avg_matrix = np.zeros((num_bins_total,num_bins_total,L))

    if same_compression:
        # For all bandwidths jointly, identify coreset for each compression bin
        hatX1_indices = compress.compress_gsn_kt(
            X1, g = g, num_bins = num_bins1, lam_sqd = lam_sqd, 
            delta = delta, seed = stat_ss[0].generate_state(1))
        hatX2_indices = compress.compress_gsn_kt(
            X2, g = g, num_bins = num_bins2, lam_sqd = lam_sqd, 
            delta = delta, seed = stat_ss[1].generate_state(1))

        # For each bandwidth, compute sum of Gaussian kernel evaluations
        # between each pair of coresets
        gaussianc.sum_gaussian_kernel_by_bin_aggregated(
            X1[hatX1_indices], X2[hatX2_indices], lam**2, avg_matrix)
    else:
        # For each bandwidth separately
        for k, bw in enumerate(lam):
            # Identify coreset for each compression bin
            hatX1_indices = compress.compress_gsn_kt(
                X1, g = g, num_bins = num_bins1, lam_sqd = lam_sqd[k:k+1], 
                delta = delta, seed = stat_ss[2*k].generate_state(1))
            hatX2_indices = compress.compress_gsn_kt(
                X2, g = g, num_bins = num_bins2, lam_sqd = lam_sqd[k:k+1], 
                delta = delta, seed = stat_ss[2*k+1].generate_state(1))

            # Compute sum of Gaussian kernel evaluations
            # between each pair of coresets
            gaussianc.sum_gaussian_kernel_by_bin(
                X1[hatX1_indices], X2[hatX2_indices], bw**2, avg_matrix[:,:,k])

    # Normalize avg_matrix by the number of kernel evaluations
    avg_matrix /= (bin_size**2) 

    # Initialize generator for generating permutations
    test_rng = np.random.default_rng(null_seed)

    # Compute signed sums of avg_matrix for each permutation
    signed_sums = np.zeros((B+B_2+1,len(lam)))
    for b in range(B+B_2+1):
        # First select permuted order
        if b == B+B_2:
            # Store original order in final permutation bin
            perm_signs = np.arange(num_bins_total, dtype=int)
        else:
            # Generate random permutation of num_bins_total indices
            perm_signs = test_rng.permutation(num_bins_total)
        # Then assign a +/-1 sign to each datapoint indicating if
        # bin was assigned a permutation index < num_bins1
        perm_signs = (perm_signs < num_bins1)*2-1

        # Next compute the signed tensor sum 
        # sum_{i,j} sign(i) sign(j) avg_matrix[i,j,:] 
        cttc.signed_tensor_sum(avg_matrix, perm_signs, signed_sums[b,:])

    # Store permuted squared MMD values by normalizing tensor sums
    all_estimator_values = dict()
    for k, bw in enumerate(lam):
        all_estimator_values[bw] = (
             signed_sums[:,k] / (num_bins1*num_bins2))
        
    # Measure runtime
    time_1 = time.time()
    total_time = time_1 - time_0
        
    # Return test results
    return AggregatedTestResults(f'actt', all_estimator_values, total_time, lam, weights, alpha=alpha, B=B, B_2=B_2, B_3=B_3)
         
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Test Results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

class TestResults:
    """Results of a test based on exchangeable replicates under the null.

    Attributes:
      name: string name associated with this test
      estimator_values: array of length B where final value represents the 
        test statistic computed on original data and others represent 
        exchangeable replicates under the null     
      alpha: nominal level of the test
      statistic_values: scalar test statistic computed on original data
      threshold_values: if original data test statistic > threshold, 
        test rejects null; otherwise, test accepts null
      rejects: 1 if the null is rejected; 0 otherwise
      total_times: total time taken to run test
      fname: file name storing a saved copy of this object
    """
    
    def __init__(self, name, estimator_values, total_time, alpha=0.05):
        """Runs hypothesis test given exchangeable test statistics under
        the null
        
        Args:
          name: string name associated with this test
          estimator_values: array of length B where final value represents the 
            test statistic computed on original data and others represent 
            exchangeable replicates under the null
          total_times: total time taken to compute the estimator values
          alpha: nominal level of the test
        """
        self.name = name
        self.alpha = alpha
        
        # Extract original data test statistic
        self.statistic_values = estimator_values[-1]
        # Reorder estimator values so that the thresh_index-th value is
        # in the correct position, any smaller values have lower indices,
        # and any larger values have larger indices
        # Note that this can be done in linear time and doesn't require 
        # sorting the entire array
        B_plus_1 = estimator_values.shape[0]
        # Note: we include -1 because indices go from 0 to B instead of 1 to B+1
        thresh_index = np.ceil((1-alpha)*B_plus_1).astype('int')-1
        self.estimator_values = np.partition(estimator_values, thresh_index) 
        # Identify the test statistic threshold / critical value
        self.threshold_values = self.estimator_values[thresh_index]   
        if self.statistic_values > self.threshold_values:
            # Always reject
            self.rejects = 1
        elif self.statistic_values == self.threshold_values:
            # Count the number of values > threshold
            num_greater = (self.estimator_values[thresh_index+1:] > 
                           self.threshold_values).sum()
            # Count the number of values < threshold
            num_less = (self.estimator_values[:thresh_index-1] < 
                         self.threshold_values).sum()
            # Reject a particular fraction of the time to ensure test has level alpha
            self.rejects = (B_plus_1*alpha - num_greater)/(
                B_plus_1 - num_greater - num_less)
        else:
            # Never reject
            self.rejects = 0
            
        # Set default values for uninitialized test members
        self.total_times = total_time
        self.fname = None
        
    def save(self,fname=None):
        """Writes object to the file self.fname after, if fname is not None,
        replacing self.fname with fname
        """
        if fname is not None:
            self.fname = fname
        pickle.dump(self, open(self.fname, 'wb'))
        
class AggregatedTestResults:
    """Results of an aggregated test based on exchangeable replicates under the null.

    Attributes:
      name: string name associated with this test
      alpha: nominal level of the test
      B: number of permutations
      B_2: number of replicates used to compute hat_u_alpha,
        which is used to obtain the threshold
      all_estimator_values: array of length B+B_2+1 where final value represents the 
        test statistic computed on original data and others represent 
        exchangeable replicates under the null
      estimator_values: array containing the last B+1 elements of all_estimator_values
      estimator_values_2: array containing the first B_2 elements of all_estimator_values
      bw: array containing the bandwidths of the aggregated test
      weights_vec: array of the same size as bw containing the weights of aggregated test
      statistic_values: scalar test statistic computed on original data
      threshold_values: used to store the threshold; modified by set_reject and also by 
        set_threshold
      rejects: 1 if the null is rejected; 0 otherwise
      total_times: total time taken to run test
      fname: file name storing a saved copy of this object
    """
    
    def __init__(self, name, all_estimator_values, total_time, bw, weights_vec, B=299, B_2=200, B_3=20, alpha=0.05):
        """Runs aggregated hypothesis test given exchangeable test statistics under
        the null, for different bandwidths
        
        Args:
          name: string name associated with this test
          all_estimator_values: array of length B where final value represents the 
            test statistic computed on original data and others represent 
            exchangeable replicates under the null
          total_times: total time taken to compute the estimator values
          bw: array containing the bandwidths of the aggregated test
          weights_vec: array of the same size as bw containing the weights of aggregated test
          B: number of permutations
          B_2: number of replicates used to compute hat_u_alpha,
            which is used to obtain the threshold
          B_3: number of iterations of the bisection method
          alpha: nominal level of the test
        """
        self.name = name
        self.alpha = alpha
        self.B = B
        self.B_2 = B_2
        self.all_estimator_values = all_estimator_values
        self.bw = bw
        self.weights_vec = weights_vec
        
        #Split into estimator_values and estimator_values_2
        self.estimator_values = dict()
        self.estimator_values_2 = dict()
        self.split_estimators()
        
        #Get statistic value
        self.statistic_values = dict()
        self.set_statistic_value()
        
        #Reorder list of values for each estimator
        self.sort_estimator_values()
        
        #Compute hat_u_alpha using the bisection method
        self.compute_hat_u_alpha(B_3, alpha)
        
        #Get reject
        self.threshold_values = dict()
        self.set_reject()
        
        #Get threshold for median bandwidth single test
        self.set_threshold(alpha)
        
        #Get reject for median bandwidth single test
        self.set_reject_median()
        
        # Set default values for uninitialized test members
        self.total_times = total_time
        self.fname = None
        
    def split_estimators(self):
        """Split the B+B_2+1 estimators into a set of B+1 used as in a regular permutation test,
        and B_2 used in compute_hat_u_alpha
        """
        for bandwidth in self.bw:
            self.estimator_values_2[bandwidth] = self.all_estimator_values[bandwidth][:self.B_2]
            self.estimator_values[bandwidth] = self.all_estimator_values[bandwidth][self.B_2:]
        
    def set_statistic_value(self):
        """Set the statistic value for each bandwidth
        """
        for bandwidth in self.bw:
            self.statistic_values[bandwidth] = self.estimator_values[bandwidth][self.B]
        
    def sort_estimator_values(self):
        """Sort the estimator values for each bandwidth
        """
        for bandwidth in self.bw:
            self.estimator_values[bandwidth][:] = np.sort(self.estimator_values[bandwidth][:])
        
    def set_threshold(self, alpha):
        """Set threshold values of the single test for each bandwidth;
        used to compute the median bandwidth test
        """
        for bandwidth in self.bw:
            threshold_position = np.ceil((1-alpha)*(self.B+1)).astype('int')
            # Note: we include -1 because indices go from 0 to B instead of 1 to B+1
            self.threshold_values[bandwidth] = self.estimator_values[bandwidth][threshold_position-1] 
        
    def set_reject(self):
        """Check whether the aggregated test rejects
        """
        n_bandwidths = self.bw.shape[0]
        quantile = np.ceil((self.B+1)*(1-self.hat_u_alpha*self.weights_vec)).astype(int)
        print(f'self.hat_u_alpha: {self.hat_u_alpha}, self.weights_vec: {self.weights_vec}, quantile: {quantile}, B+1: {self.B + 1}.')
        self.rejects = 0
        for i in range(n_bandwidths):
            self.threshold_values[self.bw[i]] = self.estimator_values[self.bw[i]][quantile[i]-1]
            if self.statistic_values[self.bw[i]] > self.estimator_values[self.bw[i]][quantile[i]-1]:
                self.rejects = 1     
        
    def compute_hat_u_alpha(self, B_3, alpha):
        """Computes hat_u_alpha using the bisection method; 
        hat_u_alpha is used to get the quantile of the threshold
        """
        n_bandwidths = self.bw.shape[0]
        u_min = 0
        u_max = 1/np.max(self.weights_vec)
        for k in range(B_3):
            u = (u_min+u_max)/2
            P_u = 0
            quantile = np.ceil((self.B+1)*(1-u*self.weights_vec)).astype(int)
            for m in range(self.B_2):
                indicator_max_value = 0
                for i in range(n_bandwidths):
                    if self.estimator_values_2[self.bw[i]][m] - self.estimator_values[self.bw[i]][quantile[i]-1] > 0:
                        indicator_max_value = 1
                        break
                P_u += indicator_max_value
            print(f'P_u (before dividing): {P_u}. self.B_2: {self.B_2}. u: {u}. self.weights_vec: {self.weights_vec}. quantile: {quantile}.')
            P_u = P_u/self.B_2
            if P_u <= alpha:
                u_min = u
            else:
                u_max = u
            print(f'P_u = {P_u}, alpha = {alpha}')
            print(f'k={k}: u_min={u_min}')
        self.hat_u_alpha = u_min
        
    def save(self,fname=None):
        """Writes object to the file self.fname after, if fname is not None,
        replacing self.fname with fname
        """
        if fname is not None:
            self.fname = fname
        pickle.dump(self, open(self.fname, 'wb'))
        
    def set_reject_median(self):
        """Checks whether the median bandwidth single tests rejects
        """
        n_bandwidths = self.bw.shape[0]
        last_bw = self.bw[n_bandwidths-1]
        if self.statistic_values[last_bw] > self.threshold_values[last_bw]:
            self.rejects_median = 1
        else:
            self.rejects_median = 0
            
class GroupResults:
    """Stores the results for a group of tests, e.g. for the group ctt, 
    which contains the tests 't'+str(g) for g in args.block_g_list
    
    Attributes:
      group_names: names of the tests in the group
      full_group_names: used to refer to the group when printing information
      group_labels: labels of the tests in the group
      B: number of permutations
      B_2: number of replicates used to compute hat_u_alpha,
        which is used to obtain the threshold in aggregated tests; 
        None for single tests
      n_bandwidths: 1 if single test, > 1 if aggregated test
      bw: bandwidth (scalar) if single test, array if of bandwidths if aggregated test
      weights_vec: None if single test, array of weights if aggregated
      fname: dictionary containing a file (string) where the results of each test 
        in the group will be stored
      file_exists: dictionary containing a Boolean variable indicating whether the files in
        fname already exist
      fname_group: string, file where the group results will be stored
      file_exists_group: True if the file fname_group already exists, False otherwise
      compute: dictionary containing a Boolean variable for each test in the group,
        indicating wether it needs to be computed
      compute_group: Boolean, True if the whole group needs to be computed; False otherwise
      group_tests: dictionary that stores the group_tests array for each test in the group
      rejects: dictionary that stores the rejects array for each test in the group
      rejects_median: dictionary that stores the rejects_median array for each test in the group
      statistic_values: dictionary that stores the statistic_values array for each test in the group
      threshold_values: dictionary that stores the threshold_values array for each test in the group
      times: dictionary that stores the times array for each test in the group
    """
    def __init__(self, n_tests, B, fname_group, file_exists_group, n_bandwidths, bw, B_2 = 0, weights_vec = None):
        self.group_names = None
        self.full_group_names = None
        self.group_labels = None
        self.B = B
        self.B_2 = B_2
        self.n_bandwidths = n_bandwidths
        self.bw = bw
        self.weights_vec = weights_vec
        self.fname = None
        self.file_exists = None
        self.fname_group = fname_group
        self.file_exists_group = file_exists_group
        self.compute = dict()
        self.compute_group = True
        self.group_tests = dict()
        self.rejects = dict()
        self.rejects_median = dict()
        self.statistic_values = dict()
        self.threshold_values = dict()
        self.times = dict()
         
    def set_group_names(self, group_names, full_group_names, group_labels):
        """Sets group names, full group names (used to print information), 
        and group labels (used in plots)
        """
        self.group_names = group_names
        self.full_group_names = full_group_names
        self.group_labels = group_labels
        
    def set_compute_group(self, no_compute, recompute):
        """Set compute_grop attribute, which specifies whether the group needs to be computed
        """
        if no_compute:
            self.compute_group = False
        elif recompute:
            self.compute_group = True
        elif self.file_exists_group:
            self.compute_group = False
        else:
            self.compute_group = True
            
    def set_compute(self, no_compute, recompute, fname, file_exists):
        """Check whether each test in the group needs to be computed
        """
        self.fname = fname
        self.file_exists = file_exists
        for name in self.group_names:
            if no_compute:
                self.compute[name] = False
            elif recompute:
                self.compute[name] = True
            elif self.file_exists[name]:
                self.compute[name] = False
            else:
                self.compute[name] = True
        
    def set_group_results(self):
        """Store the information of each test in the group in the corresponding attribute of
        the GroupResults object
        """
        if self.n_bandwidths == 1:
            for name in self.group_names:
                self.rejects[name] = self.group_tests[name].rejects
                self.statistic_values[name] = self.group_tests[name].statistic_values
                self.threshold_values[name] = self.group_tests[name].threshold_values
                self.times[name] = self.group_tests[name].total_times
        else:
            for name in self.group_names:
                self.rejects[name] = self.group_tests[name].rejects
                self.rejects_median[name] = self.group_tests[name].rejects_median
                self.times[name] = self.group_tests[name].total_times
                self.statistic_values[name] = dict()
                self.threshold_values[name] = dict()
                for bandwidth in self.bw:
                    self.statistic_values[name][bandwidth] = self.group_tests[name].statistic_values[bandwidth]
                    self.threshold_values[name][bandwidth] = self.group_tests[name].threshold_values[bandwidth]
        
    def save_results(self):
        """Save the results of the group
        """
        res = {
            'rejects': self.rejects,
            'rejects_median': self.rejects_median,
            'statistic_values': self.statistic_values,
            'threshold_values': self.threshold_values,
            'times': self.times,
            'group_names': self.group_names,
            'full_group_names': self.full_group_names,
            'group_labels': self.group_labels,
            'bw': self.bw,
        }
        
        print(f'save_results: {self.group_names}')
        print(f'file name: {self.fname_group}')

        pickle.dump(res, open(self.fname_group, 'wb'))
