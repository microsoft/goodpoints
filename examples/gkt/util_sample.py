
import numpy as np
import numpy.random as npr
import numpy.linalg as npl
import os
import pickle as pkl

'''
File containing helper functions for details about target P and
drawing samples / loading mcmc samples from file
'''
######## functions related to setting P and sampling from it ########


def sample(n, params_p, seed=None):
    """Returns n sample points drawn iid from a specified distribution
    
    Args:
      n: Number of sample points to generate
      params_p: Dictionary of distribution parameters including
        name: Distribution name in {"gauss"}
        var: Variance parameter
        d: Dimension of generated vectors
      seed: (Optional) Random seed to set prior to generation; if None,
        no seed will be set
    """
    name = params_p["name"]
    if name == "gauss":
        sig = np.sqrt(params_p["var"])
        return(sig * npr.default_rng(seed).standard_normal(size=(n, params_p["d"])))
    elif name == "unif":
        return(npr.default_rng(seed).random(size=(n, params_p["d"])))
    elif name == "mog":
        rng = npr.default_rng(seed)
        w = params_p["weights"]
        n_mix = rng.multinomial(n, w)
        for i, ni in enumerate(n_mix):
            mean = params_p["means"][i, :]
            cov = params_p["covs"][i, :, :]
            temp = rng.multivariate_normal(mean=mean, cov=cov, size=ni)
            if i == 0:
                x = temp
            else:
                x = np.vstack((x, temp))
        rng.shuffle(x)
        return(x)
    elif params_p["name"] == "diag_mog":
        rng = npr.default_rng(seed)
        w = params_p["weights"]
        d = params_p["d"]
        n_mix = rng.multinomial(n, w)
        for i, ni in enumerate(n_mix):
            mean = params_p["means"][i, :]
            cov = params_p["covs"][i] * np.eye(d)
            temp = rng.multivariate_normal(mean=mean, cov=cov, size=ni)
            if i == 0:
                x = temp
            else:
                x = np.vstack((x, temp))
        rng.shuffle(x)
        return(x)
    elif params_p["saved_samples"] == True:
        if 'Hinch' in params_p["name"]: # for this case, samples are all preloaded
            assert(params_p["include_last"])
            filename = os.path.join(params_p["data_dir"], "{}_samples_n_{}.pkl".format(params_p["name"], n))
            with open(filename, 'rb') as file:
                return(pkl.load(file))
        else:
            if '_float_step' in params_p["name"]:
                assert(params_p["include_last"])
                end = params_p["X"].shape[0]
                sample_idx = np.linspace(0, end-1, n,  dtype=int, endpoint=True)
                return(params_p["X"][sample_idx])
            else:
                end = params_p["X"].shape[0]
                # compute thinning parameter
                step_size = int(end / n)
                start = end-step_size*n
                assert(step_size>=1)
                assert(start>=0)
                if params_p["include_last"]:
                    return(params_p["X"][end-1:start:-step_size][::-1])
                else:
                    return(params_p["X"][start:end:step_size])
        
    raise ValueError("Unrecognized distribution name {}".format(params_p["name"]))
        
    raise ValueError("Unrecognized distribution name {}".format(params_p["name"]))

def compute_diag_mog_params(M=int(4), snr=3.):
    """Returns diagonal mixture of Gaussian target distribution settings for d=2
    
    Args:
      M: (Optional) Integer, number of components
      snr: (Optional) Scaling of the means    
    """
    d = int(2)
    weights = np.ones(M)
    weights /= np.sum(weights)

    # change this to set the means apart
    means = np.zeros((M, d))
    if M == 3:
        means = snr*np.array([[1., 1.], [-1., 1], [-1., -1.]])
    if M == 4: 
        means = snr*np.array([[1., 1.], [-1., 1], [-1., -1.], [1., -1.]])
    if M == 6:
        means = snr*np.array([[1., 1.], [-1., 1], [-1., -1.], [1., -1.], [0, 2.], [-2, 0.]])
    if M == 8:
        means = snr*np.array([[1., 1.], [-1., 1], [-1., -1.], [1., -1.], [0, 2.], [-2, 0.], [2, 0.], [0, -2.]])
    covs = np.ones(M)


    # compute the expected value of E[||X-Y||^2] for X, Y iid from P
    mean_sqdist = 0.
    for i in range(M):
        for j in range(M):
            temp = npl.norm(means[i])**2 + npl.norm(means[j])**2 - 2 * np.dot(means[i], means[j])
            temp += d*(covs[i]+ covs[j])
            mean_sqdist += weights[i] * weights[j] * temp
            
    params_p = {"name": "diag_mog", 
                 "weights": weights,
                "means": means,
                "covs": covs,
                "d": int(d),
                "mean_sqdist" : mean_sqdist,
               "saved_samples": False,
               "flip_Pnmax": False
               }
    return(params_p) 

def compute_mcmc_params_p(filename, nmax=int(2**15), include_last=True, profiling=False, flip_Pnmax=False):
    """Returns a dictionary for a distribution associated with samples saved in filename
    
    Args:
      filename: string, denoting a prefix to be used for the file fron which the samples are loaded
      nmax:(Optional) Integer, to define Pnmax as an approximation to P of samples
      include_last:(Optional) If True, always includes the last point from the loaded coreset, 
                  otherwise always includes the first point
      profiling:(Optional) If True, debugging mode, returns some indices        
    """
    # burn_in_parameters for the 8 settings of Lotka, and Goodwin, taken as the max value in Table S4 and S6 for the respective setting in 
    # https://arxiv.org/pdf/2005.03952.pdf
    
    burn_in_params = {
        'Goodwin_RW': int(820000),
        'Goodwin_ADA-RW': int(824000),
        'Goodwin_MALA': int(1615000),
        'Goodwin_PRECOND-MALA': int(1475000),
        'Lotka_RW': int(1512000),
        'Lotka_ADA-RW': int(1797000),
        'Lotka_MALA': int(1573000),
        'Lotka_PRECOND-MALA': int(1251000),
    }
    
    
    gl_filenames = list(burn_in_params.keys())  # goodwin lotka filenames 
        
    
    # The Hinch sample files are already preprocessed
    Hinch_filenames = ['Hinch_P_seed_1_temp_1', 'Hinch_P_seed_2_temp_1', 'Hinch_TP_seed_1_temp_8', 'Hinch_TP_seed_2_temp_8']
    Hinch_scaled_filenames = [f + '_scaled' for f in Hinch_filenames] # for samples after scaling
    
    for f in Hinch_filenames:
        burn_in_params[f] = int(0)
    for f in Hinch_scaled_filenames:
        burn_in_params[f] = int(0)

    
    med_dist_params =  {'Goodwin_RW': 0.02,
     'Goodwin_ADA-RW': 0.0201,
     'Goodwin_MALA': 0.0171,
     'Goodwin_PRECOND-MALA': 0.0205,
     'Lotka_RW': 0.0274,
     'Lotka_ADA-RW': 0.0283,
     'Lotka_MALA': 0.023,
     'Lotka_PRECOND-MALA': 0.0288, 
     'Hinch_P_seed_1_temp_1': 2.3748,
     'Hinch_P_seed_2_temp_1': 2.3311,
     'Hinch_TP_seed_1_temp_8': 7.3232, 
     'Hinch_TP_seed_2_temp_8': 7.2557,
     'Hinch_P_seed_1_temp_1_scaled': 8.0676,
     'Hinch_P_seed_2_temp_1_scaled': 8.3189,
     'Hinch_TP_seed_1_temp_8_scaled': 8.621, 
     'Hinch_TP_seed_2_temp_8_scaled': 8.6649,                        
                       }
    
    # creating dictionary items for float step size settings of Goodwin/Lotka mcmc
    for f in gl_filenames:
        g = f+'_float_step'
        burn_in_params[g] = burn_in_params[f]
        med_dist_params[g] = med_dist_params[f]
        
    assert(filename in med_dist_params.keys())
    
    params_p = {"saved_samples": True, 
                "name": filename, 
                "data_dir": "/accounts/projects/binyu/raaz.rsk/kernel_thinning/kernel_thinning_plus/data",
                "nmax": nmax,
                "burn_in": burn_in_params[filename],
                "include_last" : include_last,
                "med_dist" : med_dist_params[filename]
                }
    params_p["d"] = int(38) if 'Hinch' in filename else int(4)
    
    # flip Pnmax for Hinch cardiac experiments
    params_p["flip_Pnmax"] = True if flip_Pnmax else False
    
    # specific filename to load the coresets
    if 'Hinch' in filename:
        assert(include_last)
        
        p_name = params_p["name"]
        
        # flip the Pnmax
        if params_p["flip_Pnmax"]:
            if "seed_1" in p_name:
                p_name = p_name.replace("seed_1", "seed_2")
            elif "seed_2" in p_name:
                p_name = p_name.replace("seed_2", "seed_1")
            filename = os.path.join(params_p["data_dir"], "{}_pnmax_15.pkl".format(p_name))
        else:
            filename = os.path.join(params_p["data_dir"], "{}_pnmax_15.pkl".format(p_name))
        assert(nmax == int(2**15))
        with open(filename, "rb") as file:
            params_p["Pnmax"] = pkl.load(file)
    else:
        # for non-Hinch experiments
        
        # when dealing with step-size fixed for standard thinning
        if '_float_step' in filename:
            pkl_name = os.path.join(params_p["data_dir"], "{}_theta.pkl".format(params_p["name"].replace("_float_step", "")))
        else:
            pkl_name = os.path.join(params_p["data_dir"], "{}_theta.pkl".format(params_p["name"]))
        with open(pkl_name, "rb") as file:
            X = pkl.load(file)
        burn_in = params_p["burn_in"]
        X = X[burn_in:]
        
        # separate in odd/even indices
        idx_even = np.arange(X.shape[0]-1, 1, -2)[::-1]
        idx_odd = np.arange(X.shape[0]-2, 0, -2)[::-1]
        assert(len(set(idx_even).intersection(set(idx_odd)))==0)
        # points to be used for drawing samples
        if '_float_step' in filename:
            idx_Pnmax = np.linspace(0, len(idx_odd)-1,  nmax, dtype=int, endpoint=True)
            params_p["Pnmax"] = X[idx_odd][idx_Pnmax]
            params_p["X"] = X[idx_even]
        else:
            params_p["X"] = X[idx_even] # all even samples will be be used for drawing samples
            # compute Pnmax for evaluatibng mmd using all odd samples so that they do not overlap with the samples used for KT
            X = X[idx_odd]
            end = X.shape[0]
            nmax = params_p["nmax"]
            step_size = int(end / nmax)
            assert(step_size>=1)
            # compute Pnmax by standard  thinning
            if include_last:
                idx_Pnmax = np.arange(end-1, 0, -step_size)[:nmax][::-1] # standard thin from the end

                params_p["Pnmax"] = X[idx_Pnmax]
            else:
                idx_Pnmax = np.arange(0, end, step_size)[:nmax] # standard thin from the beginning
                params_p["Pnmax"] = X[idx_Pnmax]
            assert(len(idx_Pnmax)==nmax)
    
    if not profiling:
        return(params_p)
    else:
        return(params_p, idx_Pnmax, idx_even, idx_odd)
    
def sample_string(params_p, sample_seed):
    """Returns string summarizing the parameters of the target distribution P 
    for appending to pkl file names
    
    Args:
      params_p : Dictionary of distribution parameters recognized by sample()
      sample_seed: random seed used for generating data from P
    """
    if params_p["saved_samples"] == True:
        temp = params_p["name"]
        if "med_dist" in params_p.keys():
            if params_p["include_last"]:
                temp += "_endpt"
            else:
                temp += "_startpt"

            temp += "_nmax_" + str(int(np.log2(params_p["nmax"])))
            return(temp)
        else:
            # older file names
            return(params_p["name"])
    if params_p["name"] == "gauss":
        return("{}_var{}_seed{}".format(
                params_p["name"], params_p["var"], sample_seed))
    if params_p["name"] == "diag_mog":
        return("{}_comp{}_seed{}".format(
                params_p["name"], len(params_p["weights"]), sample_seed))
    if params_p["name"] == "mog":
        return("{}_comp{}_seed{}".format(
                params_p["name"], len(params_p["weights"]), sample_seed))
    
        
    raise ValueError("Unrecognized distribution name {}".format(params_p["name"]))


def compute_params_p(args):
    ''' 
        return dimensionality, params_p, and var_k, for the experiment
    '''
    ## P and kernel parameters ####
    if args.P == "gauss":       
        d = args.d
        var_p = 1. # Variance of P
        var_k = float(2*d) # Variance of k
        params_p = {"name": "gauss", "var": var_p, "d": int(d), "saved_samples": False,
                   "flip_Pnmax": False}
        
    if args.P == "mog":
        # d will be set to 2
        assert(args.M in [3, 4, 6, 8])
        params_p = compute_diag_mog_params(args.M)
        d = params_p["d"]
        var_k = float(2*d)
        
    if args.P == "mcmc":
        # d will be set automatically; nmax needs to be changed
        assert(args.filename is not None)
        params_p = compute_mcmc_params_p(args.filename)
        d = params_p["d"]
        var_k = (params_p["med_dist"])**2
    return(d, params_p, var_k)