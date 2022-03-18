'''File containing helper functions for details about kernel evaluations, and
evaluating mmd for a given kernel and two target distributions 
'''

import numpy as np
import numpy.random as npr
import numpy.linalg as npl
import os
import pickle as pkl
from scipy.stats import multivariate_normal
from scipy.special import gamma, kv, binom
import pathlib
from util_sample import sample_string


def kernel_eval(x, y, params_k):
    """Returns matrix of kernel evaluations kernel(xi, yi) for each row index i.
    x and y should have the same number of columns, and x should either have the
    same shape as y or consist of a single row, in which case, x is broadcasted 
    to have the same shape as y.
    
    all kernels are parameterized to have infnorm = 1
    """
    if "combo" in params_k["name"]:
        return( kernel_eval(x, y, params_k["k"]) + kernel_eval(x, y, params_k["kpower"]) )
    
    ## note below the reason to use elif; the format is modified to allow for general root kernels
    if "gauss" in params_k["name"]:
        dist_sq = np.sum((x-y)**2,axis=1)
        scale = -.5/params_k["var"]
        return(np.exp(scale*dist_sq))
    
    elif "sinc" in params_k["name"]:
        theta = 1/np.sqrt(params_k["var"])
        return(np.prod(np.sinc(theta*(x-y)), axis=1))
    
    elif params_k["name"] == "laplace": # its a-power is not laplace
        dist = np.sqrt(np.sum((x-y)**2,axis=1))
        scale = -.5/np.sqrt(params_k["var"])
        return(np.exp(scale*dist))
    elif "imq" in params_k["name"]: # currently we implement only imq power kernels; not matern
        dist_sq =  np.sum((x-y)**2,axis=1) 
        return((1+dist_sq/params_k["var"])**(-params_k["nu"]))
    
    elif ("matern" in params_k["name"]) or ("laplace" in params_k["name"]): # since laplace a-power is a matern kernel; and that's why we use elif
        nu_eff = params_k["nu"] - params_k["d"]/2.
        dist = np.sqrt(np.sum((x-y)**2,axis=1)) / np.sqrt(params_k["var"])       
        
        k_vals = np.zeros(max(x.shape[0], y.shape[0]))
        zero_dist = np.isclose(dist, 0)
        k_vals[zero_dist] = 1.

        k_vals[~zero_dist] = 2 ** (1 - nu_eff) / gamma(nu_eff) *  (np.sqrt(2 * nu_eff) * dist[~zero_dist] ) ** nu_eff *   ( kv(nu_eff, np.sqrt(2 * nu_eff) * dist[~zero_dist] ) )
        return(k_vals)
    elif "bspline" in params_k["name"]:
        dist = 1/np.sqrt(params_k["var"]) * (x-y)
        return(np.prod( bspline(dist, int(2*params_k["nu"]+2))  , axis=1) ) # params_k["nu"] denotes beta in our convention
    else:
        raise ValueError("Unrecognized kernel name {}".format(params_k["name"]))

# helper for bspline
def bspline(x, beta):
    '''
    Return bspline values of order beta with shape same as x
    given by App. I.4.1 of Kernel Thinning https://arxiv.org/pdf/2105.05842.pdf
    
    Args:
        x: shape(n, d)
        beta: integer
    Returns:
        bspline(x, beta) of shape n
    '''
    ans = np.zeros(x.shape)
    for j in range(beta+1):
        temp = (beta / 2 - j + x)
        ans += (-1)**j * binom(beta, j) * (temp)**(beta-1) * (temp>=0)
    return(ans / np.math.factorial(beta-1))
        
    
def compute_power_kernel_params_k(params_k, power):
    '''
    returns dictionary of power kernel parameter
    
    params_k: dictionary for kernel params
    power: power of the kernel to be evaluated must lie in [0.5, 1]; and needs to satisfy other conditions for different kernels 
          based on the discussion of last appendix of https://arxiv.org/pdf/2110.01593.pdf 
    
    '''
    
    params_k_power = dict()
    suffix = "_rt" if power == 0.5 else f"{power}_rt"
    params_k_power["name"]  = params_k["name"] + suffix
    
    d = params_k["d"]
    params_k_power["d"] = params_k["d"]
    
    if "gauss" in params_k["name"]:
        params_k_power["var"] = params_k["var"] * power
        return(params_k_power)
    
    if "sinc" in params_k["name"]:
        params_k_power["var"] = params_k["var"] # the var parameter doesn't change, only the kernel gets scaled; which we don't care for KT
        return(params_k_power)
    
    if "laplace" in params_k["name"]:
        params_k_power["var"] = params_k["var"]
        assert(power> d/(d+1.))
        params_k_power["nu"] = power * (d+1)/2.
        return(params_k_power)
    
    if "matern" in params_k["name"]:
        params_k_power["var"] = params_k["var"]
        nu = params_k["nu"]
        assert(power > d/(2. * nu))
        params_k_power["nu"] = power * nu
        return(params_k_power)
    
    if "imq" in params_k["name"]:  # currently we implement only sqrt imq kernels using eqn 117 from https://arxiv.org/pdf/2105.05842.pdf [our choice is valid for all nu]
        assert(power==0.5)
        params_k_power["var"] = params_k["var"]
        params_k_power["nu"] = d/4 + params_k["nu"]/2.
        return(params_k_power)
        
    if "bspline" in params_k["name"]:  # taken from last appendix of https://arxiv.org/pdf/2110.01593.pdf
        beta = params_k["nu"]
        if beta%2 == 1:
            assert(power == 0.5)
        if beta%2 == 0:
            assert(power== (beta+2)/(2*beta+2))
            params_k_power["var"] = params_k["var"]
            params_k_power["nu"] = int(beta/2)
            assert(params_k_power["nu"] >=0)
        return(params_k_power)

    raise ValueError("Unrecognized kernel name {}".format(params_k["name"]))

def compute_params_k(args, var_k, power_kernel=True, power=0.5):
    '''
        Return dictionary parameters of (target) kernel
        
        Args: A dictionary of arguments
        var_k: (Float) Variance parameter for kernel
        Power: (Optional) Boolean if power kernel should be computed; if False we return an additional empty dict
        power: (Float, Optional)Power parameter for the power kernel
    '''
    
    params_k = {"name": args.kernel, "var": var_k, "d": int(args.d)}
    if params_k["name"] in ["imq", "matern", "bspline"]: # add another argument if kernels are IMQ/Matern/Bspline
        params_k["nu"] = args.nu
    
    if power_kernel: # compute power_kernel
        params_k_power = compute_power_kernel_params_k(params_k, power)
    else: # assign empty dict
        params_k_power = dict() 
    return(params_k, params_k_power)


### functions for computing mmds

def p_kernel(y, params_k, params_p):
    """Evaluates the function Pk(.) = E_Y K(., Y)"""
    if y.ndim == 1: # if a single data point is passed as flat array, reshape it
        y = y.reshape(1, -1)
    if params_k["name"] == "gauss" and params_p["name"] == "gauss":
        ## int exp(-(x-y))^2/(2a^2)) exp(-y^2/(2b^2)) dy = (2 * pi * a^2 * b^2/(a^2+b^2) )^(d/2) * exp(-x^2/(2(a^2+b^2)))
        var = params_k["var"] + params_p["var"]
        factor = np.sqrt(params_k["var"] / var)
        d = params_k["d"]
        return(factor**d * np.exp(-npl.norm(y, axis=1)**2/(2*var)))

    if params_k["name"] == "gauss" and params_p["name"] == "mog":
        # https://arxiv.org/pdf/1501.02056.pdf Sec 2.2
        d  = params_p["d"]
        ws = params_p["weights"]
        vark = params_k["var"]
        means = params_p["means"]
        ans = np.zeros(len(y))
        for i in range(len(ws)):
            ans += ws[i] * multivariate_normal(params_p["means"][i], params_p["covs"][i] + vark * np.eye(d)).pdf(y)
        return(ans * (2 * np.pi* vark )**(d/2) ) 

    if params_k["name"] == "gauss" and params_p["name"] == "diag_mog":
        # https://arxiv.org/pdf/1501.02056.pdf Sec 2.2 or Last appendix of Kernel thinning
        d  = params_p["d"]
        ws = params_p["weights"]
        vark = params_k["var"]
        means = params_p["means"]
        ans = np.zeros(len(y))
        for i in range(len(ws)):
            ans += ws[i] * multivariate_normal(params_p["means"][i], (params_p["covs"][i] + vark) * np.eye(d)).pdf(y)
        return(ans * (2 * np.pi* vark )**(d/2) ) 
    
    if "sin" in params_p["name"] or params_p["saved_samples"] == True:
        ans = np.zeros(len(y))
        if len(y) == 1: # to speed up 
            return(np.array([np.mean(kernel_eval(params_p["Pnmax"], y,  params_k))] ))
        else:
            for x in params_p["Pnmax"]:
                ans += kernel_eval(x.reshape(1,-1), y,  params_k)
            return(ans/params_p["Pnmax"].shape[0])
        
    raise ValueError("Unrecognized kernel name {}".format(params_k["name"]))
    
def pp_kernel(params_k, params_p):
    """Returns the expression for PPk = E_X E_Y K(X, Y)"""
    
    if params_k["name"] == "gauss" and params_p["name"] == "gauss":
        d = params_p["d"]
        ans = np.sqrt(params_k["var"]/(params_k["var"] + 2*params_p["var"]))
        return(ans**d)
    
    if params_k["name"] == "gauss" and params_p["name"] == "mog":
        # see last appendix of kernel thinning for math derivation of PPk
        d  = params_p["d"]
        ws = params_p["weights"]
        vark = params_k["var"]
        means = params_p["means"]
        ans = 0.
        for i in range(len(ws)):
            for j in range(len(ws)):
                Ai = npl.inv(params_p["covs"][i] + vark * np.eye(d))
                Bj = npl.inv(params_p["covs"][j])
                Cij = Ai + Bj
                bij = Ai.dot(means[i]) + Bj.dot(means[j])
                expij = bij.dot(npl.solve(Cij, bij))
                expij -= means[i].dot(Ai.dot(means[i]))
                expij -= means[j].dot(Bj.dot(means[j]))
                expij = np.exp(0.5*expij)
                fij = np.sqrt( np.det(Ai) * np.det(Bj) / np.det(Cij) )
                ans += ws[i] * ws[j] * fij * expij
        return(ans * vark**(d/2)) 

    if params_k["name"] == "gauss" and params_p["name"] == "diag_mog":
        # see last appendix of kernel thinning
        d  = params_p["d"]
        ws = params_p["weights"]
        vark = params_k["var"]
        means = params_p["means"]
        ans = 0.
        for i in range(len(ws)):
            for j in range(len(ws)):
                Ai = 1. / (params_p["covs"][i] + vark)
                Bj = 1. / params_p["covs"][j]
                Cij = Ai + Bj
                bij = Ai * means[i] + Bj * means[j]
                expij = npl.norm(bij)**2 / Cij 
                expij -= Ai * npl.norm(means[i])**2
                expij -= Bj * npl.norm(means[j])**2
                expij = np.exp(0.5*expij)
                fij = (Ai * Bj / Cij)**(d/2)
                ans += ws[i] * ws[j] * fij * expij
        return(ans * vark**(d/2) ) 
    
    if "sin" in params_p["name"]:
        # return PnPn with respect to Sin loaded in Pnmax, and save the answer
        # load_store can be used if we know that the sin is same across experiments, which it typically is not
        # TBD Fix this part of the code
        filename =  "./data/{}_pnpn_nmax{}_k_{}_var{}.pkl".format(params_p["name"], params_p["Pnmax"].shape[0], params_k["name"], params_k["var"])
        return(pnpn_kernel(params_k, params_p["Pnmax"], load_store=False, load_from=filename))
    
    if params_p["saved_samples"] == True:
        p_name = params_p["name"]
        if params_p["flip_Pnmax"]:
            if "seed_1" in p_name:
                p_name = p_name.replace("seed_1", "seed_2")
            elif "seed_2" in p_name:
                p_name = p_name.replace("seed_2", "seed_1")
                
        filename = os.path.join(params_p["data_dir"], 
                                "{}_pp_nmax{}_k_{}_var{}.pkl".format(p_name, params_p["nmax"], params_k["name"], params_k["var"]))
        
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                ans = pkl.load(file)
        else:
            orig_xn = params_p["Pnmax"]
            ans = pnpn_kernel(params_k, orig_xn)
            with open(filename, 'wb') as file:
                pkl.dump(ans, file, protocol=pkl.HIGHEST_PROTOCOL)
        return(ans)
    
    
def ppn_kernel(params_k, params_p, xn):
    """Returns the value of Pk(x) at the points xn"""
    if xn.ndim == 1: # if a single data point is passed as flat array, reshape it
        xn = xn.reshape(1, -1)

    return(np.mean(p_kernel(xn, params_k,  params_p)))

def pnpn_kernel(params_k, xn, load_store=False, load_from=None):
    """Returns the average of all pairwise kernel evaluations"""
    ans = 0.
    if load_store and os.path.exists(load_from):
        with open(load_from, "rb") as file:
            return(pkl.load(file))
    if xn.ndim == 1: # if a single data point is passed as flat array, reshape it
        xn = xn.reshape(1, -1)
    for x in xn:
        ans += np.mean(kernel_eval(x.reshape(1,-1), xn,  params_k))
    ans /= xn.shape[0]
    if load_store:
        with open(load_from, "wb") as file:
            pkl.dump(ans, file, protocol=pkl.HIGHEST_PROTOCOL)
    return(ans)
        
        
def squared_mmd(params_k,  params_p, xn):
    """Computes the squared MMD between a sample point sequence xn and a target distribution P"""
    if xn.ndim == 1:
        xn = xn.reshape(1, -1)
    assert(params_k["d"] == params_p["d"])
    assert(params_k["d"] == xn.shape[1])
    ans = pp_kernel(params_k, params_p)
    ans -= 2*ppn_kernel(params_k, params_p, xn) 
    ans += pnpn_kernel(params_k, xn)  
    return(ans)

def get_combined_results_filename(prefix, ms, params_p, params_k_split, params_k_swap, rep_ids, delta=0.5, 
                      sample_seed=1234567, thin_seed=9876543, results_dir="results_new/combined"):
    """
    Generate the filename to load mmd results for a given set of coresets
    
    prefix: To flag the type of coreset MC/KT/KTkrt; and whether mmd or fun diff values are being stored
    ms: range of thinning rounds
    params_p: Dictionary of distribution parameters recognized by sample()
    params_k: Dictionary of kernel parameters recognized by kernel()
    params_mmd: Dictionary of kernel parameters recognized by kernel()
    rep_ids: Which replication numbers of experiment to run; the replication
        number determines the seeds set for reproducibility
    delta: delta/(4^m) is the failure probability for
        adaptive threshold sequence;
    sample_seed: (Optional) random seed is set to sample_seed + rep
        prior to generating input sample for replication rep
    thin_seed: (Optional) random seed is set to thin_seed + rep
        prior to running thinning for replication rep
    results_dir: Folder where the results are loaded from
    
    """
    # Create results directory if necessary
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # Construct results filename template with placeholder for rep value
    d = params_p["d"]
    assert(d==params_k_split["d"])
    assert(d==params_k_swap["d"])
    sample_str = sample_string(params_p, sample_seed)
    if prefix == "mc":
        split_kernel_str = ""
        swap_kernel_str =  ""
        mc_kernel_str = "{}_var{}".format(params_k_swap["name"], params_k_swap["var"])
        if "imq" in params_k_swap["name"] or "matern" in params_k_swap["name"]  or "bspline" in params_k_swap["name"]:
            mc_kernel_str += f"_nu{params_k_swap['nu']}"
    else:
        split_kernel_str = "split_{}_var{}_seed{}".format(params_k_split["name"], params_k_split["var"], thin_seed)
        if "imq" in params_k_split["name"] or "matern" in params_k_split["name"] or "bspline" in params_k_split["name"]:
            split_kernel_str += f"_nu{params_k_swap['nu']}"
        
        swap_kernel_str =  "swap_{}_var{}".format(params_k_swap["name"], params_k_swap["var"])
        if "imq" in params_k_swap["name"] or "matern" in params_k_swap["name"]:
            swap_kernel_str += f"_nu{params_k_swap['nu']}"
        mc_kernel_str = ""
        
    
    thresh_str = f"delta{delta}"
    
    
    if "sin" not in prefix:
        # if pnmax flipped (for Hinch cardiac experiments)
        if params_p["flip_Pnmax"]:
            sample_str += "_flip_Pnmax_"
        
    combined_mmd_filename = os.path.join(results_dir, f"{prefix}-{sample_str}-{mc_kernel_str}-{split_kernel_str}-{swap_kernel_str}-d{d}-m{max(ms)}-{thresh_str}-rep{len(rep_ids)}.pkl")
    return(combined_mmd_filename)
