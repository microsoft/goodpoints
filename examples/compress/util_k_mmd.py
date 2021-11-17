'''File containing helper functions for details about kernel evaluations, and
evaluating mmd for a given kernel and two target distributions 
'''

import numpy as np
import numpy.random as npr
import numpy.linalg as npl
import os
import pickle as pkl
from scipy.stats import multivariate_normal
import pathlib

from util_sample import sample_string

# for partial functions, to use kernel_eval for kernel
from functools import partial

def kernel_eval(x, y, params_k):
    """Returns matrix of kernel evaluations kernel(xi, yi) for each row index i.
    x and y should have the same number of columns, and x should either have the
    same shape as y or consist of a single row, in which case, x is broadcasted 
    to have the same shape as y.
    """
    if params_k["name"] in ["gauss", "gauss_rt"]:
        k_vals = np.sum((x-y)**2,axis=1)
        scale = -.5/params_k["var"]
        return(np.exp(scale*k_vals))
    
    raise ValueError("Unrecognized kernel name {}".format(params_k["name"]))

def compute_params_k(var_k, d,  use_krt_split=1, name="gauss"):
    '''
        return parameters, and functions for split and swap kernel;
        parameters returned should be understood by kernel_eval, p_kernel,
        ppn_kernel, and pp_kernel functions
        
        var_k: float scale for the kernel
        d: dimensionality of the problem
        use_krt_split: Whether to use krt for split
        setting: which kernel to use; kernel_eval needs to be defined for that setting
    '''
    params_k_swap = {"name": name, "var": var_k, "d": int(d)}
    if name=="gauss":
        if use_krt_split != 0: 
            params_k_split = {"name": "gauss_rt", "var": var_k/2., "d": int(d)}
        else:
            params_k_split = {"name": "gauss", "var": var_k, "d": int(d)}
    
    split_kernel = partial(kernel_eval, params_k=params_k_split)
    swap_kernel = partial(kernel_eval, params_k=params_k_swap)
    return(params_k_split, params_k_swap, split_kernel, swap_kernel)

### functions for computing mmds

def p_kernel(y, params_k, params_p):
    """Evaluates the function Pk(.) = E_Y K(., Y)"""
    if params_k["name"] == "gauss" and params_p["name"] == "gauss":
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
        # return PnPn with respect to Sin loaded in Pnmax
        filename = os.path.join(params_p["data_dir"], 
                                "pnpn_{}_n{}_k_{}_var{}.pkl".format(params_p["name"], 
                             len(params_p["Pnmax"]), params_k["name"], params_k["var"]))
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                ans = pkl.load(file)
        else:
            ans = pnpn_kernel(params_k, params_p["Pnmax"])
            with open(filename, 'wb') as file:
                pkl.dump(ans, file, protocol=pkl.HIGHEST_PROTOCOL)
        return(ans)
    
    if params_p["saved_samples"] == True:
        filename = os.path.join(params_p["data_dir"], 
                                "{}_ppnew_nmax{}_k_{}_var{}.pkl".format(params_p["name"], 
                             params_p["nmax"], params_k["name"], params_k["var"]))
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

def pnpn_kernel(params_k, xn):
    """Returns the average of all pairwise kernel evaluations"""
    ans = 0.
    for x in xn:
        ans += np.mean(kernel_eval(x.reshape(1,-1), xn,  params_k))
    return(ans/xn.shape[0])
    
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

def get_combined_mmd_filename(prefix, ms, params_p, params_k_split, params_k_swap, rep_ids, delta=0.5, 
                      sample_seed=1234567, thin_seed=9876543, results_dir="results_new/combined"):
    """
    Generate the filename to load mmd results for a given set of coresets
    
    prefix: To flag the type of coreset MC/KT/KT+/KT++
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
    split_kernel_str = "{}_var{}_seed{}".format(params_k_split["name"], params_k_split["var"], thin_seed)
    swap_kernel_str =  "{}_var{}".format(params_k_swap["name"], params_k_swap["var"])
    thresh_str = f"delta{delta}"
    combined_mmd_filename = os.path.join(results_dir, f"{prefix}-combined-mmd-{sample_str}-split{split_kernel_str}-swap{swap_kernel_str}-d{d}-m{max(ms)}-{thresh_str}-rep{len(rep_ids)}.pkl")
    return(combined_mmd_filename)