import os.path
from util_sample import sample, compute_mcmc_params_p, compute_diag_mog_params, sample_string

def get_combined_file_template(folder, prefix, d, min_input_size, max_input_size, m, params_p, params_k_split, params_k_swap, 
                      delta, experiment_seed, compressalg = None,
                      ):
    '''
    ### currently supports construct_kt_coresets, construct_compresspp_coresets, construct_st_coresets,  ###
    
    prefix: prefix for the setting
    d: dimensionality
    input_size: in log_4 scale
    m: thinning factor with kt.thin in log_2 base (typically equal to input_size)
    sample_seed: seed for generating samples
    thin_seed: seed for kt.thin 
    compress_seed: seed for compress
    
    '''
    assert(d == params_p["d"])
    assert(d == params_k_split["d"])
    assert(d == params_k_swap["d"])
    
    sample_str = sample_string(params_p, sample_seed="")
    split_kernel_str = "-split{}_var{:.3f}".format(params_k_split["name"], params_k_split["var"])
    swap_kernel_str =  "-swap{}_var{:.3f}".format(params_k_swap["name"], params_k_swap["var"])
    compress_alg_str = "-alg{}".format(compressalg) if compressalg is not None else ""
    d_str = f"-d{d}"
    size_str = f"-size{min_input_size}_{max_input_size}"
    m_str = "" if compressalg is not None else f"-m{m}"
    thresh_str = f"-delta{delta}"
    rep_str = f"-rep{{}}"
    
    file_template = os.path.join(folder, 
                                 f"{prefix}-{{}}-{sample_str}{split_kernel_str}{swap_kernel_str}"
                                 +f"{compress_alg_str}{d_str}{size_str}{m_str}{thresh_str}{rep_str}.pkl")

    return(file_template)


def get_file_template(folder, prefix, d, input_size, m, params_p, params_k_split, params_k_swap, 
                      delta, sample_seed, thin_seed=None, 
                      compress_seed=None,
                      compressalg=None, 
                      g=None,
                      ):
    '''
    ### currently supports names required by
    construct_st_coresets,
    construct_herding_coresets,
    construct_kt_coresets,
    construct_compresspp_coresets ###
    
    prefix: prefix for the setting
    d: dimensionality
    input_size: in log_4 scale
    m: thinning factor useful when prefix is KT or CompressBlowup, thinning factor in log_2 base (typically equal to input_size)
    sample_seed: seed for generating samples
    thin_seed: seed for kt.thin useful only when 
    compress_seed: seed for compress
    
    '''
    assert(d == params_p["d"])
    assert(d == params_k_swap["d"])
    
    sample_str = sample_string(params_p, sample_seed)
    
    split_kernel_str = "" # no string
    if prefix != "Herd":
        assert(d == params_k_split["d"])
        split_kernel_str = "-split{}_var{:.3f}".format(params_k_split["name"], params_k_split["var"])
    
    if prefix == "KT":
        split_kernel_str += f"_thinseed{thin_seed}"
    if prefix == "Compress++":
        split_kernel_str += f"_seed{thin_seed}"
        
    swap_kernel_str =  "-swap{}_var{:.3f}".format(params_k_swap["name"], params_k_swap["var"])
    thresh_str = ""
    if prefix != "Herd":
        thresh_str = f"-delta{delta}"
        
    d_str = f"-d{d}"
    size_str = f"-sz{input_size}"
    rep_str = f"-rep{{}}"
    
    if "Compresspp" in prefix: # since we can summy flags to in the prefix with Blowup settings
        assert(compressalg is not None)
        assert(compress_seed is not None)
        assert(g is not None)
        compress_alg_str = "-alg{}-g{}-compressseed{}".format(compressalg, g, compress_seed)
        m_str = ""
    else:
        compress_alg_str = "" # redundant
        m_str = f"-m{m}"

    file_template = os.path.join(folder, 
                                 f"{prefix}-{{}}-{sample_str}{split_kernel_str}{swap_kernel_str}"
                                 +f"{compress_alg_str}{d_str}{size_str}{m_str}{thresh_str}{rep_str}.pkl")

    return(file_template)