'''Reproduces the vignettes of generalized KT on a Slurm cluster
   by executing run_generalized_kt_experiment.ipynb with appropriate parameters
'''

import itertools
from slurmpy import Slurm
import numpy as np



def singlejob_slurm_command(prefix, temp_ids, new_fix_param_str, m_max, d, rep0, repn, computemmd, s_id,
                            M=0, filename='temp',
                            power=0.5,
                            compute_power=0,
                            target_kt=1,
                            standard_thin=0,
                            power_kt=0,
                            kt_plus=0,
                           rerun=0,
                            nu=0.5):
    '''
    Deploys a slurm job that runs thinning experiments based on the parameters, and appends the slurm id
    to temp_ids
    
    prefix: (str) prefix for slurm id description
    new_fix_param_str: basic description of the code to be run
    temp_ids: ids of slurm jobs
    m_max : (int) max size of input
    d: (int) dimension
    rep0: (int) starting index of rep
    repn: (int) number of reps (rep_ids will be rep0, rep0+1, ..., rep0+repn-1)
    computemmd: (int) whether mmd needs to be computed (anything but 0) else 0
    s_id: (int) wait for the slurm run with id s_id
    M: (int) number of components in MOG
    filename: (str) the mcmc filename
    power:(float) power of the kernel
    target_kt:  (int) if target kt needs to be run (anything but 0) else 0
    compute_power: (int) if power kernel needs to be computed (anything but 0) else 0
    standard_thin: (int) if standard thinning needs to be run (anything but 0) else 0
    power_kt: (int) if power kT needs to be run (anything but 0) else 0
    kt_plus:  (int) if kT+ needs to be run (anything but 0) else 0 [compute power must not be 0]
    rerun: (int) if experiments should be rerun (anything but 0) else 0
    nu: (float/int) a parameter for some kernels
    
    '''
    param_str = new_fix_param_str + ' -m ' + str(m_max+1) + ' -d  '  + str(d)
    param_str += ' -r0 ' + str(rep0) + ' -rn ' + str(repn)
    param_str += ' -cm ' + str(computemmd) # whether to compute mmd
    param_str += ' -M  '  + str(M) # mog number of components
    param_str += ' -f  '  + filename # mcmc filename
    param_str += ' -cp  '  + str(compute_power) # whether to compute power kernel
    param_str += ' -pow ' + str(power) # power for power kernel
    param_str += ' -tkt  '  + str(target_kt) # whether to run target KT
    param_str += ' -st  '  + str(standard_thin) # whether to run standard thin
    param_str += ' -pkt  '  + str(power_kt) # whether to run power KT
    param_str += ' -ktp  '  + str(kt_plus) # whether to run KT+
    param_str += ' -rr  '  + str(rerun) # whether to rerun
    param_str += ' -nu  '  + str(nu) # nu param for IMQ/Matern or beta param for Bspline

    s = Slurm(f"{prefix}d{d}m{m_max}r{rep0}", {"partition": partitions[idx], 
                 "c": 1
                })
    temp_ids.append(s.run(param_str, depends_on=[s_id])) # wait for the compilation of the run_ktplus_notebook
    return

def combinemmd_slurm_command(prefix, fix_param_str, m_max, d, total_reps, computemmd, temp_ids,
                             M=0, filename='temp',
                             power=0.5,
                             compute_power=0,
                            target_kt=1,
                            standard_thin=0,
                            power_kt=0,
                            kt_plus=0,
                           rerun=0, 
                            nu=0.5):
    '''
        Deploys a slurm job that combines all thinning experiment results based on the parameters
    prefix: (str) prefix for slurm id description
    fix_param_str: basic description of the code to be run
    m_max : (int) max size of input
    d: (int) dimension
    total_reps: (int) range of reps to be comibined, code will combine 0, ..., total_reps-1
    computemmd: (int) whether mmd needs to be computed (anything but 0) else 0
    temp_ids: ids of slurm jobs
    M: (int) number of components in MOG
    filename: (str) the mcmc filename
    power:(float) power of the kernel
    target_kt:  (int) if target kt needs to be run (anything but 0) else 0
    compute_power: (int) if power kernel needs to be computed (anything but 0) else 0
    standard_thin: (int) if standard thinning needs to be run (anything but 0) else 0
    power_kt: (int) if power kT needs to be run (anything but 0) else 0
    kt_plus:  (int) if kT+ needs to be run (anything but 0) else 0 [compute power must not be 0]
    rerun: (int) if experiments should be rerun (anything but 0) else 0
    nu: (float/int) a parameter for some kernels
    '''
    # combine the results once all runs done; wait for temp_ids to finish
    param_str = fix_param_str + ' -m ' + str(m_max+1) + ' -d  '  + str(d)
    param_str += ' -r0 ' + str(0) + ' -rn ' + str(total_reps)
    param_str += ' -cm ' + str(computemmd) # whether to compute mmd
    param_str += ' -scr ' + str(1) # this activates combinining
    param_str += ' -M  '  + str(M) # mog number of components
    param_str += ' -f  '  + filename # mcmc filename
    param_str += ' -cp  '  + str(compute_power) # whether to compute power kernel
    param_str += ' -pow ' + str(power) # power for power kernel
    param_str += ' -tkt  '  + str(target_kt) # whether to run target KT
    param_str += ' -st  '  + str(standard_thin) # whether to run standard thin
    param_str += ' -pkt  '  + str(power_kt) # whether to run power KT
    param_str += ' -ktp  '  + str(kt_plus) # whether to run KT+
    param_str += ' -rr  '  + str(rerun) # whether to rerun
    param_str += ' -nu  '  + str(nu) # nu param for IMQ/Matern or beta param for Bspline
    
    s = Slurm(f"C{prefix}d{d}m{m_max}r{total_reps}", {"partition": partitions[idx], 
                 "c": 1
                })
    s.run(param_str, depends_on=temp_ids)
    return

# define the slurm object
partitions = ["high", "yugroup", "jsteinhardt", "low"]
idx = 2 # which partition to pick

s = Slurm("convert", {"partition": partitions[idx], 
                 "c": 1
                })

# convert the run_kt_experiment ipython notebook into a python file
s_id = s.run('module load python; python compile_notebook.py run_generalized_kt_experiment.ipynb')

# define repetition and m parameters
total_reps = 10 # set this to the max number of repetitions
reps_per_job = 1
combine = True # whether to combine all mmd results or not
only_combine = False # whether we only run experiments to combine mmd results

m_max = 7 ## max sample size processed is 2**(2*m_max); and the output size is 2**(m_max)
computemmd = 1 #


### All experiments are run with Gauss(sigma) as k and Gauss(sigma/sqrt(2)) as krt ###
gauss_target = False # Gauss P
mog_target = False # MoG P
mcmc_target = True # MCMC P
mcmc_file_idx = range(16, 24)  # range of MCMC files that need to be run; RUN 12--16, and 16--24 separately
rerun = 0 # BUT STILL DOESN"T RERUN IF DURING COMBINING ; SO DON"T EXPECT RERUN IF TOTAL REPS = REPS_PER_JOB

all_mcmc_filenames = ['Goodwin_RW','Goodwin_ADA-RW', 'Goodwin_MALA', 'Goodwin_PRECOND-MALA', 'Lotka_RW', 'Lotka_ADA-RW', 'Lotka_MALA', 'Lotka_PRECOND-MALA','Hinch_P_seed_1_temp_1', 'Hinch_P_seed_2_temp_1', 'Hinch_TP_seed_1_temp_8', 'Hinch_TP_seed_2_temp_8', 'Hinch_P_seed_1_temp_1_scaled', 'Hinch_P_seed_2_temp_1_scaled', 'Hinch_TP_seed_1_temp_8_scaled', 'Hinch_TP_seed_2_temp_8_scaled', 
'Goodwin_RW_float_step', 'Goodwin_ADA-RW_float_step',  'Goodwin_MALA_float_step',  'Goodwin_PRECOND-MALA_float_step',  'Lotka_RW_float_step',  'Lotka_ADA-RW_float_step',  'Lotka_MALA_float_step',
 'Lotka_PRECOND-MALA_float_step']
# files to run for MCMC experiments; 
# 0-4 for Goodwin, 4-8 for Lotka, 8-12 for Hinch, 
# 12-16 for Hinch Scaled, where the entire chain was standardized coordinate wise (centered, and scaled)
# 16-24 for Goodwin/Lotka_float_step experiments with sampling indices computed using np.linspace, rather than np.arange

if gauss_target:
    ds = [2, 10, 20, 50, 100] #, 10, 20, 50, 100] # for Gauss P
#     ds = [2, 4, 10, 100] # for Gauss P
if mog_target:
    Ms = [4, 6, 8] #, 8] # M = number of mixtures for 2 dim MOG P
if mcmc_target:
    ## NOTE for Hinch /Hinch_scale MCMC experiments m_max <=8 is permitted
    mcmc_files = np.array(all_mcmc_filenames)[mcmc_file_idx] # filename denotes the MCMC setting to be loaded;
    # filename denotes the MCMC setting to be loaded; kernel k is Gauss(sigma^2), where sigma = med dist(Phat)
    # and Pstart is 2^15 sized point set obtained from standard thinning of post-burn in samples from the end
    # burn_in and sigma params are pre-loaded in the sample functions
    # samples are loaded from pkl files for Hinch dataset

# MCMC P : 4 experiments each for Goodwin/Lotka-Volterra/Hinch
#    - kernels : (1) Goodwin/Lotka:
#    -               Laplace k with KT+ (power = 0.81)
#    -               NOTE: the power kernel is matern with parameter nu_eff 
#    -               where nu_eff = power * (d+1)/2 - d/2 or power = 2*nu_eff/(d+1) + d/(d+1)
#    -               Since d = 4, we have nu_eff = 0.025
#    -           (2) Hinch:
#    -               IMQ k with KT+ with nu = 0.5 (power = 0.5)

# # check sizes of list
# for t in [power_list, target_kt_flags, power_kt_flags, kt_plus_flags, nu_list]:
#     assert(len(t) == len(kernel_list))


fix_param_str = 'module load python; python3 run_generalized_kt_experiment.py ' 

###  experiment combinations ###
# in all cases, var/gamma parameter is set automatically; 
# sigma = 1/gamma = sqrt(2d) for Gauss/MoG, and median distance in MCMC)
# for non Gauss/Laplace cases we have to specify nu parameter
    
if gauss_target:
    # Gauss P : d = 2, 10, 20, 50, 100
    #    - kernels : (1) Gauss k with t-KT and KT(rt)
    # list of kernels to be run
    kernel_list = ["gauss"] 
    # list of powers for the kernels (should be same size as kernel_list)
    power_list = [0.5]
    # whether power kernel needs to be computed (should be same size as kernel_list)
    compute_power_list = [1]
    # whether standard thinning needs to be computed (should be same size as kernel_list)
    standard_thin_flags = [1]
    # whether target KT needs to be computed (should be same size as kernel_list)
    target_kt_flags = [1]
    # whether KT+ needs to be computed (should be same size as kernel_list)
    kt_plus_flags = [0]
    # whether power KT needs to be computed (should be same size as kernel_list)
    power_kt_flags = [1]
    
    
    # run gaussian experiments
    for kk, kernel in enumerate(kernel_list):
        new_fix_param_str = fix_param_str + ' -P gauss' + ' -kernel  '  + kernel
        prefix =f"G{kernel[0]}"
        for d in ds:
            temp_ids = []
            # if reps_per_job == total_reps the goal is to generally combine
            if not only_combine:
                for i in range(0, total_reps, reps_per_job):
                    singlejob_slurm_command(prefix, temp_ids, new_fix_param_str, m_max, d, i, 
                                            reps_per_job, computemmd, s_id=s_id,
                                            compute_power=compute_power_list[kk],
                                            power=power_list[kk],
                                           target_kt=target_kt_flags[kk],
                                            standard_thin=standard_thin_flags[kk],
                                            power_kt=power_kt_flags[kk],
                                            kt_plus=kt_plus_flags[kk],
                                            rerun=rerun,
                                           )
 
            # combine the results once all runs done; wait for temp_ids to finish
            if combine:
                if computemmd==1: combinemmd_slurm_command(prefix, new_fix_param_str, m_max, 
                                            d, total_reps, computemmd, temp_ids=temp_ids,
                                            compute_power=compute_power_list[kk],
                                                        power=power_list[kk],
                                           target_kt=target_kt_flags[kk],
                                            standard_thin=standard_thin_flags[kk],
                                            power_kt=power_kt_flags[kk],
                                            kt_plus=kt_plus_flags[kk],
                                            rerun=0,)


if mog_target:
    # MoG P : M = 4, 6, 8 in R^2
    #    - kernels : (1) Gauss k with t-KT, KT(rt) (power = 0.5)
    #    -           (2) Laplace k with KT+ (power = 0.7)
    #    -               NOTE: the power kernel is matern with parameter nu_eff 
    #    -               where nu_eff = power * (d+1)/2 - d/2 or power  = 2*nu_eff/(d+1) + d/(d+1)
    #    -               Since d = 2, we have nu_eff = 0.05
    #    -           (3) IMQ k with KT+ with nu = 0.5 (power = 0.5)
    #    -           (4) B-spline with KT+ with nu = 2 (power = 2/3.) --nu is same as beta for b-spine--(power = (nu+2) / (2*nu + 2))

    # list of kernels to be run
    kernel_list = ["gauss", "laplace", "imq", "bspline"] #["gauss", "sinc", "laplace", "imq", "matern", "bspline"]
    # list of powers for the kernels (should be same size as kernel_list)
    power_list = [0.5, 0.7, 0.5, 2./3]
    # whether power kernel needs to be computed (should be same size as kernel_list)
    compute_power_list = [1, 1, 1, 1]
    # whether standard thinning needs to be computed (should be same size as kernel_list)
    standard_thin_flags = [1, 0, 0, 0] 
    # whether target KT needs to be computed (should be same size as kernel_list)
    target_kt_flags = [1, 0, 0, 0]
    # whether KT+ needs to be computed (should be same size as kernel_list)
    kt_plus_flags = [0, 1, 1, 1]
    # whether power KT needs to be computed (should be same size as kernel_list)
    power_kt_flags = [1, 0, 0, 0] # same as root kt when root_power = 0.5
    # the list of nu parameteter list (irrelevant for Gauss/Laplacae)
    nu_list = [0., 0., 0.5, 2.] # nu is same as beta for bspline
    
    # run MOG experiments
    d = 2 # doesn't matter; will be set internally automatically; just specify some int
    for kk, kernel in enumerate(kernel_list):
        new_fix_param_str = fix_param_str + ' -P mog' + ' -kernel '  + kernel
        for M in Ms:
            prefix =f"M{M}{kernel[0]}"
            temp_ids = []
            if not only_combine:
                for i in range(0, total_reps, reps_per_job):
                    singlejob_slurm_command(prefix, temp_ids, new_fix_param_str, m_max, 
                                            d, i, reps_per_job, computemmd, 
                                            s_id=s_id, M=M, 
                                            compute_power=compute_power_list[kk],
                                           power=power_list[kk],
                                           target_kt=target_kt_flags[kk],
                                            standard_thin=standard_thin_flags[kk],
                                            power_kt=power_kt_flags[kk],
                                            kt_plus=kt_plus_flags[kk],
                                           rerun=rerun,
                                           nu=nu_list[kk])
            if combine:
                if computemmd==1: combinemmd_slurm_command(prefix, new_fix_param_str, m_max, 
                                            d, total_reps, computemmd, 
                                            temp_ids=temp_ids, M=M,
                                            compute_power=compute_power_list[kk],
                                            power=power_list[kk],
                                           target_kt=target_kt_flags[kk],
                                            standard_thin=standard_thin_flags[kk],
                                            power_kt=power_kt_flags[kk],
                                            kt_plus=kt_plus_flags[kk],
                                                       rerun=0,
                                           nu=nu_list[kk])

# laplace kernel for Goodwin/Lotka-Volterra
if mcmc_target and mcmc_file_idx==range(16, 24):
    # list of kernels to be run
    kernel_list = ["laplace"] #["gauss", "sinc", "laplace", "imq", "matern", "bspline"]
    # list of powers for the kernels (should be same size as kernel_list)
    power_list = [0.81]
    # whether power kernel needs to be computed (should be same size as kernel_list)
    compute_power_list = [1]
    # whether standard thinning needs to be computed (should be same size as kernel_list)
    standard_thin_flags = [1] 
    # whether target KT needs to be computed (should be same size as kernel_list)
    target_kt_flags = [0]
    # whether KT+ needs to be computed (should be same size as kernel_list)
    kt_plus_flags = [1]
    # whether power KT needs to be computed (should be same size as kernel_list)
    power_kt_flags = [0] # same as root kt when root_power = 0.5
    # the list of nu parameteter list (irrelevant for Gauss/Laplacae)
    nu_list = [0.] # nu is same as beta for bspline

# IMQ kernel for Hinch
if mcmc_target and mcmc_file_idx==range(12, 16):
    # list of kernels to be run
    kernel_list = ["imq"] #["gauss", "sinc", "laplace", "imq", "matern", "bspline"]
    # list of powers for the kernels (should be same size as kernel_list)
    power_list = [0.5]
    # whether power kernel needs to be computed (should be same size as kernel_list)
    compute_power_list = [1]
    # whether standard thinning needs to be computed (should be same size as kernel_list)
    standard_thin_flags = [1] 
    # whether target KT needs to be computed (should be same size as kernel_list)
    target_kt_flags = [0]
    # whether KT+ needs to be computed (should be same size as kernel_list)
    kt_plus_flags = [1]
    # whether power KT needs to be computed (should be same size as kernel_list)
    power_kt_flags = [0] # same as root kt when root_power = 0.5
    # the list of nu parameteter list (irrelevant for Gauss/Laplacae)
    nu_list = [0.5] # nu is same as beta for bspline
    
if mcmc_target:
    #  run MCMC experiments
    d = 4  # doesn't matter; will be set internally automatically; just specify some int
    for kk, kernel in enumerate(kernel_list):
        new_fix_param_str = fix_param_str + ' -P mcmc' + ' -kernel '  + kernel
        
        for filename in mcmc_files:
            prefix =f"m{filename[0]}{kernel[0]}"
            temp_ids = []
            if reps_per_job != total_reps:
                for i in range(0, total_reps, reps_per_job):
                    singlejob_slurm_command(prefix, temp_ids, new_fix_param_str, m_max, d, i, 
                                            reps_per_job, computemmd, s_id=s_id, 
                                            filename=filename, power=power_list[kk],
                                            compute_power=compute_power_list[kk],
                                           target_kt=target_kt_flags[kk],
                                            standard_thin=standard_thin_flags[kk],
                                            power_kt=power_kt_flags[kk],
                                            kt_plus=kt_plus_flags[kk],
                                           rerun=rerun)
            if combine:
                if computemmd==1: combinemmd_slurm_command(prefix, new_fix_param_str, m_max, d, total_reps, 
                                                computemmd, temp_ids=temp_ids, 
                                                filename=filename, power=power_list[kk],
                                            compute_power=compute_power_list[kk],
                                           target_kt=target_kt_flags[kk],
                                            standard_thin=standard_thin_flags[kk],
                                            power_kt=power_kt_flags[kk],
                                            kt_plus=kt_plus_flags[kk],
                                                       rerun=0)        


