'''Reproduces the vignettes of Kernel Thinning (https://arxiv.org/pdf/2105.05842.pdf) on a Slurm cluster
   by executing run_kt_experiment.ipynb with appropriate parameters
'''

import itertools
from slurmpy import Slurm

# define the slurm object
partition = 'high'
s = Slurm("kt", {"partition": partition, 
                 "mem":"5G", 
                 "c": 1
                })

# convert the run_kt_experiment ipython notebook into a python file
s_id = s.run('module load python; python compile_notebook.py run_kt_experiment.ipynb')

# define repetition and m parameters
total_reps = 100 # set this to the max number of repetitions
reps_per_job = 50
m_max = 4

### All experiments are run with Gauss(sigma) as k and Gauss(sigma/sqrt(2)) as krt ###
run_gauss_experiments = True # run experiments with Gauss P
run_mog_experiments = True # run experiments with MoG P
run_mcmc_experiments = True # run experiments with MCMC P

###### Gaussian experiments ######
# if no k or filename is specified, experiments are run with Gauss(1) P in d dimensions and Gauss(1) as k
ds = [2, 3, 4]

###### MoG experiments ######
# M denotes the number of mixtures in 2 dim MOG P; kernel k is still Gauss(1) 
Ms = [4, 6, 8]

###### MCMC experiments ######
# filename denotes the MCMC setting to be loaded; kernel k is Gauss(sigma), where sigma = med dist(Phat)
# and Pstartis 2^15 sized point set obtained from standard thinning of post-burn in samples
# burn_in and sigma params are pre-loaded in the sample functions
mcmc_filenames = ['Goodwin_RW', 'Goodwin_ADA-RW', 
             'Goodwin_MALA', 
             'Goodwin_PRECOND-MALA', 'Lotka_RW', 'Lotka_ADA-RW', 
             'Lotka_MALA', 'Lotka_PRECOND-MALA']


fix_param_str = 'module load python; python3 run_kt_experiment.py ' 

if run_gauss_experiments:
    for d in ds:
        temp_ids = []
        for i in range(0, total_reps, reps_per_job):
            param_str = fix_param_str + ' -m ' + str(m_max+1) + ' -d  '  + str(d)
            param_str += ' -r0 ' + str(i) + ' -rn ' + str(reps_per_job)
            temp_ids.append(s.run(param_str, depends_on=[s_id])) # wait for the compilation of the run_kt_notebook
        
        # combine the results once all runs done; wait for temp_ids to finish
        param_str = fix_param_str + ' -m ' + str(m_max+1) + ' -d  '  + str(d)
        param_str += ' -r0 ' + str(0) + ' -rn ' + str(total_reps)
        param_str += ' -cm ' + str(True) # this activates combinining
        s.run(param_str, depends_on=temp_ids)
        
if run_mog_experiments:
    for M in Ms:
        temp_ids = []
        for i in range(0, total_reps, reps_per_job):
            param_str = fix_param_str + ' -m ' + str(m_max+1) + ' -M  '  + str(M)
            param_str += ' -r0 ' + str(i) + ' -rn ' + str(reps_per_job)
            temp_ids.append(s.run(param_str, depends_on=[s_id]))
        
        # combine the results once all runs done; wait for temp_ids to finish
        param_str = fix_param_str + ' -m ' + str(m_max+1) + ' -M  '  + str(M)
        param_str += ' -r0 ' + str(0) + ' -rn ' + str(total_reps)
        param_str += ' -cm ' + str(True) # this activates combinining
        s.run(param_str, depends_on=temp_ids)

if run_mcmc_experiments:
    for filename in mcmc_filenames:
        temp_ids = []
        for i in range(0, total_reps, reps_per_job):
            param_str = fix_param_str + ' -m ' + str(m_max+1) + ' -f  '  + filename
            param_str += ' -r0 ' + str(i) + ' -rn ' + str(reps_per_job)
            temp_ids.append(s.run(param_str, depends_on=[s_id]))
        
        # combine the results once all runs done; wait for temp_ids to finish
        param_str = fix_param_str + ' -m ' + str(m_max+1) + ' -f  '  + filename
        param_str += ' -r0 ' + str(0) + ' -rn ' + str(total_reps)
        param_str += ' -cm ' + str(True) # this activates combinining
        s.run(param_str, depends_on=temp_ids)
