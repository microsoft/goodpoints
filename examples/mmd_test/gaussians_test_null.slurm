#!/bin/bash
#SBATCH --job-name=gaussians_n_grid_job
#SBATCH --open-mode=append
#SBATCH --output=./logs/%x_%A_%a.out
#SBATCH --error=./logs/%x_%A_%a.err
#SBATCH --export=ALL
#SBATCH --time=15:00:00
#SBATCH --mem=2G

#SBATCH --array=1-400

module purge
module load python/intel/3.8.6
python ./test.py --name gaussians --task_id $SLURM_ARRAY_TASK_ID --d 10 --n 262144 --n_tests 1 --mean_diff 0.0 --B 39 --seed_0 0 --s_rff 0 --s_permute 16 --number_of_jobs 400