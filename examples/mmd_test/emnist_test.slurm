#!/bin/bash
#SBATCH --job-name=EMNIST_grid_job
#SBATCH --open-mode=append
#SBATCH --output=./logs/%x_%A_%a.out
#SBATCH --error=./logs/%x_%A_%a.err
#SBATCH --export=ALL
#SBATCH --time=22:00:00
#SBATCH --mem=3G

#SBATCH --array=1-400

module purge
module load python/intel/3.8.6
python ./test.py --task_id $SLURM_ARRAY_TASK_ID --alpha 0.05 --n 262144 --name EMNIST --d 49 --p_even 0.49 --n_tests 1 --seed_0 0 --number_of_jobs 400 --s_rff 0 --s_permute 16