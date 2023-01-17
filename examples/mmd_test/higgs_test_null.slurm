#!/bin/bash
#SBATCH --job-name=higgs_n_grid_job
#SBATCH --open-mode=append
#SBATCH --output=./logs/%x_%A_%a.out
#SBATCH --error=./logs/%x_%A_%a.err
#SBATCH --export=ALL
#SBATCH --time=00:20:00
#SBATCH --mem=2G

#SBATCH --array=1-400

module purge
module load python/intel/3.8.6
python ./test.py --name Higgs --task_id $SLURM_ARRAY_TASK_ID --n_components 2 --n 16384 --n_tests 1 --aggregated --B 299 --B_2 200 --seed_0 0 --number_of_jobs 400 --p_poisoning 1.0 --mixing