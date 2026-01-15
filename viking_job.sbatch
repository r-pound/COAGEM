#!/usr/bin/env bash
#===============================================================================
#SBATCH --job-name=pH_sens_run
#SBATCH --mail-user=rp819@york.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --account=chem-acm-2018
#SBATCH --output=%x.log
#-------------------------------------------------------------------------------
#SBATCH --partition=nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#===============================================================================
set -e

module load Miniconda3/23.5.2-0
source activate ct-env

/users/rp819/scratch/conda_environments/envs/ct-env/bin/python solve_sml_model.py 30 100 5 297 7.0
/users/rp819/scratch/conda_environments/envs/ct-env/bin/python solve_sml_model.py 30 100 5 297 7.1
/users/rp819/scratch/conda_environments/envs/ct-env/bin/python solve_sml_model.py 30 100 5 297 7.2
/users/rp819/scratch/conda_environments/envs/ct-env/bin/python solve_sml_model.py 30 100 5 297 7.3
/users/rp819/scratch/conda_environments/envs/ct-env/bin/python solve_sml_model.py 30 100 5 297 7.4
/users/rp819/scratch/conda_environments/envs/ct-env/bin/python solve_sml_model.py 30 100 5 297 7.5
/users/rp819/scratch/conda_environments/envs/ct-env/bin/python solve_sml_model.py 30 100 5 297 7.6
/users/rp819/scratch/conda_environments/envs/ct-env/bin/python solve_sml_model.py 30 100 5 297 7.7
/users/rp819/scratch/conda_environments/envs/ct-env/bin/python solve_sml_model.py 30 100 5 297 7.8
/users/rp819/scratch/conda_environments/envs/ct-env/bin/python solve_sml_model.py 30 100 5 297 7.9
/users/rp819/scratch/conda_environments/envs/ct-env/bin/python solve_sml_model.py 30 100 5 297 8
/users/rp819/scratch/conda_environments/envs/ct-env/bin/python solve_sml_model.py 30 100 5 297 8.1
/users/rp819/scratch/conda_environments/envs/ct-env/bin/python solve_sml_model.py 30 100 5 297 8.2
/users/rp819/scratch/conda_environments/envs/ct-env/bin/python solve_sml_model.py 30 100 5 297 8.3
/users/rp819/scratch/conda_environments/envs/ct-env/bin/python solve_sml_model.py 30 100 5 297 8.4
/users/rp819/scratch/conda_environments/envs/ct-env/bin/python solve_sml_model.py 30 100 5 297 8.5
