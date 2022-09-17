#!/bin/bash
#SBATCH --account=def-stanmat-ab
#SBATCH --ntasks=1               # number of MPI processes
#SBATCH --mem-per-cpu=4000M      # memory; default unit is megabytes
#SBATCH --time=5-00:00           # time (DD-HH:MM)
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --output=./output.out
srun ./run.sh