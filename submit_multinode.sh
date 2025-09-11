#!/bin/bash
#SBATCH --account=a-infra01
#SBATCH --job-name=fine-tune-apertus-70b
#SBATCH --time=02:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --output logs/slurm-%x-%j.out

set -x

srun  multi_node.sh