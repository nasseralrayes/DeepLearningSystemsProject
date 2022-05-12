#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=18:00:00
#SBATCH --mem=200GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=torch_nasser
#SBATCH --mail-type=END
#SBATCH --mail-user=nka8061@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
module load cuda/11.3.1 
source activate /scratch/nka8061/penv

cd /scratch/nka8061/DeepLearningSystemsProject
python train.py