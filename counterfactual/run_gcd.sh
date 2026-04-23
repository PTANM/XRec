#!/bin/bash
#SBATCH --job-name=gcd_counterfactual
#SBATCH --output=logs/gcd_%j.out
#SBATCH --error=logs/gcd_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --time=15:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pranati.tyagi@tamu.edu

module purge
module load GCCcore/11.3.0 Python/3.10.4

source /scratch/user/pranati.tyagi/xrec_env/bin/activate

WORKDIR=/scratch/user/pranati.tyagi/XRec-main
cd $WORKDIR

mkdir -p logs

export HF_TOKEN=${HF_TOKEN:-""}   # pass via: sbatch --export=HF_TOKEN=xxx run_gcd.sh

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

python GCD.py

echo "Job finished at: $(date)"
