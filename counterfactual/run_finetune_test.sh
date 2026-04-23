#!/bin/bash
#SBATCH --job-name=cf_lora_test
#SBATCH --output=logs/finetune_test_%j.out
#SBATCH --error=logs/finetune_test_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pranati.tyagi@tamu.edu

module purge
module load GCCcore/11.3.0 Python/3.10.4

source /scratch/user/pranati.tyagi/xrec_env/bin/activate

WORKDIR=/scratch/user/pranati.tyagi/counterfactual
cd $WORKDIR

mkdir -p logs outputs/cf_lora_test

echo "============================================"
echo "LoRA Fine-Tuning TEST RUN started at: $(date)"
echo "Node: $(hostname)"
echo "============================================"
nvidia-smi

python finetune_cf.py --test

echo "============================================"
echo "TEST run finished at: $(date)"
echo "Exit code: $?"
echo "============================================"
