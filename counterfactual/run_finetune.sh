#!/bin/bash
#SBATCH --job-name=cf_lora_finetune
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pranati.tyagi@tamu.edu

module purge
module load GCCcore/11.3.0 Python/3.10.4

source /scratch/user/pranati.tyagi/xrec_env/bin/activate
pip install openpyxl --quiet

WORKDIR=/scratch/user/pranati.tyagi/counterfactual
cd $WORKDIR

mkdir -p logs outputs/cf_lora

export HF_TOKEN=${HF_TOKEN:-""}

# Set TEST_MODE=1 for a quick 20-record smoke test; 0 (default) for full training
TEST_MODE=${TEST_MODE:-0}

echo "============================================"
echo "LoRA Fine-tuning started at: $(date)"
echo "Node: $(hostname)"
echo "TEST_MODE: $TEST_MODE"
echo "============================================"
nvidia-smi

if [ "$TEST_MODE" = "1" ]; then
    echo "Running TEST mode (20 records, 1 epoch)..."
    python finetune_cf.py --test
else
    echo "Running FULL training..."
    python finetune_cf.py
fi

echo "============================================"
echo "Fine-tuning finished at: $(date)"
echo "Exit code: $?"
echo "============================================"
