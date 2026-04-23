#!/bin/bash
#SBATCH --job-name=cf_generate
#SBATCH --output=logs/generate_%j.out
#SBATCH --error=logs/generate_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pranati.tyagi@tamu.edu

module purge
module load GCCcore/11.3.0 Python/3.10.4
source /scratch/user/pranati.tyagi/xrec_env/bin/activate

cd /scratch/user/pranati.tyagi/counterfactual
mkdir -p logs

echo "=== Generation started: $(date) ==="
nvidia-smi

python generate_cf_explanations.py \
    --lora_dir outputs/cf_lora/epoch_3 \
    --output counterfactual_finetuned_results.json

echo "=== Generation finished: $(date) | Exit: $? ==="
