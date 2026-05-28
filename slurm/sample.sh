#!/bin/bash -l
#SBATCH --partition=p.vera
#SBATCH --job-name=xray_sample
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH -o logs/sample_%j.out
#SBATCH -e logs/sample_%j.err
#SBATCH -D /vera/u/maoweyssi/Code/VeraWorkspace/tng-xray-pipeline
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maoweyssi@mpia.de

# Usage: sbatch slurm/sample.sh <out_dir> [n_halos] [seed]
#   sbatch slurm/sample.sh results/run1 30 42

OUT_DIR=${1:-results}
N=${2:-30}
SEED=${3:-42}

set -euo pipefail

mkdir -p logs "$OUT_DIR"

echo "========================================"
echo "Job:   $SLURM_JOB_ID"
echo "Node:  $SLURM_NODELIST"
echo "Start: $(date)"
echo "out=$OUT_DIR  n=$N  seed=$SEED"
echo "========================================"

module purge
module load anaconda/3/2023.03
source /vera/u/maoweyssi/Code/VeraWorkspace/XRAY_ML/xray_venv/bin/activate

python3 run_sample.py \
    --sim       tng-cluster \
    --n         "$N"        \
    --seed      "$SEED"     \
    --nthreads  "$SLURM_CPUS_PER_TASK" \
    --out-dir   "$OUT_DIR"

echo "========================================"
echo "Done: $(date)"
echo "========================================"
