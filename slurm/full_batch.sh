#!/bin/bash -l
#SBATCH --partition=p.vera
#SBATCH --cpus-per-task=72
#SBATCH --mem=220G
#SBATCH --time=24:00:00
#SBATCH -o logs/xray_%x_%A_%a.out
#SBATCH -e logs/xray_%x_%A_%a.err
#SBATCH -D /vera/u/maoweyssi/Code/VeraWorkspace/tng-xray-pipeline
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=maoweyssi@mpia.de

# Array job: one task per snapshot.
# Usage: sbatch --job-name=xray_full --array=0-12%4 slurm/full_batch.sh <out_dir> [snap_list]
#
# snap_list CSV: header row + one "snap" column per row.
# Default covers the 13 TNG-Cluster reference snapshots.

OUT_DIR=${1:?Usage: sbatch --array=0-12 slurm/full_batch.sh <out_dir>}
SNAP_LIST=${2:-halo_lists/snap_list.csv}

set -euo pipefail

if [[ ! -f "$SNAP_LIST" ]]; then
    echo "ERROR: snap list not found: $SNAP_LIST"; exit 1
fi

LINE=$(awk -v idx="$((SLURM_ARRAY_TASK_ID + 2))" 'NR==idx' "$SNAP_LIST")
if [[ -z "$LINE" ]]; then
    echo "ERROR: no row at index $SLURM_ARRAY_TASK_ID"; exit 1
fi

SIM=$(  echo "$LINE" | cut -d, -f1)
SNAP=$( echo "$LINE" | cut -d, -f2)

echo "=========================================="
echo "SIM=$SIM  SNAP=$SNAP  TASK=$SLURM_ARRAY_TASK_ID"
echo "Start: $(date)"
echo "=========================================="

module purge
module load anaconda/3/2023.03
source /vera/u/maoweyssi/Code/VeraWorkspace/XRAY_ML/xray_venv/bin/activate

mkdir -p logs "$OUT_DIR"

python3 -c "
import sys, os
sys.path.insert(0, '.')
from xray_pipeline import process_halo, SIM_CONFIGS
# Full batch logic lives here — extend as needed
print('Batch run not yet implemented; use run_sample.py for now.')
"

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
