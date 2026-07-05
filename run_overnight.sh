#!/bin/bash
# Overnight research loop — runs nohup, survives terminal disconnect.
# Runs the PINN/flow-matching experiment across 5 system variants,
# logs everything, saves results.
#
# Usage:  bash run_overnight.sh
# Logs:   results/overnight_YYYYMMDD.log

set -e
cd /homes/rdubash/scientist/pinn_paper

CONDA_PYTHON="/opt/miniconda/envs/ai_scientist/bin/python"
LOG="results/overnight_$(date +%Y%m%d_%H%M%S).log"
ANTHROPIC_KEY=$(grep "ANTHROPIC_API_KEY" /homes/rdubash/agents/.env | cut -d= -f2 | tr -d '"' | tr -d "'")
OPENAI_KEY=$(grep "OPENAI_API_KEY" /homes/rdubash/internships_emailing_v2/.env | cut -d= -f2 | tr -d '"' | tr -d "'")

export ANTHROPIC_API_KEY="$ANTHROPIC_KEY"
export OPENAI_API_KEY="$OPENAI_KEY"

echo "======================================" | tee -a "$LOG"
echo "Overnight loop started: $(date)"       | tee -a "$LOG"
echo "Log: $LOG"                             | tee -a "$LOG"
echo "======================================" | tee -a "$LOG"

# --- Run 1: Current experiment (PINN + Flow on double-well + trypsin) ---
echo "" | tee -a "$LOG"
echo "[Run 1/5] PINN vs Flow Matching — double-well + trypsin-benzamidine" | tee -a "$LOG"
$CONDA_PYTHON experiment.py 2>&1 | tee -a "$LOG"
echo "[Run 1 done: $(date)]" | tee -a "$LOG"

# --- Run 2: Data efficiency sweep (how few training points do we need?) ---
echo "" | tee -a "$LOG"
echo "[Run 2/5] Data efficiency sweep" | tee -a "$LOG"
$CONDA_PYTHON experiment_data_efficiency.py 2>&1 | tee -a "$LOG"
echo "[Run 2 done: $(date)]" | tee -a "$LOG"

# --- Run 3: Architecture comparison (MLP depth/width ablation) ---
echo "" | tee -a "$LOG"
echo "[Run 3/5] Architecture ablation" | tee -a "$LOG"
$CONDA_PYTHON experiment_arch_ablation.py 2>&1 | tee -a "$LOG"
echo "[Run 3 done: $(date)]" | tee -a "$LOG"

# --- Run 4: Beta (temperature) sweep ---
echo "" | tee -a "$LOG"
echo "[Run 4/5] Temperature sweep (beta)" | tee -a "$LOG"
$CONDA_PYTHON experiment_beta_sweep.py 2>&1 | tee -a "$LOG"
echo "[Run 4 done: $(date)]" | tee -a "$LOG"

# --- Run 5: Müller-Brown (third system) ---
echo "" | tee -a "$LOG"
echo "[Run 5/5] Müller-Brown potential" | tee -a "$LOG"
$CONDA_PYTHON experiment_muller_brown.py 2>&1 | tee -a "$LOG"
echo "[Run 5 done: $(date)]" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "======================================" | tee -a "$LOG"
echo "All runs complete: $(date)"            | tee -a "$LOG"
echo "Results in: results/"                  | tee -a "$LOG"
echo "Figures in: figures/"                  | tee -a "$LOG"
echo "======================================" | tee -a "$LOG"

# Collect all JSONs into a single summary
$CONDA_PYTHON - <<'PYEOF' 2>&1 | tee -a "$LOG"
import json, glob, os
all_results = {}
for f in sorted(glob.glob("results/*.json")):
    with open(f) as fh:
        all_results[os.path.basename(f)] = json.load(fh)
with open("results/overnight_summary.json", "w") as fh:
    json.dump(all_results, fh, indent=2, default=str)
print("Summary written to results/overnight_summary.json")
PYEOF
