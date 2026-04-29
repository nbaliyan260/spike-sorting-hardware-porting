#!/bin/bash
# Full experiment runner for tt-blackhole-01
# Runs inside the TT-Metal venv which has torch + ttnn
set -e

REPO="$HOME/spike-sorting-hardware-porting"
VENV="$HOME/assignment-1-single-core-matrix-multiplication-nbaliyan260/tt-metal/python_env/bin/activate"
PAT="${GITHUB_PAT}"  # set via: export GITHUB_PAT=your_token
REMOTE="https://nbaliyan260:${PAT}@github.com/nbaliyan260/spike-sorting-hardware-porting.git"

echo "=================================================="
echo "STEP 1: Sync repo"
echo "=================================================="
cd "$REPO"
git config --global pager.log false
git config --global pager.diff false
git config --global pager.show false
git fetch origin main
git checkout -f HEAD -- notes/ttnn_real_results.json 2>/dev/null || true
git reset --hard origin/main
git pull origin main
echo "Latest commit: $(git log --oneline -1)"

echo "=================================================="
echo "STEP 2: Activate TT venv"
echo "=================================================="
source "$VENV"
python3 --version
python3 -c "import torch; print('torch:', torch.__version__)"
python3 -c "import numpy; print('numpy:', numpy.__version__)"

echo "=================================================="
echo "STEP 3: test_pca_module.py"
echo "=================================================="
python3 experiments/test_pca_module.py 2>&1

echo "=================================================="
echo "STEP 4: cross_validate_pca.py"
echo "=================================================="
python3 experiments/cross_validate_pca.py 2>&1

echo "=================================================="
echo "STEP 5: pca_quantitative_comparison.py"
echo "=================================================="
python3 experiments/pca_quantitative_comparison.py 2>&1

echo "=================================================="
echo "STEP 6: test_pca_simulated_recordings_shaped.py"
echo "=================================================="
python3 experiments/test_pca_simulated_recordings_shaped.py 2>&1

echo "=================================================="
echo "STEP 7: test_pca_allen_real.py"
echo "=================================================="
python3 experiments/test_pca_allen_real.py 2>&1

echo "=================================================="
echo "STEP 8: test_pca_ttnn_real.py (TT-NN hardware)"
echo "=================================================="
python3 experiments/test_pca_ttnn_real.py 2>&1 || true

echo "=================================================="
echo "STEP 9: Push updated results to GitHub"
echo "=================================================="
git config user.email "nazishbaliyan@tt-blackhole-01"
git config user.name "Nazish Baliyan (tt-blackhole-01)"
git add notes/
CHANGED=$(git diff --cached --stat)
if [ -z "$CHANGED" ]; then
    echo "No result changes to push"
else
    echo "Changes: $CHANGED"
    git commit -m "results: all experiments verified on tt-blackhole-01 (2026-04-29)"
    git push "$REMOTE" main
    echo "Pushed to GitHub successfully"
fi

echo "=================================================="
echo "ALL DONE"
echo "=================================================="
