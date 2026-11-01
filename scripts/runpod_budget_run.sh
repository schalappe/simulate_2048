#!/bin/bash
# Budget run: train --mode budget until tile 2048 or the time budget, then stop the pod. Safe to rerun on restart.
BUDGET_HOURS=38
RESERVE_HOURS=0.25   # setup, final save, and evaluation overshoot
mkdir -p /workspace/runs
exec > >(while IFS= read -r line; do echo "$(date -u '+%m-%d %H:%M:%S') $line"; done | tee -a /workspace/run.log) 2>&1
[ -f /workspace/start_epoch ] || date +%s > /workspace/start_epoch
ELAPSED=$(( $(date +%s) - $(cat /workspace/start_epoch) ))
REMAINING=$(awk -v b=$BUDGET_HOURS -v r=$RESERVE_HOURS -v e=$ELAPSED 'BEGIN { h = b - r - e / 3600; print (h > 0 ? h : 0) }')
echo "=== BOOT: ${ELAPSED}s since first start, ${REMAINING} h of training left"
if awk -v h="$REMAINING" 'BEGIN { exit !(h <= 0) }'; then
    echo "=== BUDGET SPENT: stopping pod"; runpodctl stop pod "$RUNPOD_POD_ID"; sleep infinity
fi
unset LD_LIBRARY_PATH
curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1; source "$HOME/.local/bin/env"
cd /root && [ -d simulate_2048 ] || git clone -q --branch fix/stochastic-muzero-training https://github.com/schalappe/simulate_2048.git
# ##>: SIMULATE_2048_COMMIT pins the code, so a restart mid-run never resumes with newer code.
cd /root/simulate_2048 && git fetch -q origin && git checkout -q "${SIMULATE_2048_COMMIT:-HEAD}" && git log --oneline -1
export UV_PROJECT_ENVIRONMENT=/root/venv
uv sync --extra gpu -q && echo "=== SYNCED"
uv run python -u -m reinforce.train --mode budget --run-dir /workspace/runs/budget \
    --stop-at-tile 2048 --max-hours "$REMAINING"
echo "=== TRAIN EXIT $?"
echo "=== STOPPING POD"
runpodctl stop pod "$RUNPOD_POD_ID"
sleep infinity
