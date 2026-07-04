#!/bin/bash
# Quick status check for the detached comparison runs.
#   bash status.sh                  # summary across all datasets
#   bash status.sh -f cnn_kiba      # live-follow a specific log (Ctrl-C to stop)
cd "$(dirname "$0")"

if [ "${1:-}" = "-f" ]; then
    tail -f "runs/${2:-cnn_davis}.log"
    exit 0
fi

echo "===== processes ====="
pgrep -af "train.py --model" | grep -v pgrep || echo "  (no training process running)"

echo; echo "===== driver logs ====="
for d in runs/*_compare.log; do
    [ -f "$d" ] || continue
    echo "-- $d"; tail -n 2 "$d"
done

for ds in davis kiba bindingdb; do
    for m in cnn gnn; do
        log="runs/${m}_${ds}.log"
        sum="runs/${m}_${ds}_summary.json"
        [ -f "$log" ] || continue
        echo; echo "===== ${m}_${ds} ====="
        grep -E "^\[${m}_${ds}\] epoch" "$log" 2>/dev/null | tail -n 2 \
            || echo "  (no eval lines yet)"
        if [ -f "$sum" ]; then
            python -c "import json;s=json.load(open('$sum'));b=s['best'];print(f\"  -- FINISHED best: epoch {b['epoch']} | MSE {b['MSE']:.4f} | CI {b['CI']:.4f} | rm2 {b['rm2']:.4f} | {s['mean_epoch_sec']:.0f}s/epoch\")" 2>/dev/null
        fi
    done
done
