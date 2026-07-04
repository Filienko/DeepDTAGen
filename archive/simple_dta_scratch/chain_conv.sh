#!/usr/bin/env bash
# Wait for the currently-running queue (any train.py python procs) to finish,
# then launch the conv-tower experiment queue. Requires two zero-readings 20s
# apart so we don't fire during a millisecond gap between jobs.
cd "$(dirname "$0")" || exit 1
echo "CHAIN WAIT $(date)" >> runs/queue.driver.log
while true; do
  if ! ps -C python -o cmd= | grep -q train.py; then
    sleep 20
    ps -C python -o cmd= | grep -q train.py || break
  fi
  sleep 60
done
echo "CHAIN FIRE $(date) -> jobs_cnn_conv.txt" >> runs/queue.driver.log
bash run_queue.sh jobs_cnn_conv.txt 5 8
