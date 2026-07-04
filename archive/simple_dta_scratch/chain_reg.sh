#!/usr/bin/env bash
# Wait for the current queue (all train.py python procs) to finish, then launch
# the regularization sweep. Two zero-readings 20s apart so we don't fire during a
# millisecond gap between jobs.
cd "$(dirname "$0")" || exit 1
echo "CHAIN(reg) WAIT $(date)" >> runs/queue.driver.log
while true; do
  if ! ps -C python -o cmd= | grep -q train.py; then
    sleep 20
    ps -C python -o cmd= | grep -q train.py || break
  fi
  sleep 60
done
echo "CHAIN(reg) FIRE $(date) -> jobs_reg_sweep.txt" >> runs/queue.driver.log
bash run_queue.sh jobs_reg_sweep.txt 5 8
