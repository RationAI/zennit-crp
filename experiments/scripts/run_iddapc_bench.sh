#!/usr/bin/env bash
# Insertion-Deletion (DAPC) benchmark driver — Benchmark run 1.
# Chunked so each shared-A40 lock hold stays well under 10 min; releases between
# chunks (a sibling agent may grab the lock briefly). Resumable: each `run`
# skips (method,image) pairs already stored, so re-invoking continues.
set -u
PY=/home/claude/venvs/zennit-crp/bin/python
LOCK=/home/claude/.cache/zennit-gpu.lock
cd /home/claude/workspaces/zennit-crp
LOG=data/results/benchmark/_run.log
mkdir -p data/results/benchmark
echo "=== driver start $(date -u +%H:%M:%S) ===" | tee -a "$LOG"

acquire() { local i=0; while ! mkdir "$LOCK" 2>/dev/null; do sleep 30; i=$((i+1)); [ $i -gt 40 ] && { echo "LOCK TIMEOUT" | tee -a "$LOG"; return 1; }; done; }
release() { rmdir "$LOCK" 2>/dev/null; }

# chunk size per model (funny_birds models are cheaper → larger chunks)
declare -A CHUNK=( [M1]=16 [M2]=8 [M3]=16 [M4]=8 )
METHODS=lrp,chefer,rollout,rise,random

for M in M1 M2 M3 M4; do
  cs=${CHUNK[$M]}
  acquire || exit 1
  echo "[$M] select $(date -u +%H:%M:%S)" | tee -a "$LOG"
  timeout 400 $PY -m experiments.insertion_deletion_bench select --model "$M" >>"$LOG" 2>&1
  release
  for ((s=0; s<64; s+=cs)); do
    e=$((s+cs)); [ $e -gt 64 ] && e=64
    acquire || exit 1
    echo "[$M] run imgs $s..$e  $(date -u +%H:%M:%S)" | tee -a "$LOG"
    timeout 590 $PY -m experiments.insertion_deletion_bench run --model "$M" \
        --methods "$METHODS" --img-start "$s" --img-end "$e" >>"$LOG" 2>&1
    rc=$?
    release
    if [ $rc -ne 0 ]; then echo "[$M] chunk $s..$e EXIT $rc" | tee -a "$LOG"; fi
  done
  echo "[$M] done $(date -u +%H:%M:%S)" | tee -a "$LOG"
done

acquire || exit 1
$PY -m experiments.insertion_deletion_bench summarize >>"$LOG" 2>&1
release
$PY -m experiments.insertion_deletion_bench figures >>"$LOG" 2>&1
echo "=== driver done $(date -u +%H:%M:%S) ===" | tee -a "$LOG"
