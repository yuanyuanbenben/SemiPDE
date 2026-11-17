#!/usr/bin/env bash
#
# Shard the Example-2 study across CPU workers.
#
# Each shard is one process running one part of ``exp_example2.py`` with
# ``--shard i --shards N``; it appends finished jobs to its own JSONL as it goes,
# so a killed shard can be restarted and will skip what is already done.
#
# The device is the CPU on purpose.  The integration is a Python loop over
# substeps, every one a handful of operations on a small tensor, so the
# accelerator only ever sees kernel launches -- and on this (shared, saturated)
# host a launch costs far more than the arithmetic.  Measured on the forced-
# Burgers reference solve: 2.7 s on the CPU against 177 s on the GPU, 64x.
# ``OMP_NUM_THREADS=1`` keeps torch's thread pool from thrashing on tiny tensors.
#
# Usage:
#   scripts/run_example2.sh <workers> <part> [extra args...]
# e.g.
#   scripts/run_example2.sh 64 inference --repeats 50 --epochs 200
#
set -euo pipefail

WORKERS="${1:?usage: run_example2.sh <workers> <part> [args...]}"
PART="${2:?usage: run_example2.sh <workers> <part> [args...]}"
shift 2

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/runs}"
mkdir -p "$OUT_DIR"
LOG_DIR="${LOG_DIR:-$OUT_DIR/logs}"
mkdir -p "$LOG_DIR"

echo "part=$PART workers=$WORKERS out=$OUT_DIR/$PART.jsonl"

# Reproducibility: the directory must say how it was produced, since the driver
# only sees resolved args and the shard split is not recoverable from the rows.
{
  echo "date       $(date -Is)"
  echo "host       $(hostname)"
  echo "out        $OUT_DIR/$PART.jsonl"
  echo "workers    $WORKERS"
  printf 'command    OUT_DIR=%q %q %q %q' "$OUT_DIR" "$0" "$PART" "$WORKERS"
  printf ' %q' "$@"
  echo
} > "$OUT_DIR/$PART.cmd"

pids=()
for i in $(seq 0 $((WORKERS - 1))); do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$ROOT" \
    nohup python -u "$ROOT/scripts/exp_example2.py" "$PART" \
      --shard "$i" --shards "$WORKERS" \
      --out "$OUT_DIR/$PART.jsonl" \
      "$@" > "$LOG_DIR/$PART.$i.log" 2>&1 &
  pids+=($!)
done

echo "launched ${#pids[@]} shards: ${pids[*]}"
cat > "$OUT_DIR/$PART.pids" <<<"${pids[@]}"

wait
echo "all shards finished; aggregating"
PYTHONPATH="$ROOT" python -u "$ROOT/scripts/exp_example2.py" "$PART" \
  --aggregate "$OUT_DIR/$PART.jsonl" --out "$OUT_DIR/$PART.summary.json" "$@"
