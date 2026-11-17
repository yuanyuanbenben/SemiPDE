#!/usr/bin/env bash
#
# Shard one simulation study across CPU workers.
#
# Each shard is one process running one part of ``exp_study.py`` with
# ``--shard i --shards N``; it appends finished jobs to its own JSONL as it goes,
# so a killed shard can be restarted and will skip what is already done -- and
# will also skip anything whose recorded config no longer matches, which is how
# a directory cannot quietly mix a 200-epoch run with a 1000-epoch one.
#
# The device is the CPU on purpose.  The integration is a Python loop over
# substeps, every one a handful of operations on a small tensor, so the
# accelerator only ever sees kernel launches -- and on this (shared, saturated)
# host a launch costs far more than the arithmetic.  Measured on the forced-
# Burgers reference solve: 2.7 s on the CPU against 177 s on the GPU, 64x.
# ``OMP_NUM_THREADS=1`` keeps torch's thread pool from thrashing on tiny tensors.
#
# Usage:
#   scripts/run_study.sh <workers> <part> [--case CASE] [extra args...]
# e.g.
#   scripts/run_study.sh 64 inference --case burg --repeats 50 --epochs 200
#   scripts/run_study.sh 12 estimate  --case rd --n-x 16 --n-t 10 \
#       --legacy-ic --sample-sizes 64 --repeats 4 --epochs 200
#
# ``OUT_DIR`` defaults to ``runs/<case>/<part-tag>`` where the tag names the
# part; pass it explicitly to keep several grids of the same part apart.
set -euo pipefail

WORKERS="${1:?usage: run_study.sh <workers> <part> [--case CASE] [args...]}"
PART="${2:?usage: run_study.sh <workers> <part> [--case CASE] [args...]}"
shift 2

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The case decides where the results go, so it has to be read here as well as by
# the driver -- which does not parse it as an option either, for the same reason:
# it selects the parser.
CASE="burg"
args=("$@")
for i in "${!args[@]}"; do
  if [[ "${args[$i]}" == "--case" && $((i + 1)) -lt ${#args[@]} ]]; then
    CASE="${args[$((i + 1))]}"
  elif [[ "${args[$i]}" == --case=* ]]; then
    CASE="${args[$i]#--case=}"
  fi
done

OUT_DIR="${OUT_DIR:-$ROOT/runs/$CASE}"
mkdir -p "$OUT_DIR"
LOG_DIR="${LOG_DIR:-$OUT_DIR/logs}"
mkdir -p "$LOG_DIR"

echo "case=$CASE part=$PART workers=$WORKERS out=$OUT_DIR/$PART.jsonl"

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
  # Every pool has to be pinned, not just torch's: NumPy's OpenBLAS ignores
  # ``OMP_NUM_THREADS`` and spawns a per-process pool of its own (~18 threads
  # here).  With 44 shards already running, the four np/ns arms launched on
  # 2026-10-20 pushed the account to 3252 of its 4096 threads with another
  # ~780 still to spawn -- the same wedge that killed the 2026-10-18 wave.
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH="$ROOT" \
    nohup python -u "$ROOT/scripts/exp_study.py" "$PART" \
      --shard "$i" --shards "$WORKERS" \
      --out "$OUT_DIR/$PART.jsonl" \
      "$@" > "$LOG_DIR/$PART.$i.log" 2>&1 &
  pids+=($!)
done

echo "launched ${#pids[@]} shards: ${pids[*]}"
cat > "$OUT_DIR/$PART.pids" <<<"${pids[@]}"

wait
echo "all shards finished; aggregating"
PYTHONPATH="$ROOT" python -u "$ROOT/scripts/exp_study.py" "$PART" \
  --aggregate "$OUT_DIR/$PART.jsonl" --out "$OUT_DIR/$PART.summary.json" "$@"
