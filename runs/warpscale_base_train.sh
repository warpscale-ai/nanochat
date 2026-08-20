#!/bin/bash
#
# One-time prep, CPU-only. Short runs:
#   python -m nanochat.dataset -n 1
#   python -m scripts.tok_train --max-chars=200000000
# Full speedrun — the tokenizer must see the full 2B chars or val_bpb and CORE
# shift, so train it on 8 shards while the rest download:
#   python -m nanochat.dataset -n 8
#   python -m nanochat.dataset -n 170 &   # then `wait` before training — d24 at
#   python -m scripts.tok_train           # ratio 8 consumes ~150 shards
#   python -m scripts.tok_eval            # compression ratio; informational
#
#   runs/warpscale_base_train.sh baseline
#   runs/warpscale_base_train.sh profiled
set -euo pipefail

MODE="${1:-profiled}"
shift || true
case "$MODE" in
    baseline|profiled) ;;
    *) echo "usage: $0 [baseline|profiled] [extra base_train args...]" >&2; exit 2 ;;
esac

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
EXPERIMENT="${EXPERIMENT:-nanochat}"

# Short-run sizing for the 2x L4 dev box. fp8 stays off: nanochat documents it H100+.
DEPTH="${DEPTH:-10}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
NUM_ITERATIONS="${NUM_ITERATIONS:-60}"
# base_train asserts total_batch_size is a multiple of device_batch_size * max_seq_len * world_size.
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-$((DEVICE_BATCH_SIZE * MAX_SEQ_LEN * NPROC_PER_NODE * 2))}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WARPSCALE_REPO="${WARPSCALE_REPO:-$REPO_ROOT/../warpscale}"
WARPSCALE_BIN="${WARPSCALE_BIN:-$WARPSCALE_REPO/dist/warpscale}"
WARPSCALE_SRC="${WARPSCALE_SRC:-$WARPSCALE_REPO/warpscale-python/src}"

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"
cd "$REPO_ROOT"
source .venv/bin/activate

TRAIN_ARGS=(
    --depth="$DEPTH"
    --device-batch-size="$DEVICE_BATCH_SIZE"
    --max-seq-len="$MAX_SEQ_LEN"
    --total-batch-size="$TOTAL_BATCH_SIZE"
    --num-iterations="$NUM_ITERATIONS"
    # base_train forces val-bpb + CORE evals at step 0 and at last_step whenever
    # their intervals are positive; on a short run those dwarf the training itself.
    --eval-every="${EVAL_EVERY:--1}"
    --core-metric-every="${CORE_METRIC_EVERY:--1}"
    --sample-every="${SAMPLE_EVERY:--1}"
    --run="${WANDB_RUN:-dummy}"
    --model-tag="warpscale-$MODE"
    "$@"
)
TORCHRUN=(.venv/bin/torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train --)

if [[ "$MODE" == baseline ]]; then
    WS_RUN_TYPE=baseline "${TORCHRUN[@]}" "${TRAIN_ARGS[@]}"
else
    [[ -x "$WARPSCALE_BIN" ]] || { echo "error: warpscale shim not at $WARPSCALE_BIN — build it in $WARPSCALE_REPO" >&2; exit 1; }
    WS_RUN_TYPE=profiled PYTHONPATH="$WARPSCALE_SRC${PYTHONPATH:+:$PYTHONPATH}" \
        "$WARPSCALE_BIN" run --experiment "$EXPERIMENT" --run-name "nanochat-$MODE" -- \
        "${TORCHRUN[@]}" "${TRAIN_ARGS[@]}"
fi
