#!/usr/bin/env bash
# Depth follow-ups: MLP readout, retest on starts 30-39, constant control, second subset, Goal/Object.
source "$(dirname "$0")/../env.sh" && vla_activate
set -o pipefail
ROOT=$VLA_DATA/linear_followup
mkdir -p "$ROOT"

log() { echo "=== [$(date '+%F %T')] $* ==="; }

# Wait for room on the shared GPU; retry failed starts
wait_gpu() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")" -lt 7500 ]; do sleep 60; done; }
gpu_run() {
    for attempt in $(seq 12); do
        wait_gpu
        if "$@"; then return 0; fi
        log "attempt $attempt failed, retrying in 2 min"; sleep 120
    done
    log "GIVING UP: $*"; return 1
}

eval_run() {  # suite first_episode tag bundle layer
    (cd "$VLA_ROOT/openvla-oft" && python ../split/run_split_eval.py --suite "$1" --first_episode "$2" \
        --episodes_per_task 10 --bundle "$4" --tag "$3" --split_root "$ROOT/evals" \
        --interface_layer "$5" --truncate 2>&1 | grep -E '^Success|Traceback|Error')
}

pairs_dir() {  # suite layer
    case "$1" in
        libero_spatial|libero_10) echo "$VLA_DATA/linear_layers/pairs/$1/L$2" ;;
        *) echo "$ROOT/pairs/$1/L$2" ;;
    esac
}

ridge() {  # out ridge_head-args (GPU, CPU fallback)
    local out_dir=$1; shift
    local out
    out=$(PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python "$VLA_ROOT/split/ridge_head.py" "$@" \
        --out "$out_dir" 2>&1 | grep -E '^ridge|Traceback|Error'); echo "$out"
    grep -q '^ridge' <<< "$out" && return 0
    log "GPU fit failed, falling back to CPU"
    python "$VLA_ROOT/split/ridge_head.py" --device cpu "$@" --out "$out_dir" 2>&1 | grep -E '^ridge|Traceback|Error' | grep '^ridge'
}

# Wait for run_linear_compute.sh
until tail -1 "$VLA_LOGS/linear_compute.log" | grep -q 'ALL DONE'; do sleep 120; done

log "A. small MLP readout at deep layers"
for job in "libero_spatial 5000 16 12 8 4" "libero_10 13000 16 12 8"; do
    set -- $job; suite=$1; n=$2; shift 2
    for L in "$@"; do
        log "train mlp $suite L$L n=$n"
        (cd "$VLA_ROOT/split" && python train_replacement.py --arch mlp --num_queries "$n" \
            --pairs_dir "$(pairs_dir "$suite" "$L")" --out_root "$ROOT/mlp/${suite}_L$L" 2>&1 | grep -E '^mlp|Traceback|Error') || continue
        log "closed loop mlp $suite L$L"
        gpu_run eval_run "$suite" 40 "L${L}_mlp" "$ROOT/mlp/${suite}_L$L/mlp_n${n}_s0" "$L"
    done
done

log "B. replicate on starts 30-39"
for job in "libero_spatial 24 20 16 12 8 4 2" "libero_10 20 16"; do
    set -- $job; suite=$1; shift
    for L in "$@"; do
        log "fit $suite L$L (test 30-39)"
        ridge "$ROOT/heads/${suite}_L${L}_test30" --pairs_dir "$(pairs_dir "$suite" "$L")" \
            --test_from 30 --test_to 40 --val_from 40 --val_to 50 || continue
        log "closed loop $suite L$L (starts 30-39)"
        gpu_run eval_run "$suite" 30 "L${L}_ridge_test30" "$ROOT/heads/${suite}_L${L}_test30" "$L"
    done
done

log "C. constant-action control (Spatial)"
python - "$VLA_EXTRACTION/libero_spatial/pairs" "$VLA_DATA/linear_layers/heads/libero_spatial_L20" "$ROOT/heads/libero_spatial_constant" <<'EOF'
import json, shutil, sys
from pathlib import Path
import torch
pairs, template, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
acts = [torch.as_tensor(torch.load(f, weights_only=False)["actions"]).reshape(-1, 7)
        for f in sorted(pairs.glob("task*_ep*.pt")) if int(f.stem.split("_ep")[1]) < 40]
mean = torch.cat(acts).double().mean(0)
out.mkdir(parents=True, exist_ok=True)
linear = torch.nn.Linear(7 * 4096, 7)
with torch.no_grad():
    linear.weight.zero_(); linear.bias.copy_(mean.float())
torch.save(linear.state_dict(), out / "head.pt")
config = json.loads((template / "config.json").read_text())
config.update({"fit": "constant_mean_action", "pairs_dir": str(pairs), "lambda_val_l1": {}, "lambda": None,
               "metrics": {}, "num_train_queries": sum(len(a) // 8 for a in acts)})
(out / "config.json").write_text(json.dumps(config, indent=2))
print(f"constant head: mean action {mean.numpy().round(4).tolist()}")
EOF
gpu_run eval_run libero_spatial 40 constant "$ROOT/heads/libero_spatial_constant" 0

log "D. second data subset (seed 1)"
for L in 20 16; do
    for n in 100 500 1000; do
        log "fit libero_spatial L$L n=$n seed 1"
        ridge "$ROOT/heads/libero_spatial_L${L}_n${n}_s1" --pairs_dir "$(pairs_dir libero_spatial "$L")" \
            --test_from 40 --test_to 50 --val_from 0 --val_to 10 --max_queries "$n" --seed 1 || continue
        gpu_run eval_run libero_spatial 40 "L${L}_n${n}_s1" "$ROOT/heads/libero_spatial_L${L}_n${n}_s1" "$L"
    done
done

log "E. depth on Goal and Object"
for suite in libero_goal libero_object; do
    log "record $suite layers 24 20 16"
    gpu_run bash -c "set -o pipefail; cd '$VLA_ROOT/openvla-oft' && python ../split/run_split_eval.py --suite $suite \
        --split_root '$ROOT/record' --bundle '$VLA_SPLIT/$suite/model2' --save_pairs '$ROOT/pairs/$suite' \
        --record_layers 24 20 16 2>&1 | grep -E '^Success|Traceback|Error'" || continue
    # keep only the per-layer pairs
    find "$ROOT/pairs/$suite" -maxdepth 1 -name 'task*_ep*.pt' -delete
    for L in 24 20 16; do
        log "fit $suite L$L"
        ridge "$ROOT/heads/${suite}_L$L" --pairs_dir "$(pairs_dir "$suite" "$L")" \
            --test_from 40 --test_to 50 --val_from 0 --val_to 10 || continue
        gpu_run eval_run "$suite" 40 "L${L}_ridge" "$ROOT/heads/${suite}_L$L" "$L"
    done
done

log "ALL DONE"
