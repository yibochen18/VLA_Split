#!/usr/bin/env bash
# End-to-end fix on fresh test starts (RESULTS §17): fit on starts 0-29, test on 30-39.
source "$(dirname "$0")/../env.sh" && vla_activate
set -o pipefail
ROOT=$VLA_DATA/blackbox
S=libero_spatial
REC=$ROOT/record9_starts0-29/$S
FITS=$ROOT/fits_fresh30/$S
mkdir -p "$REC" "$FITS"
log() { echo "=== [$(date '+%F %T')] $* ==="; }

# Recorded queries from starts 0-29 only (file names are task<t>_ep<start>_q<k>.pt)
for f in "$ROOT/record9/$S"/task*_ep*_q*.pt; do
    ep=$(basename "$f" | sed -E 's/.*_ep([0-9]+)_q.*/\1/')
    [ "$ep" -lt 30 ] && ln -sf "$f" "$REC/"
done
log "training pool: $(ls "$REC" | wc -l) recorded queries from starts 0-29"

cd "$VLA_ROOT/openvla-oft" || exit 1
evaluate() {  # tag
    log "eval $1 (starts 30-39)"
    python ../split/run_split_eval.py --suite $S --first_episode 30 --episodes_per_task 10 \
        --bundle "$VLA_SPLIT/$S/model2" --tag "$1" --split_root "$ROOT/evals_fresh30" --blackbox "$FITS/$1.pt" 2>&1 \
        | grep -E '^Success|Traceback|Error'
}

T=L1-9_tf2_n1000
log "reconstruction fit $T"
python ../split/fit_blackbox.py --block 1 9 --kind tf2 --record_dir "$REC" --n_queries 1000 --loss scaled \
    --out "$FITS/$T.pt" > "$FITS/$T.log" 2>&1
grep -E '^loaded|^fitted|Traceback|Error' "$FITS/$T.log"
log "end-to-end fine-tune ${T}_e2e"
python ../split/fit_blackbox_e2e.py --suite $S --init "$FITS/$T.pt" --record_dir "$REC" --n_queries 1000 \
    --out "$FITS/${T}_e2e.pt" > "$FITS/${T}_e2e.log" 2>&1
grep -E '^loaded|^start|^step 3000|best|Traceback|Error' "$FITS/${T}_e2e.log" | tail -3
evaluate "${T}_e2e"
evaluate "$T"
log "ALL DONE"
