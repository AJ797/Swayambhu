#!/usr/bin/env bash
# The five remaining experiments, cheapest-and-most-valuable first.
#
#   bash scripts/morning_batch.sh            # run everything
#   bash scripts/morning_batch.sh 1 3        # run only steps 1 and 3
#
# Each step writes its own log under data/logs/ and is independent, so a
# failure in one does not block the rest. Steps are ordered so that if the
# budget runs out partway, the results you lose are the ones you care least
# about. Rough costs are estimates, not measurements.
set -u
cd "$(dirname "$0")/.."
PY="./.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
L14="Qwen/Qwen2.5-14B-Instruct"
L72="Qwen/Qwen2.5-72B-Instruct"

want() { [ $# -eq 0 ] && return 0; for a in "$@"; do [ "$a" = "$STEP" ] && return 0; done; return 1; }
run()  { echo; echo "=== [$STEP] $1 ==="; shift; "$@"; echo "   exit=$?"; }

# ---------------------------------------------------------------- 1. ablate
# The missing leg of the countermeasure story. We know bandwidth restriction
# makes collusion worse (0.190 -> 0.742) and a maverick removes it (-0.027).
# This asks whether projecting the collusion direction out of the residual
# stream breaks a cartel WITHOUT having to detect it first. Run at 72B on
# E_CANON, the strongest collusion anywhere in the study.  (~$3)
STEP=1
if want "$@"; then
  run "ablate @72B on E_CANON" env SWAY_MODEL="$L72" SWAY_GPU="A100-80GB:4" SWAY_DEVICE_MAP=auto \
    $PY -u scripts/steering.py --phase ablate \
      --direction data/cap72/direction_pre_price.npy \
      --ablate-condition E_CANON --layer 64 \
      --episodes 8 --rounds 14 --run-id ablate72 \
      > data/logs/batch1_ablate72.log 2>&1
fi

# ------------------------------------------------- 2. semantic placebo
# The control the random/shuffled vectors do NOT provide. Both placebos have
# textual difference comparable to the real contrast; placebo_coop carries the
# same cooperate-vs-compete framing but about filing reports rather than
# prices. If either induces collusion, the collusion direction is not
# price-specific and section 4.8 needs rewriting.  (~$2)
STEP=2
if want "$@"; then
  for c in placebo_coop placebo_neutral; do
    run "extract $c direction" env SWAY_MODEL="$L14" SWAY_GPU="A100-80GB" \
      $PY -u scripts/trust_direction.py --contrast $c \
        >> data/logs/batch2_placebo.log 2>&1
  done
  run "steer with placebo_coop" env SWAY_MODEL="$L14" SWAY_GPU="A100-80GB" \
    $PY -u scripts/steering.py --phase sweep \
      --direction data/main/direction_placebo_coop.npy \
      --condition E_BASE --layers 36 40 --alphas 0.2 0.4 0.6 \
      --episodes 8 --rounds 12 --run-id placebo_coop \
      >> data/logs/batch2_placebo.log 2>&1
  run "steer with placebo_neutral" env SWAY_MODEL="$L14" SWAY_GPU="A100-80GB" \
    $PY -u scripts/steering.py --phase sweep \
      --direction data/main/direction_placebo_neutral.npy \
      --condition E_BASE --layers 36 40 --alphas 0.2 0.4 0.6 \
      --episodes 8 --rounds 12 --run-id placebo_neutral \
      >> data/logs/batch2_placebo.log 2>&1
fi

# -------------------------------------------------- 3. positions="last"
# Currently we add the vector at EVERY sequence position, so the honest claim
# is activation-space prompt injection. Injecting only at the final position
# would be editing the decision state, which is the stronger reading.  (~$1)
STEP=3
if want "$@"; then
  run "collusion direction, final position only" env SWAY_MODEL="$L14" SWAY_GPU="A100-80GB" \
    $PY -u scripts/steering.py --phase sweep \
      --direction data/main/direction_pre_price.npy \
      --condition E_BASE --layers 36 40 --alphas 0.2 0.4 0.6 0.8 \
      --positions last --episodes 8 --rounds 12 --run-id poslast \
      > data/logs/batch3_poslast.log 2>&1
fi

# ------------------------------------------------------- 4. gated bank
# Collusion alone, trust alone, softmax-gated mixture and uniform-gated
# mixture at matched alpha and layer, so gating is isolated from the bank's
# span. Prediction on the record: the bank underperforms the collusion
# direction alone, because a convex mixture of two near-orthogonal (84 deg)
# unit vectors has norm ~0.71, so matched alpha delivers a smaller step
# split across one strong and one weak axis.  (~$2)
STEP=4
if want "$@"; then
  run "gated bank comparison" env SWAY_MODEL="$L14" SWAY_GPU="A100-80GB" \
    $PY -u scripts/steering.py --phase bank \
      --directions data/main/direction_pre_price.npy data/main/direction_trust.npy \
      --condition E_BASE --layer 40 --alphas 0.2 0.4 \
      --episodes 8 --rounds 12 --run-id bank14 \
      > data/logs/batch4_bank.log 2>&1
fi

# ------------------------------------------- 5. single-agent steering
# row_mask has been implemented since the first day and never used. Steer ONE
# firm of three and ask whether a single intervened agent pulls the whole
# market up. A positive result is a sharper causal claim than steering
# everyone, and a much more realistic threat model.  (~$2)
STEP=5
if want "$@"; then
  run "steer firm 0 only" env SWAY_MODEL="$L14" SWAY_GPU="A100-80GB" \
    $PY -u scripts/steering.py --phase sweep \
      --direction data/main/direction_pre_price.npy \
      --condition E_BASE --layers 40 --alphas 0.4 0.6 0.8 \
      --steer-firms 0 --episodes 8 --rounds 14 --run-id onefirm \
      > data/logs/batch5_onefirm.log 2>&1
fi

echo
echo "=== done. summaries: ==="
for f in data/logs/batch*.log; do
  [ -f "$f" ] || continue
  echo "--- $f"
  grep -E "^(baseline|  +[0-9]+ +-?[0-9]|  real |  random|  shuffled|  single:|  bank:|  ablated|  largest valid)" "$f" | tr -d '\r' | tail -14
done
