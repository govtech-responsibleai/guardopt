#!/usr/bin/env bash
# The full scoring grid: every ungated dataset, 2,000 stratified cases, three judges.
#
# Safe to re-run. Every dataset resumes from its own ledger, so a kill, a token
# expiry or a closed laptop costs one chunk; running this again picks up where it
# stopped and pays only for what is missing.
#
#   bash experiments/run_grid.sh              # the full grid
#   SAMPLE=500 bash experiments/run_grid.sh   # a smaller pass first
#
# Gated datasets are deliberately absent: wildguardmix needs HF_TOKEN and jigsaw is a
# manual Kaggle download. Add them here once those land — the grid is additive.
set -uo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-.venv311/bin/python}"
# Sized per dataset, not uniformly, because what binds is the number of UNSAFE cases
# (recall's denominator) and the unsafe share ranges from 7% to 75%:
#
#   toxicchat          7% unsafe, pool 5,083 → 4,000 cases ≈ 285 unsafe, recall ±0.058
#   openai_moderation 31% unsafe, pool 1,680 → 1,600 cases ≈ 497 unsafe, recall ±0.044
#   beavertails       57% unsafe, pool 3,021 → 2,000 cases ≈ 1,147 unsafe, ±0.029
#   unsmile           75% unsafe, pool 3,737 → 2,000 cases ≈ 1,500 unsafe, ±0.025
#
# openai_moderation is capped by the dataset itself: 1,680 rows is all there is.
# NOTE: the sample size is part of the artifact name AND changes which cases the seeded
# stratified sampler picks, so a size cannot be "topped up" later — a bigger sample is a
# different (re-scored) run. Choose it once.
SAMPLE="${SAMPLE:-}"
sample_for() {
  case "$1" in
    toxicchat) echo 4000 ;;
    openai_moderation) echo 1600 ;;
    *) echo 2000 ;;
  esac
}
SEED="${SEED:-0}"
MODELS="${MODELS:-gemini-3.5-flash-lite,azure.claude-haiku-4-5,azure.claude-sonnet-5}"
CONCURRENCY="${CONCURRENCY:-2}"
CHUNK="${CHUNK:-25}"
# The cached pools were fetched for the n=200 pilot and hold as few as 600 rows; a
# 2,000-case sample needs a bigger pool. REFRESH_POOL=1 refetches once, then the cache
# serves every later run.
POOL_SIZE="${POOL_SIZE:-6000}"
REFRESH_POOL="${REFRESH_POOL:-0}"
# A ceiling per dataset, not per run: measured spend, checked between chunks. The
# projection is ~$4.1 per dataset at n=2000, so this stops a runaway without
# stopping honest work.
MAX_SPEND="${MAX_SPEND:-12}"

DATASETS="${DATASETS:-toxicchat openai_moderation beavertails unsmile}"

echo "grid: $(echo "$MODELS" | tr ',' '\n' | wc -l | tr -d ' ') judges, sample sized per dataset"
echo "grid: datasets: ${DATASETS}"
echo "grid: per-dataset measured-spend ceiling US\$${MAX_SPEND}"
echo

for dataset in $DATASETS; do
  echo "══════════════════════════════════════════════════════════"
  echo "  ${dataset}  ($(date '+%H:%M:%S'))"
  echo "══════════════════════════════════════════════════════════"
  dataset_sample="${SAMPLE:-$(sample_for "$dataset")}"
  "$PY" -m experiments.score \
    --dataset "$dataset" \
    --sample "$dataset_sample" \
    --seed "$SEED" \
    --models "$MODELS" \
    --concurrency "$CONCURRENCY" \
    --chunk "$CHUNK" \
    --max-spend "$MAX_SPEND" \
    --pool-size "$POOL_SIZE" \
    $( [ "$REFRESH_POOL" = "1" ] && echo --refresh-pool ) \
    --yes
  status=$?
  if [ $status -ne 0 ]; then
    # One dataset failing must not abandon the rest: each has its own ledger, and
    # re-running the script retries only what is missing.
    echo "grid: ${dataset} exited ${status} — continuing with the next dataset." >&2
  fi
  echo
done

echo "grid: finished at $(date '+%H:%M:%S'). Re-run to fill any gaps."
