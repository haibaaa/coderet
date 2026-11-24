#!/usr/bin/env bash
# Reproduce the Flask experiments reported in the paper.
#   FLASK=~/r/flask experiments/run_flask.sh
# Flask must be checked out at 49943b5a94413a530761350f04e896f76e8d620e.
# Runs sequentially (CPU-only); ablation indexes are kept separate from the
# main index. Expect ~40 minutes on a 2-core laptop CPU.
set -euo pipefail
cd "$(dirname "$0")/.."
FLASK=${FLASK:-$HOME/r/flask}
OUT=experiments/results
CFG=experiments/configs
QUERIES=examples/flask_queries.jsonl
K=1,3,5,10
mkdir -p "$OUT" "$CFG"

run_eval () {  # name config modes
  local name=$1 cfg=$2 modes=$3
  uv run coderet eval "$QUERIES" --repo "$FLASK" --config "$cfg" --modes "$modes" --k "$K" \
    --output "$OUT/$name.json" | tee "$OUT/$name.txt"
}

timed_index () {  # name config
  local name=$1 cfg=$2 start end
  start=$(date +%s)
  uv run coderet index "$FLASK" --config "$cfg" > "$OUT/index-$name.json"
  end=$(date +%s)
  echo "{\"name\": \"$name\", \"wall_seconds\": $((end - start))}" > "$OUT/time-$name.json"
}

# Main configuration (default settings).
cat > "$CFG/main.toml" <<'EOF'
EOF
timed_index main "$CFG/main.toml"
run_eval main "$CFG/main.toml" dense,lexical,hybrid

# Fusion weight sweep (no re-embedding needed).
for w in 1.0 0.5 0.25 0.1 0.05; do
  printf '[search]\nlexical_weight = %s\n' "$w" > "$CFG/fusion-$w.toml"
  run_eval "fusion-$w" "$CFG/fusion-$w.toml" hybrid
done

# Ablation 1: no File/Symbol header in the embedded text.
printf '[chunking]\nembed_header = false\n[index]\npath = ".coderet/abl-noheader.db"\n' > "$CFG/noheader.toml"
timed_index noheader "$CFG/noheader.toml"
run_eval noheader "$CFG/noheader.toml" dense,lexical,hybrid

# Ablation 2: fixed 60-line windows instead of syntax-aware chunks.
printf '[chunking]\nstrategy = "window"\n[index]\npath = ".coderet/abl-window.db"\n' > "$CFG/window.toml"
timed_index window "$CFG/window.toml"
run_eval window "$CFG/window.toml" dense,lexical,hybrid

echo "done"
