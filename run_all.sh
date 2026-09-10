#!/usr/bin/env bash
# One-command reproduction. See README.md.
#   bash run_all.sh               tests + figures
#   bash run_all.sh --skip-tests  figures only
set -euo pipefail
cd "$(dirname "$0")"

SKIP_TESTS=0
for a in "$@"; do
  case "$a" in
    --skip-tests) SKIP_TESTS=1 ;;
    *) echo "unknown option $a" >&2; exit 2 ;;
  esac
done

if [ ! -x .venv/bin/python ]; then
  echo "== creating .venv from requirements-lock.txt"
  if python3 -m venv .venv 2>/dev/null && [ -x .venv/bin/pip ]; then
    .venv/bin/pip install --quiet --upgrade pip
    .venv/bin/pip install --quiet -r requirements-lock.txt
  elif command -v uv >/dev/null; then
    rm -rf .venv; uv venv --quiet .venv
    uv pip install --quiet --python .venv/bin/python -r requirements-lock.txt
  else
    echo "python3 -m venv failed (on Debian/Ubuntu install python3-venv) and uv is not available" >&2
    exit 1
  fi
fi
PY=.venv/bin/python
export PYTHONPATH=src MPLBACKEND=Agg

mkdir -p output/figures
$PY - <<'EOF' > output/environment.txt
import platform, numpy, scipy, sympy, matplotlib, networkx
print("python", platform.python_version(), platform.platform())
for m in (numpy, scipy, sympy, matplotlib, networkx):
    print(m.__name__, m.__version__)
EOF
echo "== environment"; cat output/environment.txt

if [ "$SKIP_TESTS" -eq 0 ]; then
  echo "== tests (about 4 min unloaded)"
  $PY -m pytest tests/ -q
fi

echo "== figures (about 2 min total)"
for s in generate_figures generate_supplementary_figures generate_emergence_figure \
         generate_spatial_figure generate_figure_competition generate_figure_placement \
         generate_figure_k0_sensitivity generate_figure_audience_weights generate_figure_self_assessment generate_figure_omega_k0_joint \
         generate_figure_Cn_joint generate_figure_assessment_alternatives; do
  t0=$(date +%s); $PY scripts/$s.py > /dev/null; echo "  $s $(( $(date +%s) - t0 ))s"
done
$PY scripts/compute_rapanui_energetics.py > output/rapanui_energetics.txt
echo "  (fig1_overview.pdf is a static asset exported from fig1_overview_source.svg)"
echo "== done"
