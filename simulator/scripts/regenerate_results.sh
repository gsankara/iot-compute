#!/usr/bin/env bash
# Reproduce every figure and table in docs/RESULTS.md.
#
# Run from anywhere:
#     bash scripts/regenerate_results.sh              -> results/  (gitignored)
#     bash scripts/regenerate_results.sh docs/results -> committable artefacts
#
# Writes raw CSV, JSON and PNG plus console transcripts.  Roughly two minutes on
# a laptop.  The curated tables are committed in docs/RESULTS.md; use the second
# form if you also want the raw files under version control.
#
# Override the interpreter with PYTHON=/path/to/python if the search below picks
# the wrong one.

set -euo pipefail

cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# Find a working Python 3.
#
# Not as simple as calling `python`.  On Windows that name may be absent from a
# Git Bash PATH entirely, or it may resolve to the Microsoft Store stub, which
# exists, exits zero and prints an advert instead of running anything.  So each
# candidate is *executed* and its output checked, rather than merely located.
# ---------------------------------------------------------------------------
PY=()

_try() {
    local out
    out="$("$@" -c 'import sys; print(sys.version_info[0])' 2>/dev/null)" || return 1
    [ "$out" = "3" ] || return 1
    PY=("$@")
    return 0
}

if [ -n "${PYTHON:-}" ]; then
    _try "$PYTHON" || { echo "PYTHON=$PYTHON does not run Python 3" >&2; exit 1; }
else
    # ${LOCALAPPDATA:-} rather than $LOCALAPPDATA: `set -u` would abort here on
    # a shell where the variable is not exported.
    _local="${LOCALAPPDATA:-$HOME/AppData/Local}"
    _try python \
        || _try python3 \
        || _try py -3 \
        || _try "$_local/Programs/Python/Python313/python.exe" \
        || _try "$_local/Programs/Python/Python312/python.exe" \
        || _try "$_local/Programs/Python/Python311/python.exe" \
        || {
            echo "No working Python 3 found." >&2
            echo "Tried: python, python3, py -3, and the usual Windows install paths." >&2
            echo "Set one explicitly, e.g.:" >&2
            echo "    PYTHON='/c/Users/you/AppData/Local/Programs/Python/Python313/python.exe' \\" >&2
            echo "        bash scripts/regenerate_results.sh" >&2
            exit 1
        }
fi

echo "using interpreter: ${PY[*]}"
"${PY[@]}" --version
echo

# Fail early and clearly if the package is not importable, rather than eight
# confusing tracebacks later.
if ! "${PY[@]}" -c 'import iotc' >/dev/null 2>&1; then
    echo "Cannot import iotc from $(pwd)." >&2
    echo "Install it first:  ${PY[*]} -m pip install -e '.[dev]'" >&2
    exit 1
fi

OUT="${1:-results}"
mkdir -p "$OUT/console"

echo "== 1. reach against medium and wavelength =="
"${PY[@]}" examples/01_reach_vs_medium.py | tee "$OUT/console/01_reach.txt"

echo
echo "== 2. platform comparison =="
"${PY[@]}" examples/02_platform_comparison.py | tee "$OUT/console/02_platforms.txt"

echo
echo "== 3. custom plugins, and energy-limited motion =="
"${PY[@]}" examples/03_custom_plugin.py | tee "$OUT/console/03_plugins.txt"

echo
echo "== 4. a heterogeneous liquid swarm =="
"${PY[@]}" -m iotc.cli sim configs/liquid_swarm.yaml \
    --connect --traffic 30 --until 20 --out "$OUT/liquid" \
    | tee "$OUT/console/04_liquid_swarm.txt"

echo
echo "== 5. platform design sweep =="
"${PY[@]}" -m iotc.cli bench configs/platform_study.yaml \
    --out "$OUT/platform" --seeds 2 --plot \
    | tee "$OUT/console/05_platform_sweep.txt"

echo
echo "== 6. fault campaign, with and without =="
"${PY[@]}" -m iotc.cli bench configs/fault_campaign.yaml \
    --out "$OUT/faults" --seeds 3 \
    | tee "$OUT/console/06_fault_campaign.txt"

echo
echo "== 7. surrogate accuracy against the exact model =="
"${PY[@]}" - <<'PY_EOF' > "$OUT/surrogate_config.json"
import json, yaml
cfg = yaml.safe_load(open("configs/liquid_swarm.yaml", encoding="utf-8"))
cfg["channel"] = {"type": "surrogate", "base": "fso", "turbulence": "none",
                  "min_distance": 0.5, "max_distance": 150.0}
cfg["run"]["until"] = 2.0
print(json.dumps(cfg))
PY_EOF
"${PY[@]}" -m iotc.cli sim "$OUT/surrogate_config.json" \
    --verify-surrogate --until 2 --connect \
    | tee "$OUT/console/07_surrogate.txt"

echo
echo "== 8. a single link budget =="
"${PY[@]}" -m iotc.cli link --medium water --wavelength 470 --distance 20 --sweep \
    | tee "$OUT/console/08_link_budget.txt"

echo
echo "== 9. B0 -- what each workload demands of the hardware =="
"${PY[@]}" -m iotc.cli graph --out "$OUT/compiler/b0" \
    | tee "$OUT/console/09_workload_characterization.txt"

echo
echo "== 10. B1 -- what making it fit costs =="
"${PY[@]}" -m iotc.cli tile --out "$OUT/compiler/b1" \
    | tee "$OUT/console/10_tiling.txt"

echo
echo "== 11. B2 -- cut quality and the near-linearity study =="
"${PY[@]}" -m iotc.cli partition --out "$OUT/compiler/b2" \
    | tee "$OUT/console/11_partition.txt"

echo
echo "== 12. B3 -- placing the swarm, and what the formation costs to fly =="
"${PY[@]}" -m iotc.cli place --out "$OUT/compiler/b3" \
    | tee "$OUT/console/12_placement.txt"

echo
echo "== 13. B4 -- flying the swarm without dropping its links =="
"${PY[@]}" -m iotc.cli motion --out "$OUT/compiler/b4" \
    | tee "$OUT/console/13_motion.txt"

echo
echo "== 14. B5 -- beam rounds, and what the gimbals can point at =="
"${PY[@]}" -m iotc.cli beam --out "$OUT/compiler/b5" \
    | tee "$OUT/console/14_beam.txt"

echo
echo "done -- artefacts in $OUT/"
