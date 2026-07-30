#!/usr/bin/env bash
#
# aws/run_domain_test.sh — launch the full-domain vs subdomain comparison.
#
# Thin wrapper over run_on_ec2.sh: it reuses the whole proven bootstrap (code
# tarball, conda env from the lock, log shipping, AZ walk, self-termination) and
# only swaps the command the instance runs, via --run-cmd. See aws/domain_test.sh
# for what actually happens on the instance and why it is one instance rather than
# two.
#
# Cost: inputs (~5 min) + full domain (~28 min at 24 h) + subdomain (~8 min
# expected) + a second full-domain run for the noise yardstick (~28 min) is roughly
# 69 min, about $2.60 on a g6e.2xlarge. The third run is not padding: the diffusion
# noise draw is shape-dependent, so without it the crop effect cannot be separated
# from a different random realization. See aws/domain_test.sh.
#
# All runs share the SAME GPU, so the timing comparison is apples to apples. Whether
# the subdomain also fits a cheaper A10G is inferred from its measured peak VRAM
# rather than tested directly.
#
# Usage:
#   aws/run_domain_test.sh --bucket NAME [--init-time YYYY-MM-DDTHH]
#                          [--lead-hours 24] [--height 531] [--width 903]
#                          [--dry-run] [passthrough options...]
#
# Subdomain size is constrained: height % 8 == 3 and width % 8 == 7, because the
# trained model bakes a fixed reflect-padding. src/crop_domain.py explains and
# enforces this; pick sizes with:
#   python3 src/crop_domain.py --in-dir . --out-dir . --init-time X \
#       --height H --width W --dry-run
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUCKET=""
INIT_TIME=""
LEAD_HOUR=24
SUB_H=531
SUB_W=903
# 4, not run_on_ec2.sh's default of 0. The test needs a cycle whose GFS is actually
# published: with lag 0 a 13Z init wants the 12Z cycle, which does not finish
# publishing until ~16Z, so get_bcs would fail. It also matches what hourly operation
# will really use, which is the configuration worth characterizing.
GFS_MIN_LAG_OPT=4
# Hard wall-clock cap on the whole experiment, enforced on the instance with
# timeout(1). Insurance against a hang: a command-substitution deadlock in an earlier
# version of domain_test.sh left the GPU idle for 82 minutes before it was noticed,
# and because user_data.sh only self-terminates when its command RETURNS, a hang
# burns money indefinitely. 3 h is roughly 2.5x the expected 69 min.
WALL_LIMIT=10800
# Region of interest, "N,W,S,E" in degrees. When set, the crop is sized and placed to
# contain it plus --halo cells rather than being a grid-centred --height x --width box.
# Verify the box you will actually run: the published 25.2% result was measured on a
# central/eastern US box, and both the halo behaviour and the 2 m temperature bias
# depend on which climatological regions the crop includes.
BBOX=""
HALO=40
PASSTHRU=()

while [ $# -gt 0 ]; do
    case "$1" in
        --bucket)     BUCKET="$2"; shift 2 ;;
        --init-time)  INIT_TIME="$2"; shift 2 ;;
        --lead-hours) LEAD_HOUR="$2"; shift 2 ;;
        --gfs-min-lag) GFS_MIN_LAG_OPT="$2"; shift 2 ;;
        --wall-limit)  WALL_LIMIT="$2"; shift 2 ;;
        --bbox)       BBOX="$2"; shift 2 ;;
        --halo)       HALO="$2"; shift 2 ;;
        --height)     SUB_H="$2"; shift 2 ;;
        --width)      SUB_W="$2"; shift 2 ;;
        -h|--help)    sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^#\{0,1\} \{0,1\}//'; exit 0 ;;
        *)            PASSTHRU+=("$1"); shift ;;
    esac
done

die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
[ -n "$BUCKET" ] || die "--bucket is required"

# Validate the subdomain size here, before an instance exists. The same check runs
# again on the instance, but finding out after a 5 minute boot that the height is
# wrong by one cell is a poor use of a GPU.
if [ -z "$BBOX" ]; then
python3 - "$SUB_H" "$SUB_W" <<'PY' || die "invalid subdomain size (see above)"
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath("aws")), "src"))
sys.path.insert(0, "src")
from crop_domain import valid_size
h, w = int(sys.argv[1]), int(sys.argv[2])
ok, msg = valid_size(h, w)
if not ok:
    print(f"  {msg}", file=sys.stderr)
sys.exit(0 if ok else 1)
PY
fi

# Choose a cycle whose inputs are actually all published, so the test does not fail
# on a missing GFS step. Reuses the scheduler's picker rather than guessing.
if [ -z "$INIT_TIME" ]; then
    echo "Selecting a cycle with complete inputs..."
    eval "$("${REPO_DIR}/aws/pick_cycle.sh" --lead-hours "$LEAD_HOUR" \
              --gfs-min-lag "$GFS_MIN_LAG_OPT")" \
        || die "no cycle with complete inputs right now; pass --init-time explicitly"
    echo "  using ${INIT_TIME} (GFS lag ${GFS_MIN_LAG}h)"
    GFS_MIN_LAG_OPT="$GFS_MIN_LAG"
fi

if [ -n "$BBOX" ]; then
    SUB_DESC="bbox ${BBOX} + ${HALO}-cell halo"
else
    SUB_DESC="${SUB_H} x ${SUB_W} (grid-centred)"
fi

S3_PREFIX="s3://${BUCKET}/hrrrcast/domain-test/${INIT_TIME}"

cat <<EOF

  domain test
    init time      ${INIT_TIME}
    lead hours     ${LEAD_HOUR}
    subdomain      ${SUB_DESC}
    gfs min lag    ${GFS_MIN_LAG_OPT} h
    outputs        ${S3_PREFIX}/{full,sub,full-m1}/
    expected       ~69 min, ~\$2.60 (3 forecasts on one instance)
    wall limit     ${WALL_LIMIT} s (hard timeout on the instance)

EOF

# DATAROOT is exported by user_data.sh before RUN_CMD runs.
exec "${REPO_DIR}/aws/run_on_ec2.sh" \
    --bucket "$BUCKET" \
    --init-time "$INIT_TIME" \
    --lead-hours "$LEAD_HOUR" \
    --gfs-min-lag "$GFS_MIN_LAG_OPT" \
    --run-cmd "timeout --signal=TERM --kill-after=60 ${WALL_LIMIT}s ./aws/domain_test.sh '${INIT_TIME}' '${LEAD_HOUR}' '${S3_PREFIX}' '${SUB_H}' '${SUB_W}' '${BBOX}' '${HALO}'" \
    ${PASSTHRU[@]+"${PASSTHRU[@]}"}
