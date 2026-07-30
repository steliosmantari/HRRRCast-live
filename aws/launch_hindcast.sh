#!/usr/bin/env bash
#
# aws/launch_hindcast.sh — launch a multi-cycle HRRRCast hindcast on one
# continuously-running EC2 instance.
#
# Thin wrapper over run_on_ec2.sh, same pattern as aws/run_domain_test.sh: it
# reuses the whole proven bootstrap (code tarball, conda env, log shipping, AZ
# walk, self-termination) and swaps in aws/run_hindcast.sh via --run-cmd, which
# loops run_cycle.sh over every cycle in the requested date range on that SAME
# instance instead of launching one instance per cycle.
#
# Usage:
#   aws/launch_hindcast.sh --bucket NAME --start YYYY-MM-DD --end YYYY-MM-DD \
#       --bbox N,W,S,E [options] [run_on_ec2.sh passthrough options...]
#
# Required:
#   --bucket NAME          S3 bucket for outputs and logs
#   --start YYYY-MM-DD     first cycle date (UTC), inclusive
#   --end YYYY-MM-DD       last cycle date (UTC), inclusive
#   --bbox N,W,S,E         region to crop the forecast to (see docs/subdomain.md)
#
# Hindcast options (defaults in brackets):
#   --init-hours LIST      comma list of UTC init hours              [00,06,12,18]
#   --lead-hours N         forecast length per cycle                 [24]
#   --output-hours SPEC    only keep these lead hours ("start:step:end" or a
#                          comma list, e.g. "0:3:24"); passed straight to
#                          fcst.py's --output_hours. Every hour still computes
#                          (autoregressive), only I/O is skipped.  [all hours]
#   --halo N               halo cells around --bbox                  [40]
#   --gfs-min-lag N        GFS cycle lag; 0 is correct for a hindcast since
#                          every cycle in the past is long since published [0]
#   --wall-limit SECONDS   hard timeout on the whole hindcast, enforced on the
#                          instance with timeout(1) -- insurance against a hang
#                          burning money indefinitely on an unattended
#                          multi-day run                              [see below]
#
# All other options (--instance-type, --key-name, --notify-topic,
# --no-terminate, --dry-run, --preflight-only, --region, ...) pass straight
# through to run_on_ec2.sh; see its --help.
#
# Example (SoCal, June 2026, 6-hourly cycles, 24h lead, output every 3h):
#   aws/launch_hindcast.sh --bucket mantari-cast1 \
#       --start 2026-06-01 --end 2026-06-30 --init-hours 00,06,12,18 \
#       --lead-hours 24 --output-hours 0:3:24 \
#       --bbox 35.0,-118.77,33.25,-117.0 --halo 80
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUCKET=""
START_DATE=""
END_DATE=""
INIT_HOURS="00,06,12,18"
LEAD_HOUR=24
OUTPUT_HOURS=""
BBOX=""
HALO=40
GFS_MIN_LAG_OPT=0
WALL_LIMIT=""
PASSTHRU=()

log() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --bucket)        BUCKET="$2"; shift 2 ;;
        --start)         START_DATE="$2"; shift 2 ;;
        --end)           END_DATE="$2"; shift 2 ;;
        --init-hours)    INIT_HOURS="$2"; shift 2 ;;
        --lead-hours)    LEAD_HOUR="$2"; shift 2 ;;
        --output-hours)  OUTPUT_HOURS="$2"; shift 2 ;;
        --bbox)          BBOX="$2"; shift 2 ;;
        --halo)          HALO="$2"; shift 2 ;;
        --gfs-min-lag)   GFS_MIN_LAG_OPT="$2"; shift 2 ;;
        --wall-limit)    WALL_LIMIT="$2"; shift 2 ;;
        -h|--help)       sed -n '2,44p' "${BASH_SOURCE[0]}" | sed 's/^#\{0,1\} \{0,1\}//'; exit 0 ;;
        *)               PASSTHRU+=("$1"); shift ;;
    esac
done

[ -n "$BUCKET" ]     || die "--bucket is required"
[ -n "$START_DATE" ] || die "--start YYYY-MM-DD is required"
[ -n "$END_DATE" ]   || die "--end YYYY-MM-DD is required"
[ -n "$BBOX" ]       || die "--bbox N,W,S,E is required (full-domain hindcasts are not what this script is for; use run_on_ec2.sh per cycle instead)"

[[ "$START_DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || die "--start must be YYYY-MM-DD (got '${START_DATE}')"
[[ "$END_DATE"   =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || die "--end must be YYYY-MM-DD (got '${END_DATE}')"

# --- enumerate cycles and validate the crop, before anything costs money ---
CYCLE_INFO="$(python3 -c '
import sys, datetime as d
start, end, hours = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
s = d.datetime.strptime(start, "%Y-%m-%d").date()
e = d.datetime.strptime(end, "%Y-%m-%d").date()
if e < s:
    sys.exit("--end is before --start")
for h in hours:
    if not (len(h) == 2 and h.isdigit() and 0 <= int(h) <= 23):
        sys.exit(f"invalid --init-hours entry {h!r} (want 2-digit 00-23)")
days = (e - s).days + 1
print(days * len(hours))
' "$START_DATE" "$END_DATE" "$INIT_HOURS")" || die "$CYCLE_INFO"
N_CYCLES="$CYCLE_INFO"

python3 - "$BBOX" <<'PY' || die "invalid --bbox (see above)"
import sys
parts = sys.argv[1].split(",")
if len(parts) != 4:
    sys.exit(f"--bbox must be 'N,W,S,E', got {sys.argv[1]!r}")
n, w, s, e = (float(p) for p in parts)
if not (s < n and w < e):
    sys.exit(f"--bbox 'N,W,S,E'={sys.argv[1]!r} is not a valid box (need S<N and W<E)")
PY

FIRST_HOUR="${INIT_HOURS%%,*}"
INIT_TIME="${START_DATE}T${FIRST_HOUR}"   # cosmetic only: run_on_ec2.sh needs *a* valid
                                           # init time for its notification/tagging; the
                                           # actual cycles run are the full START..END list.

# Wall-clock cap. Per-cycle timing for this exact combination (SoCal-sized crop on a
# g6e.2xlarge) has not been measured end-to-end; the closest measurements are 740s
# (12.3 min) for a SoCal crop cycle on a g5.2xlarge, and full-domain cycles on a
# g6e.2xlarge running faster than the same cycle on a g5.2xlarge (see
# experiment-2026-07-30-gpu-crop-validation/FINDINGS.md). Budgeting 20 min/cycle is a
# deliberately generous margin over that 12.3 min figure, not a measured number for
# this setup -- treat the estimate below as approximate.
if [ -z "$WALL_LIMIT" ]; then
    WALL_LIMIT=$(( N_CYCLES * 20 * 60 ))
fi
EST_MIN=$(( N_CYCLES * 13 ))   # rough, see note above
EST_HOURS_LOW=$(( EST_MIN / 60 ))

SUB_DESC="bbox ${BBOX} + ${HALO}-cell halo"

cat <<EOF

  hindcast
    cycles         ${N_CYCLES}  (${START_DATE}..${END_DATE} @ ${INIT_HOURS}Z)
    lead hours     ${LEAD_HOUR}  (output_hours: ${OUTPUT_HOURS:-every hour})
    domain         ${SUB_DESC}
    gfs min lag    ${GFS_MIN_LAG_OPT} h
    wall limit     ${WALL_LIMIT} s ($(( WALL_LIMIT / 3600 )) h, hard timeout on the instance)
    rough estimate ~${EST_HOURS_LOW}+ h wall clock -- UNVERIFIED for this exact
                   instance/crop combination; see the note in this script.
                   A single test cycle first (aws/run_on_ec2.sh --bbox ...) is the
                   way to get a real number before committing to all ${N_CYCLES}.

EOF

exec "${REPO_DIR}/aws/run_on_ec2.sh" \
    --bucket "$BUCKET" \
    --init-time "$INIT_TIME" \
    --lead-hours "$LEAD_HOUR" \
    --members 1 \
    --bbox "$BBOX" \
    --halo "$HALO" \
    --gfs-min-lag "$GFS_MIN_LAG_OPT" \
    --run-cmd "timeout --signal=TERM --kill-after=60 ${WALL_LIMIT}s ./aws/run_hindcast.sh '${START_DATE}' '${END_DATE}' '${INIT_HOURS}' '${LEAD_HOUR}' '${OUTPUT_HOURS}'" \
    ${PASSTHRU[@]+"${PASSTHRU[@]}"}
