#!/usr/bin/env bash
#
# aws/run_hindcast.sh — run many HRRRCast cycles back to back, on one instance.
#
# One-off on-demand runs launch a fresh instance per cycle (aws/run_on_ec2.sh);
# a hindcast over weeks/months of history does not want that (per-instance
# bootstrap and AMI/env setup would dominate the bill). This instead loops
# run_cycle.sh over every requested init time on the SAME already-provisioned
# instance, one cycle after another, and keeps going past a failed cycle
# instead of stopping the whole hindcast for it.
#
# Meant to be driven through aws/launch_hindcast.sh, which supplies the
# instance (aws/run_on_ec2.sh --run-cmd) and therefore the environment this
# script reads as overrides: DATAROOT, S3_OUTPUT, SUB_BBOX, SUB_HALO,
# GFS_MIN_LAG, NC_COMPLEVEL, NC_LSD, PURGE_LOCAL, MAKE_BCS_WORKERS (see
# aws/user_data.sh). Every one of those has the same default here as
# run_cycle.sh uses on its own, so this also runs sensibly by hand.
#
# Usage:
#   aws/run_hindcast.sh START_DATE END_DATE INIT_HOURS LEAD_HOURS [OUTPUT_HOURS]
#
#   START_DATE, END_DATE   YYYY-MM-DD, inclusive
#   INIT_HOURS             comma list of 2-digit UTC hours, e.g. "00,06,12,18"
#   LEAD_HOURS             forecast length per cycle
#   OUTPUT_HOURS           optional --output_hours spec passed to fcst.py
#                          ("start:step:end" or a comma list); default: every
#                          hour written, same as a plain run_cycle.sh call.
#
# Example (SoCal, all of June 2026, 6-hourly cycles, 24h lead, output every 3h):
#   SUB_BBOX="35.0,-118.77,33.25,-117.0" SUB_HALO=80 GFS_MIN_LAG=0 \
#     ./aws/run_hindcast.sh 2026-06-01 2026-06-30 00,06,12,18 24 0:3:24
#
# A cycle that fails (missing archive file, transient stage error) is logged
# and skipped; the hindcast continues to the next cycle rather than aborting.
# Per-cycle input/intermediate files are deleted after each cycle regardless of
# outcome -- PURGE_LOCAL only removes fcst.py's own NetCDF after upload, not
# the get_ics/get_bcs/make_ics/make_bcs/crop working files, which would
# otherwise accumulate across the whole run and fill the disk.
set -uo pipefail

START_DATE="${1:?START_DATE (YYYY-MM-DD) required}"
END_DATE="${2:?END_DATE (YYYY-MM-DD) required}"
INIT_HOURS="${3:?INIT_HOURS required, e.g. 00,06,12,18}"
LEAD_HOUR="${4:?LEAD_HOURS required}"
OUTPUT_HOURS="${5:-}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATAROOT="${DATAROOT:-$(pwd)}"
export OUTPUT_HOURS

mkdir -p "${DATAROOT}/logs"
SUMMARY="${DATAROOT}/logs/hindcast-summary.tsv"
: > "$SUMMARY"
printf 'init_time\tstatus\telapsed_s\n' >> "$SUMMARY"

log()  { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mWARNING:\033[0m %s\n' "$*" >&2; }
die()  { printf '\n\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# --- validate date range and build the cycle list ---------------------------
IFS=',' read -ra HOUR_LIST <<< "$INIT_HOURS"
for h in "${HOUR_LIST[@]}"; do
    [[ "$h" =~ ^[0-9]{2}$ ]] && [ "$h" -ge 0 ] && [ "$h" -le 23 ] \
        || die "invalid init hour '${h}' in --init-hours (want 2-digit 00-23)"
done

# python3 date arithmetic: available everywhere this runs (the instance and any
# workstation), unlike GNU vs BSD `date -d`/`-v` relative-time syntax.
CYCLE_DATES="$(python3 -c '
import sys, datetime as d
start, end = sys.argv[1], sys.argv[2]
s = d.datetime.strptime(start, "%Y-%m-%d").date()
e = d.datetime.strptime(end, "%Y-%m-%d").date()
if e < s:
    sys.exit("END_DATE is before START_DATE")
cur = s
out = []
while cur <= e:
    out.append(cur.strftime("%Y-%m-%d"))
    cur += d.timedelta(days=1)
print(" ".join(out))
' "$START_DATE" "$END_DATE")" || die "$CYCLE_DATES"

CYCLES=()
for day in $CYCLE_DATES; do
    for h in "${HOUR_LIST[@]}"; do
        CYCLES+=("${day}T${h}")
    done
done
TOTAL=${#CYCLES[@]}
[ "$TOTAL" -gt 0 ] || die "cycle list is empty"

log "Hindcast: ${TOTAL} cycles, ${START_DATE}..${END_DATE} @ ${INIT_HOURS}Z, lead=${LEAD_HOUR}h, output_hours=${OUTPUT_HOURS:-all}"
echo "  DATAROOT=${DATAROOT}"
echo "  SUB_BBOX=${SUB_BBOX:-<full domain>}  SUB_HALO=${SUB_HALO:-40}"
echo "  S3_OUTPUT=${S3_OUTPUT:-<none>}  GFS_MIN_LAG=${GFS_MIN_LAG:-0}"

N_OK=0
N_FAIL=0
i=0
HINDCAST_T0=$(date +%s)
for INIT_TIME in "${CYCLES[@]}"; do
    i=$((i + 1))
    DATE="${INIT_TIME%%T*}"; DATE="${DATE//-/}"
    HOUR="${INIT_TIME#*T}"
    CYCLE_LOG="${DATAROOT}/logs/cycle-${DATE}_${HOUR}.out"

    log "[${i}/${TOTAL}] cycle ${INIT_TIME}Z"
    T0=$(date +%s)
    if "${REPO_DIR}/run_cycle.sh" "$INIT_TIME" "$LEAD_HOUR" 1 1 "$REPO_DIR" "$DATAROOT" NO "" \
            > "$CYCLE_LOG" 2>&1; then
        RC=0
        N_OK=$((N_OK + 1))
    else
        RC=$?
        N_FAIL=$((N_FAIL + 1))
        warn "cycle ${INIT_TIME}Z failed (rc=${RC}); see logs/$(basename "$CYCLE_LOG"). Continuing."
    fi
    ELAPSED=$(( $(date +%s) - T0 ))
    printf '%s\t%s\t%ds\n' "$INIT_TIME" "$([ "$RC" -eq 0 ] && echo ok || echo "failed(rc=${RC})")" "$ELAPSED" >> "$SUMMARY"

    # Per-cycle cleanup, independent of outcome. PURGE_LOCAL (if set) only deletes
    # fcst.py's own NetCDF after S3 upload; everything else run_cycle.sh writes
    # under DATAROOT for this cycle is scratch that the next cycle does not need.
    rm -rf "${DATAROOT:?}/${DATE}/${HOUR}" "${DATAROOT}/${DATE}/${HOUR}-sub" \
           "${DATAROOT}/subrun/${DATE}"
done

TOTAL_ELAPSED_MIN=$(( ( $(date +%s) - HINDCAST_T0 ) / 60 ))
log "Hindcast complete: ${N_OK} ok, ${N_FAIL} failed, ${TOTAL} total, ${TOTAL_ELAPSED_MIN} min wall clock."
echo "Summary: ${SUMMARY}"

# A hindcast where nothing at all succeeded is a systemic problem (bad bbox,
# broken credentials, wrong dates) worth surfacing as a failed run; a handful of
# missing archive files among 100+ cycles is not.
[ "$N_OK" -gt 0 ] || die "every cycle failed; see ${SUMMARY} and the per-cycle logs"
exit 0
