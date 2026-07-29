#!/usr/bin/env bash
#
# aws/run_plots.sh — plot a forecast cycle from NetCDF held in S3, independently
# of the forecast run that produced it.
#
# Why separate: plotting is pure CPU work that produces thousands of PNGs (a 12h
# single-member cycle produced ~5,000 plus animations, most of the disk footprint
# of a run). Doing it inline would keep a GPU instance busy running matplotlib,
# and it forces the forecast to retain every NetCDF locally. Split out, the
# forecast streams NetCDF to S3 and deletes as it goes, and plots can be made
# later, on a cheap CPU box, for any cycle still in the bucket, as many times as
# you like.
#
# src/plot.py only needs <forecast_dir>/<YYYYMMDD>/<HH>/hrrrcast_<member>_f<HH>.nc,
# which is exactly the layout the uploader preserves, so this is a sync plus a
# call to the same plot stage run_cycle.sh uses.
#
# Usage:
#   aws/run_plots.sh --s3-input s3://BUCKET/hrrrcast/out --init-time 2026-07-20T00 \
#                    [--lead-hours 24] [--members 0] [--s3-output s3://BUCKET/hrrrcast/plots] \
#                    [--variables all|surface|pressure] \
#                    [--work-dir DIR] [--keep-netcdf] [--animate]
#
# --variables surface plots only the surface fields, skipping the pressure-level
# variables at 20 levels. That is roughly 120 of the ~170 figures per lead hour,
# so it is far quicker and much smaller.
#
# Runs anywhere the hrrrcast env exists: a CPU EC2 instance, or your laptop.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

S3_INPUT=""
S3_PLOTS=""
INIT_TIME=""
LEAD_HOUR=24
MEMBERS="0"
VARIABLES="all"
WORK_DIR=""
KEEP_NETCDF="NO"
ANIMATE="NO"

log() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --s3-input)    S3_INPUT="$2"; shift 2 ;;
        --s3-output)   S3_PLOTS="$2"; shift 2 ;;
        --init-time)   INIT_TIME="$2"; shift 2 ;;
        --lead-hours)  LEAD_HOUR="$2"; shift 2 ;;
        --members)     MEMBERS="$2"; shift 2 ;;
        --variables)   VARIABLES="$2"; shift 2 ;;
        --work-dir)    WORK_DIR="$2"; shift 2 ;;
        --keep-netcdf) KEEP_NETCDF="YES"; shift ;;
        --animate)     ANIMATE="YES"; shift ;;
        -h|--help)     sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)             die "Unknown option: $1 (try --help)" ;;
    esac
done

[ -n "$S3_INPUT" ]  || die "--s3-input is required (e.g. s3://my-bucket/hrrrcast/out)"
[ -n "$INIT_TIME" ] || die "--init-time is required (YYYY-MM-DDTHH)"
[[ "$INIT_TIME" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}$ ]] \
    || die "--init-time must be YYYY-MM-DDTHH (got '${INIT_TIME}')"
command -v aws >/dev/null 2>&1 || die "aws CLI not found."
# Validate before the sync: a typo here would otherwise surface only after pulling
# many GB from S3, which costs time and egress.
case "$VARIABLES" in
    all|surface|pressure) ;;
    *) die "--variables must be one of: all, surface, pressure (got '${VARIABLES}')" ;;
esac

DATE=${INIT_TIME%%T*}; DATE=${DATE//-/}
HOUR=${INIT_TIME#*T}
[ -n "$WORK_DIR" ] || WORK_DIR="$(mktemp -d -t hrrrcast-plots)"
export AWS_PAGER=""

log "Plotting ${INIT_TIME} f00-f${LEAD_HOUR}, members ${MEMBERS}"
echo "  source:  ${S3_INPUT}/${DATE}/${HOUR}/"
echo "  vars:    ${VARIABLES}"
echo "  workdir: ${WORK_DIR}"

# shellcheck disable=SC1091
source "${REPO_DIR}/etc/env_mac.sh"

# --- pull the NetCDF -------------------------------------------------------
mkdir -p "${WORK_DIR}/${DATE}/${HOUR}"
log "Syncing NetCDF from S3"
aws s3 sync "${S3_INPUT}/${DATE}/${HOUR}/" "${WORK_DIR}/${DATE}/${HOUR}/" \
    --exclude '*' --include 'hrrrcast_*.nc' \
    || die "s3 sync failed"

NFILES="$(find "${WORK_DIR}/${DATE}/${HOUR}" -name 'hrrrcast_*.nc' | wc -l | tr -d ' ')"
[ "$NFILES" -gt 0 ] || die "No hrrrcast_*.nc found under ${S3_INPUT}/${DATE}/${HOUR}/"
log "Synced ${NFILES} NetCDF files"

# A run that died partway leaves fewer hours than requested. Plot what is there
# rather than failing, but say so, so a short plot set is never mistaken for a
# complete one. Compare against one member's worth of hours: MEMBERS may be a
# range ("0-2"), so the exact expected count is not worth parsing here, and
# fewer files than a single member needs is unambiguously incomplete.
MIN_EXPECTED=$(( LEAD_HOUR + 1 ))
if [ "$NFILES" -lt "$MIN_EXPECTED" ]; then
    printf '\033[1;33mWARNING:\033[0m found %s NetCDF files but f00-f%s needs at least %s per member. The forecast may be incomplete; plotting the available hours only.\n' \
        "$NFILES" "$LEAD_HOUR" "$MIN_EXPECTED"
fi

mkdir -p "${WORK_DIR}/logs"

# --- plot ------------------------------------------------------------------
log "Running plot stage"
python3 "${REPO_DIR}/src/plot.py" "${INIT_TIME}" "${LEAD_HOUR}" \
    --members "${MEMBERS}" --variables "${VARIABLES}" \
    --forecast_dir "${WORK_DIR}" --output_dir "${WORK_DIR}" \
    2>&1 | tee "${WORK_DIR}/logs/plot.out"

if [ "$ANIMATE" == "YES" ]; then
    log "Building animations"
    python3 "${REPO_DIR}/src/make_animations.py" \
        2>&1 | tee "${WORK_DIR}/logs/make_animations.out" \
        || printf '\033[1;33mWARNING:\033[0m animation step failed; plots are still available.\n'
fi

# --- deliver ---------------------------------------------------------------
if [ -n "$S3_PLOTS" ]; then
    log "Uploading plots to ${S3_PLOTS}/${DATE}/${HOUR}/"
    aws s3 sync "${WORK_DIR}/${DATE}/${HOUR}/" "${S3_PLOTS}/${DATE}/${HOUR}/" \
        --exclude '*' --include '*.png' --include '*.gif' \
        || die "plot upload failed"
fi

if [ "$KEEP_NETCDF" == "NO" ]; then
    log "Removing synced NetCDF (already in S3); pass --keep-netcdf to retain"
    find "${WORK_DIR}/${DATE}/${HOUR}" -name 'hrrrcast_*.nc' -delete
fi

log "Plots complete under ${WORK_DIR}/${DATE}/${HOUR}/"
