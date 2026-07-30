#!/usr/bin/env bash
#
# aws/pick_cycle.sh — decide which cycle an unattended hourly run should produce,
# and refuse to launch when the inputs are not actually on S3 yet.
#
# Emits shell-sourceable assignments on stdout and a human-readable trace on
# stderr, so a scheduler can do:
#
#     eval "$(aws/pick_cycle.sh)" || exit 0        # exit 0 = nothing to do
#     aws/run_on_ec2.sh --init-time "$INIT_TIME" --gfs-min-lag "$GFS_MIN_LAG" ...
#
# Exit codes are the interface:
#   0  a cycle is ready; INIT_TIME / GFS_MIN_LAG / GFS_CYCLE printed on stdout
#   1  hard error (bad arguments, no network, unreadable bucket)
#   3  nothing to do: no cycle has complete inputs yet, or the newest one is
#      already present in S3. This is the normal quiet case, not a failure.
#
# WHY THIS EXISTS. A g6e.2xlarge costs about $2.24/h. Launching one and letting it
# discover a missing GFS file wastes roughly a dollar and, worse, produces a
# truncated forecast rather than a clean failure -- get_bcs logs the failed
# download and exits 2, but a partially populated input directory can still carry
# make_bcs some distance. Two HEAD requests against public buckets cost nothing
# and take under a second, so availability is checked before any instance starts.
#
# TIMING, measured 2026-07-29 against the live buckets (last-modified headers):
#   HRRR analysis  hour H         published at about H+0:51, tight spread
#   GFS cycle CC   f000-f036      complete at about CC+4:00 (f003 at +3:36,
#                                 f024 at +3:58; the block lands together rather
#                                 than progressively)
# The consequence is that the newest GFS cycle is NOT usable when the HRRR
# analysis appears, which is what GFS_MIN_LAG exists to solve. See
# src/get_bcs.py gfs_cycle_for() for the full argument.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# HRRR_BASE_URL / GFS_BASE_URL are overridable only so the cycle-search fallback
# can be exercised: every GFS cycle in the past is complete, so the only way to
# reach the "step back a cycle" path is to point a probe somewhere the files are
# not. Leave both unset in production. GFS URLs come from get_bcs itself (see
# gfs_manifest), so that override is applied there rather than here.
HRRR_BASE="${HRRR_BASE_URL:-https://noaa-hrrr-bdp-pds.s3.amazonaws.com}"

LEAD_HOUR=24
GFS_MIN_LAG=4          # 4 is what makes every hour launchable; see above
S3_OUTPUT=""           # when set, an already-produced cycle is skipped
MAX_LOOKBACK=6         # how many hours back to consider before giving up
GFS_MAX_EXTRA_CYCLES=2 # Extra 6 h steps back through GFS cycles when one is
                       # incomplete, so a late or partial GFS feed degrades instead
                       # of dropping the hour. This is a fallback, NOT a free
                       # option: at lag 4 the base cycle uses f005-f033, one step
                       # back uses f011-f039 and two back f017-f045, against a
                       # trained range of f001-f029. Set to 0 if you would rather
                       # skip an hour than publish a forecast forced by GFS steps
                       # the network has never seen.
NOW_OVERRIDE=""        # testing hook: pretend "now" is this UTC YYYY-MM-DDTHH:MM
FORCE=NO

usage() {
    sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^#\{0,1\} \{0,1\}//'
    cat <<'EOF'

Options:
  --lead-hours N        Forecast length; sets the last GFS step needed (default 24)
  --gfs-min-lag N       Cycle shift, 0-5 (default 4)
  --s3-output URI       s3://bucket/prefix; skip cycles already present there
  --max-lookback N      Hours to walk back looking for a ready cycle (default 6)
  --gfs-max-extra-cycles N  Extra 6 h steps back through GFS cycles (default 2)
  --now YYYY-MM-DDTHH:MM  Treat this UTC time as now (testing)
  --force               Report the cycle even if inputs are incomplete
  -h, --help            This text
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --lead-hours)   LEAD_HOUR="$2"; shift 2 ;;
        --gfs-min-lag)  GFS_MIN_LAG="$2"; shift 2 ;;
        --s3-output)    S3_OUTPUT="$2"; shift 2 ;;
        --max-lookback) MAX_LOOKBACK="$2"; shift 2 ;;
        --gfs-max-extra-cycles) GFS_MAX_EXTRA_CYCLES="$2"; shift 2 ;;
        --now)          NOW_OVERRIDE="$2"; shift 2 ;;
        --force)        FORCE=YES; shift ;;
        -h|--help)      usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 1 ;;
    esac
done

say() { printf '%s\n' "$*" >&2; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

case "$GFS_MIN_LAG" in ''|*[!0-9]*) die "--gfs-min-lag must be an integer 0-5" ;; esac
# 0-5 is the range that selects a cycle without skipping one. Skipping whole cycles
# is what --gfs-max-extra-cycles does, applied on top of this base.
[ "$GFS_MIN_LAG" -le 5 ] || die "--gfs-min-lag must be 0-5; use --gfs-max-extra-cycles to skip cycles"
case "$GFS_MAX_EXTRA_CYCLES" in ''|*[!0-9]*) die "--gfs-max-extra-cycles must be a non-negative integer" ;; esac
# get_bcs refuses a lag above 23, so cap the search rather than emit a value it
# would reject.
[ $(( GFS_MIN_LAG + 6 * GFS_MAX_EXTRA_CYCLES )) -le 23 ] \
    || die "--gfs-min-lag + 6*--gfs-max-extra-cycles must be <= 23 (get_bcs rejects larger lags)"

# --- date arithmetic -------------------------------------------------------
# GNU date and BSD date disagree on relative-time syntax, and this script has to
# run both on the Mac (interactive testing) and on Amazon Linux / Ubuntu (the
# scheduler). Python is present in every one of those places.
udate() {  # udate <epoch> <strftime>
    python3 -c 'import sys,datetime as d
print(d.datetime.fromtimestamp(int(sys.argv[1]),d.timezone.utc).strftime(sys.argv[2]))' "$1" "$2"
}

if [ -n "$NOW_OVERRIDE" ]; then
    NOW_EPOCH=$(python3 -c 'import sys,datetime as d
print(int(d.datetime.strptime(sys.argv[1],"%Y-%m-%dT%H:%M").replace(tzinfo=d.timezone.utc).timestamp()))' \
        "$NOW_OVERRIDE") || die "--now must be UTC YYYY-MM-DDTHH:MM"
else
    NOW_EPOCH=$(date -u +%s)
fi
say "now (UTC)          $(udate "$NOW_EPOCH" '%Y-%m-%dT%H:%M')Z"
say "lead hours         ${LEAD_HOUR}"
say "gfs min lag        ${GFS_MIN_LAG} h"

# --- availability probes ---------------------------------------------------
# A ranged GET of one byte, not a HEAD. Both are cheap, but some S3 fronting
# layers answer HEAD from a cache that can be stale by minutes, and a cycle that
# looks present but is still being written is exactly the failure this is meant
# to prevent. 200 and 206 both mean the object is readable.
exists() {
    local code
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -r 0-0 "$1")
    case "$code" in 200|206) return 0 ;; *) return 1 ;; esac
}

# Ask get_bcs itself which GFS files a run would need, rather than reconstructing
# the range here. An earlier version guessed the last step as GSTART+LEAD+6, which
# over-probed by up to 6 steps and could reject a cycle over a file the run would
# never open. get_gfs_urls() is the authority, including the extra
# next-synoptic-hour file it appends at the final lead hour.
#
# Prints "CYCLE <yyyymmdd> <hh> <age_hours>" then one URL per line.
gfs_manifest() {  # $1 = INIT_TIME (YYYY-MM-DDTHH), $2 = lag
    PYTHONPATH="${REPO_DIR}/src" python3 -c '
import os, sys, datetime as d
from get_bcs import get_gfs_urls, gfs_cycle_for, Config
Config.GFS_BASE_URL = os.environ.get("GFS_BASE_URL", Config.GFS_BASE_URL)
t, lag, lead = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
run = d.datetime.strptime(t, "%Y-%m-%dT%H")
c = gfs_cycle_for(run, lag)
print("CYCLE", c.strftime("%Y%m%d"), c.strftime("%H"),
      int((run - c).total_seconds() // 3600))
for url, _ in get_gfs_urls(t[0:4], t[5:7], t[8:10], t[11:13], lead, lag):
    print(url)' "$1" "$2" "$LEAD_HOUR"
}

# --- candidate loop --------------------------------------------------------
# Two nested searches, and the nesting order is the point:
#
#   outer: the HRRR hour, freshest first. This is the initial condition and the
#          thing whose freshness the product is judged on, so it moves last.
#   inner: the GFS cycle, newest usable first, stepping back a whole 6 h at a
#          time.
#
# An earlier version had no inner loop: it computed one GFS cycle arithmetically
# and, if that cycle was incomplete, stepped the HRRR hour back instead. Because
# four of every six consecutive hours map to the same GFS cycle, that rejected
# candidate after candidate on the identical missing file, threw away up to three
# hours of initial-condition freshness, and only recovered when the hour happened
# to cross a 6 h boundary. Holding the HRRR hour and stepping the cycle is both
# faster and gives up the right thing.
CHOSEN=""
for (( back=1; back<=MAX_LOOKBACK; back++ )); do
    H_EPOCH=$(( (NOW_EPOCH / 3600) * 3600 - back * 3600 ))
    IDATE=$(udate "$H_EPOCH" '%Y%m%d')
    IHOUR=$(udate "$H_EPOCH" '%H')
    INIT_TIME="${IDATE:0:4}-${IDATE:4:2}-${IDATE:6:2}T${IHOUR}"

    say ""
    say "candidate -${back}h      ${INIT_TIME}Z"

    # 1. HRRR analysis for this hour.
    if ! exists "${HRRR_BASE}/hrrr.${IDATE}/conus/hrrr.t${IHOUR}z.wrfprsf00.grib2"; then
        say "  hrrr prs f00      MISSING"; continue
    fi
    if ! exists "${HRRR_BASE}/hrrr.${IDATE}/conus/hrrr.t${IHOUR}z.wrfsfcf00.grib2"; then
        say "  hrrr sfc f00      MISSING"; continue
    fi
    say "  hrrr f00          ok"

    # 2. Previous hour's f01 files, which supply APCP and VVEL (the analysis has
    #    neither in usable form). A missing f01 does not fail the download stage
    #    loudly enough -- make_ics logs a warning and carries on with whatever it
    #    has -- so it is checked here where it can still stop the run.
    P_EPOCH=$(( H_EPOCH - 3600 ))
    PDATE=$(udate "$P_EPOCH" '%Y%m%d'); PHOUR=$(udate "$P_EPOCH" '%H')
    if ! exists "${HRRR_BASE}/hrrr.${PDATE}/conus/hrrr.t${PHOUR}z.wrfsfcf01.grib2" \
       || ! exists "${HRRR_BASE}/hrrr.${PDATE}/conus/hrrr.t${PHOUR}z.wrfprsf01.grib2"; then
        say "  hrrr f01 (${PHOUR}Z)    MISSING (APCP/VVEL source)"; continue
    fi
    say "  hrrr f01 (${PHOUR}Z)    ok"

    # 3. Idempotency, before the GFS probing because it is one API call against
    #    the run's own bucket and settles the question cheaply. If this cycle
    #    already has output, the scheduler has already done the work (a duplicate
    #    tick, or a manual run). Re-running would double the bill for an
    #    identical product.
    if [ -n "$S3_OUTPUT" ]; then
        PREFIX="${S3_OUTPUT%/}/${IDATE}/${IHOUR}/"
        N=$(aws s3 ls "$PREFIX" 2>/dev/null | grep -c '\.nc$')
        if [ "${N:-0}" -gt 0 ]; then
            say "  s3 output         ALREADY PRESENT (${N} .nc under ${PREFIX})"
            [ "$FORCE" = "YES" ] || continue
            say "                    (--force: would overwrite)"
        else
            say "  s3 output         absent, nothing produced yet"
        fi
    fi

    # 4. GFS, newest usable cycle first. EVERY file the run will open is probed,
    #    not just the last one. The previous version probed only the final step on
    #    the grounds that the f000-f036 block publishes together (measured: f003 at
    #    +3:36, f024 at +3:58), which is true on a good day and useless on the day
    #    this check is meant to catch: a hole in the middle of the block would have
    #    passed. 25 ranged GETs against a public bucket cost nothing.
    for (( k=0; k<=GFS_MAX_EXTRA_CYCLES; k++ )); do
        LAG=$(( GFS_MIN_LAG + 6 * k ))
        MANIFEST=$(gfs_manifest "$INIT_TIME" "$LAG") \
            || die "could not build the GFS manifest from src/get_bcs.py"
        read -r _ GDATE GHOUR GSTART <<<"$(printf '%s\n' "$MANIFEST" | head -1)"
        URLS=$(printf '%s\n' "$MANIFEST" | tail -n +2)
        NEED=$(printf '%s\n' "$URLS" | grep -c .)

        MISSING=0; FIRST_MISSING=""
        while IFS= read -r u; do
            [ -n "$u" ] || continue
            if ! exists "$u"; then
                MISSING=$(( MISSING + 1 ))
                [ -n "$FIRST_MISSING" ] || FIRST_MISSING="$u"
            fi
        done <<<"$URLS"

        LABEL="GFS ${GDATE} ${GHOUR}Z (${GSTART}h old, lag=${LAG})"
        if [ "$MISSING" -eq 0 ]; then
            say "  ${LABEL}: all ${NEED} files present"
            CHOSEN="$INIT_TIME"; CHOSEN_GFS="${GDATE}${GHOUR}"
            CHOSEN_AGE="$GSTART"; CHOSEN_LAG="$LAG"; CHOSEN_NEED="$NEED"
            break
        fi
        say "  ${LABEL}: ${MISSING}/${NEED} files MISSING"
        say "                    first: ${FIRST_MISSING##*/}"
        if [ "$FORCE" = "YES" ]; then
            say "                    (--force: proceeding anyway)"
            CHOSEN="$INIT_TIME"; CHOSEN_GFS="${GDATE}${GHOUR}"
            CHOSEN_AGE="$GSTART"; CHOSEN_LAG="$LAG"; CHOSEN_NEED="$NEED"
            break
        fi
    done

    [ -n "$CHOSEN" ] && break
done

if [ -z "$CHOSEN" ]; then
    say ""
    say "No cycle with complete inputs in the last ${MAX_LOOKBACK} h. Nothing to do."
    exit 3
fi

say ""
say "SELECTED           ${CHOSEN}Z  (GFS ${CHOSEN_GFS}Z, ${CHOSEN_AGE}h old, ${CHOSEN_NEED} files verified)"

# GFS_MIN_LAG is the lag that was actually settled on, which is the base lag plus
# 6 h for each cycle the search had to skip. Emitting the base value here instead
# would silently send the run at a different cycle than the one just verified.
printf 'INIT_TIME=%s\n'   "$CHOSEN"
printf 'GFS_MIN_LAG=%s\n' "$CHOSEN_LAG"
printf 'GFS_CYCLE=%s\n'   "$CHOSEN_GFS"
printf 'LEAD_HOUR=%s\n'   "$LEAD_HOUR"
