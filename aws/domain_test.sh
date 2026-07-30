#!/usr/bin/env bash
#
# aws/domain_test.sh — full-domain vs subdomain inference, on ONE instance.
#
# Answers one question: does running HRRRCast on a cropped domain give a usable
# forecast, and how much cheaper is it?
#
# The motivation is cost. A full-domain 24 h forecast holds 43.9 GB of VRAM and takes
# 28.4 min on an L40S (g6e.2xlarge, $2.24/h). Activation memory scales with H*W, so a
# quarter-area subdomain should fit a 24 GB A10G (g5.2xlarge, about half the price)
# and run roughly 4x faster. If fidelity holds, hourly operation drops from ~$875 to
# well under $200/month. That is a bigger lever than spot.
#
# WHY ONE INSTANCE AND NOT TWO
#   Both forecasts must consume bit-identical inputs, otherwise the comparison
#   confounds "the model dislikes a smaller domain" with "the inputs differed".
#   So the input stages run ONCE at full domain, and src/crop_domain.py slices the
#   resulting npz. It also halves the cost and removes a second capacity gamble.
#
# WHAT IS BEING MEASURED
#   1. Fidelity. The model has 39 squeeze-excitation blocks that take a
#      GlobalAveragePooling2D over the whole field to gate channels, so cropping
#      changes the interior solution and not just the edges. Separately there is no
#      lateral boundary condition, so fine-scale inflow is lost. Both effects show up
#      as full-vs-cropped differences that should grow with lead time and shrink with
#      distance from the crop boundary. src/compare_domains.py quantifies that.
#   2. Runtime, per lead hour, from the fcst logs.
#   3. Peak VRAM, sampled during each forecast.
#
# THREE forecasts write to distinct S3 prefixes:
#   .../full/     full domain, member 0
#   .../sub/      subdomain,   member 0
#   .../full-m1/  full domain, member 1 -- the stochastic-noise yardstick, see below
#
# --purge_local is passed to both, so NetCDF is deleted after a confirmed upload. The
# full run alone is 10.1 GiB and the inputs another ~30 GiB, and cropping needs the
# full-domain npz to still be on disk, so retaining output as well would put the
# 200 GB root volume under real pressure for no benefit.
#
# Usage (on the instance; aws/run_domain_test.sh launches it):
#   aws/domain_test.sh INIT_TIME LEAD_HOURS S3_PREFIX [HEIGHT] [WIDTH] [BBOX] [HALO]
#
# BBOX ("N,W,S,E") overrides HEIGHT/WIDTH and places the crop on that region.
set -uo pipefail

INIT_TIME="${1:?INIT_TIME (YYYY-MM-DDTHH) required}"
LEAD_HOUR="${2:?LEAD_HOURS required}"
S3_PREFIX="${3:?S3 prefix required}"
SUB_H="${4:-531}"
SUB_W="${5:-903}"
# Optional: size and place the crop from a region of interest instead of H/W. Without
# it the box is centred on the CONUS grid, which would not test the region you care
# about. Format "N,W,S,E" in degrees.
SUB_BBOX="${6:-}"
SUB_HALO="${7:-40}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATAROOT="${DATAROOT:-$(pwd)}"
MODEL="${MODEL:-${REPO_DIR}/net-diffusion/model.keras}"
STATS="${STATS:-${REPO_DIR}/net-diffusion/normalize-stats.nc}"
NC_COMPLEVEL="${NC_COMPLEVEL:-1}"
NC_LSD="${NC_LSD:-2}"
GFS_MIN_LAG="${GFS_MIN_LAG:-4}"

DATE="${INIT_TIME%%T*}"; DATE="${DATE//-/}"
HOUR="${INIT_TIME#*T}"
STAMP="${DATE}_${HOUR}"
CYCLE_DIR="${DATAROOT}/${DATE}/${HOUR}"
SUB_DIR="${DATAROOT}/${DATE}/${HOUR}-sub"
mkdir -p "${DATAROOT}/logs"

log()  { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
die()  { printf '\n\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
secs() { date -u +%s; }

RESULTS="${DATAROOT}/logs/domain-test-results.txt"
: > "$RESULTS"
note() { printf '%s\n' "$*" | tee -a "$RESULTS"; }

if [ -n "$SUB_BBOX" ]; then
    note "domain test  init=${INIT_TIME}  lead=${LEAD_HOUR}h  bbox=${SUB_BBOX} halo=${SUB_HALO}"
else
    note "domain test  init=${INIT_TIME}  lead=${LEAD_HOUR}h  subdomain=${SUB_H}x${SUB_W}"
fi
note "gfs_min_lag=${GFS_MIN_LAG}  nc_complevel=${NC_COMPLEVEL}  nc_lsd=${NC_LSD}"
note "gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
note ""

# --- VRAM sampler ----------------------------------------------------------
# nvidia-smi peak is not retained across processes, so sample it. 2 s is plenty
# given a ~36 s per-lead-hour cadence and costs nothing.
#
# Sets VRAM_PID rather than echoing the pid, and redirects the subshell's stdout.
# The obvious version,
#     start_vram_sampler() { ( while :; do ...; done ) & echo $!; }
#     pid=$(start_vram_sampler "$f")
# DEADLOCKS, and did: the backgrounded infinite loop inherits the write end of the
# command-substitution pipe, so $(...) waits forever for an EOF that never arrives.
# The script hung before reaching the first forecast and burned 82 minutes of idle
# GPU time. Assigning to a global avoids the substitution entirely, and the
# >/dev/null closes the inherited descriptor so no future caller can reintroduce it.
VRAM_PID=""
start_vram_sampler() {  # $1 = output file; sets VRAM_PID
    ( while :; do
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$1" 2>/dev/null
        sleep 2
      done ) >/dev/null 2>&1 &
    VRAM_PID=$!
}
peak_vram() { sort -n "$1" 2>/dev/null | tail -1; }

# --- stage 1: inputs, ONCE, at full domain --------------------------------
log "Input stages (full domain, shared by both forecasts)"
T0=$(secs)
python3 "${REPO_DIR}/src/get_ics.py" "${INIT_TIME}" --base_dir "${DATAROOT}" \
    > "${DATAROOT}/logs/get-ics.out" 2>&1 || die "get_ics failed"
python3 "${REPO_DIR}/src/get_bcs.py" "${INIT_TIME}" "${LEAD_HOUR}" --base_dir "${DATAROOT}" \
    --gfs_min_lag_hours "${GFS_MIN_LAG}" \
    > "${DATAROOT}/logs/get-bcs.out" 2>&1 || die "get_bcs failed"
python3 "${REPO_DIR}/src/make_ics.py" "${STATS}" "${INIT_TIME}" \
    --base_dir "${DATAROOT}" --output_dir "${DATAROOT}" \
    > "${DATAROOT}/logs/make-ics.out" 2>&1 || die "make_ics failed"
python3 "${REPO_DIR}/src/make_bcs.py" "${STATS}" "${INIT_TIME}" "${LEAD_HOUR}" \
    --base_dir "${DATAROOT}" --output_dir "${DATAROOT}" \
    --hrrr_grid_file "${DATE}/${HOUR}/hrrr_${DATE}_${HOUR}_surface.grib2" \
    > "${DATAROOT}/logs/make-bcs.out" 2>&1 || die "make_bcs failed"
T_INPUT=$(( $(secs) - T0 ))
note "inputs           ${T_INPUT} s  ($((T_INPUT/60)) min)"
grep -E "GFS cycle:" "${DATAROOT}/logs/get-bcs.out" | tail -1 | tee -a "$RESULTS"

# --- stage 2: crop --------------------------------------------------------
if [ -n "$SUB_BBOX" ]; then
    log "Cropping to contain ${SUB_BBOX} with a ${SUB_HALO}-cell halo"
    CROP_ARGS=(--bbox "$SUB_BBOX" --halo "$SUB_HALO")
else
    log "Cropping inputs to ${SUB_H}x${SUB_W} (centred on the grid)"
    CROP_ARGS=(--height "$SUB_H" --width "$SUB_W")
fi
python3 "${REPO_DIR}/src/crop_domain.py" --in-dir "$CYCLE_DIR" --out-dir "$SUB_DIR" \
    --init-time "$INIT_TIME" "${CROP_ARGS[@]}" \
    2>&1 | tee -a "$RESULTS" || die "crop_domain failed"

# The cropped run needs the same surface GRIB present for any grid metadata lookups,
# and fcst.py reads its inputs from <base_dir>/<DATE>/<HH>/. Rather than teach fcst.py
# about a suffixed directory, run the cropped forecast in a separate DATAROOT whose
# cycle directory holds the cropped npz.
SUB_ROOT="${DATAROOT}/subrun"
mkdir -p "${SUB_ROOT}/${DATE}/${HOUR}" "${SUB_ROOT}/logs"
cp "${SUB_DIR}/hrrr_${STAMP}.npz" "${SUB_DIR}/gfs_${STAMP}.npz" "${SUB_ROOT}/${DATE}/${HOUR}/"
cp "${SUB_DIR}/subdomain.json" "${SUB_ROOT}/" 2>/dev/null || true

# --- forecasts ------------------------------------------------------------
# --num_members 2 for every run, not 1. Phase shifts are indexed by member id, so
# `--num_members 1 --members 1` is now a clear ValueError (member 1 does not exist in
# a 1-member ensemble); before the fix it raised KeyError(1), logged as the baffling
# "Forecast failed: 1". With num_members=2 the phase sequence is [0.0, 0.0], so
# members 0 and 1 get the SAME GFS phase and differ only in their noise seed, which
# is exactly what the yardstick needs. Member 0's phase is 0.0 either way, so this
# does not change the full/sub runs.
run_forecast() {   # $1 = label, $2 = base dir, $3 = s3 subprefix, $4 = member id
    local label="$1" root="$2" sub="$3" member="${4:-0}"
    local vram="${DATAROOT}/logs/vram-${label}.txt" t0 t1
    : > "$vram"
    start_vram_sampler "$vram"
    log "Forecast: ${label} (member ${member})"
    t0=$(secs)
    # NC_LSD guarded: empty means lossless, and fcst.py's argparse (type=int)
    # rejects an empty string outright rather than treating it as "off".
    local lsd_flag=()
    [ -n "$NC_LSD" ] && lsd_flag=(--nc_least_significant_digit "$NC_LSD")
    python3 "${REPO_DIR}/src/fcst.py" "$MODEL" "$INIT_TIME" "$LEAD_HOUR" \
        --num_members 2 --members "$member" --batch_size 1 --log_level INFO \
        --nc_complevel "$NC_COMPLEVEL" "${lsd_flag[@]}" \
        --s3_output "${S3_PREFIX}/${sub}" --purge_local \
        --base_dir "$root" --output_dir "$root" \
        > "${DATAROOT}/logs/fcst-${label}.out" 2>&1
    local rc=$?
    t1=$(secs)
    kill "$VRAM_PID" 2>/dev/null
    local elapsed=$(( t1 - t0 ))
    local pk; pk=$(peak_vram "$vram")
    if [ $rc -ne 0 ]; then
        note "${label}: FAILED after ${elapsed} s (rc=${rc}), peak VRAM ${pk} MiB"
        note "  tail of ${DATAROOT}/logs/fcst-${label}.out:"
        tail -25 "${DATAROOT}/logs/fcst-${label}.out" | sed 's/^/    /' | tee -a "$RESULTS"
        return 1
    fi
    note "${label}: ${elapsed} s ($((elapsed/60)) min), "\
"$(python3 -c "print(f'{${elapsed}/${LEAD_HOUR}:.1f}')") s per lead hour, peak VRAM ${pk} MiB"
    return 0
}

# THREE runs, not two, and the third is the point.
#
# The diffusion sampler uses tf.random.stateless_normal(shape, seed=[member, hour]).
# That is deterministic for a given member and hour, BUT the draw depends on the
# tensor SHAPE: a 531x903 noise field is not the corresponding window of a
# 1059x1799 one. So the cropped run necessarily gets a different noise realization,
# and a plain full-vs-sub difference measures the crop effect PLUS a different
# random draw, with no way to separate them.
#
# The fix is a yardstick. Run the full domain twice with different member IDs; their
# difference is the model's own stochastic spread on identical inputs. The decision
# then rests on a comparison that is actually meaningful:
#
#     is  |sub_m0 - full_m0|  smaller than  |full_m1 - full_m0|  ?
#
# If yes, cropping costs no more than swapping ensemble members, which is well
# inside the uncertainty the product already carries. If the crop error is several
# times the spread, it is a real degradation.
#
# Cost of the third run is ~28 min / ~$1.05. Without it the experiment cannot
# answer the question it was built for, so it is not optional.
FULL_OK=0; SUB_OK=0; REF_OK=0
run_forecast full "$DATAROOT" full 0 && FULL_OK=1
# Run the cropped case even if the full one failed: it is the cheaper and more
# interesting result, and a full-domain failure here would be a regression worth
# seeing separately rather than a reason to abandon the experiment.
run_forecast sub "$SUB_ROOT" sub 0 && SUB_OK=1
# The noise yardstick. Skipped if the full-domain run failed, since there would be
# nothing to compare it against.
if [ "$FULL_OK" -eq 1 ]; then
    run_forecast full-m1 "$DATAROOT" full-m1 1 && REF_OK=1
fi

# --- summary --------------------------------------------------------------
note ""
note "full(m0) ok: ${FULL_OK}   sub(m0) ok: ${SUB_OK}   full(m1) noise ref ok: ${REF_OK}"
note "outputs: ${S3_PREFIX}/{full,sub,full-m1}/"
aws s3 cp "$RESULTS" "${S3_PREFIX}/results.txt" --quiet || true
aws s3 cp "${SUB_DIR}/subdomain.json" "${S3_PREFIX}/subdomain.json" --quiet || true
for l in full sub full-m1; do
    aws s3 cp "${DATAROOT}/logs/fcst-${l}.out" "${S3_PREFIX}/fcst-${l}.log" --quiet || true
done
log "Domain test complete"
cat "$RESULTS"
[ "$FULL_OK" -eq 1 ] && [ "$SUB_OK" -eq 1 ] && [ "$REF_OK" -eq 1 ]
