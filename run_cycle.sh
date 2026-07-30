#!/usr/bin/env bash
#
# run_cycle.sh — run one HRRRCast forecast cycle locally on this machine.
#
# Local (non-SLURM) counterpart to submit_all.sh. It runs the same pipeline
# stages, in the same dependency order, with the same src/*.py commands the
# jobs/job-*.sh templates use, but sequentially in one process instead of as
# sbatch jobs with afterok dependencies (there is no scheduler or GPU array on
# a workstation). Stages are fatal: the cycle stops on the first failure,
# mirroring submit_all.sh's submit_with_check.
#
# Usage (same positional interface as submit_all.sh):
#   ./run_cycle.sh <INIT_TIME> <LEAD_HOUR> <N_ENSEMBLES> <N_GPUS> <PACKAGEROOT> <DATAROOT> <RUNPLOT> <ENVMODE>
#
#   INIT_TIME    init time, YYYY-MM-DDTHH (e.g. 2024-05-06T23)   [default 2024-07-17T23]
#   LEAD_HOUR    number of forecast hours                        [default 18]
#   N_ENSEMBLES  ensemble members                                [default 1]
#   N_GPUS       accepted for interface parity; ignored (all members run in one
#                process here — there is no GPU job array locally)  [default 1]
#   PACKAGEROOT  repo root                                        [default: script dir]
#   DATAROOT     working/output data root                         [default: $PWD]
#   RUNPLOT      YES|NO, run plotting stage                        [default YES]
#   ENVMODE      OPN uses etc/env_emc.sh; otherwise etc/env_mac.sh [default local]
#
# Forecast (src/fcst.py) options are overridable via environment variables
# (defaults match fcst.py):
#   BATCH_SIZE=1  LOG_LEVEL=INFO  PMM_ALPHA=0.7  NOISE_RHO=0.9
#   NO_DIFFUSION=NO   (YES -> --no_diffusion, deterministic model)
#
# Output/delivery options (defaults keep the original local behavior):
#   NO_GRIB2=YES      GRIB2 is OFF by default; set NO_GRIB2=NO to also write GRIB2
#                     (adds ~4 GB/cycle and needs wgrib2)
#   NC_COMPLEVEL=0    (1-9 enables NetCDF zlib compression; 0 = uncompressed.
#                      Measured: level 1 gives 1.54x on these fields, and higher
#                      levels add almost nothing, so 1 is the useful setting.)
#   NC_LSD=           (LOSSY quantization to N decimal digits before compression.
#                      Measured: N=2 gives 3.8x at max abs error 0.004 in native
#                      units, N=3 gives 3.0x at 0.0005. Empty = off.)
#   OUTPUT_HOURS=     (Only build/write/upload these lead hours; the rollout still
#                      computes every hour in between (autoregressive state needs
#                      it), only the I/O is skipped. "start:step:end", e.g.
#                      "0:3:24" for f00,f03,...,f24, or a comma list. Empty = every
#                      hour, unchanged from before this option existed.)
#   S3_OUTPUT=        (s3://bucket/prefix; uploads each file as it is written)
#   PURGE_LOCAL=NO    (YES -> delete local copies after upload; needs S3_OUTPUT.
#                      Incompatible with RUNPLOT=YES in the same cycle: plotting
#                      reads the NetCDF back. Run plots as a separate job.)
#   OVERLAP_FCST=NO   (YES -> start the forecast before the input stages so its
#                      TensorFlow import and model load, ~5 min and independent of
#                      the input data, run while the inputs are being prepared. The
#                      forecast blocks on a sentinel until they are ready. Saves
#                      roughly the whole of that startup; the GPU is otherwise idle
#                      for the entire input phase.)
#
# make_bcs regridding worker cap (avoids OOM on small-RAM hosts):
#   MAKE_BCS_WORKERS  (unset -> one worker per lead hour; set 1 or 2 on 16 GiB)
#
# Example: 6-hour, single-member cycle
#   ./run_cycle.sh 2024-05-06T23 6 1
# Example: deterministic, 3-member, DEBUG logging
#   NO_DIFFUSION=YES LOG_LEVEL=DEBUG ./run_cycle.sh 2024-05-06T23 6 3
#
set -euo pipefail

INIT_TIME=${1:-"2024-07-17T23"}
LEAD_HOUR=${2:-18}
N_ENSEMBLES=${3:-1}
N_GPUS=${4:-1}                       # interface parity only; unused locally
PACKAGEROOT=${5:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
DATAROOT=${6:-"$(pwd)"}
RUNPLOT=${7:-"YES"}
ENVMODE=${8:-""}

# Runtime knobs (same defaults as submit_all.sh).
export NETCDF2GRIB_SECTION3=${NETCDF2GRIB_SECTION3:-}
export WGRIB2=${WGRIB2:-wgrib2}      # provided by the conda env on this machine
export PMM_POLL_SECONDS=${PMM_POLL_SECONDS:-60}
export PMM_MIN_AGE_SECONDS=${PMM_MIN_AGE_SECONDS:-90}

# make_bcs GFS->HRRR regridding worker cap. Default (unset) is one worker per
# lead hour, which can OOM on small-RAM hosts. Set MAKE_BCS_WORKERS=1 or 2 on
# memory-constrained instances (e.g. a 16 GiB g5.xlarge). Exported so make_bcs.py
# picks it up; only exported when set, so the default behavior is unchanged.
[ -n "${MAKE_BCS_WORKERS:-}" ] && export MAKE_BCS_WORKERS

# Forecast options (src/fcst.py) — env-var overridable; defaults match fcst.py.
#   model_path / inittime / lead_hours    <- MODEL / INIT_TIME / LEAD_HOUR (above)
#   --num_members                         <- N_ENSEMBLES (above)
#   --members                             <- MEMBER_RANGE (below)
#   --base_dir / --output_dir             <- DATAROOT (above)
BATCH_SIZE=${BATCH_SIZE:-1}          # --batch_size
LOG_LEVEL=${LOG_LEVEL:-INFO}         # --log_level (DEBUG|INFO|WARNING|ERROR)
PMM_ALPHA=${PMM_ALPHA:-0.7}          # --pmm_alpha (nudge toward PMM mean, 0..1)
NOISE_RHO=${NOISE_RHO:-0.9}          # --noise_rho (noise blend/correlation, 0..1)
NO_DIFFUSION=${NO_DIFFUSION:-NO}     # --no_diffusion when YES (deterministic model)
# NO_NUDGING was removed: upstream deleted nudging from src/fcst.py entirely
# (commit fd13a09), so passing --no_nudging is now an argparse error.

# GFS cycle selection (src/get_bcs.py --gfs_min_lag_hours). 0 = newest cycle, which
# is what every run to date used. GFS takes about 4 h to publish a cycle while the
# HRRR analysis lands in about 51 min, so with 0 the freshest GFS is still missing
# for 12 of the 24 hours in a day. Set GFS_MIN_LAG=4 for a fixed hourly schedule;
# aws/pick_cycle.sh emits the value it decided on.
GFS_MIN_LAG=${GFS_MIN_LAG:-0}

# Subdomain cropping. Unset -> full 1059x1799 domain, exactly as before.
#
# SUB_BBOX="N,W,S,E" crops to a box containing that region plus SUB_HALO cells on
# every side, sized and placed by src/crop_domain.py. SUB_HEIGHT/SUB_WIDTH crop to a
# fixed grid-centred box instead; SUB_BBOX wins if both are given.
#
# The input stages always run at full domain, because the GFS->HRRR regridding
# weights are fixed at 1059x1799. Only the forecast runs on the crop, which is where
# the cost is: inference scales with H*W (9.8x faster on a 1.2% box, measured), and a
# crop fits a 24 GB A10G where the full grid does not.
#
# Sizes are constrained to H % 8 == 3 and W % 8 == 7 by the model's baked
# reflect-padding; crop_domain.py enforces it and refuses illegal sizes. See
# docs/subdomain.md.
SUB_BBOX=${SUB_BBOX:-}
SUB_HALO=${SUB_HALO:-40}
SUB_HEIGHT=${SUB_HEIGHT:-}
SUB_WIDTH=${SUB_WIDTH:-}


# Output/delivery knobs (defaults preserve the original local behavior).
# GRIB2 off by default: NetCDF is what everything downstream reads, and GRIB2 adds
# ~4 GB per cycle plus a wgrib2 dependency for no consumer in this pipeline.
NO_GRIB2=${NO_GRIB2:-YES}            # NO -> also write GRIB2 (passes --grib2)
NC_COMPLEVEL=${NC_COMPLEVEL:-0}      # --nc_complevel; 0 = uncompressed (original behavior)
NC_LSD=${NC_LSD:-}                   # --nc_least_significant_digit (LOSSY; empty = off)
OUTPUT_HOURS=${OUTPUT_HOURS:-}       # --output_hours (e.g. "0:3:24"); empty = every lead hour
S3_OUTPUT=${S3_OUTPUT:-}             # s3://bucket/prefix; empty disables upload
PURGE_LOCAL=${PURGE_LOCAL:-NO}       # YES -> --purge_local (requires S3_OUTPUT)
OVERLAP_FCST=${OVERLAP_FCST:-NO}     # YES -> start fcst early so its model load
                                     # overlaps the input stages (see below)

# Assemble the store_true forecast flags (empty-array-safe under set -u).
FCST_FLAGS=()
[ "$NO_DIFFUSION" == "YES" ] && FCST_FLAGS+=(--no_diffusion)
[ "$NO_GRIB2"     != "YES" ] && FCST_FLAGS+=(--grib2)
[ -n "$NC_LSD" ]             && FCST_FLAGS+=(--nc_least_significant_digit "$NC_LSD")
[ -n "$OUTPUT_HOURS" ]       && FCST_FLAGS+=(--output_hours "$OUTPUT_HOURS")
[ -n "$S3_OUTPUT" ]          && FCST_FLAGS+=(--s3_output "$S3_OUTPUT")
[ "$PURGE_LOCAL"  == "YES" ] && FCST_FLAGS+=(--purge_local)

# init-time date/hour components (mirrors job-make-bcs.sh).
DATE=${INIT_TIME%%T*}; DATE=${DATE//-/}
HOUR=${INIT_TIME#*T}

# Where the forecast reads its inputs and writes its outputs. Full-domain runs use
# DATAROOT unchanged; a cropped run gets its own root, because fcst.py resolves inputs
# as <base_dir>/<YYYYMMDD>/<HH>/ and both the full-domain and cropped npz would
# otherwise collide on that one path.
if [ -n "$SUB_BBOX" ] || { [ -n "$SUB_HEIGHT" ] && [ -n "$SUB_WIDTH" ]; }; then
    SUBDOMAIN=YES
    FCST_ROOT="${DATAROOT}/subrun"
    if [ -n "$SUB_BBOX" ]; then
        # A named region plus a halo is what src/crop_grib2.py needs to crop the RAW
        # inputs before make_ics.py/make_bcs.py run, which is what actually saves
        # their cost rather than only the forecast's.
        CROP_MODE=raw
        CROP_ARGS=(--bbox "$SUB_BBOX" --halo "$SUB_HALO")
        SUB_DESC="bbox ${SUB_BBOX} + ${SUB_HALO}-cell halo"
    else
        # A fixed grid-centred box has no region to protect with a halo, so it stays
        # on the older path: crop the .npz AFTER make_ics.py/make_bcs.py run at full
        # domain (src/crop_domain.py). That does nothing for input-stage cost, only
        # for the forecast itself.
        CROP_MODE=npz
        CROP_DIR="${DATAROOT}/${DATE}/${HOUR}-sub"
        CROP_ARGS=(--height "$SUB_HEIGHT" --width "$SUB_WIDTH")
        SUB_DESC="${SUB_HEIGHT} x ${SUB_WIDTH} (grid-centred)"
    fi
else
    SUBDOMAIN=NO
    FCST_ROOT="${DATAROOT}"
    SUB_DESC="full domain"
    # Half a request is a mistake, not a default: silently running the full domain
    # because only one of the two was set would waste an entire run.
    if [ -n "$SUB_HEIGHT" ] || [ -n "$SUB_WIDTH" ]; then
        echo "ERROR: SUB_HEIGHT and SUB_WIDTH must be set together (got '${SUB_HEIGHT}' and '${SUB_WIDTH}')." >&2
        exit 1
    fi
fi

log() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# --- environment (mirrors "source ${PACKAGEROOT}/etc/env.sh" in the jobs) ---
if [ "$ENVMODE" == "OPN" ]; then
    # shellcheck disable=SC1091
    source "${PACKAGEROOT}/etc/env_emc.sh"
else
    # shellcheck disable=SC1091
    source "${PACKAGEROOT}/etc/env_mac.sh"
fi

# PURGE_LOCAL deletes each NetCDF after upload, so nothing is left for plot.py
# to read. Refuse the combination rather than producing an empty plot stage.
if [ "$PURGE_LOCAL" == "YES" ] && [ "$RUNPLOT" == "YES" ]; then
    die "PURGE_LOCAL=YES removes the NetCDF that the plot stage reads. Set RUNPLOT=NO and run plots separately (aws/run_plots.sh)."
fi
if [ "$PURGE_LOCAL" == "YES" ] && [ -z "$S3_OUTPUT" ]; then
    die "PURGE_LOCAL=YES requires S3_OUTPUT; it would otherwise discard the outputs."
fi
# compute_pmm reads every member's NetCDF back off disk, so it has the same
# conflict with PURGE_LOCAL as plotting does.
if [ "$PURGE_LOCAL" == "YES" ] && [ "$N_ENSEMBLES" -ge 2 ]; then
    die "PURGE_LOCAL=YES removes the NetCDF that the ensemble PMM stage reads. Run PMM as a separate job for multi-member cycles."
fi

MODEL="${PACKAGEROOT}/net-diffusion/model.keras"
STATS="${PACKAGEROOT}/net-diffusion/normalize-stats.nc"
[ -f "$MODEL" ] || die "Model not found: $MODEL (run install_env_mac.sh to pull the LFS object)."
[ -f "$STATS" ] || die "Normalization stats not found: $STATS"

mkdir -p "${DATAROOT}/logs"
cd "$DATAROOT"

# Single process handles all members (no GPU array); full member range.
MEMBER_RANGE="0-$((N_ENSEMBLES-1))"

echo "PACKAGEROOT=$PACKAGEROOT, DATAROOT=$DATAROOT"
echo "INIT_TIME=$INIT_TIME, LEAD_HOUR=$LEAD_HOUR, N_ENSEMBLES=$N_ENSEMBLES, MEMBER_RANGE=$MEMBER_RANGE, RUNPLOT=$RUNPLOT"

# run_stage NAME <command...> : tee to logs/NAME.out, fatal on failure.
run_stage() {
    local name="$1"; shift
    log "Stage: ${name}"
    echo "+ $*" | tee "${DATAROOT}/logs/${name}.out"
    if ! "$@" 2>&1 | tee -a "${DATAROOT}/logs/${name}.out"; then
        die "Stage '${name}' failed (see ${DATAROOT}/logs/${name}.out)."
    fi
}

# Stages 1-4 prepare the inputs. Grouped in a function so the forecast can either
# follow them (default) or run concurrently with them (OVERLAP_FCST=YES).
# gfs_to_hrrr_weights.nc's name is hardcoded inside make_bcs.py (not a --flag), and
# it is looked up relative to whatever directory the process is CWD'd in when make_bcs
# runs, which is DATAROOT (this script cd's there before Stage 1). xESMF's
# reuse_weights=True means: if that file exists, USE IT, regardless of whether it was
# built for this run's target grid. A full-domain weights file and a subdomain one are
# different shapes, and make_bcs.py has no size check of its own -- xESMF raises
# "invalid entry in coordinates array" if the shapes disagree (confirmed by testing:
# it fails loudly rather than silently misregridding), but a run that dies there is
# still a run that dies. Two different subdomain boxes collide the same way.
#
# Handled here, without touching make_bcs.py: key a cache of weights files by the
# target grid actually in play (the crop's y0/x0/height/width, or "full" for no crop),
# stage the right one in before calling make_bcs.py, and put whatever it built back in
# the cache afterward -- then remove the generic filename so it cannot be picked up by
# a DIFFERENT box on a later run in this same DATAROOT. This preserves the reuse this
# script has always relied on (repeat runs at the same box regenerate nothing) while
# making a box change safe instead of a landmine for whoever runs next.
WEIGHTS_FILE="gfs_to_hrrr_weights.nc"     # must match make_bcs.py's hardcoded name
WEIGHTS_CACHE="${DATAROOT}/.weights_cache"
stage_weights_for() {   # $1 = cache key
    mkdir -p "$WEIGHTS_CACHE"
    rm -f "$WEIGHTS_FILE"
    # A cache miss (first run at this box) is normal, not an error: under
    # set -e, a bare `[ -f X ] && cp ...` as a standalone statement returns 1
    # and kills the whole script the moment the file is absent. `if` guards
    # against that regardless of which branch is taken.
    if [ -f "${WEIGHTS_CACHE}/$1.nc" ]; then
        cp "${WEIGHTS_CACHE}/$1.nc" "$WEIGHTS_FILE"
    fi
}
save_weights_as() {   # $1 = cache key
    if [ -f "$WEIGHTS_FILE" ]; then
        cp "$WEIGHTS_FILE" "${WEIGHTS_CACHE}/$1.nc"
    fi
    rm -f "$WEIGHTS_FILE"
}

run_input_stages() {
    # --- Stage 1: get ICs  (jobs/job-get-ics.sh) ----------------------------
    run_stage get-ics \
        python3 "${PACKAGEROOT}/src/get_ics.py" "${INIT_TIME}" --base_dir "${DATAROOT}"

    # --- Stage 2: get BCs  (jobs/job-get-bcs.sh) ----------------------------
    run_stage get-bcs \
        python3 "${PACKAGEROOT}/src/get_bcs.py" "${INIT_TIME}" "${LEAD_HOUR}" --base_dir "${DATAROOT}" \
            --gfs_min_lag_hours "${GFS_MIN_LAG}"

    HRRR_GRID_FILE="${DATE}/${HOUR}/hrrr_${DATE}_${HOUR}_surface.grib2"
    ICS_BASE_DIR="${DATAROOT}"
    # Raw-crop mode writes make_ics/make_bcs output directly into FCST_ROOT, already
    # at the cropped size. Grid-centred (npz) crop mode and full domain both write to
    # DATAROOT at full size; npz mode then crops that into FCST_ROOT as Stage 4b below.
    if [ "$SUBDOMAIN" == "YES" ] && [ "$CROP_MODE" == "raw" ]; then
        ICS_OUTPUT_DIR="${FCST_ROOT}"
    else
        ICS_OUTPUT_DIR="${DATAROOT}"
    fi
    WEIGHTS_KEY="full"

    # --- Stage 2b: crop the RAW GRIB2, when a region+halo was requested -----
    # Cropping here, before make_ics.py/make_bcs.py run, is what actually saves
    # money: measured, input staging is ~450s of a subdomain forecast's wall clock
    # (55%), because make_ics.py reads every field with pygrib over the full
    # 1059x1799 grid and make_bcs.py's xESMF regridder targets that many output
    # points, regardless of how small the eventual forecast domain is. Cropping
    # the .npz AFTER preprocessing (src/crop_domain.py, still used by
    # aws/domain_test.sh for fidelity experiments, where full-domain and cropped
    # runs must share bit-identical inputs) does nothing for that cost.
    #
    # make_ics.py and make_bcs.py are not modified: pygrib and grbs[1].latlons()
    # read whatever grid is actually in the file, and their hardcoded-1059x1799
    # shape check is a logged warning, never fatal (src/make_ics.py:434,
    # src/make_bcs.py:765). See src/crop_grib2.py's module docstring for how the
    # crop is done (wgrib2 -ijsmall_grib) and how it was verified: 162 of 170
    # surface fields bit-identical to the corresponding window of the full file,
    # the rest at GRIB2 repacking precision (1e-5 to 1e-18 relative), not
    # resampling error.
    #
    # net-diffusion/normalize-stats.nc carries LAND/OROG stats added for this:
    # make_ics.py falls back to computing them FROM WHATEVER DATA IT IS GIVEN when
    # the norm file lacks an entry, which is fine at full domain (same domain every
    # time) but means a raw crop and the full domain disagree on their own local
    # land fraction. Fixed stats make both runs agree; verified to change nothing
    # else (see docs/subdomain.md).
    if [ "$SUBDOMAIN" == "YES" ] && [ "$CROP_MODE" == "raw" ]; then
        # make_ics.py always resolves its inputs as <base_dir>/<DATE>/<HOUR>/..., the
        # same as get_ics.py's own output layout, and it is not told otherwise here
        # (no code change). crop_grib2.py writes flat into whatever --out-dir it is
        # given, so that has to BE the nested <DATE>/<HOUR> leaf; ICS_BASE_DIR is the
        # directory above it, which is what make_ics.py's --base_dir must be.
        RAW_CROP_DIR="${DATAROOT}/${DATE}/${HOUR}-sub-raw"
        run_stage crop-inputs \
            python3 "${PACKAGEROOT}/src/crop_grib2.py" \
                --in-dir "${DATAROOT}/${DATE}/${HOUR}" \
                --out-dir "${RAW_CROP_DIR}/${DATE}/${HOUR}" \
                "${CROP_ARGS[@]}"
        ICS_BASE_DIR="${RAW_CROP_DIR}"
        HRRR_GRID_FILE="${RAW_CROP_DIR}/${DATE}/${HOUR}/hrrr_${DATE}_${HOUR}_surface.grib2"
        WEIGHTS_KEY="$(python3 -c "import json;m=json.load(open('${RAW_CROP_DIR}/${DATE}/${HOUR}/subdomain.json'));print(f\"{m['height']}x{m['width']}_y{m['y0']}_x{m['x0']}\")")"
        mkdir -p "${FCST_ROOT}/${DATE}/${HOUR}"
        # Carry the crop definition next to the outputs so a consumer can tell which
        # part of the field is product and which is halo to discard.
        cp "${RAW_CROP_DIR}/${DATE}/${HOUR}/subdomain.json" "${FCST_ROOT}/${DATE}/${HOUR}/" 2>/dev/null || true
    fi

    # --- Stage 3: make ICs  (depends on get-ics/crop-inputs) ----------------
    # ICS_BASE_DIR is the cropped-raw directory in raw-crop mode, DATAROOT
    # otherwise. In raw-crop mode this writes directly into FCST_ROOT already at
    # the cropped size; a plain full-domain or grid-centred run writes at full
    # size to DATAROOT, and the grid-centred (npz) crop mode below crops it after.
    run_stage make-ics \
        python3 "${PACKAGEROOT}/src/make_ics.py" "${STATS}" "${INIT_TIME}" \
            --base_dir "${ICS_BASE_DIR}" --output_dir "${ICS_OUTPUT_DIR}"

    # --- Stage 4: make BCs  (depends on get-bcs) ----------------------------
    # --base_dir stays DATAROOT even when cropping: GFS is not cropped by this
    # script (it is read once per lead hour regardless of the HRRR-side box, so
    # cropping it is a separate, unimplemented optimisation; see docs/subdomain.md).
    # --hrrr_grid_file is the cropped surface grib2 in raw-crop mode, which is what
    # actually shrinks the xESMF regrid: it defines the TARGET grid, and the
    # regridder's cost scales with target points, not with how much of GFS was
    # read to reach them.
    stage_weights_for "$WEIGHTS_KEY"
    run_stage make-bcs \
        python3 "${PACKAGEROOT}/src/make_bcs.py" "${STATS}" "${INIT_TIME}" "${LEAD_HOUR}" \
            --base_dir "${DATAROOT}" --output_dir "${ICS_OUTPUT_DIR}" \
            --hrrr_grid_file "${HRRR_GRID_FILE}"
    save_weights_as "$WEIGHTS_KEY"

    # --- Stage 4b: crop the .npz, for the grid-centred (npz) mode only ------
    # Deliberately inside run_input_stages: with OVERLAP_FCST=YES the forecast is
    # already running and blocked on the sentinel, which is touched only after this
    # returns. So the crop is covered by the same barrier as the input stages, and
    # the forecast cannot start reading a half-written cropped npz.
    if [ "$SUBDOMAIN" == "YES" ] && [ "$CROP_MODE" == "npz" ]; then
        run_stage crop \
            python3 "${PACKAGEROOT}/src/crop_domain.py" \
                --in-dir "${DATAROOT}/${DATE}/${HOUR}" --out-dir "${CROP_DIR}" \
                --init-time "${INIT_TIME}" "${CROP_ARGS[@]}"

        # fcst.py resolves inputs as <base_dir>/<YYYYMMDD>/<HH>/, so the cropped npz
        # have to sit at that path under their own root.
        mkdir -p "${FCST_ROOT}/${DATE}/${HOUR}"
        cp "${CROP_DIR}/hrrr_${DATE}_${HOUR}.npz" \
           "${CROP_DIR}/gfs_${DATE}_${HOUR}.npz" \
           "${FCST_ROOT}/${DATE}/${HOUR}/" || die "staging cropped inputs failed"
        cp "${CROP_DIR}/subdomain.json" "${FCST_ROOT}/${DATE}/${HOUR}/" 2>/dev/null || true
    fi
}

FCST_CMD=(python3 "${PACKAGEROOT}/src/fcst.py" "${MODEL}" "${INIT_TIME}" "${LEAD_HOUR}"
          --num_members "${N_ENSEMBLES}" --members "${MEMBER_RANGE}"
          --batch_size "${BATCH_SIZE}" --log_level "${LOG_LEVEL}"
          --pmm_alpha "${PMM_ALPHA}" --noise_rho "${NOISE_RHO}"
          --nc_complevel "${NC_COMPLEVEL}"
          ${FCST_FLAGS[@]+"${FCST_FLAGS[@]}"}
          --base_dir "${FCST_ROOT}" --output_dir "${FCST_ROOT}")

if [ "$OVERLAP_FCST" == "YES" ]; then
    # The forecast's startup (TF import, GPU init, model load) needs no input data,
    # so start it now and let it block on a sentinel. The sentinel is created only
    # after the input stages have all succeeded, which is why the forecast waits on
    # it rather than on the npz files themselves: the files appear before they are
    # complete, so watching them would race.
    READY="${FCST_ROOT}/${DATE}/${HOUR}/.inputs_ready"
    mkdir -p "$(dirname "$READY")"; rm -f "$READY"

    log "Stage: fcst (started early; will load the model, then wait for inputs)"
    echo "+ ${FCST_CMD[*]} --wait_for_input ${READY}" > "${DATAROOT}/logs/fcst.out"
    "${FCST_CMD[@]}" --wait_for_input "$READY" --wait_timeout 3600 \
        >> "${DATAROOT}/logs/fcst.out" 2>&1 &
    FCST_PID=$!
    # Never leave the forecast orphaned if an input stage dies (run_stage calls die).
    trap '[ -n "${FCST_PID:-}" ] && kill "$FCST_PID" 2>/dev/null; exit' EXIT INT TERM

    run_input_stages

    log "Inputs ready; releasing the forecast"
    touch "$READY"

    if wait "$FCST_PID"; then
        trap - EXIT INT TERM; FCST_PID=""
        log "Stage fcst complete (see ${DATAROOT}/logs/fcst.out)"
    else
        rc=$?
        trap - EXIT INT TERM; FCST_PID=""
        die "Stage 'fcst' failed with status ${rc} (see ${DATAROOT}/logs/fcst.out)."
    fi
else
    run_input_stages

    # --- Stage 5: forecast  (depends on make-ics + make-bcs) ---------------
    run_stage fcst "${FCST_CMD[@]}"
fi

# --- Stage 6: plot members  (depends on fcst; jobs/job-plot.sh) -------------
if [ "$RUNPLOT" == "YES" ]; then
    run_stage plot \
        python3 "${PACKAGEROOT}/src/plot.py" "${INIT_TIME}" "${LEAD_HOUR}" \
            --members "${MEMBER_RANGE}" --forecast_dir "${FCST_ROOT}" --output_dir "${FCST_ROOT}"
fi

# --- Stage 7: ensemble PMM (+ mean/spread plot) -----------------------------
if [ "$N_ENSEMBLES" -ge 2 ]; then
    run_stage compute-pmm \
        python3 "${PACKAGEROOT}/src/compute_pmm.py" "${INIT_TIME}" "${LEAD_HOUR}" \
            --forecast_dir "${FCST_ROOT}" --output_dir "${FCST_ROOT}" --n_ensembles "${N_ENSEMBLES}"

    if [ "$RUNPLOT" == "YES" ]; then
        # non-array plot path plots the mean/spread (jobs/job-plot.sh: "avg spr")
        run_stage plot-pmm \
            python3 "${PACKAGEROOT}/src/plot.py" "${INIT_TIME}" "${LEAD_HOUR}" \
                --members avg spr --forecast_dir "${FCST_ROOT}" --output_dir "${FCST_ROOT}"
    fi
fi

log "Forecast cycle complete (${SUB_DESC}). Outputs under ${FCST_ROOT}/${DATE}/${HOUR}/  (logs in ${DATAROOT}/logs/)"
