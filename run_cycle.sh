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
#   NO_GRIB2=NO       (YES -> NetCDF only; drops ~4 GB/cycle and the wgrib2 dep)
#   NC_COMPLEVEL=0    (1-9 enables NetCDF zlib compression; 0 = uncompressed.
#                      Measured: level 1 gives 1.54x on these fields, and higher
#                      levels add almost nothing, so 1 is the useful setting.)
#   NC_LSD=           (LOSSY quantization to N decimal digits before compression.
#                      Measured: N=2 gives 3.8x at max abs error 0.004 in native
#                      units, N=3 gives 3.0x at 0.0005. Empty = off.)
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

# Output/delivery knobs (defaults preserve the original local behavior).
NO_GRIB2=${NO_GRIB2:-NO}             # YES -> --no_grib2, NetCDF only
NC_COMPLEVEL=${NC_COMPLEVEL:-0}      # --nc_complevel; 0 = uncompressed (original behavior)
NC_LSD=${NC_LSD:-}                   # --nc_least_significant_digit (LOSSY; empty = off)
S3_OUTPUT=${S3_OUTPUT:-}             # s3://bucket/prefix; empty disables upload
PURGE_LOCAL=${PURGE_LOCAL:-NO}       # YES -> --purge_local (requires S3_OUTPUT)
OVERLAP_FCST=${OVERLAP_FCST:-NO}     # YES -> start fcst early so its model load
                                     # overlaps the input stages (see below)

# Assemble the store_true forecast flags (empty-array-safe under set -u).
FCST_FLAGS=()
[ "$NO_DIFFUSION" == "YES" ] && FCST_FLAGS+=(--no_diffusion)
[ "$NO_GRIB2"     == "YES" ] && FCST_FLAGS+=(--no_grib2)
[ -n "$NC_LSD" ]             && FCST_FLAGS+=(--nc_least_significant_digit "$NC_LSD")
[ -n "$S3_OUTPUT" ]          && FCST_FLAGS+=(--s3_output "$S3_OUTPUT")
[ "$PURGE_LOCAL"  == "YES" ] && FCST_FLAGS+=(--purge_local)

# init-time date/hour components (mirrors job-make-bcs.sh).
DATE=${INIT_TIME%%T*}; DATE=${DATE//-/}
HOUR=${INIT_TIME#*T}

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
run_input_stages() {
    # --- Stage 1: get ICs  (jobs/job-get-ics.sh) ----------------------------
    run_stage get-ics \
        python3 "${PACKAGEROOT}/src/get_ics.py" "${INIT_TIME}" --base_dir "${DATAROOT}"

    # --- Stage 2: get BCs  (jobs/job-get-bcs.sh) ----------------------------
    run_stage get-bcs \
        python3 "${PACKAGEROOT}/src/get_bcs.py" "${INIT_TIME}" "${LEAD_HOUR}" --base_dir "${DATAROOT}" \
            --gfs_min_lag_hours "${GFS_MIN_LAG}"

    # --- Stage 3: make ICs  (depends on get-ics; jobs/job-make-ics.sh) ------
    run_stage make-ics \
        python3 "${PACKAGEROOT}/src/make_ics.py" "${STATS}" "${INIT_TIME}" \
            --base_dir "${DATAROOT}" --output_dir "${DATAROOT}"

    # --- Stage 4: make BCs  (depends on get-bcs; jobs/job-make-bcs.sh) ------
    run_stage make-bcs \
        python3 "${PACKAGEROOT}/src/make_bcs.py" "${STATS}" "${INIT_TIME}" "${LEAD_HOUR}" \
            --base_dir "${DATAROOT}" --output_dir "${DATAROOT}" \
            --hrrr_grid_file "${DATE}/${HOUR}/hrrr_${DATE}_${HOUR}_surface.grib2"
}

FCST_CMD=(python3 "${PACKAGEROOT}/src/fcst.py" "${MODEL}" "${INIT_TIME}" "${LEAD_HOUR}"
          --num_members "${N_ENSEMBLES}" --members "${MEMBER_RANGE}"
          --batch_size "${BATCH_SIZE}" --log_level "${LOG_LEVEL}"
          --pmm_alpha "${PMM_ALPHA}" --noise_rho "${NOISE_RHO}"
          --nc_complevel "${NC_COMPLEVEL}"
          ${FCST_FLAGS[@]+"${FCST_FLAGS[@]}"}
          --base_dir "${DATAROOT}" --output_dir "${DATAROOT}")

if [ "$OVERLAP_FCST" == "YES" ]; then
    # The forecast's startup (TF import, GPU init, model load) needs no input data,
    # so start it now and let it block on a sentinel. The sentinel is created only
    # after the input stages have all succeeded, which is why the forecast waits on
    # it rather than on the npz files themselves: the files appear before they are
    # complete, so watching them would race.
    READY="${DATAROOT}/${DATE}/${HOUR}/.inputs_ready"
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
            --members "${MEMBER_RANGE}" --forecast_dir "${DATAROOT}" --output_dir "${DATAROOT}"
fi

# --- Stage 7: ensemble PMM (+ mean/spread plot) -----------------------------
if [ "$N_ENSEMBLES" -ge 2 ]; then
    run_stage compute-pmm \
        python3 "${PACKAGEROOT}/src/compute_pmm.py" "${INIT_TIME}" "${LEAD_HOUR}" \
            --forecast_dir "${DATAROOT}" --output_dir "${DATAROOT}" --n_ensembles "${N_ENSEMBLES}"

    if [ "$RUNPLOT" == "YES" ]; then
        # non-array plot path plots the mean/spread (jobs/job-plot.sh: "avg spr")
        run_stage plot-pmm \
            python3 "${PACKAGEROOT}/src/plot.py" "${INIT_TIME}" "${LEAD_HOUR}" \
                --members avg spr --forecast_dir "${DATAROOT}" --output_dir "${DATAROOT}"
    fi
fi

log "Forecast cycle complete. Outputs under ${DATAROOT}/${DATE}/${HOUR}/  (logs in ${DATAROOT}/logs/)"
