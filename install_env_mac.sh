#!/usr/bin/env bash
#
# install_env_mac.sh — build the HRRRCast conda environment on Apple Silicon
# (macOS / osx-arm64, CPU-only). Counterpart to install_env_ursa.sh (HPC/Linux).
#
# This machine has no NVIDIA GPU and no arm64 build of tensorflow==2.15.0, so the
# environment is CPU TensorFlow (from PyPI) plus a Keras 2 shim (tf-keras +
# TF_USE_LEGACY_KERAS=1) so the pretrained Keras-2 model in net-diffusion/ loads.
# See environment.mac.yaml for the full rationale.
#
# Usage:
#   ./install_env_mac.sh            # create (or update) the env, then verify
#   ENV_NAME=hrrrcast ./install_env_mac.sh
#   SKIP_LFS=1 ./install_env_mac.sh # skip the git-lfs model pull
#
set -euo pipefail

ENV_NAME="${ENV_NAME:-hrrrcast}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${REPO_DIR}/environment.mac.yaml"

log()  { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mWARNING:\033[0m %s\n' "$*" >&2; }
die()  { printf '\n\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# --- sanity checks ---------------------------------------------------------
[[ "$(uname -s)" == "Darwin" ]] || die "This script targets macOS. Use install_env_ursa.sh on HPC/Linux."
[[ "$(uname -m)" == "arm64" ]]  || die "This script targets Apple Silicon (arm64), got $(uname -m)."
[[ -f "$ENV_FILE" ]]            || die "Missing $ENV_FILE"
command -v conda >/dev/null 2>&1 || die "conda not found on PATH. Install Miniconda/Miniforge first."

# Resolve package downloads against the arm64 index.
export CONDA_SUBDIR=osx-arm64

# --- create or update the environment --------------------------------------
# conda's libmamba solver intermittently segfaults (SIGSEGV) mid-solve on
# macOS/arm64. Retry, then fall back to the classic solver, before giving up.
run_conda_env() {
    local attempt
    for attempt in 1 2 3; do
        if "$@"; then return 0; fi
        warn "conda solve failed/crashed (attempt ${attempt}/3); retrying..."
    done
    warn "Retrying once more with the classic solver (CONDA_SOLVER=classic)."
    CONDA_SOLVER=classic "$@"
}

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "Environment '$ENV_NAME' exists — updating from $ENV_FILE (pruning removed deps)"
    run_conda_env conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
    log "Creating environment '$ENV_NAME' from $ENV_FILE"
    run_conda_env conda env create -n "$ENV_NAME" -f "$ENV_FILE"
fi

# Pin the subdir on the env so future `conda install` stays arm64.
conda config --env --set subdir osx-arm64 2>/dev/null || true

# --- activate --------------------------------------------------------------
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# --- Keras 2 shim ----------------------------------------------------------
# TF 2.21 defaults to Keras 3; net-diffusion/model.keras was saved under Keras 2
# with custom registered layers (src/resnet.py). Route tf.keras -> tf-keras.
log "Pinning TF_USE_LEGACY_KERAS=1 into the environment"
conda env config vars set TF_USE_LEGACY_KERAS=1 -n "$ENV_NAME" >/dev/null
conda activate "$ENV_NAME"   # reactivate so the var takes effect now

# --- pull the LFS-backed model weights -------------------------------------
# model.keras is a Git-LFS object (~200 MB). Requires network + SSH access to
# the GitHub remote and the object being present there. FATAL: the install is
# not considered complete without the model. Set SKIP_LFS=1 to opt out.
if [[ "${SKIP_LFS:-0}" != "1" ]]; then
    git -C "$REPO_DIR" rev-parse --git-dir >/dev/null 2>&1 \
        || die "$REPO_DIR is not a git checkout; cannot pull the LFS model."
    git lfs version >/dev/null 2>&1 \
        || die "git-lfs unavailable in this shell; cannot pull the LFS model."
    log "Fetching Git-LFS model weights (net-diffusion/model.keras)"
    git -C "$REPO_DIR" lfs install --local \
        || die "git lfs install failed."
    git -C "$REPO_DIR" lfs fetch --all origin \
        || die "git lfs fetch failed (check SSH access to origin and that the object exists)."
    git -C "$REPO_DIR" lfs checkout || true   # no-op here: repo has no .gitattributes LFS rule
    model="$REPO_DIR/net-diffusion/model.keras"
    [[ -f "$model" ]] || die "model.keras missing from the working tree."
    # This repo never committed a .gitattributes LFS rule, so `lfs checkout` will
    # not smudge the pointer even after the object is cached. Convert it directly
    # from the local LFS cache via `lfs smudge` (no network) and verify.
    if [[ "$(wc -c < "$model")" -lt 100000 ]]; then
        log "Resolving model.keras pointer from the LFS cache (git lfs smudge)"
        git -C "$REPO_DIR" lfs smudge < "$model" > "${model}.tmp" \
            || { rm -f "${model}.tmp"; die "git lfs smudge failed; object not in cache. Run: git -C '$REPO_DIR' lfs fetch --all origin"; }
        [[ "$(wc -c < "${model}.tmp")" -ge 100000 ]] \
            || { rm -f "${model}.tmp"; die "smudged model.keras is still a pointer; the LFS object was not fetched."; }
        mv "${model}.tmp" "$model"
    fi
    log "model.keras present ($(wc -c < "$model") bytes)"
fi

# --- cartopy shapefiles (for plotting) -------------------------------------
log "Downloading Cartopy Natural Earth shapefiles"
python -c "import cartopy.io.shapereader as s; s.natural_earth()" \
    || die "cartopy shapefile download failed."

# --- verify the stack imports ----------------------------------------------
log "Verifying the installed stack"
python - <<'PY'
import importlib, sys
mods = ["numpy","pandas","xarray","dask","zarr","netCDF4","h5netcdf","cartopy",
        "xesmf","esmpy","pygrib","grib2io","skimage","tensorflow"]
bad = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:                       # noqa: BLE001
        bad.append(f"{m}: {e}")
import tensorflow as tf
print("tensorflow", tf.__version__,
      "| devices:", [d.device_type for d in tf.config.list_physical_devices()])
# Under TF_USE_LEGACY_KERAS=1 this must resolve to tf_keras (Keras 2), which is
# what loads the pretrained model.
route = tf.keras.__name__
print("tf.keras ->", route)
if "tf_keras" not in route:
    bad.append(f"tf.keras did not route to Keras 2 (got {route}); check TF_USE_LEGACY_KERAS")
if bad:
    print("\nFAILED CHECKS:", *bad, sep="\n  ")
    sys.exit(1)
print("\nAll core modules imported OK.")
PY

# --- confirm the model loads (FATAL) ---------------------------------------
model="$REPO_DIR/net-diffusion/model.keras"
[[ -f "$model" ]] && [[ "$(wc -c < "$model")" -ge 100000 ]] \
    || die "net-diffusion/model.keras not present as a real file; cannot smoke-test model load."
log "Smoke-testing model load"
MODEL_PATH="$model" SRC_DIR="$REPO_DIR/src" python - <<'PY'
import os, sys
# The model uses custom layers registered via @register_keras_serializable in
# src/resnet.py; they must be imported before load_model can deserialize them
# (this mirrors what src/fcst.py does).
sys.path.insert(0, os.environ["SRC_DIR"])
import tensorflow as tf
import resnet  # noqa: F401  (registers custom Keras layers)
try:
    m = tf.keras.models.load_model(os.environ["MODEL_PATH"], safe_mode=False, compile=False)
    print(f"Model loaded OK: {type(m).__name__}, {m.count_params():,} params")
except Exception as e:                           # noqa: BLE001
    print("Model load FAILED:", e); sys.exit(1)
PY

log "Done. Activate with:  conda activate ${ENV_NAME}"
echo "    Pipeline entrypoint:  python src/fcst.py --help"
