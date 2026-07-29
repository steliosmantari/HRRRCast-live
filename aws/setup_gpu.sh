#!/usr/bin/env bash
#
# aws/setup_gpu.sh — provision the HRRRCast conda env on a GPU EC2 instance
# (Linux x86_64). MVP path: get one GPU box running a real forecast to validate
# GPU parity and timing before committing to any orchestration.
#
# Target region: us-east-1 (co-located with the public NOAA HRRR/GFS S3 buckets
# noaa-hrrr-bdp-pds / noaa-gfs-bdp-pds, so input data is fast and egress-free).
#
# Prerequisites on the instance:
#   - NVIDIA driver + CUDA already present. Easiest: launch from the
#     "AWS Deep Learning Base GPU AMI" (Ubuntu). `nvidia-smi` must work.
#   - This repo checked out; run this script from anywhere inside it.
#
# Usage:
#   MODEL_S3=s3://my-bucket/hrrrcast/model.keras ./aws/setup_gpu.sh   # stage model from S3 (preferred)
#   ./aws/setup_gpu.sh                                                # else fall back to git-lfs
#
set -euo pipefail

ENV_NAME="${ENV_NAME:-hrrrcast}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${REPO_DIR}/aws/environment.aws.yaml"

log() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# --- sanity: GPU present ---------------------------------------------------
command -v nvidia-smi >/dev/null 2>&1 \
    || die "nvidia-smi not found. Launch on a GPU instance with NVIDIA drivers (e.g. AWS Deep Learning Base GPU AMI)."
log "GPU visible to the OS:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# --- conda / miniforge -----------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    log "Installing Miniforge"
    curl -fsSL -o /tmp/miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
fi
# shellcheck disable=SC1091
source "$(conda info --base 2>/dev/null || echo "$HOME/miniforge3")/etc/profile.d/conda.sh"

# Make conda available in *later* shells too, not just this one. Without this a
# fresh login shell has no conda on PATH, so `conda activate hrrrcast` and
# etc/env_mac.sh (which run_cycle.sh sources) both fail with
# "conda not found on PATH". Idempotent.
conda init bash >/dev/null 2>&1 || log "conda init bash failed; later shells may not find conda"

# --- create/update env (GPU TF 2.15 via CONDA_OVERRIDE_CUDA) ---------------
# Prefer the explicit lock file when present. Two reasons:
#   1. Reproducibility. environment.aws.yaml pins only tensorflow, so a rebuild
#      months from now can resolve different versions of xarray, dask, esmpy and
#      the rest, silently changing numerical behavior. The lock pins all 339
#      packages to exact builds.
#   2. Speed. An @EXPLICIT lock is a URL list, so conda skips dependency solving
#      entirely and just downloads and links.
# Regenerate after editing environment.aws.yaml:
#   conda-lock lock --file aws/environment.aws.yaml --platform linux-64 \
#       --lockfile aws/conda-lock.yml
#   conda-lock render --kind explicit -p linux-64 aws/conda-lock.yml
LOCK_FILE="${REPO_DIR}/aws/conda-linux-64.lock"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "Env '$ENV_NAME' already exists; leaving it untouched"
    echo "    To rebuild it from the lock: conda env remove -n ${ENV_NAME} && $0"
elif [ -f "$LOCK_FILE" ] && [ "${USE_LOCK:-YES}" == "YES" ]; then
    log "Creating env '$ENV_NAME' from the explicit lock ($(grep -c '^https' "$LOCK_FILE") packages, no solve)"
    CONDA_OVERRIDE_CUDA="12.0" conda create -y -n "$ENV_NAME" --file "$LOCK_FILE" \
        || die "conda create from lock failed. Retry with USE_LOCK=NO to solve from ${ENV_FILE}."
else
    log "Creating env '$ENV_NAME' by solving $ENV_FILE (unpinned; prefer the lock file)"
    CONDA_OVERRIDE_CUDA="12.0" conda env create -n "$ENV_NAME" -f "$ENV_FILE"
fi
conda activate "$ENV_NAME"

# Confirm the GPU build of TensorFlow actually got installed. This takes about a
# second and closes a failure mode that is otherwise silent and very expensive:
# conda ships cpu_* and cuda*_ builds of tensorflow 2.15 under the same version,
# and the GPU one is only chosen when a CUDA runtime is visible to the solver.
# Pick up the CPU build and everything still installs, imports and runs -- at
# roughly 1,700 s per lead hour instead of 35 s, so a 24-hour forecast becomes an
# 11-hour one. Caught exactly this while generating the lock file, where
# conda-lock ignored CONDA_OVERRIDE_CUDA (see aws/virtual-packages.yaml).
TF_BUILD="$(conda list --no-pip '^tensorflow$' 2>/dev/null | awk '$1=="tensorflow" {print $3}')"
log "TensorFlow build: ${TF_BUILD:-unknown}"
case "$TF_BUILD" in
    *cuda*) : ;;
    cpu_*)  die "The CPU build of TensorFlow is installed (${TF_BUILD}). Inference would be ~50x slower.
    Regenerate the lock with aws/virtual-packages.yaml, or create the env with CONDA_OVERRIDE_CUDA=12.0." ;;
    *)      log "WARNING: could not determine the TensorFlow build; expected a cuda* build." ;;
esac

# Record exactly what got installed, so a future run can be compared against this
# one. Without it, "did the environment change?" is unanswerable after the fact.
mkdir -p "${REPO_DIR}/logs"
conda list --explicit > "${REPO_DIR}/logs/conda-installed.txt" 2>/dev/null || true

# --- stage the model -------------------------------------------------------
MODEL="${REPO_DIR}/net-diffusion/model.keras"
if [ -n "${MODEL_S3:-}" ]; then
    log "Staging model from ${MODEL_S3}"
    command -v aws >/dev/null 2>&1 || die "aws CLI not found; needed for MODEL_S3."
    aws s3 cp "$MODEL_S3" "$MODEL"
elif [ "$(wc -c < "$MODEL" 2>/dev/null || echo 0)" -lt 100000 ]; then
    log "MODEL_S3 not set; pulling model.keras via git-lfs (needs GitHub auth on this instance)"
    git -C "$REPO_DIR" lfs install --local

    # This repo has no committed .gitattributes LFS rule, so `lfs checkout` will
    # not smudge the pointer even once the object is cached. Resolve it from the
    # local LFS cache directly.
    smudge_model() {
        git -C "$REPO_DIR" lfs checkout net-diffusion/model.keras >/dev/null 2>&1 || true
        if [ "$(wc -c < "$MODEL" 2>/dev/null || echo 0)" -lt 100000 ]; then
            if git -C "$REPO_DIR" lfs smudge < "$MODEL" > "${MODEL}.tmp" 2>/dev/null; then
                mv "${MODEL}.tmp" "$MODEL"
            else
                rm -f "${MODEL}.tmp"
            fi
        fi
    }

    # 1) Try origin (your fork). GitHub serves fork LFS via the fork network for
    #    authenticated users.
    git -C "$REPO_DIR" lfs fetch origin --include="net-diffusion/model.keras" 2>/dev/null || true
    smudge_model

    # 2) Fall back to the canonical upstream, which holds the blob and serves it
    #    (this is where the object actually lives). Override MODEL_LFS_UPSTREAM
    #    if your fork's upstream differs.
    if [ "$(wc -c < "$MODEL" 2>/dev/null || echo 0)" -lt 100000 ]; then
        UPSTREAM_REPO="${MODEL_LFS_UPSTREAM:-https://github.com/NOAA-GSL/HRRRCast-live.git}"
        log "origin did not serve the LFS blob; trying upstream ${UPSTREAM_REPO}"
        git -C "$REPO_DIR" -c "lfs.url=${UPSTREAM_REPO}/info/lfs" \
            lfs fetch origin --include="net-diffusion/model.keras" \
            || die "git lfs fetch failed from both origin and upstream. Check GitHub auth / that the object exists."
        smudge_model
    fi
fi
[ "$(wc -c < "$MODEL" 2>/dev/null || echo 0)" -ge 100000 ] \
    || die "model.keras not staged. Set MODEL_S3=s3://... or ensure git-lfs access."
log "model.keras present ($(wc -c < "$MODEL") bytes)"

# --- verify GPU TensorFlow + model load ------------------------------------
# Importing TensorFlow, initializing the GPU and loading the 50M-parameter model
# costs about 3.5 minutes (measured on a g6e.2xlarge), and src/fcst.py does exactly
# the same work again a few minutes later. So this is a genuinely useful check when
# setting up a new account or debugging by hand, and pure duplicated cost on an
# automated run -- fcst.py fails loudly if the GPU is absent or the model will not
# load, so skipping it here loses no coverage.
#
# Default on, so the manual path documented in README_aws_mvp.md is unchanged.
# aws/user_data.sh sets VERIFY_GPU=NO for unattended runs.
VERIFY_GPU="${VERIFY_GPU:-YES}"
if [ "$VERIFY_GPU" == "YES" ]; then
    log "Verifying GPU TensorFlow and model load"
    cd "$REPO_DIR"
    python - <<'PY'
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
import tensorflow as tf
import resnet  # registers custom Keras layers
gpus = tf.config.list_physical_devices("GPU")
print("tensorflow", tf.__version__, "| GPUs:", [g.name for g in gpus] or "NONE")
assert gpus, "No GPU visible to TensorFlow. Check the CUDA build and drivers."
m = tf.keras.models.load_model("net-diffusion/model.keras", safe_mode=False, compile=False)
print(f"Model loaded OK: {type(m).__name__}, {m.count_params():,} params")
PY
else
    # Cheap substitutes for the two things the full check would have caught, at a
    # cost of a second rather than 3.5 minutes.
    log "Skipping GPU TensorFlow verification (VERIFY_GPU=NO); fcst.py performs the real load"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
        || die "nvidia-smi failed; the GPU is not usable."
    [ "$(wc -c < "$MODEL" 2>/dev/null || echo 0)" -ge 100000 ] \
        || die "model.keras looks truncated."
fi

log "Setup complete. Activate with:  conda activate ${ENV_NAME}"
echo "    Run one cycle:  ./run_cycle.sh 2024-05-06T23 6 1"
echo "    (run_cycle.sh sources etc/env_mac.sh, which just activates '${ENV_NAME}';"
echo "     TF_USE_LEGACY_KERAS is a harmless no-op under TF 2.15.)"
