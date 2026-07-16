#!/bin/bash
# Local macOS environment activation for HRRRCast.
# Parallels etc/env.sh (HPC/GSD) and etc/env_emc.sh (EMC): those source a
# site-specific miniconda and `conda activate hrrrcast`. This one activates the
# conda env built by install_env_mac.sh on this machine.
#
# Override the env name with HRRRCAST_ENV if needed.
HRRRCAST_ENV="${HRRRCAST_ENV:-hrrrcast}"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH; run install_env_mac.sh first." >&2
    return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HRRRCAST_ENV"

# TF 2.21 defaults to Keras 3; the pretrained model is Keras 2. Route tf.keras
# to tf-keras (also pinned into the env by install_env_mac.sh).
export TF_USE_LEGACY_KERAS=1
