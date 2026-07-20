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

# Keras 2 shim: enable it ONLY when tf-keras is actually installed. TF >= 2.16
# defaults to Keras 3; forcing TF_USE_LEGACY_KERAS=1 without the tf_keras package
# makes tf.keras unimportable ("Keras cannot be imported"), which breaks model
# loading. The arm64 Mac env ships tf-keras, so route tf.keras to it and load the
# Keras-2 model via the legacy path. The AWS GPU env (Keras 3, no tf-keras) loads
# the same model natively once fcst.py imports the custom layers, so leave the
# legacy path off there.
if python -c "import tf_keras" >/dev/null 2>&1; then
    export TF_USE_LEGACY_KERAS=1
else
    export TF_USE_LEGACY_KERAS=0
fi
