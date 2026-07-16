# HRRRCast on macOS / Apple Silicon (Local Runbook)

This is a companion to [README.md](README.md). It documents the changes added to
run HRRRCast on a single Apple Silicon Mac (arm64, CPU only), and the impact of
running locally instead of on the HPC/GPU pipeline described in the main README.

The main README remains the source of truth for the science, channels, APCP
logic, diagnostics, and output naming. Nothing here changes those. This file
covers only the local install, the local run driver, and the platform
differences.

Verified on macOS 26.5 (arm64), 2026-07-15: environment builds, the pretrained
model loads (50,237,524 parameters), and the full forecast cycle runs through
preprocessing and CPU inference.

## Table of Contents

- [What Changed](#what-changed)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [End-to-End Pipeline (Local)](#end-to-end-pipeline-local)
- [Dependency Choices and Why](#dependency-choices-and-why)
- [Model Weights (Git LFS)](#model-weights-git-lfs)
- [Model Code Change: TimeCondLayer](#model-code-change-timecondlayer)
- [Impact of Running Locally](#impact-of-running-locally)
- [Troubleshooting (Local)](#troubleshooting-local)
- [File Manifest](#file-manifest)

## What Changed

New files (none of the original HPC files were removed; the SLURM path still works):

| File | Purpose |
|------|---------|
| `install_env_mac.sh` | Local installer; arm64/CPU counterpart to `install_env_ursa.sh`. |
| `environment.mac.yaml` | conda-forge environment spec for osx-arm64. |
| `requirements.txt` | pip-style dependency reference (kept in sync with the yaml). |
| `etc/env_mac.sh` | Env activation shim; parallels `etc/env.sh` and `etc/env_emc.sh`. |
| `run_cycle.sh` | Local forecast-cycle driver; non-SLURM counterpart to `submit_all.sh`. |

One edit to existing source:

| File | Change |
|------|--------|
| `src/resnet.py` | `TimeCondLayer.call` now normalizes negative gather indices so the same code runs on CPU and GPU. See [below](#model-code-change-timecondlayer). |

## Installation

### Prerequisites

- Apple Silicon Mac (arm64), macOS.
- Miniconda or Miniforge (`conda` on PATH).
- Git with SSH access to the model remote (for the Git LFS weights).

There is no CUDA on this platform; TensorFlow runs on CPU.

### Install

```bash
./install_env_mac.sh
```

The script:

1. Creates or updates the `hrrrcast` conda env from `environment.mac.yaml`
   (`CONDA_SUBDIR=osx-arm64`), with a retry plus classic-solver fallback for the
   intermittent libmamba solver crash on arm64.
2. Pins `TF_USE_LEGACY_KERAS=1` into the env (Keras 2 shim; see below).
3. Pulls the Git LFS model object and resolves the pointer from the local cache.
4. Downloads Cartopy Natural Earth shapefiles.
5. Verifies the full stack imports and that the pretrained model loads.

Every step is fatal: the script stops with a nonzero exit and a clear `ERROR:`
message on any failure. `SKIP_LFS=1 ./install_env_mac.sh` skips the model pull.

### Activate

```bash
conda activate hrrrcast
```

or, from scripts, `source etc/env_mac.sh` (mirrors how the job scripts source
`etc/env.sh`). This also exports `TF_USE_LEGACY_KERAS=1`.

## Quick Start

Run one forecast cycle locally. `run_cycle.sh` uses the same positional
interface as `submit_all.sh`:

```bash
./run_cycle.sh <INIT_TIME> <LEAD_HOUR> <N_ENSEMBLES> <N_GPUS> <PACKAGEROOT> <DATAROOT> <RUNPLOT> <ENVMODE>
```

Example (6-hour, single-member cycle):

```bash
./run_cycle.sh 2024-05-06T23 6 1
```

`N_GPUS` is accepted for interface parity but unused locally: there is no GPU
job array, so a single process runs the full `0..N_ENSEMBLES-1` member range.

### Forecast options

All `src/fcst.py` arguments are exposed as environment variables, with defaults
matching `fcst.py`:

| Variable | fcst.py argument | Default |
|----------|------------------|---------|
| `BATCH_SIZE` | `--batch_size` | `1` |
| `LOG_LEVEL` | `--log_level` | `INFO` |
| `PMM_ALPHA` | `--pmm_alpha` | `0.7` |
| `NOISE_RHO` | `--noise_rho` | `0.9` |
| `NO_DIFFUSION` | `--no_diffusion` when `YES` | `NO` |
| `NO_NUDGING` | `--no_nudging` when `YES` | `YES` (matches `jobs/job-fcst.sh`) |

Example (deterministic, 3-member, DEBUG logging):

```bash
NO_DIFFUSION=YES LOG_LEVEL=DEBUG ./run_cycle.sh 2024-05-06T23 6 3
```

## End-to-End Pipeline (Local)

`run_cycle.sh` runs the same stages as `submit_all.sh`, in the same dependency
order, calling the same `src/*.py` with the same arguments the `jobs/job-*.sh`
templates use. The difference is orchestration: stages run sequentially in one
process instead of as `sbatch` jobs with `afterok` dependencies.

| Stage | Script | Depends on | HPC equivalent |
|-------|--------|-----------|----------------|
| get-ics | `src/get_ics.py` | - | `jobs/job-get-ics.sh` |
| get-bcs | `src/get_bcs.py` | - | `jobs/job-get-bcs.sh` |
| make-ics | `src/make_ics.py` | get-ics | `jobs/job-make-ics.sh` |
| make-bcs | `src/make_bcs.py` | get-bcs | `jobs/job-make-bcs.sh` |
| fcst | `src/fcst.py` | make-ics, make-bcs | `jobs/job-fcst.sh` |
| plot | `src/plot.py` | fcst (if `RUNPLOT=YES`) | `jobs/job-plot.sh` |
| compute-pmm | `src/compute_pmm.py` | fcst (if `N_ENSEMBLES>=2`) | `jobs/job-compute-pmm.sh` |

Each stage tees its output to `${DATAROOT}/logs/<stage>.out` and is fatal: the
cycle stops on the first failure (this mirrors `submit_with_check` in
`submit_all.sh`). Outputs follow the same naming as the main README, under
`${DATAROOT}/<YYYYMMDD>/<HH>/`.

## Dependency Choices and Why

The upstream `environment.yaml` targets Linux/HPC and does not resolve on
osx-arm64. `environment.mac.yaml` differs as follows, each item forced by the
platform:

- **python 3.12**: the only version with osx-arm64 conda-forge builds across the
  full geoscience stack (`grib2io`, `pygrib`, `xesmf`/`esmpy`, `wgrib2`).
- **TensorFlow from PyPI, not conda**: the upstream `==2.15.0` pin has no arm64
  build. The install uses `tensorflow==2.21.0` (CPU) plus `tf-keras==2.21.0`.
- **Keras 2 shim**: TF 2.21 defaults to Keras 3, but the pretrained
  `net-diffusion/model.keras` was saved under Keras 2 with custom registered
  layers. `TF_USE_LEGACY_KERAS=1` routes `tf.keras` to `tf-keras` so the model
  loads. This is set both in the env (`conda env config vars`) and in
  `etc/env_mac.sh`.
- **`numpy<2.5`**: the pip TensorFlow 2.21 wheel is built against numpy < 2.5;
  numpy 2.5.x breaks the import (abort). 2.4.x is verified.
- **`dask-core` instead of `dask`**: the `dask` metapackage pulls `dask-expr`,
  which requires `pyarrow`. The `pyarrow` 24.x arm64 build aborts (SIGABRT) on
  import, which crashes anything importing pandas (including TensorFlow via
  tf-keras). xarray only needs `dask-core`.
- **`pandas<3`**: conda-forge pandas 3.0 also pulls `pyarrow`. 2.3.x does not.
- **`xesmf>=0.8.7`**: older xesmf (0.6.0) does `import ESMF`, but `esmpy` 8.4+
  renamed that module to `esmpy`; recent xesmf imports it correctly.

## Model Weights (Git LFS)

`net-diffusion/model.keras` is a Git LFS object (about 203 MB). Two local
gotchas the installer handles:

- **Forks do not copy LFS objects.** The blob is served by the canonical remote
  (`NOAA-GSL/HRRRCast-live`), not by a personal fork. `install_env_mac.sh`
  fetches from `origin`.
- **No committed `.gitattributes` LFS rule.** Because the repo never committed a
  `*.keras filter=lfs` rule, `git lfs checkout` will not smudge the pointer even
  after the object is cached. The installer resolves the pointer directly from
  the local LFS cache with `git lfs smudge` and verifies the size.

If you pull the model manually:

```bash
git lfs fetch --all origin
git lfs smudge < net-diffusion/model.keras > net-diffusion/model.keras.tmp
mv net-diffusion/model.keras.tmp net-diffusion/model.keras
```

## Model Code Change: TimeCondLayer

`TimeCondLayer` (in `src/resnet.py`) selects its time-conditioning channels with
`tf.gather(inputs, self.time_mask, axis=-1)`, and the saved model stores
`time_mask = [-2, -1]` (the last two channels).

`tf.gather` does not wrap negative indices. The GPU kernel does not bounds-check
and tolerates out-of-range indices; the CPU kernel validates and raises
`indices = -1 is not in [0, N)`. So the model ran on the H100 it was validated
on but failed on CPU.

The fix normalizes negative indices to positive before the gather:

```python
n_channels = tf.shape(inputs)[-1]
mask = tf.convert_to_tensor(self.time_mask, dtype=tf.int32)
mask = tf.where(mask < 0, mask + n_channels, mask)
time_feats = tf.gather(inputs, mask, axis=-1)
```

Properties:

- **Same codebase on CPU and GPU**: the computed indices are identical and
  in-range on both.
- **No model re-export**: this is a call-time change; `get_config` and the stored
  `time_mask` are unchanged, so existing `.keras` files load as-is.
- **Correctness note**: if the GPU kernel had been returning zeros for the
  out-of-range indices rather than the intended last two channels, the lead-time
  and ensemble conditioning would have been degraded even on GPU. The patch makes
  the intended behavior explicit on both platforms.

`src/resnet.py` is the only source file changed for local support; propagate it
to the GPU/HPC checkouts.

## Impact of Running Locally

- **CPU only**: `fcst.py` logs "No GPUs used, running on CPU only." Diffusion
  inference is many denoising steps at full grid resolution, so CPU forecasts are
  slow. Expect long wall times even for short leads and single members. This is a
  performance characteristic, not an error. Preprocessing (get/make ICs and BCs)
  runs in minutes.
- **No scheduler**: no `sbatch`, no `atparse` templating, no job dependencies or
  arrays. `run_cycle.sh` runs the stages in one process.
- **Ensemble members run serially** in a single process rather than distributed
  across a GPU array.
- **Network at runtime**: `get_ics.py` and `get_bcs.py` still download HRRR and
  GFS from public sources during the run; the install does not pre-stage data.
- **wgrib2** comes from the conda env (`WGRIB2=wgrib2`) instead of an HPC module.

For production or timely forecasts, use the HPC/GPU path in the main README. The
local path is intended for development, preprocessing, and small validation runs.

## Troubleshooting (Local)

- **`Segmentation fault` on `import tensorflow` or during solve**: usually a
  `pyarrow` 24.x arm64 abort pulled in transitively, or an intermittent libmamba
  solver crash. `install_env_mac.sh` avoids the first (via `dask-core` and
  `pandas<3`) and retries the second. Rebuild with `./install_env_mac.sh`.
- **`Aborted` on `import tensorflow` after a change**: check that `numpy` is
  still `<2.5` (`python -c "import numpy; print(numpy.__version__)"`).
- **`ModuleNotFoundError: No module named 'ESMF'`**: xesmf drifted to an old
  build; ensure `xesmf>=0.8.7`.
- **`indices = -1 is not in [0, N)` in `TimeCondLayer`**: the `src/resnet.py`
  patch is missing on this checkout. Apply it (see above).
- **Model load fails with `'str' object is not callable`**: the custom layers in
  `src/resnet.py` were not imported before `load_model`. Import `resnet` (with
  `src/` on `sys.path`) first; `run_cycle.sh` and the pipeline do this.
- **`model.keras` is a 134-byte text file**: it is still an LFS pointer. See
  [Model Weights](#model-weights-git-lfs).

## File Manifest

```
install_env_mac.sh      # local installer (arm64/CPU)
environment.mac.yaml    # conda-forge env spec (osx-arm64)
requirements.txt        # pip-style dependency reference
etc/env_mac.sh          # env activation shim
run_cycle.sh            # local forecast-cycle driver
src/resnet.py           # TimeCondLayer negative-index fix (edited)
```
