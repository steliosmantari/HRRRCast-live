# HRRRCast (Live Pipeline)

HRRRCast is a neural network-based, high‑resolution regional weather forecasting system leveraging HRRR analyses/forecasts and GFS boundary conditions. The live pipeline now features unified logging utilities, per‑variable/level normalization, enhanced APCP (precipitation) sourcing, HRRR→model downsampling, GFS→HRRR interpolation, diffusion (probabilistic) and deterministic model support, and NetCDF→GRIB2 export.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Running on AWS](#running-on-aws)
- [Ensemble and PMM Support](#ensemble-and-pmm-support)
- [End‑to‑End Pipeline](#end-to-end-pipeline)
- [Model Usage](#model-usage)
- [Data & Channels](#data--channels)
- [Diagnostic Variables](#diagnostic-variables)
- [APCP Handling Logic](#apcp-handling-logic)
- [GRIB2 Export](#grib2-export)
- [Outputs & Naming](#outputs--naming)
- [Available Models](#available-models)
- [Logging & Utilities](#logging--utilities)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Support](#support)

## Installation

### Prerequisites

- Miniconda3 or Anaconda
- CUDA-compatible GPU (recommended) or CPU
- Internet connection (for initial setup)

### Standard Installation (GPU/CPU with Internet)

1. Install Miniconda3 if not already installed
2. Clone this repository and navigate to the project directory
3. Install the environment using the provided configuration:

```bash
conda env create -f environment.yaml
conda activate hrrrcast
```

### HPC Installation (No Internet on Compute Nodes)

For HPC environments like Ursa where compute nodes lack internet access:

```bash
./install_env_ursa.sh
```

This script handles CUDA availability simulation on login nodes.

### Post-Installation Configuration

1. **Configure Environment Paths**: Edit the environment files in the `etc/` directory to match your conda installation directory

2. **Download Cartopy Shapefiles** (for plotting functionality):
   ```bash
   python -c "import cartopy.io.shapereader as shpreader; shpreader.natural_earth()"
   ```

## Quick Start

### Running Forecasts

Use the provided submission script to run forecasts:

```bash
./submit_all.sh <INIT_TIME> <LEAD_HOUR> <N_ENSEMBLES> <N_GPUS> <ACCNR>
```

- `INIT_TIME`: Initialization time in format `YYYY-MM-DDTHH` (e.g., `2024-05-06T23`)
- `LEAD_HOUR`: Number of forecast hours (e.g., `6`)
- `N_ENSEMBLES`: Number of ensemble members to run (default: `1`)
- `N_GPUS`: Number of GPUs to use for parallel forecast jobs (default: `1`)
- `ACCNR`: (Optional) Account number for SLURM jobs (default: `gsd-hpcs`)

**Example**: Run a 6-hour ensemble forecast with 10 members on 2 GPUs starting from May 6, 2024 at 23:00 UTC:
```bash
./submit_all.sh 2024-05-06T23 6 10 2
```

### Manual Forecast, Plotting & GRIB Export

#### Forecast

You can run the forecast script directly:

```bash
python src/fcst.py <model_path> <inittime> <lead_hours> --members 0-2 --output_dir <output_dir> [--no_diffusion] [--base_dir <dir>]
```

- `model_path`: Path to the trained model (e.g., `net-diffusion/model.keras`)
- `inittime`: Initialization time (e.g., `2024-05-06T23`)
- `lead_hours`: Number of forecast hours (e.g., `6`)
- `--members`: List or range of ensemble member IDs (e.g., `0-2 4 6-7`)
- `--no_diffusion`: Use deterministic model (default is diffusion/ensemble)
- `--base_dir`: Base directory for input files (default: `./`)
- `--output_dir`: Output directory for forecast files (default: `./`)

#### Plotting

To plot the forecast output for all hours 1 to N for each member:

```bash
python src/plot.py <inittime> <lead_hour> --members 0-2 --forecast_dir <forecast_dir> --output_dir <output_dir>
```

- `inittime`: Initialization time (e.g., `2024-05-06T23`)
- `lead_hour`: Maximum forecast hour to plot (e.g., `6`)
- `--members`: List or range of member IDs (e.g., `0-2 4 pmm`)
- `--forecast_dir`: Directory containing forecast files (default: `./`)
- `--output_dir`: Output directory for plots (default: `./`)

**Note:** This will generate plots for all hours from 1 to `lead_hour` (inclusive) for each member, saving each hour's plots in a separate subdirectory.

#### GRIB2 Output (default)

Forecasts run via `src/fcst.py` write **both NetCDF and GRIB2** outputs by default (per-hour files during rollout). GRIB2 export uses `grib2io`, `eccodes`, and system `wgrib2`.

If you need a standalone conversion utility, use `src/nc2grib.py` (see `Netcdf2Grib`).

## Running on AWS

An on-demand AWS path exists: one command launches a GPU instance, runs a cycle,
streams NetCDF to S3, and terminates itself. Plotting is a separate job.

```bash
aws/run_on_ec2.sh --bucket <your-bucket>                    # 24 h, latest cycle
aws/run_on_ec2.sh --bucket <your-bucket> --preflight-only   # check the account, launch nothing
aws/status.sh --live                                        # what is running right now
```

Full documentation, including account setup, measured timings, and the hardware
constraints, is in **[aws/README_aws.md](aws/README_aws.md)**. The essentials:

| | |
|---|---|
| Instance | `g6e.2xlarge` (L40S 48 GB). **24 GB GPUs cannot run this model** |
| Throughput | ~36 s per lead hour on the L40S, versus ~1,720 s on Apple Silicon CPU |
| A 24 h forecast | ~50-60 min wall clock, ~9 GB of NetCDF |
| Deliverable | NetCDF to S3, one file per lead hour, uploaded as it is written |

### Pipeline options added for the AWS path

All of these work locally too, and all default to the previous behavior:

| `run_cycle.sh` env var | `src/fcst.py` argument | Default | Purpose |
|---|---|---|---|
| `NO_GRIB2` | `--no_grib2` | `NO` | NetCDF only; also drops the `wgrib2` dependency |
| `NC_COMPLEVEL` | `--nc_complevel` | `0` | NetCDF zlib level. Measured: level 1 gives 1.54x on these fields and higher levels add nothing |
| `NC_LSD` | `--nc_least_significant_digit` | unset | **Lossy** quantization. `2` gives 3.8x for max abs error 0.0039 in native units |
| `S3_OUTPUT` | `--s3_output` | unset | Upload each file to S3 as it is written |
| `PURGE_LOCAL` | `--purge_local` | `NO` | Delete the local copy after a confirmed upload |
| `OVERLAP_FCST` | `--wait_for_input` | `NO` | Start the forecast before the input stages so its model load overlaps them (see below) |

`src/plot.py` gained `--variables {all,surface,pressure}`. Surface-only is 52 of
the ~173 figures per lead hour, so roughly 30% of the output and runtime.

`MAKE_BCS_WORKERS` caps the `make_bcs` regridding pool. On AWS it is derived
automatically from instance RAM; see [aws/user_data.sh](aws/user_data.sh). The
binding constraint is memory, not cores: each worker holds ~15 GB and the parent
holds ~1.25 GB per lead hour until the npz is written.

### Performance notes on `make_bcs`

`make_bcs` dominated wall clock on a 24-lead-hour cycle. Four changes took it from
~42 min to ~14 min, all measured on AWS at 24 lead hours with 2 workers:

| change | stage total |
|---|---|
| original | ~42 min |
| single pass instead of two (see below) | 22.6 min |
| float32 storage, no compression, no unused raw arrays | 14.1 min |
| worker pool sized from measured memory (7 instead of 2) | 7.3 min |
| regridding batched into one call per lead hour | **4.2 min** |

All five are measured on AWS at 24 lead hours on a `g6e.2xlarge`: a **10x** reduction
in the stage, which took a full 24-hour forecast from 79.5 min to ~35 min. Every
step was verified against a reference npz before being accepted.

**1. Every GFS file was regridded twice.** `process_pressure_levels()` and
`process_surface_variables()` each called `process_single_lead_hour()` with
identical arguments, and that function already returns both result sets, so each
call discarded exactly what the other kept. `process_all_variables()` does one
pass. Output is byte-identical; verified by comparing every array in the npz.

**2. Arrays were stored float64 but consumed as float32.** `src/fcst.py` does
`tf.convert_to_tensor(..., dtype=tf.float32)` on load, so half of every byte was
discarded. Storing float32 halves worker memory, the pickle payload per lead hour,
the parent's accumulated arrays (measured 29.5 -> 14.6 GB peak RSS) and the npz
(13.4 -> 7.7 GB). Verified equivalent: max relative difference 3e-08, about 0.25
ulp of float32.

**3. The npz was compressed.** `np.savez_compressed` achieved only ~1.15x on these
fields while costing **7.7 minutes** of single-threaded zlib. This file is scratch,
deleted when the run ends, so that was a poor trade. Plain `np.savez` writes it in
**12 seconds**. With float32 the uncompressed file is smaller than the old
compressed one anyway.

**4. Raw HRRR-grid values were built and returned but never used.** Every call site
discarded them, so they cost worker memory and an extra pickle per lead hour.

`save_preprocessed_data` also preallocates its output array rather than building a
list and calling `np.array()` on it, which had made a second full copy of every
lead hour.

**5. The worker pool was capped far below what memory allowed.** The cap was set
from a measurement that predated changes 2 and 4, and so conflated worker memory
with the parent's float64 accumulation: it assumed 15 GB per worker when the
measured peak is 1.89 GB (96 samples, top five 1.70-1.89). The pool is now sized
from actual instance RAM, giving 7 workers instead of 2 on a `g6e.2xlarge` and
halving the stage again.

Scaling is sublinear: 3.5x the workers gave 2x the speedup. Load average was 2.47
with 7 workers on 8 cores, so the workers were mostly blocked rather than computing.
Profiling one lead hour showed why, and it was not what we assumed:

| phase | seconds | share |
|---|---|---|
| `pg.open` (GRIB index scan) | 0.38 | 0.9% |
| decode (`select` + `.values`) | 18.17 | 52% |
| regrid, per-field loop (28 fields) | 16.74 | 48% |
| **regrid, batched (same 28 fields)** | **0.79** | — |

**6. Regridding was done once per 2-D field: ~43 xESMF calls per lead hour.** Each
call re-streamed the sparse weight matrix from memory. `interpolate_many_to_hrrr_grid()`
stacks the fields and makes **one** call, measured 21x faster on that step and
bit-identical (max difference exactly 0). Both loops were restructured into decode ->
batch regrid -> transform, preserving field order so the channel layout is unchanged.

Note this profile also **ruled out GFS field subsetting** as a way to speed up
`make_bcs`: `pg.open` is 0.9% of the work, and pygrib's `select()` already decodes
only the requested fields, so a smaller input file would not reduce the 52% spent
decoding. Subsetting would still cut the ~15 GB download in `get_bcs`, which is a
separate stage.

With regrid down to ~2% of a lead hour, decode now dominates and is cleanly
CPU-bound per field, so worker scaling should be closer to linear than it was.

### Overlapping the forecast's startup with input preparation

`OVERLAP_FCST=YES` starts `src/fcst.py` *before* the input stages instead of after.
The forecast's startup -- TensorFlow import, GPU initialization, and deserializing
the 50M-parameter model -- takes about 4 minutes and depends on none of the input
data, while the GPU sits idle for the whole input phase.

`fcst.py` therefore loads the model first, then blocks on `--wait_for_input <path>`
until `run_cycle.sh` creates a sentinel. The sentinel is created only after every
input stage has succeeded, which is why the forecast waits on it rather than on the
npz files: those appear before they are complete, so watching them would race.

The overlap hides `min(model load, input phase)`, so which side dominates depends on
forecast length. Both cases were measured on AWS:

| lead hours | input phase | model load | outcome |
|---|---|---|---|
| 6 | ~2.7 min | ~3.8 min | inputs ready **69 s before** the model loaded; input phase costs nothing. Run 14.7 -> 12.5 min |
| 24 | ~6 min | ~3.7 min | forecast **waited 206 s** for inputs; the model load is fully hidden instead. Run 79.5 -> 28.4 min overall |

So at 6 lead hours the input stages are effectively free and further `make_bcs` work
would buy nothing, while at the production length of 24 lead hours `make_bcs` is still
on the critical path and savings there still convert to wall clock.

If any input stage fails, `run_cycle.sh` kills the background forecast via an EXIT
trap and exits non-zero -- verified by forcing a `get-ics` failure and confirming no
orphaned process remains.

## Ensemble and PMM Support

- For diffusion/ensemble forecasts, use `--members` to specify which ensemble members to run and plot.
- The system supports ranges (e.g., `0-2`), comma-separated, and non-integer IDs (e.g., `pmm` for ensemble mean).
- The PMM (Probability-Matched Mean) is computed and plotted automatically when running in ensemble mode.

## End-to-End Pipeline

| Stage | Script | Key Actions |
|-------|--------|-------------|
| 1. Download HRRR analyses + prior hour f01 surface | `src/get_ics.py` | Fetches pressure & surface GRIB plus previous hour 1h surface forecast (for APCP fallback) |
| 2. Build IC dataset | `src/make_ics.py` | Reads HRRR GRIB, applies per‑variable / per‑level normalization, log transforms, APCP replacement strategy, saves `.npz` |
| 3. Download GFS boundary GRIBs | `src/get_bcs.py` | Selects appropriate synoptic cycle(s); can ensure required f006 and window coverage |
| 4. Build BC dataset | `src/make_bcs.py` | Interpolates GFS fields to downsampled HRRR grid (xESMF), normalizes, APCP future synoptic sourcing, saves `.npz` |
| 5. Run forecast | `src/fcst.py` | Loads IC + BC arrays, assembles inputs, runs deterministic or diffusion model, writes per-hour NetCDF **and GRIB2** outputs |
| 6. Plot results | `src/plot.py` | Parallel (per lead hour) map plots for pressure & surface variables + summary panels |
| 7. (Optional) Standalone GRIB2 export | `src/nc2grib.py` | Converts NetCDF member/mean outputs to GRIB2 with parameter metadata |

All scripts use centralized utilities in `src/utils.py` for logging (`setup_logging`), directory creation, datetime validation, and resilient downloading.

## Model Usage

### Loading Models

Load trained models using TensorFlow/Keras:

```python
import tensorflow as tf

model = tf.keras.models.load_model("net-deterministic/model.keras", safe_mode=False, compile=False)
```

### Input/Output Dimensions

The spatial grid (530×900) represents every other grid point from the original HRRR grid (1059×1799).

## Data & Channels

Channel counts are dynamic and driven by configuration in `make_ics.py` / `make_bcs.py`. Use those scripts (or `fcst.py`) to confirm the exact channel counts for a given model. The default configuration in `make_ics.py` is:

| Category | Components | Count (default) |
|----------|------------|-----------------|
| Pressure-level variables | 6 vars × 20 levels (UGRD,VGRD,VVEL,TMP,HGT,SPFH) | 120 |
| Surface dynamic variables | 18 (PRES, MSLMA, REFC, T2M, UGRD10M, VGRD10M, UGRD80M, VGRD80M, D2M, TCDC, LCDC, MCDC, HCDC, VIS, APCP, HGTCC, CAPE, CIN) | 18 |
| Static constants | LAND, OROG | 2 |
| Lead time (per step, autoregressive) | 1 | 1 |
| Total model input (IC) | 120 + 18 + 2 + 1 | 141 |

The forecast model typically predicts only the dynamic meteorological fields (pressure-level + surface set, excluding static + lead-time). The exact predicted channel count is inferred automatically in `fcst.py` and depends on the model configuration.

## Diagnostic Variables

Diagnostics are computed in [src/diagnostics.py](src/diagnostics.py) via `compute_diagnostics()`. You can run all diagnostics or select a subset with `include`/`exclude` flags.

Available diagnostic groups (see function docstrings for full variable lists):

- **Surface thermodynamics**: R2M, SPFH2M, POT2M
- **Column-integrated**: PWAT
- **Precipitation diagnostics**: CRAIN, CFRZR, and related masks/fractions
- **Wind diagnostics**: GUST, GUST_FACTOR, GUST_CONV, WIND_10M, WIND_MAX
- **Convective diagnostics**: shear, helicity, vorticity, storm motion, updraft helicity, and vertical velocity extrema
- **Vertical profile**: 0°C isotherm height/pressure and RH_0C

## APCP Handling Logic

Accumulated precipitation (APCP / total precipitation) is not reliable directly from the HRRR analysis or isolated GFS lead files for sub‑hour windows, so the pipeline applies tiered sourcing:

1. **Initial Conditions (`make_ics.py`)**: Replace analysis APCP with prior hour 1‑hour forecast accumulation file (`*_surface_f01.grib2`) downloaded by `get_ics.py`.
2. **Boundary Conditions (`make_bcs.py`)**: For each valid time, attempt to replace APCP with the field from the nearest future synoptic GFS cycle (> valid time). If that GRIB file exists it is interpolated and substituted; otherwise keep current lead’s APCP.
3. **(Optional future)**: If cumulative fields from consecutive future hours are available, compute 1‑hour increments (difference of cumulative precipitation); current implementation substitutes directly (documented for transparency).

Logging clearly notes when APCP is substituted (INFO) or when fallback occurs (DEBUG/WARNING).

## GRIB2 Export

GRIB2 export is handled in `src/fcst.py` during forecasts. For standalone conversion, `nc2grib.py` converts NetCDF forecast outputs to GRIB2 with:
* Parameter overrides (`GRIB_PARAM_OVERRIDE`) and center metadata
* Cube attribute mapping (`ATTR_MAPS`)
* Optional index generation via `wgrib2` (`.idx` files)

Dependencies: `grib2io`, `eccodes`, `wgrib2`. These are optional and not required for core inference/plotting.

## Outputs & Naming

Forecast outputs are written per hour into:

```
<output_dir>/<YYYYMMDD>/<HH>/
```

Where `<YYYYMMDD>` and `<HH>` come from the initialization time.

**NetCDF** (per hour):

- Members: `hrrrcast_mem<NN>_f<HH>.nc` (e.g., `hrrrcast_mem0_f03.nc`)
- Ensemble mean: `hrrrcast_avg_f<HH>.nc`

**GRIB2** (per hour):

- Members: `hrrrcast.m<NN>.t<HH>z.pgrb2.f<HH>`
- Ensemble mean: `hrrrcast.avg.t<HH>z.pgrb2.f<HH>`

Hour `f00` is written for the initial state when per-hour outputs are enabled.

## Available Models

| Model | Use |
|-------|------------|
| net-diffusion | For probabilistic forecast |

## Logging & Utilities

All major scripts (`get_ics.py`, `make_ics.py`, `get_bcs.py`, `make_bcs.py`, `fcst.py`, `plot.py`, `nc2grib.py`) use centralized helpers in `src/utils.py`:

| Function | Purpose |
|----------|---------|
| `setup_logging(level)` | Idempotent root logger config |
| `validate_datetime(str)` | Flexible datetime parsing → padded components |
| `make_directory(path)` | Recursive directory creation |
| `download_file_with_retry(url, path, ...)` | Simple resilient downloader with progress |

Customize log verbosity with `--log_level` on each CLI.

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Use the smaller model or reduce batch size
2. **Missing Cartopy Shapefiles**: Run the cartopy download command in post-installation
3. **Environment Path Issues**: Verify conda paths in `etc/` configuration files
4. **Missing Optional Libraries**: Plotting works without Cartopy (falls back); GRIB2 export requires extra libs
5. **Model Loading Errors**: Ensure `safe_mode=False` when loading models

### Performance Tips

- Use GPU acceleration when available
- For large-scale runs, consider batch processing
- Monitor memory usage during rollout forecasts

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License
MIT License. See [LICENSE](LICENSE).

## Citation

If you use HRRRCast in your research, please cite:

    @misc{abdi2025hrrrcastdatadrivenemulatorregional,
          title={HRRRCast: a data-driven emulator for regional weather forecasting at convection allowing scales}, 
          author={Daniel Abdi and Isidora Jankov and Paul Madden and Vanderlei Vargas and Timothy A. Smith and Sergey Frolov and Montgomery Flora and Corey Potvin},
          year={2025},
          eprint={2507.05658},
          archivePrefix={arXiv},
          primaryClass={physics.ao-ph},
          url={https://arxiv.org/abs/2507.05658}, 
    }

## Support

For questions or issues not covered in this README, please open an issue in the repository or contact the development team.

---

*This README reflects the live pipeline as of 2026-02-23. Refer to source code and the cited paper for deeper architectural details.*
