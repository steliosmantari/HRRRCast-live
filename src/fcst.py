#!/usr/bin/env python3
"""
Weather Forecast Runner Script

This script loads preprocessed GRIB data and runs the neural network forecast model.
This stage is GPU-intensive and handles the autoregressive model inference.

Usage:
    python run_forecast.py <model_path> <preprocessed_data> <lead_hours> [--output_dir DIR]
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
import tensorflow as tf
import xarray as xr
import pandas as pd

from nc2grib import Netcdf2Grib
import s3io

# Import custom modules (assuming they exist)
try:
    import resnet
except ImportError as e:
    logging.warning(f"Could not import custom modules: {e}")

from diffusion_params import (
    NUM_DIFFUSION_STEPS,
    NUM_INFERENCE_STEPS,
    INFERENCE_STEPS,
    compute_epsilon,
    ddpm,
    ddim,
    ddim_heun,
    dpmpp_2m,
)
from transform import (
    inverse_log_transform_array,
    inverse_neg_log_transform_array,
)
import utils
from utils import setup_logging
from diagnostics import compute_diagnostics
from cf_attributes import apply_cf_attributes, get_cf_encoding
from compute_pmm import compute_PMM

logger = None


def make_noise(
    anchor_noise: tf.Tensor, hour: int, member_seed: int, rho: float = 0.9
) -> tf.Tensor:
    """Generate blended previous-state + hourly innovation noise.
    
    Args:
        anchor_noise: Previous member noise state, shape (1, H, W, C)
        hour: Lead time hour (used for seed)
        member_seed: Member ID (used for seed)
        rho: Blend factor (1.0 = pure anchor, 0.0 = pure innovation)
    
    Returns:
        Tensor same shape as anchor_noise, blended with innovation
    """
    sigma = tf.sqrt(tf.constant(1.0 - rho**2, dtype=tf.float32))
    eps = tf.random.stateless_normal(
        shape=tf.shape(anchor_noise),
        dtype=tf.float32,
        seed=[member_seed, hour],
    )
    return rho * anchor_noise + sigma * eps


class PreprocessedDataLoader:
    """Handles loading and validation of preprocessed data."""
    
    def __init__(self, preprocessed_file: str):
        self.preprocessed_file = preprocessed_file
        self.data = None
        self.metadata = None
        self._load_data()
    
    def _load_data(self):
        """Load preprocessed data from file."""
        if not os.path.exists(self.preprocessed_file):
            raise FileNotFoundError(f"Preprocessed data file not found: {self.preprocessed_file}")
        
        try:
            logger.info(f"Loading preprocessed data from {self.preprocessed_file}")
            self.data = np.load(self.preprocessed_file)
            
            # Extract metadata
            self.metadata = {
                'init_year': str(self.data['init_year']),
                'init_month': str(self.data['init_month']),
                'init_day': str(self.data['init_day']),
                'init_hh': str(self.data['init_hh']),
                'init_datetime': str(self.data['init_datetime']),
                'pl_vars': self.data['pl_vars'].tolist(),
                'sfc_vars': self.data['sfc_vars'].tolist(),
                'levels': self.data['levels'].tolist(),
                'grid_height': int(self.data['grid_height']),
                'grid_width': int(self.data['grid_width']),
                'downsample_factor': int(self.data['downsample_factor']),
                'norm_file': str(self.data['norm_file'])
            }
            
            logger.info("Preprocessed data loaded successfully")
            logger.info(f"Model input shape: {self.data['model_input'].shape}")
            logger.info(f"Initialization time: {self.metadata['init_datetime']}")
            
        except Exception as e:
            logger.error(f"Error loading preprocessed data: {e}")
            raise
    
    def get_model_input(self) -> np.ndarray:
        """Get the model input array."""
        return self.data['model_input']
    
    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get latitude and longitude arrays."""
        return self.data['lats'], self.data['lons']
    
    def get_init_datetime(self) -> datetime:
        """Get initialization datetime."""
        return datetime.fromisoformat(self.metadata['init_datetime'])


class ForecastModel:
    """Handles model loading and inference."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self._setup_tf_environment()
        self._load_model()

    def _setup_tf_environment(self) -> None:
        """ 
        Set up the TensorFlow environment for optimal performance.
        """
        # use only 1 gpu
        num_gpus = 1
        # Improved CPU/GPU device handling
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            logger.info(f"Num GPUs available: {len(gpus)}")
            tf.config.set_visible_devices(gpus[:num_gpus], "GPU")
            visible_gpus = tf.config.get_visible_devices("GPU")
            logger.info(f"Using GPUs: {[gpu.name for gpu in visible_gpus]}")
            for gpu in tf.config.get_visible_devices("GPU"):
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("GPU memory growth set for all visible GPUs.")
        else:
            tf.config.set_visible_devices([], "GPU")
            logger.warning("No GPUs used, running on CPU only.")

        # set JIT compilation of graphs
        tf.config.optimizer.set_jit(True)
    
    def _load_model(self):
        """Load the TensorFlow model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model = tf.keras.models.load_model(
                self.model_path, 
                safe_mode=False, 
                compile=False
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Make prediction using the loaded model."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            return self.model(input_data, training=False)
        except Exception as e:
            logger.error(f"Error during model prediction: {e}")
            raise


class WeatherForecaster:
    """Handles the forecasting pipeline."""

    def __init__(
        self,
        data_loader_hrrr: PreprocessedDataLoader,
        data_loader_gfs: PreprocessedDataLoader,
        num_members: int,
        members: List[int],
        batch_size: int,
        use_diffusion: bool,
        lead_hours: int,
        predicted_channels: Optional[int] = None,
        gfs_channels: Optional[int] = None,
        static_channels: Optional[int] = None,
        pmm_alpha: float = 0.65,
        diffusion_sampler: str = "dpmpp-2m",
        noise_rho: float = 0.9,
        write_grib2: bool = False,
        nc_complevel: int = 0,
        nc_least_significant_digit: Optional[int] = None,
        s3_uploader: Optional["s3io.S3Uploader"] = None,
        output_hours: Optional[Set[int]] = None,
    ):
        self.data_loader_hrrr = data_loader_hrrr
        self.data_loader_gfs = data_loader_gfs
        self.metadata = data_loader_hrrr.metadata
        self.num_members = num_members
        self.members = members
        self.batch_size = batch_size
        self.use_diffusion = use_diffusion
        self.lead_hours = lead_hours
        self.pmm_alpha = pmm_alpha
        self.diffusion_sampler = diffusion_sampler
        self.noise_rho = noise_rho
        self.write_grib2 = write_grib2
        self.nc_complevel = nc_complevel
        self.nc_least_significant_digit = nc_least_significant_digit
        self.s3_uploader = s3_uploader
        # None (default) writes/uploads every lead hour, unchanged from before this
        # option existed. When set, only these hours are built and written; the
        # rollout itself still computes every hour in between (autoregressive state
        # depends on it), only the I/O is skipped.
        self.output_hours = output_hours
        # Names of files S3Uploader gave up on. Appended from the single-threaded
        # nc_executor and the grib2 executor; list.append is atomic under the GIL, and
        # the list is only read after both have been joined.
        self.failed_uploads: List[str] = []
        # Lead hours whose GRIB2 conversion failed. Appended from the grib2
        # executor, whose futures are never awaited; checked after the drain.
        self.failed_grib2: List[str] = []

        # log-transform variables list
        self.LOG_TRANSFORM_VARS = [
            "VIS",
            "APCP",
            "HGTCC",
            "CAPE",
        ]
        self.NEG_LOG_TRANSFORM_VARS = [
            "CIN",
        ]

        # Derive dynamic channel counts if not provided
        pl_vars = self.metadata["pl_vars"]
        sfc_vars = self.metadata["sfc_vars"]
        levels = self.metadata["levels"]
        default_predicted = len(pl_vars) * len(levels) + len(sfc_vars)
        self.input_shape = data_loader_hrrr.get_model_input().shape
        hrrr_channels = self.input_shape[-1]
        nlat, nlon = self.input_shape[1], self.input_shape[2] 

        if predicted_channels is None:
            predicted_channels = default_predicted
        if gfs_channels is None:
            gfs_channels = data_loader_gfs.get_model_input().shape[-1]
        if static_channels is None:
            static_channels = max(hrrr_channels - predicted_channels, 0)

        self.predicted_channels = predicted_channels
        self.gfs_channels = gfs_channels
        self.static_channels = static_channels

        # Load normalization file and construct per-channel mean/std vectors consistent with preprocessing
        norm_file = self.metadata["norm_file"]
        try:
            ds_norm = xr.open_dataset(norm_file)
            self._init_channel_stats(ds_norm)
            ds_norm.close()
            logger.info(
                f"Normalization file loaded and channel stats constructed: {norm_file}"
            )
        except Exception as e:
            logger.error(f"Error loading/processing normalization file: {e}")
            raise

        # Initialize per-member base noise anchors (on-the-fly generation in predict)
        logger.info(f"Initializing member anchors for {len(self.members)} members")
        self.member_anchors: Dict[int, tf.Tensor] = {}

        # Generate the base anchor for each member
        for member in self.members:
            if self.use_diffusion:
                anchor = tf.random.stateless_normal(
                    shape=(nlat, nlon, self.predicted_channels),
                    dtype=tf.float32,
                    seed=[member, 0]
                )
                anchor = tf.expand_dims(anchor, axis=0)
            else:
                anchor = tf.random.stateless_uniform(
                    shape=(),
                    minval=0.0,
                    maxval=1.0,
                    dtype=tf.float32,
                    seed=[member, 0]
                )
                anchor = tf.tile(
                    tf.reshape(anchor, (1, 1, 1, 1)),
                    [1, nlat, nlon, 1]
                )
            self.member_anchors[member] = anchor

        logger.info("Member anchors initialized; noise will be generated on-the-fly during inference")


    def _init_channel_stats(self, ds_norm: xr.Dataset):
        """Build flattened mean/std vectors matching channel ordering in preprocessing.

        Ordering used in make_ics preprocessing:
          1. Pressure-level vars in the order (UGRD, VGRD, VVEL, TMP, HGT, SPFH) for each level.
          2. Surface vars in the order stored in metadata['sfc_vars'] (no constants).
        Constants (e.g., LAND, OROG) were appended in preprocessing but are not predicted
        by the diffusion / deterministic heads (first 74 channels). We still include them
        at the tail of the vectors if present so slicing remains safe.
        """
        pl_vars = self.metadata['pl_vars']
        sfc_vars = self.metadata['sfc_vars']
        levels = self.metadata['levels']

        fallback_mins_raw, fallback_maxs_raw = self.get_variable_bounds()

        raw_means: List[float] = []
        raw_stds: List[float] = []
        raw_mins: List[float] = []
        raw_maxs: List[float] = []
        channel_idx = 0

        # Pressure-level variables
        for var in pl_vars:
            if var not in ds_norm.variables:
                # Fallback: fill with zeros/ones to avoid crash
                logger.warning(f"Normalization stats missing for pressure var {var}; using mean=0,std=1")
                for _ in levels:
                    raw_means.append(0.0)
                    raw_stds.append(1.0)
                    raw_mins.append(float(fallback_mins_raw[channel_idx]))
                    raw_maxs.append(float(fallback_maxs_raw[channel_idx]))
                    channel_idx += 1
                continue

            stats = ds_norm[var].values  # shape (stat, level)
            # Safeguard shape
            if stats.shape[0] < 2:
                logger.warning(f"Stats for {var} malformed; using zeros/ones")
                for _ in levels:
                    raw_means.append(0.0)
                    raw_stds.append(1.0)
                    raw_mins.append(float(fallback_mins_raw[channel_idx]))
                    raw_maxs.append(float(fallback_maxs_raw[channel_idx]))
                    channel_idx += 1
                continue

            # If level dimension differs, broadcast or truncate
            nlev_stats = stats.shape[1] if stats.ndim > 1 else 1
            for i, lvl in enumerate(levels):
                if i < nlev_stats:
                    stat_mean = float(stats[0, i])
                    stat_std = float(stats[1, i]) if float(stats[1, i]) != 0 else 1.0
                    stat_min = float(stats[2, i]) if stats.shape[0] > 2 and i < nlev_stats else float(fallback_mins_raw[channel_idx])
                    stat_max = float(stats[3, i]) if stats.shape[0] > 3 and i < nlev_stats else float(fallback_maxs_raw[channel_idx])
                    if np.isnan(stat_min):
                        stat_min = float(fallback_mins_raw[channel_idx])
                    if np.isnan(stat_max):
                        stat_max = float(fallback_maxs_raw[channel_idx])
                else:
                    stat_mean = 0.0
                    stat_std = 1.0
                    stat_min = float(fallback_mins_raw[channel_idx])
                    stat_max = float(fallback_maxs_raw[channel_idx])

                raw_means.append(stat_mean)
                raw_stds.append(stat_std)
                raw_mins.append(stat_min)
                raw_maxs.append(stat_max)
                channel_idx += 1

        # Surface variables (single value per variable)
        for var in sfc_vars:
            if var not in ds_norm.variables:
                logger.warning(f"Normalization stats missing for surface var {var}; using mean=0,std=1")
                raw_means.append(0.0)
                raw_stds.append(1.0)
                raw_mins.append(float(fallback_mins_raw[channel_idx]))
                raw_maxs.append(float(fallback_maxs_raw[channel_idx]))
                channel_idx += 1
                continue

            stats = ds_norm[var].values  # expect (stat, ...)
            if stats.shape[0] < 2:
                logger.warning(f"Stats for {var} malformed; using mean=0,std=1")
                raw_means.append(0.0)
                raw_stds.append(1.0)
                raw_mins.append(float(fallback_mins_raw[channel_idx]))
                raw_maxs.append(float(fallback_maxs_raw[channel_idx]))
                channel_idx += 1
                continue

            stat_mean = float(np.nanmean(stats[0]))
            stat_std = float(np.nanmean(stats[1])) if np.nanmean(stats[1]) != 0 else 1.0
            stat_min = float(np.nanmean(stats[2])) if stats.shape[0] > 2 else float(fallback_mins_raw[channel_idx])
            stat_max = float(np.nanmean(stats[3])) if stats.shape[0] > 3 else float(fallback_maxs_raw[channel_idx])
            if np.isnan(stat_min):
                stat_min = float(fallback_mins_raw[channel_idx])
            if np.isnan(stat_max):
                stat_max = float(fallback_maxs_raw[channel_idx])

            raw_means.append(stat_mean)
            raw_stds.append(stat_std)
            raw_mins.append(stat_min)
            raw_maxs.append(stat_max)
            channel_idx += 1

        self.raw_means = np.array(raw_means, dtype=np.float32)
        self.raw_stds = np.array(raw_stds, dtype=np.float32)
        self.raw_mins = np.array(raw_mins, dtype=np.float32)
        self.raw_maxs = np.array(raw_maxs, dtype=np.float32)

        self.channel_means = self.raw_means
        self.channel_stds = self.raw_stds
        self.channel_mins = (self.raw_mins - self.channel_means) / self.channel_stds
        self.channel_maxs = (self.raw_maxs - self.channel_means) / self.channel_stds

    def denormalize(self, output: np.ndarray) -> np.ndarray:
        """Convert model output back to physical units using stored per-channel stats.

        output: shape (1, H, W, C_out) or (H,W,C_out). We slice stats to C_out.
        """
        try:
            if output.ndim == 3:
                output = output[None, ...]
            C_out = output.shape[-1]
            means = self.channel_means[:C_out][None, None, None, :]
            stds = self.channel_stds[:C_out][None, None, None, :]
            return np.squeeze(output * stds + means)
        except Exception as e:
            logger.error(f"Error in denormalization: {e}")
            raise

    def _apply_inverse_transforms(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply inverse transforms to variables stored in log/signed-log space.

        Returns the same dataset instance with values modified in place for the affected
        variables. Safe to call when variables are absent.
        """
        try:
            for var in self.LOG_TRANSFORM_VARS:
                if var in ds.variables:
                    ds[var] = inverse_log_transform_array(ds[var])
            for var in self.NEG_LOG_TRANSFORM_VARS:
                if var in ds.variables:
                    ds[var] = inverse_neg_log_transform_array(ds[var])
        except Exception as e:
            logger.error(f"Failed applying inverse transforms: {e}")
        return ds

    def build_single_hour_dataset(
        self,
        init_datetime: datetime,
        hour: int,
        lats: np.ndarray,
        lons: np.ndarray,
        forecast_norm: np.ndarray,
    ) -> xr.Dataset:
        """Build an xarray.Dataset for a single lead hour from a normalized forecast slice.

        Args:
            init_datetime: initialization datetime
            hour: lead time in hours (int)
            lats, lons: 2D latitude/longitude arrays (Ny, Nx)
            forecast_norm: normalized model output for this hour, shape (1, Ny, Nx, C)

        Returns:
            xr.Dataset with dims (lead_time=1, time=1, [level], latitude, longitude)
        """
        t0 = time.time()

        # Denormalize to physical units
        denorm = self.denormalize(forecast_norm)
        # Ensure shape (time=1, Ny, Nx, C)
        if denorm.ndim == 3:
            denorm = denorm[None, ...]
        lead_times = [hour]
        valid_times = [init_datetime + timedelta(hours=int(t)) for t in lead_times]
        ds_hour = self.create_xarray_dataset(init_datetime, lead_times, lats, lons, denorm)

        # Inject constants if present in preprocessed NPZ (repeat across lead_time length 1)
        for cname in ["LAND", "OROG"]:
            raw_key = f"{cname}_raw"
            if hasattr(self.data_loader_hrrr, "data") and raw_key in self.data_loader_hrrr.data.files and cname not in ds_hour:
                cvals = self.data_loader_hrrr.data[raw_key].astype(np.float32)
                const_4d = np.tile(cvals[None, None, :, :], (len(lead_times), 1, 1, 1))
                ds_hour[cname] = xr.DataArray(
                    const_4d,
                    dims=("lead_time", "time", "latitude", "longitude"),
                    coords={
                        "lead_time": ("lead_time", lead_times),
                        "time": ("time", valid_times),
                        "latitude": (("latitude", "longitude"), lats),
                        "longitude": (("latitude", "longitude"), lons),
                    },
                    name=cname,
                )
                logger.debug(f"Injected constant field {cname} for hour {hour}")

        # Apply inverse transforms to recover physical units
        ds_hour = self._apply_inverse_transforms(ds_hour)

        # compute diagnostics
        ds_hour = compute_diagnostics(ds_hour)

        # Apply CF-compliant long_name and units to all variables
        ds_hour = apply_cf_attributes(ds_hour, init_datetime=init_datetime)

        build_time = time.time() - t0
        logger.info(f"Build output dataset in {build_time:.3f}s")

        return ds_hour

    def write_single_hour_netcdf(
        self,
        init_datetime: datetime,
        hour: int,
        ds_hour: xr.Dataset,
        output_dir: str,
        member: Union[int, str],
    ) -> None:
        """Write a NetCDF file for a single lead time.

        Returns the output file path.
        """
        t0 = time.time()

        init_year = self.metadata['init_year']
        init_month = self.metadata['init_month']
        init_day = self.metadata['init_day']
        init_hh = self.metadata['init_hh']
        date_str = f"{init_year}{init_month}{init_day}/{init_hh}"
        utils.make_directory(f"{output_dir}/{date_str}")
        outdir = Path(f"{output_dir}/{date_str}")
        outdir.mkdir(parents=True, exist_ok=True)
        mem_str = str(member)
        if mem_str not in {"avg", "spr"}:
            mem_str = f"m{int(member):02d}"
        nc_path = outdir / f"hrrrcast_{mem_str}_f{hour:02d}.nc"

        # CF-compliant encoding (fill values, CF time units, no _FillValue on
        # coordinates per CF 2.5.1) comes from upstream and is the base.
        encoding = get_cf_encoding(ds_hour, init_datetime)

        # This fork's compression and quantization is then layered on top of the DATA
        # VARIABLES only. The two are orthogonal: CF governs metadata and fill values,
        # these settings govern how the bytes are stored, and both are needed.
        #
        # Output is ~1.36 GB per lead hour uncompressed at the full 1059x1799
        # grid. Measured on a real f01 file from a 12h run:
        #   complevel=1 -> 879 MB (1.54x), 11.5s;  complevel=6 -> 871 MB (1.56x), 18.8s
        # Lossless deflate does little on float32 continuous fields, and level 1
        # captures essentially all of it, so there is no reason to go higher.
        # Quantizing first is what actually shrinks these files:
        #   lsd=2 -> 355 MB (3.82x), max abs error 0.0039 in native units
        #   lsd=3 -> 460 MB (2.95x), max abs error 0.00049
        # That is lossy, so it is opt-in and off by default.
        # complevel=0 with no lsd reproduces the original uncompressed behavior.
        if self.nc_complevel > 0 or self.nc_least_significant_digit is not None:
            per_var = {}
            if self.nc_complevel > 0:
                per_var["zlib"] = True
                per_var["complevel"] = self.nc_complevel
            if self.nc_least_significant_digit is not None:
                per_var["least_significant_digit"] = self.nc_least_significant_digit
            # Only variables get_cf_encoding() already listed, which deliberately
            # excludes the scalar `grid_mapping` container: compressing it is
            # pointless and quantizing it would be wrong. Coordinates are likewise
            # left alone, since least_significant_digit on the int32 `level` axis or
            # on the CF time axis would corrupt them.
            for name in ds_hour.data_vars:
                if name in encoding:
                    encoding[name].update(per_var)
        ds_hour.to_netcdf(nc_path, encoding=encoding)

        write_time = time.time() - t0
        size_mb = nc_path.stat().st_size / 1e6
        logger.info(
            f"Wrote NetCDF in {write_time:.3f}s ({size_mb:.1f} MB, "
            f"complevel={self.nc_complevel}, lsd={self.nc_least_significant_digit}) : {nc_path}"
        )

        # Stream off-box as soon as the file is closed, so a run that dies at
        # hour 20 still delivered hours 0-19.
        #
        # The return value is RECORDED, not discarded. S3Uploader.upload() returns
        # False after exhausting its retries; ignoring that made a run whose every
        # upload was refused (an IAM policy missing the output prefix) still exit 0
        # and report success, having delivered nothing. Under an hourly schedule that
        # produces empty cycles with a green status, which is the worst failure mode
        # available. The count is checked once at the end of the rollout, so delivery
        # failures do not abort a forecast that is still producing useful local
        # output -- see the failed_uploads check after the drain.
        if self.s3_uploader is not None:
            if not self.s3_uploader.upload(nc_path, output_dir):
                self.failed_uploads.append(nc_path.name)

    def write_single_hour_grib2(
        self,
        init_datetime: datetime,
        hour: int,
        ds_hour: xr.Dataset,
        output_dir: str,
        member: Union[int, str],
    ) -> None:
        """Write a GRIB2 file for a single lead time using Netcdf2Grib.

        Netcdf2Grib iterates over available time points; with a single-hour dataset,
        it will produce only the requested f{hour:02d} product.
        """
        t0 = time.time()

        init_year = self.metadata['init_year']
        init_month = self.metadata['init_month']
        init_day = self.metadata['init_day']
        init_hh = self.metadata['init_hh']
        date_str = f"{init_year}{init_month}{init_day}/{init_hh}"
        utils.make_directory(f"{output_dir}/{date_str}")
        outdir = Path(f"{output_dir}/{date_str}")
        outdir.mkdir(parents=True, exist_ok=True)
        mem_str = str(member)
        if mem_str not in {"avg", "spr"}:
            mem_str = f"m{int(member):02d}"
        grib2_path = outdir / f"hrrrcast.{mem_str}.t{init_datetime.hour:02d}z.pgrb2.f{hour:02d}"

        converter = Netcdf2Grib()
        # Ensure ds_hour has exactly one lead_time equal to 'hour'
        if 'lead_time' in ds_hour.coords:
            try:
                # If needed, overwrite lead_time coord to match the requested hour
                ds_hour = ds_hour.assign_coords(lead_time=("lead_time", [hour]))
            except Exception:
                pass
        converter.save_grib2(init_datetime, ds_hour, str(grib2_path))

        write_time = time.time() - t0
        logger.info(f"Wrote GRIB2 in {write_time:.3f}s : {grib2_path}")

        if self.s3_uploader is not None:
            if not self.s3_uploader.upload(grib2_path, output_dir):
                self.failed_uploads.append(grib2_path.name)
            # nc2grib writes a wgrib2 .idx sidecar next to the GRIB2 when wgrib2
            # is available. Ship it too; a GRIB2 delivered without its index is a
            # broken pair for consumers that expect one.
            idx_path = Path(f"{grib2_path}.idx")
            if idx_path.is_file():
                if not self.s3_uploader.upload(idx_path, output_dir):
                    self.failed_uploads.append(idx_path.name)

    def get_variable_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (mins, maxs) numpy arrays each shaped (output_channels,).
        """
        raw_bounds = {
            "UGRD":    (-120, 120),
            "VGRD":    (-120, 120),
            "VVEL":    (-30, 30),
            "TMP":     (180, 340),
            "HGT":     (-600, 20000),
            "SPFH":    (0, 0.05),
            "PRES":    (50000, 110000),
            "MSLMA":   (50000, 110000),
            "REFC":    (0, 80),
            "T2M":     (180, 340),
            "UGRD10M": (-100, 100),
            "VGRD10M": (-100, 100),
            "UGRD80M": (-100, 100),
            "VGRD80M": (-100, 100),
            "D2M":     (180, 340),
            "TCDC":    (0, 100),
            "LCDC":    (0, 100),
            "MCDC":    (0, 100),
            "HCDC":    (0, 100),
            "VIS":     (0, 100000),
            "APCP":    (0, 500),
            "HGTCC":   (0, 20000),
            "CAPE":    (0, 7000),
            "CIN":     (-2000, 0),
        }
        mins = []
        maxs = []
        num_levels = len(self.metadata['levels'])
        # Merge 3D and 2D targets into a single loop
        for i, var in enumerate(raw_bounds):
            vmin, vmax = raw_bounds[var]
            if var in self.LOG_TRANSFORM_VARS:
                vmin = np.log1p(vmin)
                vmax = np.log1p(vmax)
            elif var in self.NEG_LOG_TRANSFORM_VARS:
                vmin = np.sign(vmin) * np.log1p(abs(vmin))
                vmax = np.sign(vmax) * np.log1p(abs(vmax))
            # Repeat for each pressure level if 3D, else once
            n_levels = num_levels if i < 6 else 1
            for _ in range(n_levels):
                mins.append(vmin)
                maxs.append(vmax)
        return np.array(mins, dtype=np.float32), np.array(maxs, dtype=np.float32)


    @staticmethod
    def compute_time_features(init_times_np, lead_times_np):
        # Ensure inputs are array-like
        if not isinstance(init_times_np, (list, np.ndarray)):
            init_times_np = [init_times_np]
        if lead_times_np is not None and not isinstance(lead_times_np, (list, np.ndarray)):
            lead_times_np = [lead_times_np]
        # compute valid times
        time_coord = pd.to_datetime(init_times_np)
        if lead_times_np is not None:
            time_coord += pd.to_timedelta(lead_times_np, unit='h')
        # compute cyclical features
        hours = pd.DatetimeIndex(time_coord).hour.astype(np.float32)
        doy = pd.DatetimeIndex(time_coord).dayofyear.astype(np.float32)
        # version masks
        v4 = (time_coord >= np.datetime64("2021-03-23T00")).astype(np.float32)
        v3 = ((time_coord >= np.datetime64("2018-07-12T00")) & (time_coord < np.datetime64("2021-03-23T00"))).astype(np.float32)
        gfs_v15 = ((time_coord >= np.datetime64("2019-06-01T00")) & (time_coord < np.datetime64("2021-03-23T00"))).astype(np.float32)
        # Stack features into shape [B, 7]
        features = np.stack([
            np.sin(2 * np.pi * hours / 24.0).astype(np.float32),
            np.cos(2 * np.pi * hours / 24.0).astype(np.float32),
            np.sin(2 * np.pi * doy / 365.0).astype(np.float32),
            np.cos(2 * np.pi * doy / 365.0).astype(np.float32),
            v4.astype(np.float32),
            v3.astype(np.float32),
            gfs_v15.astype(np.float32),
        ], axis=-1)
        return features

    def date_encoding_tensor(self, init_times_np, lead_times_np):
        """Compute cyclical time encodings and HRRR version masks"""

        def get_encoding_tensor(enc):
            enc = tf.cast(enc, dtype=tf.float32)
            batch_size, lat, lon = tf.shape(enc)[0], self.input_shape[1], self.input_shape[2]
            enc = tf.reshape(enc, (batch_size, 1, 1, 7))
            enc = tf.broadcast_to(enc, (batch_size, lat, lon, 7))
            return enc

        enc = self.compute_time_features(init_times_np, lead_times_np)
        enc = get_encoding_tensor(enc)
        return enc

    def _next_member_noise(self, member: int, hour: int) -> tf.Tensor:
        """Advance a member's noise state by one hour and return it."""
        next_noise = make_noise(self.member_anchors[member], hour, member, self.noise_rho)
        self.member_anchors[member] = next_noise
        return next_noise

    def predict(self, model: ForecastModel, X: tf.Tensor, members: Union[int, List[int]], hour: int) -> tf.Tensor:
        """Predict using diffusion or CRPS model.
        
        Args:
            model: ForecastModel to use for predictions
            X: Input tensor, shape (batch_size, H, W, C)
            members: Single member ID (int) for batch_size=1, or list of member IDs for batch_size>1
                    Length must match batch_size of X.
            hour: Lead time hour to retrieve noise for
        
        Returns:
            Predicted tensor of shape (batch_size, H, W, predicted_channels)
        """
        if self.use_diffusion:
            num_output_channels = self.predicted_channels
            start = self.predicted_channels + self.gfs_channels

            # Advance each member's noise state in-order (hour-to-hour correlation)
            Xn_list = [self._next_member_noise(m, hour) for m in members]
            Xn = tf.concat(Xn_list, axis=0)  # (batch_size, H, W, C)

            def build_diffusion_input(
                X_in: tf.Tensor,
                X_noisy: tf.Tensor,
                step_t: tf.Tensor,
            ) -> tf.Tensor:
                step_encoding = tf.fill(
                    tf.concat([tf.shape(X_in)[:-1], [1]], axis=0),
                    tf.cast(step_t / NUM_DIFFUSION_STEPS, X_in.dtype),
                )
                X_out = tf.concat(
                    [
                        X_in[:, :, :, :start],
                        X_noisy,
                        X_in[:, :, :, start + num_output_channels :-2],
                        step_encoding,
                        X_in[:, :, :, -1:],
                    ],
                    axis=-1,
                )
                return X_out

            def predict_x0_and_epsilon(
                X_in: tf.Tensor,
                X_noisy: tf.Tensor,
                step_t: tf.Tensor,
            ) -> tuple[tf.Tensor, tf.Tensor]:
                X_model = build_diffusion_input(X_in, X_noisy, step_t)
                x_0_pred = model.predict(X_model)
                eps = compute_epsilon(X_noisy, x_0_pred, step_t)
                return x_0_pred, eps

            # iterate over diffusion steps
            prev_x0 = None
            prev_h = None
            for t_ in range(NUM_INFERENCE_STEPS):
                ti = NUM_INFERENCE_STEPS - 1 - t_
                t = INFERENCE_STEPS[ti]

                # compute predicted noise epsilon at this step
                x0_t, epsilon_t = predict_x0_and_epsilon(X, Xn, t)

                # Choose diffusion sampler
                if ti == 0:
                    Xn = x0_t
                else:
                    if self.diffusion_sampler == "ddim-heun":
                        x_tm1_pred = ddim(Xn, epsilon_t, ti, seed=members, eta=0.0)

                        tm1 = tf.gather(list(INFERENCE_STEPS), ti - 1)
                        _, epsilon_tm1 = predict_x0_and_epsilon(X, x_tm1_pred, tm1)

                        Xn = ddim_heun(Xn, epsilon_t, epsilon_tm1, ti, seed=members, eta=0.0)
                    elif self.diffusion_sampler == "dpmpp-2m":
                        Xn, prev_x0, prev_h = dpmpp_2m(Xn, x0_t, ti, prev_x0=prev_x0, prev_h=prev_h)
                    elif self.diffusion_sampler == "ddim":
                        Xn = ddim(Xn, epsilon_t, ti, seed=members, eta=0.0)
                    else:
                        Xn = ddpm(Xn, epsilon_t, ti, seed=members)

            return Xn
        else:
            # CRPS branch also advances per-member noise state in-order
            Xn_list = [self._next_member_noise(m, hour) for m in members]
            Xn = tf.concat(Xn_list, axis=0)  # (batch_size, H, W, C)
            X = tf.concat(
                [
                    X[:, :, :, :-2],
                    Xn,
                    X[:, :, :, -1:]
                ],
                axis=-1,
            )
            y = model.predict(X)
            return y

    def autoregressive_rollout(self, initial_input: np.ndarray, forcing_input: np.ndarray, model: ForecastModel, 
                             target_hour: int,
                             output_dir: Optional[str] = None,
                             init_datetime: Optional[datetime] = None) -> Dict[int, Dict]:
        """Perform greedy autoregressive rollout with overlapped I/O.

        Persist single-hour NetCDF and GRIB2 files for each lead hour, including f00
        representing the initial state. I/O is done in background threads to overlap
        with forecasting.
        
        Uses correlated noise for each lead time: different noise vectors per lead hour
        but with temporal correlation (rho=0.9 by default) to maintain coherence
        across the forecast period.
        """
        logger.info(f"Starting autoregressive rollout for {target_hour} hours")
        
        # Initial input (updated during rollout)
        X = tf.convert_to_tensor(initial_input, dtype=tf.float32)
        
        # Track state_from_hour per member
        state_from_hour = {
            member: tf.identity(X[0:1, :, :, :self.predicted_channels]) for member in self.members
        }

        start_pred_noise = self.predicted_channels + self.gfs_channels

        lats, lons = self.data_loader_hrrr.get_coordinates()

        # Local helpers to build hourly datasets and fan them out to file writers.
        def write_hour_nc(hour: int, ds_hour: xr.Dataset, member: int) -> None:
            """Write NetCDF for a given hour."""
            try:
                self.write_single_hour_netcdf(init_datetime, hour, ds_hour, output_dir, member)
                logger.debug(f"Completed writing NetCDF hour {hour} for member {member}")
            except Exception as e:
                logger.error(f"Failed writing NetCDF hour {hour} for member {member}: {e}")

        def write_hour_grib2(hour: int, ds_hour: xr.Dataset, member: int) -> None:
            """Write GRIB2 for a given hour."""
            try:
                self.write_single_hour_grib2(init_datetime, hour, ds_hour, output_dir, member)
                logger.debug(f"Completed writing GRIB2 hour {hour} for member {member}")
            except Exception as e:
                # Recorded, not just logged. grib2_executor.submit()'s future is never
                # awaited, so an exception raised here would otherwise vanish into an
                # unexamined future and the run would report success having written no
                # GRIB2 at all. Same failure mode as the discarded upload result.
                logger.error(f"Failed writing GRIB2 hour {hour} for member {member}: {e}")
                self.failed_grib2.append(f"f{hour:02d} member {member}")

        def build_and_submit_hour_outputs(hour: int, data: np.ndarray, member: int) -> None:
            """Build the dataset in a worker, then submit both file writes."""
            if self.output_hours is not None and hour not in self.output_hours:
                return
            try:
                ds_hour = self.build_single_hour_dataset(init_datetime, hour, lats, lons, data)
            except Exception as e:
                logger.error(f"Failed building dataset for hour {hour} member {member}: {e}")
                return

            nc_executor.submit(write_hour_nc, hour, ds_hour, member)
            if self.write_grib2:
                grib2_executor.submit(write_hour_grib2, hour, ds_hour, member)

        build_executor = ThreadPoolExecutor(max_workers=len(self.members))
        nc_executor = ThreadPoolExecutor(max_workers=1) # Serial: HDF5 not thread safe
        grib2_executor = ThreadPoolExecutor(max_workers=len(self.members)) # Parallel: locked in save_grib2()
        build_futures = []

        # Semaphore to limit pending build tasks and prevent OOM from too many queued forecasts
        max_pending_builds = 8
        build_semaphore = threading.Semaphore(max_pending_builds)
        
        def submit_with_backpressure(hour: int, data_tensor: tf.Tensor, member: int) -> None:
            """Submit build task with semaphore-based backpressure control.
            The tensor->numpy copy happens AFTER semaphore acquisition to prevent
            memory buildup when the queue is full.
            """
            build_semaphore.acquire()  # Block if queue is full
            data = data_tensor.numpy()  # Copy from GPU to CPU only after semaphore acquired
            future = build_executor.submit(build_and_submit_hour_outputs, hour, data, member)
            future.add_done_callback(lambda _: build_semaphore.release())  # Release when done
            build_futures.append(future)

        # Write out hour 0 (f00) products representing the initial state for each member
        for member in self.members:
            submit_with_backpressure(0, state_from_hour[member], member)

        # phase shift of GFS forcing input
        #
        # members_sorted must be the ACTUAL member IDs being run, not range(n).
        # phase_angle is looked up below as phase_angle[member], so keying it by
        # position broke any run whose member IDs are not 0..n-1: `--members 1` with
        # num_members=1 built {0: 0.0} and then raised KeyError(1) at the first
        # rollout hour. Because str(KeyError(1)) is just "1", the log read
        # "Forecast failed: 1", which looks like an exit status rather than a missing
        # dict key and is genuinely hard to place.
        #
        # self.num_members is still used for the phase spacing, so a member subset of
        # a larger planned ensemble keeps the spread it would have had.
        num_members = self.num_members
        members_sorted = sorted(self.members)
        half_count = (num_members // 2 - ((num_members + 1) % 2)) # Half count for symmetry
        step = 1.0 / half_count if half_count > 0 else 0.0
        seq = []
        seq.append(0.0)  # Always include zero phase shift for the first member
        if num_members % 2 == 0:
            seq.append(0.0)
        for i in range(half_count):
            seq.append(step * (i + 1))  # Positive phase shifts
            seq.append(-step * (i + 1))  # Negative phase shifts
        # Index seq by the member's OWN id, not by its position in this run's subset.
        # seq has exactly num_members entries by construction, and a member's phase is
        # a property of the member, so `--members 3` out of 5 must get the same phase
        # it would get in a full 5-member run. Keying by subset position would make a
        # single-member rerun silently disagree with the ensemble it came from.
        if any(m >= num_members for m in members_sorted):
            raise ValueError(
                f"member id(s) {[m for m in members_sorted if m >= num_members]} are "
                f">= --num_members ({num_members}); phase shifts are only defined for "
                f"ids 0..{num_members - 1}. Raise --num_members to match.")
        phase_angle = {member: seq[member] for member in members_sorted}

        # rollout hour
        rollout_hour = 6

        # Process all hourly steps
        # The loop is wrapped so a mid-rollout failure still flushes whatever
        # finished. Otherwise output is silently lost: builder threads submit their
        # write to nc_executor, and once the main thread has raised, interpreter
        # shutdown makes ThreadPoolExecutor.submit() fail with "cannot schedule new
        # futures after interpreter shutdown". Seen on a GPU OOM at hour 1 -- the f00
        # dataset finished building 2s *after* the exception and was discarded, so a
        # run with valid hour-0 output delivered nothing. Per-hour streaming is only
        # worth having if partial results survive a failure.
        try:
            for hour in range(1, target_hour + 1):
                from_hour = ((hour - 1) // rollout_hour) * rollout_hour
                step = hour - from_hour
                logger.info(f"Forecasting hour {hour:2d}: from hour {from_hour:2d} using step {step}h")

                # date encoding
                date_encoding = self.date_encoding_tensor(init_datetime, hour)
                lead_encoding = tf.fill(
                    tf.concat([tf.shape(X)[:-1], [1]], axis=0),
                    tf.cast(step / 6.0, tf.float32),
                )

                # NOTE: forcing_input no longer includes hour 0, so hour=1 corresponds to index 0
                X_base = tf.concat(
                    [
                        X[:, :, :, start_pred_noise:-(7 + 2)],
                        date_encoding,
                        X[:, :, :, -2:-1],
                        lead_encoding,
                    ],
                    axis=-1,
                )

                # Process members in batches
                batch_size = self.batch_size
                for batch_start in range(0, len(self.members), batch_size):
                    batch_end = min(batch_start + batch_size, len(self.members))
                    batch_members_list = self.members[batch_start:batch_end]
                
                    # Collect inputs for this batch of members
                    batch_X_members = []
                
                    for member in batch_members_list:
                        # apply phase shift to forcing input index for this member
                        phase_width = from_hour // 12
                        phase_shift = round(phase_width * phase_angle[member])
                        forcing_idx = hour - 1 + phase_shift
                        forcing_idx = np.clip(forcing_idx, 0, forcing_input.shape[0] - 1)

                        # Assemble input for this member
                        X_member = tf.concat(
                            [
                                state_from_hour[member],
                                forcing_input[forcing_idx:forcing_idx + 1, :, :, :],
                                X_base,
                            ],
                            axis=-1,
                        )
                        batch_X_members.append(X_member)
                
                    # Stack batch inputs
                    X_batch = tf.concat(batch_X_members, axis=0)
                
                    # Predict next-hour fields for entire batch using hour-specific noise
                    t0 = time.time()
                    y_batch = self.predict(model, X_batch, batch_members_list, hour=hour)
                    predict_time = time.time() - t0
                    logger.info(f"Hour {hour}, batch {batch_start//batch_size + 1}: predict took {predict_time:.3f}s")
                
                    y_batch = tf.clip_by_value(
                        y_batch,
                        self.channel_mins[:y_batch.shape[-1]],
                        self.channel_maxs[:y_batch.shape[-1]]
                    )
                
                    # Process outputs for each member in batch
                    for batch_idx, member in enumerate(batch_members_list):
                        y = y_batch[batch_idx:batch_idx+1]
                    
                        # When we reach a 6-hour boundary, update the reference 
                        # state for this member
                        if hour % rollout_hour == 0:
                            state_from_hour[member] = y

                        # Store the output for this hour and member, then submit write task
                        submit_with_backpressure(hour, y, member)  # Pass tensor, not numpy array

        finally:
            # Wait for all background work to complete. Build tasks must drain first so
            # they cannot submit into executors that are already shutting down.
            logger.info(f"Waiting for {len(build_futures)} background build operations to complete...")
            for future in as_completed(build_futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Background build operation failed: {e}")
            build_executor.shutdown(wait=True)
            nc_executor.shutdown(wait=True)
            grib2_executor.shutdown(wait=True)
            logger.info("All background output operations completed")

        # Delivery is part of the job. Checked here, after the drain, rather than at
        # the first failure: a forecast that is still computing correctly should finish
        # and leave its local files behind, and per-hour streaming means an outage in
        # the middle of a run may resolve by the end. But the process must NOT exit 0
        # having delivered nothing, which is what happened when this return value was
        # ignored -- an IAM policy missing the output prefix produced two runs that
        # reported success with zero objects in S3.
        if self.failed_grib2:
            n = len(self.failed_grib2)
            sample = ", ".join(self.failed_grib2[:5])
            raise RuntimeError(
                f"{n} GRIB2 file(s) requested by --grib2 were not written "
                f"(first: {sample}{', ...' if n > 5 else ''}). NetCDF output is "
                "unaffected; drop --grib2 if GRIB2 is not required.")

        if self.failed_uploads:
            n = len(self.failed_uploads)
            sample = ", ".join(self.failed_uploads[:5])
            raise RuntimeError(
                f"{n} file(s) were written locally but could not be uploaded to S3 "
                f"(first: {sample}{', ...' if n > 5 else ''}). The forecast itself "
                "completed; this is a delivery failure. Local copies were retained.")

        logger.info("Autoregressive rollout completed")

    def create_xarray_dataset(self, init_datetime: datetime, lead_times: List[int],
                            lats: np.ndarray, lons: np.ndarray, data: np.ndarray) -> xr.Dataset:
        """Convert numpy array to xarray.Dataset.
        
        Args:
            init_datetime: Forecast initialization time
            lead_times: List of lead times in hours (e.g., [0, 1, 2] for f00, f01, f02)
            lats, lons: 2D latitude/longitude arrays (Ny, Nx)
            data: Forecast data with shape (n_times, Ny, Nx, C)
        
        Returns:
            xr.Dataset with CF-1.6 compliant forecast structure with separate time
            and lead_time dimensions for flexibility in multi-hour output files.
        """
        data_vars = {}
        var_index = 0

        pl_vars = self.metadata['pl_vars']
        sfc_vars = self.metadata['sfc_vars']
        # Cast pressure levels to int32 (CF-1.6 disallows int64 for coordinates).
        levels = np.asarray(self.metadata['levels'], dtype=np.int32)

        # Compute valid times for all forecast steps
        valid_times = [init_datetime + timedelta(hours=int(t)) for t in lead_times]

        # Pressure-level variables: (lead_time, time, level, latitude, longitude)
        # CF conventions: non-standard dimensions (lead_time) come before T, Z, Y, X
        for pl_var in pl_vars:
            pl_data = np.transpose(data[..., var_index:var_index+len(levels)], (0, 3, 1, 2))
            # Add lead_time dimension at axis 0 (leftmost position)
            pl_data = np.expand_dims(pl_data, axis=0)
            data_vars[pl_var] = xr.DataArray(
                pl_data,
                dims=("lead_time", "time", "level", "latitude", "longitude"),
                coords={
                    "lead_time": ("lead_time", lead_times),
                    "time": ("time", valid_times),
                    "level": ("level", levels),
                    "latitude": (("latitude", "longitude"), lats),
                    "longitude": (("latitude", "longitude"), lons),
                },
                name=pl_var
            )
            var_index += len(levels)

        # Surface variables: (lead_time, time, latitude, longitude)
        # CF conventions: non-standard dimensions (lead_time) come before T, Y, X
        for sfc_var in sfc_vars:
            sfc_data = data[..., var_index]
            # Add lead_time dimension at axis 0 (leftmost position)
            sfc_data = np.expand_dims(sfc_data, axis=0)
            data_vars[sfc_var] = xr.DataArray(
                sfc_data,
                dims=("lead_time", "time", "latitude", "longitude"),
                coords={
                    "lead_time": ("lead_time", lead_times),
                    "time": ("time", valid_times),
                    "latitude": (("latitude", "longitude"), lats),
                    "longitude": (("latitude", "longitude"), lons),
                },
                name=sfc_var
            )
            var_index += 1

        ds = xr.Dataset(data_vars)

        # CF-1.6 §4.4.1: scalar forecast_reference_time (model initialization time).
        ds = ds.assign_coords(
            forecast_reference_time=xr.DataArray(
                np.datetime64(init_datetime, "ns"),
            ),
        )

        return ds
    
    def run_forecast(self, model: ForecastModel, lead_hours: int, model_input: np.ndarray, output_dir: str = "./"):
        """Run the forecasting pipeline with per-hour streaming outputs. Requires precomputed model_input.

        This function now avoids building a single multi-hour xarray Dataset and avoids bulk NetCDF/GRIB2 writes.
        Instead, per-hour NetCDF/GRIB2 files are written during the autoregressive rollout.
        """
        try:
            init_datetime = self.data_loader_hrrr.get_init_datetime()

            logger.info(f"Running forecast for {init_datetime} with {lead_hours} hour lead time")
            logger.info(f"Model input shape: {model_input.shape}")
            logger.info(self.metadata)

            # Run autoregressive forecast and write per-hour outputs to disk
            self.autoregressive_rollout(
                model_input,
                self.data_loader_gfs.get_model_input(),
                model,
                lead_hours,
                output_dir=output_dir,
                init_datetime=init_datetime,
            )
            logger.info("Forecast completed successfully (per-hour outputs written during rollout)")
        except Exception as e:
            logger.error(f"Forecast failed: {e}")
            raise

def run_weather_forecast(forecaster: WeatherForecaster, model: ForecastModel, lead_hours: int, model_input: np.ndarray, output_dir: str):
    """Run forecast for a single member. Requires precomputed model_input."""
    try:
        forecaster.run_forecast(model, lead_hours, model_input, output_dir)
    except Exception as e:
        logger.error(f"Forecast failed : {e}")
        raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Weather Forecast Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("model_path", help="Path to the trained model")
    parser.add_argument('inittime', help='Forecast initialization time in format YYYY-MM-DDTHH (e.g., "2024-05-06T23")')
    parser.add_argument("lead_hours", type=int, help="Lead time in hours")
    parser.add_argument("--num_members", type=int, default=1, help="Number of ensemble members to generate")
    parser.add_argument("--members", nargs='+', required=True, help="List of ensemble member IDs (e.g., 0 1 2 or 0,1,2)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for model inference")
    parser.add_argument("--no_diffusion", default=False, action="store_true", help="Turn off diffusion")
    parser.add_argument("--base_dir", default="./", help="Base directory for input preprocessed files")
    parser.add_argument("--output_dir", default="./", help="Output directory for forecast files")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--pmm_alpha", type=float, default=0.7,
                        help="Nudge factor toward PMM mean for member outputs (0..1)")
    parser.add_argument("--noise_rho", type=float, default=0.9,
                        help="Noise blend/correlation parameter (0..1)")
    # GRIB2 is opt-in. NetCDF is the deliverable everything downstream reads, GRIB2
    # roughly doubles write time and output volume, and no consumer in this pipeline
    # needs it, so paying for it by default is wrong. --no_grib2 is still accepted
    # because run_cycle.sh, domain_test.sh and other callers pass it; it is now
    # redundant rather than an error.
    parser.add_argument("--grib2", default=False, action="store_true",
                        help="Also write GRIB2 alongside the NetCDF (requires wgrib2; "
                             "roughly doubles output volume). Off by default")
    parser.add_argument("--no_grib2", default=False, action="store_true",
                        help="Deprecated and redundant: GRIB2 is off unless --grib2 is given. "
                             "Accepted for compatibility; overrides --grib2 if both appear")
    parser.add_argument("--nc_complevel", type=int, default=0, choices=range(0, 10),
                        metavar="0-9",
                        help="NetCDF zlib compression level; 0 writes uncompressed")
    parser.add_argument("--nc_least_significant_digit", type=int, default=None, metavar="N",
                        help="Lossy: quantize NetCDF variables to N decimal digits before "
                             "compression (much smaller files; lsd=2 gives ~3.8x at max abs "
                             "error 0.004 in native units). Off by default")
    parser.add_argument("--wait_for_input", default=None, metavar="PATH",
                        help="Load the model first, then block until PATH exists before reading the "
                             "preprocessed npz files. Lets the forecast's TensorFlow import and "
                             "model load (~5 min) overlap with the input-preparation stages")
    parser.add_argument("--wait_timeout", type=int, default=3600, metavar="SECONDS",
                        help="Give up waiting for --wait_for_input after this long")
    parser.add_argument("--s3_output", default=None, metavar="s3://BUCKET/PREFIX",
                        help="Upload each output file to this S3 prefix as it is written")
    parser.add_argument("--purge_local", default=False, action="store_true",
                        help="Delete the local copy after a confirmed S3 upload (requires --s3_output)")
    parser.add_argument("--output_hours", default=None, metavar="SPEC",
                        help="Only build/write/upload these lead hours; the rollout still computes "
                             "every hour in between (autoregressive state needs it), only the I/O "
                             "is skipped. SPEC is 'start:step:end' (e.g. '0:3:24' for f00,f03,...,f24) "
                             "or an explicit comma list (e.g. '0,6,12,18,24'). Default: every hour")

    args = parser.parse_args()
    if args.purge_local and not args.s3_output:
        parser.error("--purge_local requires --s3_output (it would otherwise just discard the outputs)")
    if args.output_hours is not None:
        try:
            args.output_hours = parse_output_hours(args.output_hours)
        except ValueError as e:
            parser.error(str(e))
    return args


def parse_output_hours(spec: str) -> Set[int]:
    """Parse --output_hours SPEC into a set of lead hours.

    'start:step:end' (all ints, end inclusive) or a comma list of ints.
    """
    if spec.count(":") == 2:
        start_s, step_s, end_s = spec.split(":")
        try:
            start, step, end = int(start_s), int(step_s), int(end_s)
        except ValueError:
            raise ValueError(f"--output_hours '{spec}': 'start:step:end' parts must be integers")
        if step <= 0:
            raise ValueError(f"--output_hours '{spec}': step must be positive")
        return set(range(start, end + 1, step))
    try:
        return {int(p.strip()) for p in spec.split(",") if p.strip() != ""}
    except ValueError:
        raise ValueError(f"--output_hours '{spec}': not a valid 'start:step:end' or comma list")


def wait_for_input_sentinel(path: str, timeout_s: int) -> None:
    """Block until `path` appears, or raise once timeout_s has elapsed.

    Used with --wait_for_input so the expensive, data-independent part of startup
    (TensorFlow import, GPU init, loading the 50M-parameter model: ~5 min measured)
    can run while the input stages are still preparing the npz files. The caller
    creates the sentinel only after those stages have succeeded, so its presence
    means the inputs are complete -- checking for the npz files directly would race
    against them being written.
    """
    waited = 0.0
    interval = 2.0
    if os.path.exists(path):
        logger.info(f"Inputs already present ({path}); no wait needed")
        return
    logger.info(f"Model is loaded; waiting for inputs to be ready ({path}, timeout {timeout_s}s)")
    t0 = time.time()
    while not os.path.exists(path):
        if waited >= timeout_s:
            raise TimeoutError(
                f"Waited {timeout_s}s for {path} and it never appeared. The input stages "
                "either failed or are slower than expected."
            )
        time.sleep(interval)
        waited = time.time() - t0
        if int(waited) % 60 < interval:
            logger.info(f"still waiting for inputs ({waited:.0f}s elapsed)")
    logger.info(f"Inputs ready after {waited:.1f}s of waiting")


def main():
    """Main execution function."""
    global logger
    args = parse_arguments()
    logger = setup_logging(args.log_level)

    try:
        # Build the S3 uploader first: if delivery is configured but broken,
        # fail now rather than after hours of GPU time.
        uploader = s3io.make_uploader(args.s3_output, purge_local=args.purge_local)

        # Parse members argument (support space/comma separated and ranges like 0-2)
        def expand_member_arg(m):
            result = []
            for part in m.split(","):
                part = part.strip()
                if "-" in part:
                    start, end = part.split("-")
                    result.extend(list(range(int(start), int(end)+1)))
                elif part != "":
                    result.append(int(part))
            return result
        members = []
        for m in args.members:
            members.extend(expand_member_arg(m))
        members = sorted(set(members))  # Remove duplicates and sort

        # Load preprocessed data and model ONCE
        init_datetime, init_year, init_month, init_day, init_hh = utils.validate_datetime(args.inittime)
        date_str = f"{init_year}{init_month}{init_day}/{init_hh}"
        filedate_str = f"{init_year}{init_month}{init_day}_{init_hh}"
        hrrr_preprocessed_file = f"{args.base_dir}/{date_str}/hrrr_{filedate_str}.npz"
        gfs_preprocessed_file = f"{args.base_dir}/{date_str}/gfs_{filedate_str}.npz"
        # Model FIRST, data second. Loading the model needs no input data and takes
        # ~5 min (TF import, GPU init, 50M-parameter deserialization), so doing it
        # before the npz reads is what allows --wait_for_input to hide that cost
        # behind the input-preparation stages. Ordering is otherwise irrelevant.
        model = ForecastModel(args.model_path)

        if args.wait_for_input:
            wait_for_input_sentinel(args.wait_for_input, args.wait_timeout)

        data_loader_hrrr = PreprocessedDataLoader(hrrr_preprocessed_file)
        data_loader_gfs = PreprocessedDataLoader(gfs_preprocessed_file)

        # Precompute model_input ONCE
        model_input_hrrr = data_loader_hrrr.get_model_input()
        model_input_gfs = data_loader_gfs.get_model_input()

        pl_vars = data_loader_hrrr.metadata["pl_vars"]
        sfc_vars = data_loader_hrrr.metadata["sfc_vars"]
        levels = data_loader_hrrr.metadata["levels"]
        predicted_channels = len(pl_vars) * len(levels) + len(sfc_vars)
        gfs_channels = model_input_gfs.shape[-1]
        static_channels = max(model_input_hrrr.shape[-1] - predicted_channels, 0)

        nlat = model_input_hrrr.shape[1]
        nlon = model_input_hrrr.shape[2]
        date_channel = np.ones((1, nlat, nlon, 7), dtype=model_input_hrrr.dtype)
        lead_channel = np.ones((1, nlat, nlon, 1), dtype=model_input_hrrr.dtype)
        step_channel = np.ones((1, nlat, nlon, 1), dtype=model_input_hrrr.dtype)

        if not args.no_diffusion:
            rand_channel = np.ones((1, nlat, nlon, predicted_channels), dtype=model_input_hrrr.dtype)
            model_input = np.concatenate(
                [
                    model_input_hrrr[:, :, :, :predicted_channels],
                    model_input_gfs[0:1, :, :, :],
                    rand_channel,
                    model_input_hrrr[:, :, :, predicted_channels:],
                    date_channel,
                    step_channel,
                    lead_channel
                ],
                axis=-1
            )
        else:
            model_input = np.concatenate(
                [
                    model_input_hrrr[:, :, :, :predicted_channels],
                    model_input_gfs[0:1, :, :, :],
                    model_input_hrrr[:, :, :, predicted_channels:],
                    date_channel,
                    step_channel,
                    lead_channel
                ],
                axis=-1
            )
        
        forecaster = WeatherForecaster(data_loader_hrrr, data_loader_gfs,
                                        args.num_members, members,
                                        args.batch_size, not args.no_diffusion,
                                        args.lead_hours,
                                        predicted_channels=predicted_channels,
                                        gfs_channels=gfs_channels,
                                        static_channels=static_channels,
                                        pmm_alpha=args.pmm_alpha,
                                        noise_rho=args.noise_rho,
                                        write_grib2=args.grib2 and not args.no_grib2,
                                        nc_complevel=args.nc_complevel,
                                        nc_least_significant_digit=args.nc_least_significant_digit,
                                        s3_uploader=uploader,
                                        output_hours=args.output_hours)
        run_weather_forecast(
            forecaster, model, args.lead_hours, model_input, args.output_dir
        )
        logger.info(f"All forecasts complete.")
        
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
