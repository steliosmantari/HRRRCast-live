#!/usr/bin/env python3
"""
Regional forecast visualization driver.

Reuses the plotting machinery in ``plot.py`` (ForecastPlotter) but restricts
the output to one or more regional domains defined by a lat/lon bounding box
(N/W/S/E). For each per-hour NetCDF file it regrids the 2D curvilinear
(Lambert) grid onto a regular Cartesian lat/lon grid spanning exactly the
requested box, then produces the same pressure level, surface, and summary
plots that ``plot.py`` makes, into a per-domain output directory.

Regridding uses barycentric interpolation on a Delaunay triangulation of the
source points. The triangulation and target barycentric weights are computed
once per (file, domain) and reused across every field/level, so interpolating
all variables is just a set of vectorized weighted sums.

Usage:
    python plot_domains.py <init_time> <lead_hour> --member m00 \
        --forecast_dir DIR --output_dir DIR

The domains are defined in DOMAINS below.
"""

import argparse
import logging
import os
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import xarray as xr
from scipy.spatial import Delaunay

import plot as plotmod
import utils
from plot import ForecastPlotter, ForecastPlotterConfig
from utils import setup_logging

# Domains requested: (name, N, W, S, E). dx = target grid spacing (deg);
# HRRR native resolution is ~3 km (~0.03 deg).
DOMAINS = [
    {"name": "northeast",  "N": 42.0, "W": -86.0,    "S": 35.0,   "E": -73.0,   "dx": 0.03},
    {"name": "socal",      "N": 35.0, "W": -118.77,  "S": 33.25,  "E": -117.0,  "dx": 0.03},
]


def subset_to_box(ds: xr.Dataset, N: float, W: float, S: float, E: float,
                  pad: float = 0.1) -> xr.Dataset:
    """Subset a dataset with 2D latitude/longitude coords to the tightest
    (latitude, longitude) index window covering the lat/lon box, plus a small
    pad so the box interior stays inside the source point cloud's convex hull.

    Returns None if the box does not overlap the grid.
    """
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    mask = (lat >= S - pad) & (lat <= N + pad) & (lon >= W - pad) & (lon <= E + pad)
    if not mask.any():
        return None
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1 = int(rows[0]), int(rows[-1])
    c0, c1 = int(cols[0]), int(cols[-1])
    return ds.isel(latitude=slice(r0, r1 + 1), longitude=slice(c0, c1 + 1))


def _build_bary_interpolator(src_lon, src_lat, tgt_lon, tgt_lat):
    """Build a reusable linear (barycentric) interpolator from scattered source
    points onto target points. Returns a function mapping a flat source-value
    array to a flat target-value array; points outside the source convex hull
    become NaN. Delaunay triangulation and barycentric weights are computed once
    here and reused for every field.
    """
    src_pts = np.column_stack([src_lon.ravel(), src_lat.ravel()])
    tri = Delaunay(src_pts)
    tgt_pts = np.column_stack([tgt_lon.ravel(), tgt_lat.ravel()])

    simplex = tri.find_simplex(tgt_pts)
    valid = simplex >= 0
    # Barycentric coordinates of each target point within its containing simplex.
    Tm = tri.transform[simplex]                     # (ntgt, 3, 2): [affine(2x2); offset]
    delta = tgt_pts - Tm[:, 2, :]
    bary2 = np.einsum("nij,nj->ni", Tm[:, :2, :], delta)
    bary = np.concatenate([bary2, 1.0 - bary2.sum(axis=1, keepdims=True)], axis=1)  # (ntgt, 3)
    verts = tri.simplices[simplex]                  # (ntgt, 3)

    def interp(values_flat):
        v = np.asarray(values_flat, dtype=float)[verts]   # (ntgt, 3)
        out = np.einsum("ni,ni->n", bary, v)
        out[~valid] = np.nan
        return out

    return interp


def regrid_to_latlon(ds: xr.Dataset, N: float, W: float, S: float, E: float,
                     dx: float = 0.03) -> xr.Dataset:
    """Regrid a dataset on a 2D curvilinear grid onto a regular Cartesian
    lat/lon grid spanning exactly [S, N] x [W, E] at spacing ``dx`` (degrees).

    Every data variable whose last two dims are (latitude, longitude) is
    interpolated; leading dims (time, lead_time, level) are preserved. Returns
    None if the box does not overlap the grid.
    """
    ds_bb = subset_to_box(ds, N, W, S, E)
    if ds_bb is None:
        return None

    lat_t = np.arange(S, N + dx / 2.0, dx)
    lon_t = np.arange(W, E + dx / 2.0, dx)
    ny, nx = lat_t.size, lon_t.size
    LON, LAT = np.meshgrid(lon_t, lat_t)

    interp = _build_bary_interpolator(
        ds_bb["longitude"].values, ds_bb["latitude"].values, LON, LAT)

    data_vars = {}
    for name, da in ds_bb.data_vars.items():
        if da.dims[-2:] != ("latitude", "longitude"):
            continue
        lead_shape = da.shape[:-2]
        flat = da.values.reshape(-1, da.shape[-2] * da.shape[-1])
        out = np.empty((flat.shape[0], ny * nx), dtype=float)
        for i in range(flat.shape[0]):
            out[i] = interp(flat[i])
        data_vars[name] = (da.dims, out.reshape(lead_shape + (ny, nx)))

    coords = {
        "latitude": ("latitude", lat_t),
        "longitude": ("longitude", lon_t),
    }
    for c in ("time", "lead_time", "level"):
        if c in ds_bb.coords:
            coords[c] = ds_bb[c]

    return xr.Dataset(data_vars, coords=coords)


def plot_lead_hour_domain(h, ds_path, domain, init_datetime, init_year, init_month,
                          init_day, init_hh, output_dir, date_str, member, config_dict):
    """Plot all variables for a single lead hour restricted to one domain."""
    if plotmod.logger is None:
        plotmod.logger = setup_logging()

    config = ForecastPlotterConfig()
    for k, v in config_dict.items():
        setattr(config, k, v)
    plotter = ForecastPlotter(config)

    ds = xr.open_dataset(ds_path, decode_timedelta=True)
    try:
        ds_sub = regrid_to_latlon(ds, domain["N"], domain["W"], domain["S"],
                                  domain["E"], dx=domain.get("dx", 0.03))
        if ds_sub is None:
            plotmod.logger.warning(
                f"Domain '{domain['name']}' does not overlap grid for f{h:02d}; skipping")
            return
        timestamp_str = f"{init_year}-{init_month}-{init_day} {init_hh}:00 UTC"
        output_subdir = f"{output_dir}/{date_str}/domain_{domain['name']}/mem{member}_lead{h:02d}h"
        utils.make_directory(output_subdir)
        plotter.plot_pressure_level_variables(ds_sub, h, output_subdir, timestamp_str)
        plotter.plot_surface_variables(ds_sub, h, output_subdir, timestamp_str)
        plotter.create_summary_plot(ds_sub, h, output_subdir, timestamp_str)
        plotmod.logger.info(
            f"[{domain['name']}] plots for lead hour {h} saved to: {output_subdir}")
    finally:
        ds.close()


def plot_forecast_domains(datetime_str, lead_hour, member, forecast_dir="./", output_dir="./"):
    """Plot hours 1..lead_hour for every domain in DOMAINS, in parallel."""
    init_datetime, init_year, init_month, init_day, init_hh = utils.validate_datetime(datetime_str)
    date_str = f"{init_year}{init_month}{init_day}/{init_hh}"
    lead_hour_int = int(lead_hour)

    mem_str = str(member)
    if mem_str not in {"avg", "spr"} and not mem_str.startswith("m"):
        mem_str = f"m{int(member):02d}"

    config = ForecastPlotterConfig()
    config_dict = config.__dict__

    args_list = []
    for h in range(1, lead_hour_int + 1):
        ds_path = f"{forecast_dir}/{date_str}/hrrrcast_{mem_str}_f{h:02d}.nc"
        if not os.path.exists(ds_path):
            plotmod.logger.warning(f"Skipping hour f{h:02d}: file not found {ds_path}")
            continue
        for domain in DOMAINS:
            args_list.append((h, ds_path, domain, init_datetime, init_year, init_month,
                              init_day, init_hh, output_dir, date_str, mem_str, config_dict))

    n_workers = max(1, min(len(args_list), (os.cpu_count() or 4)))
    log_level = logging.getLevelName(logging.getLogger().getEffectiveLevel())
    plotmod.logger.info(
        f"Parallel plotting {len(args_list)} (hour x domain) tasks using {n_workers} workers")

    errors = []
    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=plotmod._init_worker, initargs=(log_level,)) as executor:
        futures = {executor.submit(plot_lead_hour_domain, *a): (a[0], a[2]["name"]) for a in args_list}
        for future in as_completed(futures):
            h, dname = futures[future]
            try:
                future.result()
            except Exception as e:
                errors.append((h, dname, e))
                plotmod.logger.error(f"Error plotting f{h:02d} [{dname}]: {e}")

    if errors:
        failed = ", ".join(f"f{h:02d}[{d}]" for h, d, _ in sorted(errors, key=lambda x: (x[0], x[1])))
        raise RuntimeError(f"Plotting failed for {len(errors)} of {len(args_list)} task(s): {failed}")
    plotmod.logger.info(f"Domain plotting completed for hours 1..{lead_hour_int}.")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot forecast variables restricted to regional domains",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inittime", help='Forecast init time, e.g. "2026-07-20T00"')
    parser.add_argument("lead_hour", help="Max lead hour (plots 1..lead_hour)")
    parser.add_argument("--member", default="m00", help="Member ID (e.g. m00, avg, spr)")
    parser.add_argument("--forecast_dir", default="./", help="Directory containing forecast files")
    parser.add_argument("--output_dir", default="./", help="Output directory for plots")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    return parser.parse_args()


def main():
    args = parse_arguments()
    plotmod.logger = setup_logging(args.log_level)
    plot_forecast_domains(
        datetime_str=args.inittime,
        lead_hour=args.lead_hour,
        member=args.member,
        forecast_dir=args.forecast_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
