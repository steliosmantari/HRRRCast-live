#!/usr/bin/env python3
"""GFS cycle selection and file manifest for HRRRCast boundary conditions.

Deliberately dependency-free: standard library only, no logging setup, no
argparse, no side effects on import. This module is imported by three very
different callers and each of them constrains it:

  src/get_bcs.py        the pipeline stage that actually downloads the files
  aws/pick_cycle.sh     the scheduler's availability probe (via get_bcs)
  aws/lambda/handler.py the hourly launcher, which runs in AWS Lambda where
                        neither `requests` nor `dateutil` is available -- so it
                        cannot import get_bcs, which needs both through utils

Before this module existed the rule lived only in get_bcs.py, which meant the
Lambda would have had to reimplement it. Two copies of "which GFS cycle backs
this forecast" would eventually disagree, and the symptom would be a forecast
silently built from a cycle nobody intended. Keep the rule here, and keep this
module importable from a bare Python.
"""
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

GFS_BASE_URL = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"

# The synoptic hours GFS runs at. Not configurable; it is a property of GFS.
CYCLE_HOURS = (0, 6, 12, 18)


def gfs_cycle_for(valid_dt: datetime, min_lag_hours: int = 0) -> datetime:
    """Pick the GFS cycle that supplies boundary conditions for an HRRRCast run
    initialized at valid_dt.

    GFS runs only at 00/06/12/18Z, so the cycle is always the most recent synoptic
    hour at or before valid_dt (optionally shifted back, see below).

    Why min_lag_hours exists. Measured publication latency on the public buckets:
    the HRRR analysis for hour H appears at about H+0:51, while a GFS cycle does not
    finish publishing its f000-f036 block until about cycle+4:00 (one cycle measured
    directly at +3:41, so +4:00 is a conservative round number). The newest GFS
    cycle is therefore NOT available at the moment the HRRR analysis lands -- at 07Z
    the 06Z GFS is still hours from complete. Under min_lag_hours=0, 12 of the 24
    hours in a day cannot start on time and 4 more have exactly zero margin; they
    must either wait up to four hours (making the HRRR initial condition that much
    staler) or be skipped.

    min_lag_hours shifts cycle selection back BEFORE rounding down to a synoptic
    hour, trading GFS freshness for a launch slot in every hour:

        0  (default)  Newest cycle. Correct for on-demand and retrospective runs,
                      where publication latency is irrelevant because the data is
                      long since complete. Matches the f001-f029 forecast-hour
                      range every run to date has used.
        4             Every hour can launch as soon as the HRRR analysis lands. The
                      GFS cycle is then 4-10 h old and forecast hours f005-f033 are
                      used, which extends past the f001-f029 range the network was
                      trained on.

    Values above 5 skip whole cycles. That is intentional and used by
    aws/pick_cycle.sh as a fallback when the newest usable cycle turns out to be
    incomplete on S3; it degrades further out of the trained range, so it is a
    last resort rather than a free option.

    Returns the cycle initialization datetime (day rollover handled by timedelta).
    """
    shifted = valid_dt - timedelta(hours=min_lag_hours)
    return shifted.replace(hour=(shifted.hour // 6) * 6,
                           minute=0, second=0, microsecond=0)


def get_gfs_urls(year: str, month: str, day: str, hour: str, lead_hours: int,
                 gfs_min_lag_hours: int = 0,
                 base_url: Optional[str] = None) -> List[Tuple[str, str]]:
    """Generate GFS download URLs and filenames for boundary conditions, skipping the 0th hour."""
    urls = []
    cycle_hours = list(CYCLE_HOURS)
    base = base_url if base_url is not None else GFS_BASE_URL

    # Files are named by VALID time, so shifting the cycle changes which forecast
    # hours are fetched but not what the downstream stages see on disk.
    run_dt = datetime(int(year), int(month), int(day), int(hour))
    init_dt = gfs_cycle_for(run_dt, gfs_min_lag_hours)
    init_date_str = init_dt.strftime("%Y%m%d")
    init_cycle = init_dt.hour
    cycle_str = f"{init_cycle:02d}"

    # Offset from the cycle to the run's initialization time, in whole hours.
    start_forecast_hour = int((run_dt - init_dt).total_seconds() // 3600)

    # Generate URLs for all forecast hours from start to start + lead_hours, skipping 0th hour
    for fh_ in range(1, lead_hours + 1):
        fh = fh_ + start_forecast_hour
        forecast_str = f"{fh:03d}"
        url = f"{base}/gfs.{init_date_str}/{cycle_str}/atmos/gfs.t{cycle_str}z.pgrb2.0p25.f{forecast_str}"

        # Calculate valid time for this forecast hour
        valid_dt = init_dt + timedelta(hours=fh)
        valid_str = valid_dt.strftime("%Y%m%d_%H")

        filename = f"gfs_{valid_str}.grib2"
        urls.append((url, filename))

        if fh_ == lead_hours:
            # if valid_dt is not a synoptic hour, also get the next synoptic hour file
            if valid_dt.hour not in cycle_hours:
                next_syn_hour = min([c for c in cycle_hours if c > valid_dt.hour], default=0)
                if next_syn_hour == 0:
                    next_syn_dt = valid_dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                else:
                    next_syn_dt = valid_dt.replace(hour=next_syn_hour, minute=0, second=0, microsecond=0)

                next_fh = int((next_syn_dt - init_dt).total_seconds() // 3600)
                next_forecast_str = f"{next_fh:03d}"
                next_url = f"{base}/gfs.{init_date_str}/{cycle_str}/atmos/gfs.t{cycle_str}z.pgrb2.0p25.f{next_forecast_str}"
                next_valid_str = next_syn_dt.strftime("%Y%m%d_%H")
                next_filename = f"gfs_{next_valid_str}.grib2"
                urls.append((next_url, next_filename))

    return urls


# --- HRRR side -------------------------------------------------------------
HRRR_BASE_URL = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"


def get_hrrr_urls(init_dt: datetime,
                  base_url: Optional[str] = None) -> List[Tuple[str, str]]:
    """The four HRRR files an initial condition needs, as (url, purpose) pairs.

    Two are the analysis itself. The other two are 1 h forecasts from the PREVIOUS
    cycle, whose valid time equals this analysis time: APCP has no meaning in an
    f00 analysis (there is no preceding accumulation interval) and HRRR's analysis
    does not carry usable VVEL, so both come from the previous hour's f01. See
    src/get_ics.py, which is what actually downloads them.

    Kept here rather than in get_ics.py for the same reason as the GFS manifest:
    the Lambda cannot import get_ics (it needs `requests` through utils), and a
    second copy of this list would drift.
    """
    base = base_url if base_url is not None else HRRR_BASE_URL
    prev = init_dt - timedelta(hours=1)
    d, h = init_dt.strftime("%Y%m%d"), init_dt.strftime("%H")
    pd, ph = prev.strftime("%Y%m%d"), prev.strftime("%H")
    return [
        (f"{base}/hrrr.{d}/conus/hrrr.t{h}z.wrfprsf00.grib2", "analysis pressure"),
        (f"{base}/hrrr.{d}/conus/hrrr.t{h}z.wrfsfcf00.grib2", "analysis surface"),
        (f"{base}/hrrr.{pd}/conus/hrrr.t{ph}z.wrfprsf01.grib2", f"{ph}Z f01, VVEL"),
        (f"{base}/hrrr.{pd}/conus/hrrr.t{ph}z.wrfsfcf01.grib2", f"{ph}Z f01, APCP"),
    ]
