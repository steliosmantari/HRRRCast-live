#!/usr/bin/env python3
"""
GFS Lateral Boundary Conditions Downloader
Downloads GFS GRIB2 files for lateral boundary conditions.
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import utils
from utils import setup_logging, create_output_directory, download_file_with_retry
# The cycle-selection rule and file manifest live in gfs_cycle so that AWS Lambda
# can import them too; Lambda has neither `requests` nor `dateutil`, which utils
# needs, so it cannot import this module. See src/gfs_cycle.py.
from gfs_cycle import GFS_BASE_URL as _GFS_BASE_URL, gfs_cycle_for
from gfs_cycle import get_gfs_urls as _get_gfs_urls

# -------------------------------
# Configuration
# -------------------------------
class Config:
    """Configuration class for GFS data downloader."""
    
    # Base URLs
    GFS_BASE_URL = _GFS_BASE_URL
    
    # Retry settings
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    TIMEOUT = 300    # seconds


# -------------------------------
# GFS Download Functions
# -------------------------------
def get_gfs_urls(year: str, month: str, day: str, hour: str, lead_hours: int,
                 gfs_min_lag_hours: int = 0) -> List[Tuple[str, str]]:
    """Thin wrapper over gfs_cycle.get_gfs_urls that honours Config.GFS_BASE_URL.

    The indirection is not decoration: aws/pick_cycle.sh mutates
    Config.GFS_BASE_URL to point the availability probe at a dead path, which is
    the only way to exercise its cycle-search fallback (every GFS cycle in the past
    is complete). Passing the value through here keeps that hook working while the
    rule itself stays in one place.
    """
    return _get_gfs_urls(year, month, day, hour, lead_hours, gfs_min_lag_hours,
                         base_url=Config.GFS_BASE_URL)


def download_gfs_files(year: str, month: str, day: str, hour: str, lead_hours: int, output_dir: Path,
                       gfs_min_lag_hours: int = 0) -> List[bool]:
    """Download GFS GRIB2 files for boundary conditions."""
    logger = logging.getLogger(__name__)

    if lead_hours == 0:
        logger.info(f"Downloading GFS data for {year}-{month}-{day} {hour}:00 UTC")
    else:
        logger.info(f"Downloading GFS boundary conditions: {year}-{month}-{day} {hour}:00 UTC + {lead_hours} hours")

    # Log the resolved cycle explicitly. Which GFS cycle backed a given forecast is
    # not recoverable from the output later (the files are named by valid time), so
    # without this line a run made with a non-zero lag is indistinguishable from one
    # made with the default.
    cycle = gfs_cycle_for(datetime(int(year), int(month), int(day), int(hour)), gfs_min_lag_hours)
    logger.info(f"GFS cycle: {cycle:%Y-%m-%d %H}Z "
                f"(min_lag={gfs_min_lag_hours}h, age at init "
                f"{int((datetime(int(year), int(month), int(day), int(hour)) - cycle).total_seconds() // 3600)}h)")

    urls = get_gfs_urls(year, month, day, hour, lead_hours, gfs_min_lag_hours)
    logging.info(f"Total files to download: {len(urls)}")
    for url in urls:
        logger.info(f"{url[0]} -> {url[1]}")
        
    results = []
    
    # Use ThreadPoolExecutor for multiple files
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_url = {
            executor.submit(download_file_with_retry, url, str(output_dir / filename)): (url, filename)
            for url, filename in urls
        }
        
        for future in as_completed(future_to_url):
            url, filename = future_to_url[future]
            try:
                result = future.result()
                results.append(result)
                if result:
                    logger.info(f"Downloaded: {filename}")
            except Exception as e:
                logger.error(f"Error downloading {filename}: {e}")
                results.append(False)
    
    logger.info(f"GFS downloads completed: {sum(results)}/{len(results)} successful")
    return results

# -------------------------------
# Main Functions
# -------------------------------
def download_gfs_data(datetime_str: str, lead_hours: int, base_dir: str = "./",
                      gfs_min_lag_hours: int = 0) -> dict:
    """Download GFS boundary condition data for specified date and time."""
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    init_datetime, year, month, day, hour = utils.validate_datetime(datetime_str)
    date_str = f"{year}{month}{day}/{hour}"
    
    # Create output directory
    output_dir = create_output_directory(base_dir, date_str)
    logger.info(f"Output directory: {output_dir}")
    
    results = {'gfs': []}
    
    # Download GFS data
    try:
        gfs_results = download_gfs_files(year, month, day, hour, lead_hours, output_dir,
                                         gfs_min_lag_hours)
        results['gfs'] = gfs_results
    except Exception as e:
        logger.error(f"Error downloading GFS data: {e}")
        results['gfs'] = [False]
    
    return results

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Download GFS lateral boundary conditions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python get_lbcs.py 2024-01-15T12 0    # Single file
  python get_lbcs.py 2024-01-15T12 24   # 24-hour boundary conditions
  python get_lbcs.py 2024-01-15T12 48 --base_dir /data/weather
  python get_lbcs.py 2024-01-15T12 36 --log_level DEBUG
        """
    )
    
    parser.add_argument('inittime',
                       help='Forecast initialization time in format YYYY-MM-DDTHH (e.g., "2024-05-06T23")')
    parser.add_argument('lead_hours', type=int, help='Lead time in hours for boundary conditions')
    parser.add_argument('--base_dir', default='./', help='Base directory for downloads (default: ./)')
    parser.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set logging level (default: INFO)')
    parser.add_argument('--gfs_min_lag_hours', type=int, default=0,
                       help='Shift GFS cycle selection back this many hours before rounding '
                            'down to 00/06/12/18Z. 0 (default) uses the newest cycle, which is '
                            'unavailable for 12 of 24 hours because GFS takes ~4 h to publish. '
                            'Use 4 for a fixed hourly schedule; see gfs_cycle_for().')

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    # Validate lead_hours
    if args.lead_hours < 0:
        logger.error("Lead hours must be >= 0")
        sys.exit(1)

    # A negative lag would ask for a cycle that does not exist yet. The upper bound
    # allows deliberately skipping whole cycles: aws/pick_cycle.sh steps back a full
    # 6 h at a time when the newest usable cycle turns out to be incomplete on S3,
    # and expresses the cycle it settled on as a lag. 23 h is the point past which
    # GFS forecast hours would exceed f047 for a 24 h run, well outside anything the
    # network has seen.
    if not 0 <= args.gfs_min_lag_hours <= 23:
        logger.error("--gfs_min_lag_hours must be between 0 and 23 "
                     "(negative asks for a future cycle; >23 leaves the trained range)")
        sys.exit(1)

    try:
        # Download GFS data
        results = download_gfs_data(
            args.inittime, args.lead_hours, args.base_dir, args.gfs_min_lag_hours
        )
        
        # Summary
        total_successful = sum(results['gfs'])
        total_attempted = len(results['gfs'])
        
        logger.info(f"Download summary: {total_successful}/{total_attempted} files successful")
        
        if total_successful == 0:
            logger.error("No files were downloaded successfully")
            sys.exit(1)
        elif total_successful < total_attempted:
            logger.warning("Some downloads failed")
            sys.exit(2)
        else:
            logger.info("All downloads completed successfully")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    main()
