"""GRIB2 write/decode round-trip geolocation.

Needs a full-domain and a cropped NetCDF from the same cycle:
    export HRRRCAST_TEST_DATA=/path/containing/full/and/sub
    python tests/test_grib2_roundtrip.py
"""
import sys, os, datetime
sys.path.insert(0, "src")
import numpy as np, xarray as xr, grib2io
from nc2grib import Netcdf2Grib

S = os.environ.get("HRRRCAST_TEST_DATA", "")
if not S:
    sys.exit("set HRRRCAST_TEST_DATA to a dir holding full/ and sub/ "
             "hrrrcast_m00_f01.nc from a domain-test run")
fails = []
def check(n, c, d=""):
    print(f"  {'PASS' if c else 'FAIL'}  {n}" + (f"  {d}" if d else ""))
    if not c: fails.append(n)

out_crop, out_full = f"{S}/rt_crop.grib2", f"{S}/rt_full.grib2"
for tag, src, out in [("CROP 155x151", f"{S}/sub/hrrrcast_m00_f01.nc", out_crop),
                      ("FULL 1059x1799", f"{S}/full/hrrrcast_m00_f01.nc", out_full)]:
    print(f"\n=== {tag} ===")
    ds = xr.open_dataset(src)
    ds1 = ds.isel(time=0, lead_time=0) if "time" in ds.dims else ds
    if os.path.exists(out): os.unlink(out)
    conv = Netcdf2Grib()
    conv.save_grib2(datetime.datetime(2026,7,29,0), ds, out)
    check("file written", os.path.exists(out) and os.path.getsize(out) > 0,
          f"{os.path.getsize(out)/1e6:.1f} MB" if os.path.exists(out) else "")
    if not os.path.exists(out):
        ds.close(); continue

    with grib2io.open(out) as f:
        msgs = len(f)
        m = f[0]
        glat, glon = m.grid()          # decoded lat/lon from Section 3 alone
    print(f"  messages: {msgs}")
    nlat, nlon = ds.latitude.values, ds.longitude.values
    check("decoded shape == NetCDF shape", glat.shape == nlat.shape, f"{glat.shape} vs {nlat.shape}")
    if glat.shape == nlat.shape:
        dlat = np.abs(glat - nlat).max()
        glon180 = np.where(glon > 180, glon - 360, glon)
        dlon = np.abs(glon180 - nlon).max()
        # 3 km grid: 0.01 deg is ~1 km, well inside a grid cell
        check("max |lat| error < 0.01 deg", dlat < 0.01, f"{dlat:.6f}")
        check("max |lon| error < 0.01 deg", dlon < 0.01, f"{dlon:.6f}")
        km = dlat * 111.0
        print(f"  worst-case geolocation error: {km:.3f} km ({km/3.0*100:.1f}% of a grid cell)")
    ds.close()

# grib2io also drops hashed *.grib2ioidx sidecars next to the file, so glob rather
# than guessing the suffix.
import glob
for tmp in (out_crop, out_full):
    for f in [tmp] + glob.glob(tmp + "*"):
        if os.path.exists(f):
            os.unlink(f)
print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILURES: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
