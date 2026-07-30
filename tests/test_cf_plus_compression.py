"""CF encoding and this fork's compression/quantization must coexist.

The upstream CF-compliance merge replaced the encoding dict wholesale; this fork had
put compression and quantization in the same place. Both are needed, so fcst.py now
layers per-variable compression on top of get_cf_encoding(). This checks that the
composition is actually valid to xarray and does not corrupt coordinates.

    export HRRRCAST_TEST_DATA=/path/containing/full/and/sub
    python tests/test_cf_plus_compression.py
"""
import os, sys, datetime
sys.path.insert(0, "src")
import numpy as np, xarray as xr
from cf_attributes import apply_cf_attributes, get_cf_encoding

S = os.environ.get("HRRRCAST_TEST_DATA", "")
if not S:
    sys.exit("set HRRRCAST_TEST_DATA")
fails = []
def check(n, c, d=""):
    print(f"  {'PASS' if c else 'FAIL'}  {n}" + (f"  {d}" if d else ""))
    if not c: fails.append(n)

init = datetime.datetime(2026, 7, 29, 0)
src = f"{S}/sub/hrrrcast_m00_f01.nc"
ds = xr.open_dataset(src).load()
ds = apply_cf_attributes(ds, init_datetime=init)

def build_encoding(ds, complevel, lsd):
    """Exactly the logic now in fcst.write_single_hour_netcdf."""
    enc = get_cf_encoding(ds, init)
    if complevel > 0 or lsd is not None:
        per_var = {}
        if complevel > 0:
            per_var["zlib"] = True; per_var["complevel"] = complevel
        if lsd is not None:
            per_var["least_significant_digit"] = lsd
        for name in ds.data_vars:
            if name in enc:
                enc[name].update(per_var)
    return enc

print("TEST: CF-only vs CF+compression both write, and compression still shrinks")
sizes = {}
enc_lsd = build_encoding(ds, 1, 2)
for tag, cl, lsd in [("cf-only", 0, None), ("cf+zlib1", 1, None), ("cf+zlib1+lsd2", 1, 2)]:
    out = f"{S}/enc_{tag}.nc"
    if os.path.exists(out): os.unlink(out)
    enc = build_encoding(ds, cl, lsd)
    try:
        ds.to_netcdf(out, encoding=enc)
        sizes[tag] = os.path.getsize(out)
        check(f"{tag} written", True, f"{sizes[tag]/1e6:.2f} MB")
    except Exception as e:
        check(f"{tag} written", False, f"{type(e).__name__}: {e}")
check("zlib shrinks vs cf-only", sizes.get("cf+zlib1", 9e9) < sizes.get("cf-only", 0),
      f"{sizes.get('cf+zlib1',0)/1e6:.2f} < {sizes.get('cf-only',0)/1e6:.2f} MB")
# NOT asserted here: that lsd shrinks the file further. This fixture was itself
# written with least_significant_digit=2 (netCDF4 reports the attribute on every
# variable), so re-quantizing it is idempotent -- the round-trip error below is
# exactly 0.0 and the size does not fall. The size effect of lsd is isolated on
# synthetic unquantized data at the end of this file instead.
check("lsd is plumbed into the encoding", all(
          enc_lsd[v].get("least_significant_digit") == 2
          for v in ds.data_vars if v in enc_lsd),
      "all data vars")

print("\nTEST: CF metadata survives compression")
chk = xr.open_dataset(f"{S}/enc_cf+zlib1+lsd2.nc", decode_times=False)
check("Conventions attr present", "Conventions" in chk.attrs, chk.attrs.get("Conventions", ""))
check("time units are CF hours-since", "hours since" in chk.time.attrs.get("units", ""),
      chk.time.attrs.get("units", ""))

print("\nTEST: compression applied to data vars, NOT to coords or grid_mapping")
enc = build_encoding(ds, 1, 2)
dv = [v for v in ds.data_vars if v in enc]
check("every encoded data var got zlib", all(enc[v].get("zlib") for v in dv), f"{len(dv)} vars")
for coord in ("time", "level", "x", "y", "latitude", "longitude"):
    if coord in enc:
        bad = "zlib" in enc[coord] or "least_significant_digit" in enc[coord]
        check(f"coord '{coord}' untouched", not bad, str(enc[coord]))
if "grid_mapping" in ds.data_vars:
    check("grid_mapping not compressed", "grid_mapping" not in enc or
          "zlib" not in enc.get("grid_mapping", {}))
    check("grid_mapping excluded by get_cf_encoding", "grid_mapping" not in get_cf_encoding(ds, init))

print("\nTEST: quantization error is within the lsd=2 contract")
ref = xr.open_dataset(src).load()
worst = 0.0
for v in ("T2M", "PRES", "UGRD10M"):
    if v in chk and v in ref:
        d = float(np.nanmax(np.abs(np.asarray(chk[v].values, "f8") - np.asarray(ref[v].values, "f8"))))
        worst = max(worst, d)
        print(f"    {v}: max abs diff {d:.6f}")
check("worst error consistent with lsd=2", worst < 0.05, f"{worst:.6f}")

chk.close(); ref.close(); ds.close()
print("\nTEST: lsd does shrink UNQUANTIZED data (isolated from the pre-quantized fixture)")
rng = np.random.default_rng(0)
syn = xr.Dataset({"f": (("y", "x"), (rng.random((400, 400), dtype=np.float32) * 300 + 200))})
syn_sizes = {}
for tag, e in [("zlib1", {"f": {"zlib": True, "complevel": 1}}),
               ("zlib1+lsd2", {"f": {"zlib": True, "complevel": 1,
                                     "least_significant_digit": 2}})]:
    o = f"{S}/syn_{tag}.nc"
    if os.path.exists(o): os.unlink(o)
    syn.to_netcdf(o, encoding=e)
    syn_sizes[tag] = os.path.getsize(o)
    print(f"    {tag}: {syn_sizes[tag]/1e3:.1f} kB")
check("lsd shrinks unquantized float32", syn_sizes["zlib1+lsd2"] < syn_sizes["zlib1"],
      f"{syn_sizes['zlib1+lsd2']/1e3:.1f} < {syn_sizes['zlib1']/1e3:.1f} kB")
syn.close()

import glob
for f in glob.glob(f"{S}/enc_*.nc") + glob.glob(f"{S}/syn_*.nc"): os.unlink(f)
print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILURES: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
