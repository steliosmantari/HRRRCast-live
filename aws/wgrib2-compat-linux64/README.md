# wgrib2 compatibility shims (linux-64 only)

**What:** five small shared libraries, extracted once from old conda-forge packages
and checked in, so `wgrib2` (used by `src/crop_grib2.py` for `-ijsmall_grib`) can run
on the AWS Linux environment despite depending on ABI-old libraries this repo's
`aws/conda-linux-64.lock` cannot provide.

| shim | from | needed because |
|---|---|---|
| `libjasper.so.1` | `jasper-1.900.1-hff1ad4c_5` | `pygrib` hard-requires `jasper>=4.1.0`, a different SONAME than any `wgrib2` build links against |
| `libjpeg.so.9` | `jpeg-9c-h14c3975_1001` | `libjasper.so.1` needs it; the environment's `libjpeg-turbo` only provides `.so.8` |
| `libnetcdf.so.13` | `libnetcdf-4.6.1-2` | the environment's `libnetcdf` is `4.9.3`, SONAME `.so.22` |
| `libhdf5.so.101` | `hdf5-1.10.2-hc401514_1` | `libnetcdf.so.13` needs it; the environment's `hdf5` is `1.14.6`, SONAME `.so.310` |
| `libhdf5_hl.so.100` | `hdf5-1.10.2-hc401514_1` | same package as `libhdf5.so.101`, same reason |

**Why not touch `aws/conda-linux-64.lock` instead:** `pygrib` needs a *newer* jasper
than any `wgrib2` build tolerates. There is no single choice of `jasper` version that
satisfies both, so the shared, hash-pinned lock the rest of the pipeline depends on
cannot be the fix without breaking `pygrib` everywhere, not just for cropping.

**Why this exact set and no more.** Confirmed by walking the actual ELF `NEEDED` /
`SONAME` entries with `objdump -p` (not conda's declared metadata, which does not
even list `jasper` as a `wgrib2` dependency at all) until every edge terminates in
either one of the five files above, a base glibc library (`libc`, `libm`, `libz`,
`libdl`, `libpthread`, `librt` — always present), or a package already correctly
installed by the lock (`libpng`, `mysql-connector-c`, `hdf4`, `libcurl` all matched
on first check; no further shims were needed for them):

```
wgrib2            -> libjasper.so.1, libnetcdf.so.13, libpng16.so.16 (lock, matches),
                     libmysqlclient.so.18 (lock, matches), + base libs
libjasper.so.1    -> libjpeg.so.9, + base libs
libjpeg.so.9      -> base libs only
libnetcdf.so.13   -> libmfhdf.so.0 (lock, matches), libdf.so.0 (lock, matches),
                     libhdf5_hl.so.100, libhdf5.so.101,
                     libcurl.so.4 (lock, matches), + base libs
libhdf5.so.101    -> base libs only
libhdf5_hl.so.100 -> libhdf5.so.101, + base libs
```

**How they're used:** `src/crop_grib2.py` adds this directory to `LD_LIBRARY_PATH`
*only* for the `wgrib2` subprocess call, and only if the directory exists (a no-op
everywhere else, including macOS). The environment's own `jasper`, `netcdf`, and
`hdf5` — the versions `pygrib`, `make_bcs.py`'s xESMF regridding, and everything
else actually run against — are never touched.

**Regenerating, if a future `wgrib2` build changes its requirements:**

```bash
mkdir -p /tmp/wgrib2check && cd /tmp/wgrib2check
curl -sL -o w.tar.bz2 "https://conda.anaconda.org/conda-forge/linux-64/wgrib2-<version>.tar.bz2"
mkdir w && tar xjf w.tar.bz2 -C w
objdump -p w/bin/wgrib2 | grep NEEDED
```

Cross-check each entry against `aws/conda-linux-64.lock`'s currently pinned versions
the same way (download, extract, `objdump -p lib/<name> | grep SONAME`) before
assuming a shim is needed for it.
