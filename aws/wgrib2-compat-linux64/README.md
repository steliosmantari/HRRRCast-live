# wgrib2 jasper compatibility shim (linux-64 only)

**What:** `lib/libjasper.so.1` and `lib/libjpeg.so.9`, two small shared libraries
extracted from old conda-forge packages (`jasper-1.900.1-hff1ad4c_5`,
`jpeg-9c-h14c3975_1001`).

**Why they exist:** every conda-forge `wgrib2` build for linux-64 (checked: `2.0.5-2`,
the one this repo locks, and `2.0.7-h16864d7_1001`) is dynamically linked against
`libjasper.so.1`. `aws/conda-linux-64.lock` cannot provide that: `pygrib` hard-requires
`jasper >=4.1.0,<5.0a0`, and jasper's shared-library SONAME changed between the 1.x
and 4.x lines, so no single jasper install can satisfy both `pygrib` and `wgrib2` at
once. Confirmed by inspecting the actual ELF `NEEDED`/`SONAME` entries
(`objdump -p`) of both `wgrib2` builds and of `jasper-1.900.1`'s own
`libjasper.so.1.0.0` — not by guessing from conda's declared metadata, which does not
even list `jasper` as a `wgrib2` dependency at all. `libjasper.so.1.0.0` in turn needs
`libjpeg.so.9`, which the environment's `libjpeg-turbo` only provides as `.so.8`, so a
matching old `libjpeg` had to come along too. Both shims need nothing beyond glibc:

```
libjasper.so.1  NEEDED: libm.so.6, libjpeg.so.9, libc.so.6
libjpeg.so.9    NEEDED: libc.so.6
```

**How they're used:** `src/crop_grib2.py` adds this directory to `LD_LIBRARY_PATH`
*only* for the `wgrib2` subprocess call, and only if the directory exists (a no-op
everywhere else, including macOS). The environment's own `jasper` package — the one
`pygrib` needs — is never touched, so nothing else in the pipeline is affected.

**Regenerating, if a future `wgrib2` build changes its jasper requirement:**

```bash
mkdir -p /tmp/wgrib2check && cd /tmp/wgrib2check
curl -sL -o w.tar.bz2 "https://conda.anaconda.org/conda-forge/linux-64/wgrib2-<version>.tar.bz2"
mkdir w && tar xjf w.tar.bz2 -C w
objdump -p w/bin/wgrib2 | grep NEEDED   # confirm it still needs libjasper.so.1
```

If a build ever declares a *different* SONAME, or drops the jasper dependency
entirely, this directory is no longer needed and `crop_grib2.py`'s `_COMPAT_LIB`
reference can be removed.
