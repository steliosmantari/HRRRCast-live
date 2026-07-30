# Running HRRRCast on a subdomain

Cropping the domain makes inference cheaper: the cost of a step scales with `H*W`.
A quarter-area subdomain runs about 3.3x faster per lead hour and writes 4.2x
faster, and may fit hardware with less VRAM.

**This works, with one caveat.** Measured on 2026-07-29 17Z at 24 h lead against a
matched full-domain run, a 25.2% crop produced errors indistinguishable from the
model's own ensemble spread for surface pressure, 10 m wind, reflectivity and
precipitation, and a systematic **cold bias of about 0.5 to 1.1 K in 2 m
temperature**. See the "What it costs you" section for the numbers and the reasoning.

---

## 0. The short way: name a region, run one command

If you know the area you care about, do not compute grid indices and do not run the
stages by hand. Give a region and a halo to whichever driver you are already using, and
it sizes and places a legal box for you.

**Locally, or on an instance:**

```bash
SUB_BBOX=35.0,-118.77,33.25,-117.0 SUB_HALO=40 \
    ./run_cycle.sh 2026-07-29T17 24 1 1 "$PWD" "$PWD" NO
```

**On AWS, one on-demand forecast:**

```bash
aws/run_on_ec2.sh --bucket mantari-cast1 --init-time 2026-07-29T17 --lead-hours 24 \
    --instance-type g5.2xlarge \
    --bbox 35.0,-118.77,33.25,-117.0 --halo 40
```

**On AWS, every hour:** the same flags plus `--stage-scheduler`, which bakes the box into
the staged user-data template. No Lambda change is needed.

```bash
aws/run_on_ec2.sh --bucket mantari-cast1 --lead-hours 24 \
    --bbox 35.0,-118.77,33.25,-117.0 --halo 40 --stage-scheduler
```

All three crop the *raw* HRRR GRIB2 before the input stages run (see "Cropping the
HRRR-side inputs" below), not just the forecast: `make_ics.py` runs 4.6x faster and
`make_bcs.py` 1.5-1.6x faster on the crop, measured. GFS itself is not cropped, so
its per-lead-hour read stays the same size regardless of the box. A cropped run
writes to its own root, `$DATAROOT/subrun/<YYYYMMDD>/<HH>/`, so cropped and
full-domain output never collide. Use `SUB_HEIGHT`/`SUB_WIDTH` (together) for a
fixed grid-centred box instead of a region; that mode has no region to protect
with a halo, so it stays on the older, slower path of cropping the full-domain
`.npz` after the input stages run (still used by `aws/domain_test.sh`'s fidelity
experiments, where full-domain and cropped runs must share bit-identical inputs).

### Sizing or checking a box without running anything

`crop_domain.py` is the piece underneath, and it is worth running on its own before you
spend money on an instance:

```bash
python3 src/crop_domain.py \
    --in-dir 20260729/17 --out-dir 20260729/17-socal \
    --init-time 2026-07-29T17 \
    --bbox 35.0,-118.77,33.25,-117.0 --halo 40      # add --dry-run to only report
```

`--bbox` overrides `--height/--width/--y0/--x0`. It finds every grid cell inside the
region, takes the index bounding box (a superset, since the HRRR grid is Lambert
conformal so a lat/lon box is curved in index space), adds `--halo` cells on every
side, and rounds **up** to the nearest legal size so the halo is never eaten by
rounding.

It reports the halo it actually achieved on each side. Clamping at a grid edge can
silently thin one side, and a thin side is exactly where crop error concentrates:

```
  halo achieved      south=40  north=40  west=12!  east=40
  NOTE: halo is thinner than requested on west (clamped at the grid edge).
```

The box, the requested region and the achieved halo are all recorded in
`subdomain.json`, so a downstream consumer can tell which part of the output is
product and which is halo to discard.

Read sections 1 and 2 anyway: they explain why the size rule exists and how the halo
number was arrived at. Sections 3 and 4 are the manual stage-by-stage sequence, useful
for understanding what the drivers above do, and for debugging.

## 1. Pick a legal subdomain size

**The size is constrained. This is not negotiable and a wrong size does not fail
cleanly.**

The trained model bakes a *fixed* reflect-padding of `[[3,2],[1,0]]` (H+5, W+1) and
then downsamples by 8 through three stride-2 poolings:

```
1059 + 5 = 1064 = 8 x 133
1799 + 1 = 1800 = 8 x 225
```

Because that padding is a constructor argument stored inside `model.keras`, it does
not adapt to a different input size. Any subdomain must satisfy:

```
(H + 5) % 8 == 0   ->   H % 8 == 3
(W + 1) % 8 == 0   ->   W % 8 == 7
```

A size that violates this misaligns the UNet skip connections after upsampling. That
surfaces either as a shape error deep inside the model or, worse, as silently
shifted output. Note `512x512` is **not** legal; round numbers usually are not.

Check a candidate before doing anything else:

```bash
python3 src/crop_domain.py --in-dir . --out-dir . --init-time 2026-07-29T17 \
    --height 531 --width 903 --dry-run
```

The tool refuses illegal sizes and prints the nearest legal ones:

```
ERROR: invalid subdomain size.
  height 530: (h+5) must be divisible by 8 (h % 8 == 3); nearest valid: [523, 531]
  width 900:  (w+1) must be divisible by 8 (w % 8 == 7); nearest valid: [895, 903]
```

Some legal sizes:

| H x W | area | notes |
|---|---|---|
| 1059 x 1799 | 100% | the full grid, which of course satisfies the rule |
| 795 x 1351 | 56.4% | conservative crop, generous halo |
| **531 x 903** | **25.2%** | the tested configuration |
| 379 x 647 | 12.9% | aggressive; halo shrinks, see below |

## 2. Leave a halo

The model has no lateral boundary condition. GFS forcing is supplied at every grid
point including the crop edge, and it does most of what a boundary condition would,
which is why a crop works at all. What is lost is the fine, convective-scale
structure advecting in from outside.

Measured: T2M error is elevated within roughly the first 10 to 40 cells of the edge
and flat beyond, so **a halo of 20 to 40 cells (60 to 120 km) captures nearly all the
available improvement.** Size the box as your region of interest plus at least that
much on every side, and treat the outer ~40 cells as scratch.

That is far cheaper than a naive advection argument implies. At 20 to 30 m/s air
travels 1700 to 2600 km in 24 h, which would contaminate any small box throughout if
the model relied on inflow alone. It does not.

## 3. Crop the inputs

`src/fcst.py` already derives its grid size from the input arrays
(`nlat, nlon = input_shape[1], input_shape[2]`), so **it needs no changes at all.**
`make_ics.py` and `make_bcs.py` do hardcode 1059x1799, but there is no need to touch
them: run the input stages once at full domain, then crop the resulting `.npz`.

```bash
# normal input stages, full domain
python3 src/get_ics.py  2026-07-29T17 --base_dir .
python3 src/get_bcs.py  2026-07-29T17 24 --base_dir . --gfs_min_lag_hours 4
python3 src/make_ics.py net-diffusion/normalize-stats.nc 2026-07-29T17 \
    --base_dir . --output_dir .
python3 src/make_bcs.py net-diffusion/normalize-stats.nc 2026-07-29T17 24 \
    --base_dir . --output_dir . \
    --hrrr_grid_file 20260729/17/hrrr_20260729_17_surface.grib2

# crop, centred on the grid by default
python3 src/crop_domain.py \
    --in-dir 20260729/17 --out-dir 20260729/17-sub \
    --init-time 2026-07-29T17 --height 531 --width 903
```

Place the box explicitly by grid index or by lat/lon:

```bash
--y0 264 --x0 448                      # south-west corner, grid indices
--centre-lat 39.0 --centre-lon -95.0   # or centre it on a location
```

Cropping by shape, not by name, means a field added to `make_ics` or `make_bcs` later
is cropped automatically. A spatial array in an unexpected axis position raises rather
than being passed through at full size, which would produce a broken file instead of
an error.

`crop_domain.py` also writes `subdomain.json` recording the box, which
`compare_domains.py` uses to align a cropped run against a full one.

## 3b. Or let run_cycle.sh do it

Sections 3 and 4 spell out the manual sequence. `run_cycle.sh` does exactly those steps,
in one call, using the knobs from section 0. Two details it handles that are easy to get
wrong by hand:

- **The forecast gets its own base directory** (`$DATAROOT/subrun`), because `fcst.py`
  resolves inputs as `<base_dir>/<YYYYMMDD>/<HH>/` and the full-domain and cropped npz
  would otherwise collide on that one path.
- **With `OVERLAP_FCST=YES`** the forecast starts early and blocks on a sentinel. The
  crop runs inside the same input phase, so the sentinel is touched only after cropping
  finishes and the forecast cannot read a half-written cropped npz.

It also copies `subdomain.json` next to the outputs, and the plotting and PMM stages
follow the forecast to `subrun` rather than reading the full-domain root.

Verified bit-identical to the manual sequence: given the same cropped inputs, all 70
variables match to 0.

## 4. Run the forecast

`fcst.py` reads its inputs from `<base_dir>/<YYYYMMDD>/<HH>/`, so give the cropped run
its own base directory rather than teaching `fcst.py` about a suffixed one:

```bash
mkdir -p subrun/20260729/17
cp 20260729/17-sub/{hrrr,gfs}_20260729_17.npz subrun/20260729/17/

python3 src/fcst.py net-diffusion/model.keras 2026-07-29T17 24 \
    --num_members 1 --members 0 --batch_size 1 \
    --nc_complevel 1 --nc_least_significant_digit 2 \
    --base_dir subrun --output_dir subrun
```

**GRIB2 output works on a subdomain.** It did not until 2026-07-30: `nc2grib.py`
built Section 3 from a hardcoded `Nx=1799, Ny=1059` *and* a hardcoded first grid point,
so a crop was rejected by grib2io with a shape mismatch and no file was produced.
Section 3 is now derived from the dataset's own `latitude`/`longitude`: only the number
of data points, `Nx`, `Ny`, `La1` and `Lo1` depend on the domain, and everything else
(`LoV`, `LaD`, `Latin1/2`, `Dx/Dy`, earth shape, scanning mode) is a property of the
projection and invariant under cropping.

Verified by round-trip: writing GRIB2 from a 155x151 crop and decoding it back with
grib2io reproduces the NetCDF latitudes and longitudes exactly, and the full-domain
Section 3 is bit-identical to what the old hardcoded path produced.

GRIB2 is nonetheless **off by default everywhere**, on a crop and at full domain
alike: `fcst.py` needs `--grib2`, and `run_cycle.sh` and
`aws/run_subdomain_forecast.sh` need `NO_GRIB2=NO`. NetCDF is what everything
downstream reads, and GRIB2 roughly doubles write time and output volume (a 155x151
crop writes 4.1 MB of GRIB2 per lead hour against 5.5 MB of NetCDF; a full-domain hour
is 324 MB against 435 MB). Ask for it only if a consumer needs it.

## 5. What it costs you

Measured on one cycle (2026-07-29 17Z, 24 h, 1 member, 531x903 = 25.2%), against a
matched full-domain run on the same instance with bit-identical inputs.

### Fidelity

The comparison needs care, because the diffusion sampler draws noise with
`tf.random.stateless_normal(shape, seed=[member, hour])`. That is deterministic per
member and hour, but the draw depends on the tensor SHAPE, so a cropped run gets a
different realization no matter what. A raw full-vs-cropped difference therefore
measures the crop effect *plus* a different random draw.

The fix is a yardstick: a second full-domain run with a different member ID. Its
difference from the first is the model's own stochastic spread.

**Crop/noise ratio: median 1.02, max 1.49**, i.e. cropping costs about as much as swapping
ensemble members.

| variable | random error / noise floor | bias | verdict |
|---|---|---|---|
| PRES | 0.99 -> 1.01 | +14 -> +18 Pa | within noise |
| UGRD10M | 0.99 -> 1.03 | -0.04 -> -0.13 m/s | within noise |
| REFC | 1.00 -> 0.98 | -0.03 -> -0.54 dBZ | within noise |
| APCP | 0.73 -> 0.83 | negligible | within noise |
| **T2M** | 1.02 -> 1.13 | **-0.48 -> -1.06 K** | **real cold bias** |

The T2M bias is the one thing to plan around. It is most plausibly the 39
squeeze-excitation blocks: each takes a `GlobalAveragePooling2D` over the whole field
and uses it to gate channels, so cropping changes 39 domain-mean vectors. A box that
excludes the cool north and the hot desert southwest has a different mean temperature
than CONUS. Because it is a bias rather than lost skill, calibration can remove it,
but no amount of halo will.

### Speed and size

| | full | sub (25.2%) | ratio |
|---|---|---|---|
| wall clock, 24 h | 1308 s | 543 s | 2.4x |
| inference only (less ~210 s model load) | 45.8 s/h | 13.9 s/h | 3.3x |
| NetCDF write, mean | 20.0 s | 4.75 s | 4.2x |
| total output | 10.9 GB | 2.77 GB | 3.9x |

Inference scales at 3.3x, near the 4x area ratio. End-to-end is only 2.4x because
model load is a fixed ~3.5 min, which is 39% of the cropped run's wall clock against
16% of the full run's. A baked AMI or a persistent model server is worth more in a
subdomain deployment than in a full-domain one.

### VRAM: ignore the reported number, a crop fits 24 GB

Every run plateaus at 43953 MiB, 95.4% of an L40S, with median equal to max, and it
does this for a 1.2% crop just as much as for the full domain. That is TensorFlow's
allocator, not demand: `fcst.py` enables memory growth but also sets
`tf.config.optimizer.set_jit(True)`, and XLA claims a large fixed pool regardless of
the work. **The long-quoted "needs 48 GB" figure measures the allocator.** Do not size
hardware from it, or from `nvidia-smi`.

**Measured (2026-07-30): a 155x151 crop completes a 24 h forecast on a `g5.2xlarge`
(A10G 24 GB), status `success`, zero OOM, inside the 20,795 MB TensorFlow reports
usable.** That is the same limit it exhausted at full domain, where the failure was
genuine: one `1059 x 1799 x 139` float32 activation is 1,058,646,016 bytes.

Steady-state `predict()` per lead hour, hour 1 excluded because it carries the XLA
compile, all measured the same way from the `fcst` logs:

| case | hardware | steady state | vs full |
|---|---|---|---|
| full CONUS 1059x1799 | L40S 48 GB | 35.73 s/h | 1.0x |
| 25% crop 531x903 | L40S 48 GB | 9.00 s/h | 4.0x |
| 1.2% crop 155x151 | L40S 48 GB | 3.64 s/h | 9.8x |
| 1.2% crop 155x151 | **A10G 24 GB** | **3.97 s/h** | 9.0x |

The 25% crop is exactly linear in area. The 1.2% crop is sublinear (9.8x on an 80x area
reduction) because per-call overhead independent of area starts to dominate. The A10G is
within 9% of the L40S at this size, at 54% of the hourly price.

At a 1.2% box inference stops being the cost. On the A10G run: input staging ~450 s
(55%), model load and XLA compile ~230 s (28%), **inference 128 s (16%)**, NetCDF writes
10 s. Cropping the input stage, or a baked AMI, is where the next saving is.

## 5b. Worked example: Southern California

Region requested: `N 35.0, W -118.77, S 33.25, E -117.0` (roughly the LA basin
through to San Diego and inland).

To run it, end to end:

```bash
# locally (24 h, 1 member, no plots)
SUB_BBOX=35.0,-118.77,33.25,-117.0 SUB_HALO=40 \
    ./run_cycle.sh 2026-07-29T17 24 1 1 "$PWD" "$PWD" NO

# on AWS, on the cheaper 24 GB instance a crop now fits
aws/run_on_ec2.sh --bucket mantari-cast1 --init-time 2026-07-29T17 --lead-hours 24 \
    --instance-type g5.2xlarge --bbox 35.0,-118.77,33.25,-117.0 --halo 40
```

To see the box it picks without running a forecast:

```bash
python3 src/crop_domain.py --in-dir 20260729/17 --out-dir 20260729/17-socal \
    --init-time 2026-07-29T17 --bbox 35.0,-118.77,33.25,-117.0 --halo 40 --dry-run
```

which reports:

| | |
|---|---|
| region on the grid | rows [399:474], cols [244:311] = 75 x 67 cells (225 x 201 km) |
| **subdomain** | **155 x 151, `--y0 359 --x0 202`** |
| area | **1.2%** of CONUS |
| corners | SW (31.65, -119.66), NE (36.58, -116.00) |
| halo achieved | south 40, north 40, west 42, east 42 cells |
| npz size | 1029 MB -> 13 MB (IC), 1951 MB -> 22 MB (BC per 6 leads) |

Verified: all 3,544 grid cells of the requested region are present in the crop, and
the region sits 40+ cells from every edge.

Halo variants, if you want more margin:

| halo | size | area | y0, x0 |
|---|---|---|---|
| 20 cells (60 km) | 115 x 111 | 0.67% | 379, 222 |
| **40 cells (120 km)** | **155 x 151** | **1.23%** | **359, 202** |
| 80 cells (240 km) | 235 x 231 | 2.85% | 319, 162 |

### Two things change at this size

**This box has now been measured directly** (2026-07-30, cycle 2026-07-29 00Z, 24 h,
against a matched full-domain run and a noise yardstick on the same instance):

| | 531x903 (25.2%) | **155x151 SoCal (1.2%)** |
|---|---|---|
| crop/noise ratio | median 1.02, max 1.49 | median **1.10**, max 1.95 |
| T2M bias, f01 -> f24 | -0.48, -1.06, -0.60, -0.93 K | **-0.11, -0.24, +0.20, -0.13 K** |
| worst variable | T2M (systematic cold bias) | **UGRD10M at f24** (ratio 1.50, -1.22 m/s) |

A box 20x smaller costs only slightly more against the noise floor. The expectation
that the T2M bias would grow was **wrong**: it nearly vanished and changed sign. The
bias tracks **where** the box sits relative to the CONUS mean, not how small it is, so
it still has to be measured per box (see section 6) but small does not imply bad. What
degrades instead is 10 m wind at long leads: at f24 the crop-versus-full difference is
as large as the field's own spatial standard deviation. The noise floor for that field
is itself 0.67 of sd, so it is loosely constrained either way, but track it if
long-lead surface wind matters.

The halo guidance is unchanged and now confirmed on two different boxes: error is
elevated within roughly the first 20 cells and flat beyond.

**Inference stops being the bottleneck.** Measured, not extrapolated, on the A10G run
of this box:

| stage | time | share |
|---|---|---|
| input staging, full domain | ~450 s | 55% |
| model load + XLA compile | ~230 s | 28% |
| **inference, 24 lead hours** | **128 s** | **16%** |
| NetCDF write, 25 files | 10 s | 1% |

(An earlier version of this table extrapolated inference by area and put its share at
3%. The real figure is 16%: the 1.2% crop scales 9.8x rather than 80x, because per-call
overhead that does not depend on area starts to dominate at this size. The conclusion
is the same, just less extreme.)

The 24 h forecast becomes cheap, and the run is then dominated by **downloading and
preprocessing full-CONUS GFS and HRRR** (~450 s) and **loading the model** (~230 s).
Optimising the model further would be pointless; the targets are the input stage
(below) and a baked AMI or persistent model server to amortise the ~230 s load.

### Cropping the HRRR-side inputs, not just the outputs of preprocessing

`SUB_BBOX` (via `run_cycle.sh` or `run_on_ec2.sh --bbox`) crops the *raw* HRRR
GRIB2 before `make_ics.py`/`make_bcs.py` run, rather than cropping their full-domain
`.npz` output afterward. `make_ics.py` and `make_bcs.py` are not modified:
`src/crop_grib2.py` crops the GRIB2 files with `wgrib2 -ijsmall_grib`, which
recomputes Section 3 (Nx, Ny, La1, Lo1) for the smaller grid the same way
`nc2grib.py` does on the way out, and both scripts read whatever grid is actually
in the file. Their hardcoded-1059x1799 shape check is a logged warning, never
fatal (`src/make_ics.py:434`, `src/make_bcs.py:765`).

Measured, isolated, repeated:

| stage | full domain | HRRR-side crop (1.2% box) | speedup |
|---|---|---|---|
| `make_ics.py` | 34.7 s | 7.6-7.9 s | **4.6x** |
| `make_bcs.py`, 1 lead hour | 34.2 s | 22.6 s | 1.5x |
| `make_bcs.py`, 6 lead hours | 49.8 s | 30.8 s | 1.6x |

`make_ics.py` scales almost with area, since it is mostly pygrib field reads over
the grid. `make_bcs.py` scales much less, because **GFS itself is not cropped**:
it is read once per lead hour at its native global resolution regardless of the
HRRR-side box, and that read is now the larger share of make_bcs's cost. Only the
xESMF regrid *target* (the cropped `--hrrr_grid_file`) got smaller. Cropping GFS
too is the next lever and is not implemented; it needs a margin sized for the
bilinear stencil at GFS's own (coarser) resolution, not the HRRR halo.

**Verified equivalent, not merely faster.** Given the *same* cropped inputs,
`make_ics.py`'s and `make_bcs.py`'s output is bit-identical to running them by
hand (70 of 70 pressure/surface variables, worst diff 0). Comparing a
raw-cropped run against the corresponding window of a full-domain run of the
*same* cycle: worst |diff| 4.75e-5 on the IC (5 of 140 channels touched, all at
GRIB2 repacking precision — the same 1e-5 to 1e-18 relative noise already
measured on the raw GRIB2 fields themselves, not resampling error) and 1.96e-4 on
the BC (a different-sized xESMF sparse matrix computes the same bilinear weights
in a different floating-point order). Both are far below the model's own
operating precision and the crop/noise ratios above.

**LAND and OROG needed a normalization fix to get there.** `make_ics.py` falls
back to computing a variable's mean/std *from whatever data it is given* when
`normalize-stats.nc` lacks an entry (`src/make_ics.py:333`), which it does for
these two constant fields. At full domain that is harmless: the CONUS land
fraction and terrain are the same every cycle. But a raw crop's *local* land
fraction differs from CONUS's, so before the fix, a full-domain run and a
same-cycle crop disagreed on LAND's own normalization (0.223 in normalized
units, the single largest discrepancy found) and OROG's (0.8, entirely from a
domain-mean shift, not the field itself). OROG is bit-identical across every HRRR
cycle checked; LAND varies by under 0.02% of pixels cycle to cycle (coastal/lake
mask changes). `net-diffusion/normalize-stats.nc` now carries fixed LAND/OROG
entries (mean/std/min/max, in the same `('stat',)` shape every other surface
variable uses), computed once across three cycles spanning different seasons.
This is the only change to a shipped data file this work made; it moves 2 of 140
IC channels by at most 0.000177 for an ordinary **full-domain** run and is not
a code change to either excluded script.

**A caching hazard, handled without touching make_bcs.py.** `make_bcs.py`
hardcodes its regridding weights filename (`gfs_to_hrrr_weights.nc`,
`src/make_bcs.py:120`) with `reuse_weights=True`, looked up relative to whatever
directory the process is CWD'd in. A full-domain weights file and a subdomain one
are different shapes; xESMF fails loudly (`invalid entry in coordinates array`)
rather than silently misregridding when they disagree, but a run that dies there
is still a run that dies, and two different subdomain boxes collide the same way.
`run_cycle.sh` keys a cache of weights files by the target grid actually in play
(`WEIGHTS_KEY`, either `full` or `<height>x<width>_y<y0>_x<x0>`), stages the right
one in before calling `make_bcs.py`, and removes the generic filename afterward so
a later run at a different box cannot pick it up by accident.

## 6. Verifying your own crop

Do not trust a new box or a new season without checking it:

```bash
# verify a specific region, which is what you normally want
aws/run_domain_test.sh --bucket BUCKET --lead-hours 24 \
    --bbox 35.0,-118.77,33.25,-117.0 --halo 40

# or a grid-centred box of a given size
aws/run_domain_test.sh --bucket BUCKET --lead-hours 24 --height 531 --width 903
```

`--bbox` matters here: without it the crop is centred on the CONUS grid, so you would
be measuring a box over Kansas rather than the one you intend to run.

This runs all three forecasts on one instance (~75 min, ~$2.80), so they share
bit-identical inputs. Then:

```bash
python3 src/compare_domains.py \
    --full-dir DIR/full --sub-dir DIR/sub --ref2-dir DIR/m1 \
    --subdomain-json DIR/subdomain.json --leads 1,6,12,24 --out-dir OUT
```

Read two things first:

1. **The crop/noise ratio.** Near 1 means the crop is inside the model's own
   uncertainty. Several times that is a real degradation.
2. **The deepest-interior bin at the shortest lead.** Error there cannot come from
   missing inflow, so if it is well above the noise floor the cause is global (the
   squeeze-excitation gating) and no halo will fix it.

Without `--ref2-dir` the tool warns, correctly, that the crop RMSE cannot be
attributed to cropping at all.

## Caveats on the published numbers

One cycle, one season, one box. The T2M bias in particular should depend on which
climatological regions the box includes, so a winter case and a differently placed box
are worth testing before relying on the 1 K figure. The full-domain run is the
reference, not truth: this measures agreement with the full-domain forecast, not skill
against observations. Distance bins mix all four edges, so an inflow effect
concentrated on the upwind edge would be diluted.

## Related

| file | role |
|---|---|
| [../src/crop_domain.py](../src/crop_domain.py) | crops IC/BC npz; enforces the size rule; also exposes `size_and_place_bbox()`, shared with crop_grib2.py |
| [../src/crop_grib2.py](../src/crop_grib2.py) | crops the RAW HRRR GRIB2 before make_ics.py/make_bcs.py run (`wgrib2 -ijsmall_grib`) |
| [../src/compare_domains.py](../src/compare_domains.py) | the fidelity analysis |
| [../aws/domain_test.sh](../aws/domain_test.sh) | three-run experiment, on-instance |
| [../aws/run_domain_test.sh](../aws/run_domain_test.sh) | launcher for the above |
| [../aws/run_subdomain_forecast.sh](../aws/run_subdomain_forecast.sh) | one production forecast on a crop; drive with `run_on_ec2.sh --run-cmd` |

## Experiment records

The measurements in this document come from two archived experiments. Each holds the
raw logs, the comparison JSON, figures, and the exact code state.

| experiment | what it established |
|---|---|
| `s3://mantari-cast1/hrrrcast/experiments/2026-07-29-subdomain/` | 25.2% box: crop/noise ratio ~1, T2M cold bias, halo 20-40 cells, VRAM reading is an allocator artifact |
| `s3://mantari-cast1/hrrrcast/experiments/2026-07-30-socal-a10g/` | a crop runs on a 24 GB A10G; the 1.2% SoCal box holds; the T2M bias tracks placement, not size |

Read the second one's `FINDINGS.md` first: it revises two inferences from the first.
