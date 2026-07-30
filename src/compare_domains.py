#!/usr/bin/env python3
"""Quantify what cropping the domain costs, by differencing a full-domain and a
subdomain forecast of the same cycle.

The decision this informs: a quarter-area subdomain should fit a 24 GB A10G
(g5.2xlarge, about half the hourly price of g6e.2xlarge) and run roughly 4x faster,
taking hourly operation from ~$875/month to well under $200. Worth doing only if the
forecast is still usable.

TWO DISTINCT ERROR SOURCES, AND WHY THE DIAGNOSTIC IS SHAPED THIS WAY

  1. Missing inflow. There is no lateral boundary condition. GFS forcing is present
     at every point, so the synoptic constraint survives, but fine-scale features
     advecting in from outside are absent. This error enters at the crop edge and
     propagates inward at the wind speed, so it should look like a front moving in
     from the boundary: small in the deep interior at short lead, filling the domain
     by 24 h. Diagnosed by binning error against distance from the crop boundary.

  2. Squeeze-excitation gating. 39 blocks take a GlobalAveragePooling2D over the whole
     field and use it to gate channels. Cropping changes those 39 domain-mean vectors,
     which perturbs the solution EVERYWHERE at once, including the deep interior at
     f01. Diagnosed by the intercept: error in the deepest interior bin at the
     shortest lead. If that is already large, cropping is not viable at any halo size,
     because no amount of halo fixes a global-statistics change.

  Distinguishing them matters. Source 1 is fixable by cropping a bigger box and
  trusting a smaller interior. Source 2 is not fixable at all without retraining.

The full-domain run is the reference, not truth. Both runs share bit-identical inputs
(see aws/domain_test.sh), so every difference is attributable to the crop.

Usage:
    python compare_domains.py --full-dir DIR --sub-dir DIR --subdomain-json FILE \\
        [--leads 1,6,12,24] [--out-dir DIR]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import xarray as xr

# Variables worth reporting on, chosen to span smooth/large-scale through
# noisy/convective, because the crop should hurt them very differently.
VARS = [
    ("T2M", "2 m temperature", "K"),
    ("PRES", "surface pressure", "Pa"),
    ("UGRD10M", "10 m u wind", "m/s"),
    ("REFC", "composite reflectivity", "dBZ"),
    ("APCP", "1 h precipitation", "mm"),
]

NAVY, BLUE, CYAN = "#042E60", "#074FB5", "#00B5FF"


def load(d, lead):
    hits = sorted(glob.glob(os.path.join(d, f"hrrrcast_m*_f{lead:02d}.nc")))
    if not hits:
        return None
    return xr.open_dataset(hits[0])


def boundary_distance(h, w):
    """Grid cells to the nearest edge of the subdomain, as a (h, w) array."""
    yy = np.minimum(np.arange(h), h - 1 - np.arange(h))[:, None]
    xx = np.minimum(np.arange(w), w - 1 - np.arange(w))[None, :]
    return np.minimum(np.broadcast_to(yy, (h, w)), np.broadcast_to(xx, (h, w)))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full-dir", required=True)
    p.add_argument("--sub-dir", required=True)
    p.add_argument("--subdomain-json", required=True)
    p.add_argument("--ref2-dir", help="second full-domain run (different member) for "
                                      "the stochastic noise floor; strongly recommended")
    p.add_argument("--leads", default="1,6,12,24")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--km-per-cell", type=float, default=3.0,
                   help="HRRR grid spacing, for reporting distances in km [3]")
    a = p.parse_args()

    box = json.load(open(a.subdomain_json))
    y0, x0, H, W = box["y0"], box["x0"], box["height"], box["width"]
    leads = [int(x) for x in a.leads.split(",")]
    os.makedirs(a.out_dir, exist_ok=True)

    print(f"subdomain rows [{y0}:{y0+H}] cols [{x0}:{x0+W}]  "
          f"{H}x{W} = {box['area_fraction']:.1%} of the full grid")
    print(f"init {box['init_time']}   leads {leads}\n")

    dist = boundary_distance(H, W)
    # Bin edges in grid cells. Non-uniform on purpose: the interesting behavior is
    # near the boundary, and the deepest bin is the one that isolates the
    # squeeze-excitation effect from the inflow effect.
    edges = [0, 5, 10, 20, 40, 80, 160, max(265, dist.max() + 1)]
    labels = [f"{edges[i]}-{edges[i+1]}" for i in range(len(edges) - 1)]

    rows = []
    per_bin = {}          # (var, lead) -> list of rmse per bin
    for lead in leads:
        dsf, dss = load(a.full_dir, lead), load(a.sub_dir, lead)
        dsr = load(a.ref2_dir, lead) if a.ref2_dir else None
        if dsf is None or dss is None:
            print(f"  f{lead:02d}: MISSING "
                  f"({'full' if dsf is None else ''}{' sub' if dss is None else ''})")
            continue
        for var, longname, units in VARS:
            if var not in dsf or var not in dss:
                continue
            full = np.asarray(dsf[var].values[0, 0])[y0:y0 + H, x0:x0 + W]
            sub = np.asarray(dss[var].values[0, 0])
            if sub.shape != full.shape:
                print(f"  shape mismatch for {var} at f{lead:02d}: "
                      f"sub {sub.shape} vs cropped-full {full.shape}; skipping")
                continue
            diff = sub - full
            denom = np.std(full)
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            noise = np.nan
            if dsr is not None and var in dsr:
                ref2 = np.asarray(dsr[var].values[0, 0])[y0:y0 + H, x0:x0 + W]
                noise = float(np.sqrt(np.mean((ref2 - full) ** 2)))
            rows.append(dict(lead=lead, var=var, units=units,
                             rmse=rmse, bias=float(np.mean(diff)),
                             maxabs=float(np.max(np.abs(diff))),
                             sd_full=float(denom), noise=noise,
                             ratio=float(rmse / noise) if noise and np.isfinite(noise) and noise > 0 else np.nan,
                             frac=float(rmse / denom) if denom > 0 else np.nan))
            bin_rmse = []
            for i in range(len(edges) - 1):
                m = (dist >= edges[i]) & (dist < edges[i + 1])
                bin_rmse.append(float(np.sqrt(np.mean(diff[m] ** 2))) if m.any() else np.nan)
            per_bin[(var, lead)] = bin_rmse

    if not rows:
        print("no comparable output found; nothing to report", file=sys.stderr)
        sys.exit(1)

    # --- table 1: overall error, normalized by the field's own variability ---
    print(f"{'lead':>5} {'variable':<10} {'crop RMSE':>10} {'noise':>10} {'ratio':>7} "
          f"{'bias':>10} {'sd(full)':>10} {'RMSE/sd':>8}")
    print("-" * 78)
    for r in rows:
        nz = f"{r['noise']:10.4f}" if np.isfinite(r["noise"]) else f"{'-':>10}"
        rt = f"{r['ratio']:7.2f}" if np.isfinite(r["ratio"]) else f"{'-':>7}"
        print(f"f{r['lead']:02d}   {r['var']:<10} {r['rmse']:10.4f} {nz} {rt} "
              f"{r['bias']:+10.4f} {r['sd_full']:10.4f} {r['frac']:8.3f}")
    if not a.ref2_dir:
        print("\n  WARNING: no --ref2-dir, so the noise floor is unknown and the crop\n"
              "  RMSE above CANNOT be attributed to cropping. Rerun with the\n"
              "  full-m1 output from aws/domain_test.sh.")
    else:
        rr = [r["ratio"] for r in rows if np.isfinite(r["ratio"])]
        if rr:
            print(f"\n  crop/noise ratio: median {np.median(rr):.2f}, max {max(rr):.2f}")
            if np.median(rr) <= 1.5:
                print("  -> cropping costs about as much as swapping ensemble members.")
            elif np.median(rr) <= 3:
                print("  -> cropping costs a few times the stochastic spread; usable with care.")
            else:
                print("  -> cropping costs much more than the stochastic spread: a real degradation.")

    # --- table 2: error vs distance from the crop boundary -------------------
    print(f"\nRMSE by distance from the crop boundary "
          f"(bins in grid cells; 1 cell = {a.km_per_cell:g} km)")
    hdr = f"{'variable':<10} {'lead':>5} " + " ".join(f"{l:>9}" for l in labels)
    print(hdr); print("-" * len(hdr))
    for (var, lead), vals in sorted(per_bin.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        print(f"{var:<10} f{lead:02d}   " +
              " ".join(f"{v:9.4f}" if np.isfinite(v) else f"{'-':>9}" for v in vals))

    # --- the interpretation the decision actually needs ---------------------
    print("\n--- reading the numbers ---")
    for var, longname, units in VARS:
        k = (var, min(leads))
        if k not in per_bin:
            continue
        # Deepest POPULATED bin, not simply the last one. A small box has no
        # cells in the outer bins: a 155x151 crop is at most 75 cells from a
        # boundary, so the 80-160 and 160-265 bins are empty and per_bin[k][-1]
        # is NaN. Taking the last bin unconditionally printed "nan" for every
        # variable and made the summary unreadable on small subdomains.
        populated = [(i, v) for i, v in enumerate(per_bin[k]) if np.isfinite(v)]
        if not populated:
            continue
        deep_i, deep = populated[-1]
        edge = per_bin[k][0]
        deep_label = labels[deep_i]
        r0 = next(r for r in rows if r["var"] == var and r["lead"] == min(leads))
        sd = r0["sd_full"]
        print(f"  {var:<9} at f{min(leads):02d}: deepest-bin ({deep_label} cells) RMSE "
              f"{deep:.4f} {units} ({deep/sd*100 if sd else float('nan'):.1f}% of sd)  "
              f"edge RMSE {edge:.4f}")
        # Judge the deep interior against the NOISE FLOOR, not against the field's
        # spatial spread. "deep RMSE > 5% of sd" flags any field whose spread is
        # small, which is most of them: on the 155x151 SoCal box it fired for four
        # of five variables including ones whose crop/noise ratio was ~1.0, i.e.
        # pure ensemble noise. The gating signature is a deep interior that stays
        # WORSE THAN THE NOISE while being no better than the edge.
        noise = r0["noise"]
        flat = np.isfinite(edge) and edge > 0 and deep / edge > 0.75
        if np.isfinite(noise) and noise > 0 and deep / noise > 1.25 and flat:
            print(f"    ^ deepest bin is {deep/noise:.2f}x the noise floor and no better "
                  f"than the edge ({deep:.4f} vs {edge:.4f}): consistent with the "
                  "squeeze-excitation gating change, which no halo can fix.")
        elif np.isfinite(noise) and noise > 0 and deep / noise <= 1.25:
            print(f"    deepest bin is {deep/noise:.2f}x the noise floor: "
                  "indistinguishable from the model's own ensemble spread.")

    # --- plots --------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib unavailable; skipping plots)")
        return

    nvar = len({r["var"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # left: RMSE/sd vs lead
    for var, longname, units in VARS:
        pts = [(r["lead"], r["frac"]) for r in rows if r["var"] == var]
        if pts:
            axes[0].plot(*zip(*sorted(pts)), "o-", label=var)
    axes[0].set_xlabel("lead hour"); axes[0].set_ylabel("RMSE / sd(full domain)")
    axes[0].set_title("Cost of cropping vs lead time\n(normalized by each field's own variability)",
                      fontsize=10, color=NAVY, weight="bold", loc="left")
    axes[0].legend(frameon=False, fontsize=8); axes[0].grid(alpha=.3)

    # right: RMSE vs distance from boundary, longest lead
    L = max(leads)
    centres = [(edges[i] + edges[i + 1]) / 2 * a.km_per_cell for i in range(len(edges) - 1)]
    for var, longname, units in VARS:
        if (var, L) in per_bin:
            v = per_bin[(var, L)]
            sd = next(r["sd_full"] for r in rows if r["var"] == var and r["lead"] == L)
            if sd > 0:
                axes[1].plot(centres, [x / sd for x in v], "o-", label=var)
    axes[1].set_xlabel(f"distance from crop boundary (km)")
    axes[1].set_ylabel("RMSE / sd")
    axes[1].set_xscale("log")
    axes[1].set_title(f"Where the error lives, f{L:02d}\n"
                      "falling to the right = inflow error; flat = global gating error",
                      fontsize=10, color=NAVY, weight="bold", loc="left")
    axes[1].legend(frameon=False, fontsize=8); axes[1].grid(alpha=.3)

    out = os.path.join(a.out_dir, "domain_comparison.png")
    fig.tight_layout(); fig.savefig(out, dpi=140, facecolor="white")
    print(f"\nwrote {out}")

    # NaN is not valid JSON, and empty distance bins produce it. Sanitize to null so
    # the file can actually be read back; an earlier version wrote a file that
    # json.load() rejected.
    def clean(o):
        if isinstance(o, np.generic):      # numpy int64/float64 are NOT json types
            o = o.item()
        if isinstance(o, float):
            return None if not np.isfinite(o) else o
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o

    # Serialize fully BEFORE opening the file, then write and read back. json.dump
    # writes incrementally, so a TypeError partway through leaves a truncated file
    # on disk and still prints nothing about it. That is exactly what happened to
    # the 2026-07-29 experiment: `edges` ends in max(265, dist.max() + 1), which is
    # a numpy int64 whenever the box is at least 530 cells across, clean() passed
    # numpy scalars through untouched, and the archived JSON was cut off mid-array
    # in both the local bundle and the S3 copy.
    payload = json.dumps(clean(dict(box=box, rows=rows,
                                    bins=dict(edges=edges, labels=labels,
                                              rmse={f"{v}_f{l:02d}": r
                                                    for (v, l), r in per_bin.items()}))),
                         indent=2)
    jpath = os.path.join(a.out_dir, "domain_comparison.json")
    with open(jpath, "w") as f:
        f.write(payload)
    with open(jpath) as f:
        json.load(f)                       # fail loudly here, not on the reader
    print(f"wrote {jpath}")


if __name__ == "__main__":
    main()
