#!/usr/bin/env python3
"""Crop preprocessed HRRRCast inputs to a subdomain, for cheaper inference.

Motivation: a full-domain 24 h forecast holds 43.9 GB of VRAM and takes 28.4 min on
an L40S. Activation memory scales with H*W, so a quarter-area subdomain should fit a
24 GB A10G (g5.2xlarge, roughly half the hourly price of g6e.2xlarge) and run about
4x faster. This script produces the cropped inputs so that claim can be tested
against a full-domain run using bit-identical source data.

WHY CROP THE NPZ RATHER THAN THE PIPELINE
    src/fcst.py already derives its grid size from the input arrays
    (`nlat, nlon = input_shape[1], input_shape[2]`), so it needs no changes at all.
    make_ics.py and make_bcs.py do hardcode 1059x1799, but for an experiment there
    is no reason to touch them: run the input stages once at full domain, then crop.
    Both forecasts then share exactly the same source data, which is what makes the
    comparison meaningful. Cropping the pipeline instead would also change the xESMF
    regridding weights, confounding "does the model tolerate a smaller domain" with
    "did the interpolation change".

THE SIZE CONSTRAINT IS NOT NEGOTIABLE
    The trained model bakes a FIXED reflect-padding of [[3,2],[1,0]] (H+5, W+1) and
    then downsamples by 8 through three stride-2 poolings:
        1059 + 5 = 1064 = 8 x 133
        1799 + 1 = 1800 = 8 x 225
    Because that padding is a constructor argument stored in model.keras, it does not
    adapt. Any subdomain must therefore satisfy

        (H + 5) % 8 == 0   ->   H % 8 == 3
        (W + 1) % 8 == 0   ->   W % 8 == 7

    A size that violates this does not raise a clean error; the UNet's skip
    connections misalign after upsampling, which surfaces as a shape mismatch deep in
    the model or, worse, as silently shifted output. This script refuses such sizes.

WHAT THIS CANNOT FIX
    The model has 39 squeeze-excitation blocks, each taking a GlobalAveragePooling2D
    over the WHOLE field and using it to gate channels. Cropping changes those 39
    domain-mean vectors, so the cropped forecast differs from the full-domain one in
    the interior, not merely near the edges. Separately, there is no lateral boundary
    condition: GFS forcing is present at every point (so the synoptic constraint
    survives) but fine-scale features advecting in from outside are absent. Measuring
    both effects is the entire point of the comparison; see src/compare_domains.py.

Usage:
    python crop_domain.py --in-dir 20260729/13 --out-dir 20260729/13-sub \\
        --init-time 2026-07-29T13 --y0 264 --x0 448 --height 531 --width 903

    python crop_domain.py ... --centre-lat 39.0 --centre-lon -95.0 \\
        --height 531 --width 903        # place the box by lat/lon instead

    python crop_domain.py --in-dir 20260729/17 --out-dir 20260729/17-socal \\
        --init-time 2026-07-29T17 --bbox 35.0,-118.77,33.25,-117.0 --halo 40
        # size and place the box from a region of interest, adding a halo and
        # rounding up to the nearest legal size. Usually what you want.
"""
import argparse
import json
import os
import sys

import numpy as np

# From the trained model's ReflectPadLayer/UnpadLayer and its three stride-2 poolings.
PAD_H = 5      # [[3, 2]]
PAD_W = 1      # [[1, 0]]
DOWNSAMPLE = 8

FULL_H, FULL_W = 1059, 1799


def next_valid_size(h: int, w: int) -> tuple:
    """Smallest legal (H, W) that is >= (h, w).

    Used when sizing a box from a region of interest plus a halo: the requested
    extent is a lower bound, and the model's padding rule decides the actual size.
    Rounding UP rather than to the nearest keeps the requested halo intact.
    """
    # int() at the boundary: callers derive the target from np.where, which yields
    # numpy int64, and that propagates into subdomain.json where json.dump rejects it.
    h, w = int(h), int(w)
    while (h + PAD_H) % DOWNSAMPLE:
        h += 1
    while (w + PAD_W) % DOWNSAMPLE:
        w += 1
    return h, w


def valid_size(h: int, w: int):
    """Return (ok, message). See the module docstring for why this is fixed."""
    bad = []
    if (h + PAD_H) % DOWNSAMPLE:
        need = [x for x in range(h - 8, h + 9) if x > 0 and (x + PAD_H) % DOWNSAMPLE == 0]
        bad.append(f"height {h}: (h+{PAD_H}) must be divisible by {DOWNSAMPLE} "
                   f"(h % 8 == 3); nearest valid: {need}")
    if (w + PAD_W) % DOWNSAMPLE:
        need = [x for x in range(w - 8, w + 9) if x > 0 and (x + PAD_W) % DOWNSAMPLE == 0]
        bad.append(f"width {w}: (w+{PAD_W}) must be divisible by {DOWNSAMPLE} "
                   f"(w % 8 == 7); nearest valid: {need}")
    return (not bad), "; ".join(bad)


def size_and_place_bbox(lats: np.ndarray, lons: np.ndarray, bbox: str, halo: int):
    """Size and place a legal box containing `bbox` plus `halo` cells on every side.

    Shared by this script's --bbox path and src/crop_grib2.py, so a GRIB2 crop (sized
    from a raw grid's own lat/lon) and an npz crop land on the identical box for the
    same region and halo. Prints the same diagnostics either way and exits(1) on the
    same errors this script has always raised for a bad box.

    Returns (y0, x0, height, width, meta) where meta is the dict this script has
    always written to subdomain.json (minus the npz-specific init_time/area_fraction,
    added by the caller).
    """
    H, W = lats.shape
    try:
        n_, w_, s_, e_ = [float(v) for v in bbox.split(",")]
    except ValueError:
        print("ERROR: --bbox must be N,W,S,E in degrees", file=sys.stderr)
        sys.exit(1)
    # The HRRR grid is Lambert conformal, so a lat/lon box is a curved region in
    # index space. Take the index bounding box of every cell inside the region;
    # that is a superset of the request, which is what "must contain" needs.
    inside = ((lats >= s_) & (lats <= n_) & (lons >= w_) & (lons <= e_))
    if not inside.any():
        print(f"ERROR: no grid cell falls inside N={n_} W={w_} S={s_} E={e_}. "
              "Outside the HRRR CONUS domain?", file=sys.stderr)
        sys.exit(1)
    iy, ix = np.where(inside)
    ry0, ry1, rx0, rx1 = iy.min(), iy.max(), ix.min(), ix.max()
    roi_h, roi_w = ry1 - ry0 + 1, rx1 - rx0 + 1

    want_h, want_w = roi_h + 2 * halo, roi_w + 2 * halo
    height, width = next_valid_size(want_h, want_w)
    if height > H or width > W:
        print(f"ERROR: region plus {halo}-cell halo needs {height}x{width}, "
              f"larger than the {H}x{W} grid. Reduce --halo.", file=sys.stderr)
        sys.exit(1)
    # Centre on the region, then clamp inside the grid without changing the size,
    # so the legality of (height, width) still holds.
    y0 = max(0, min((ry0 + ry1) // 2 - height // 2, H - height))
    x0 = max(0, min((rx0 + rx1) // 2 - width // 2, W - width))
    y0 = int(max(0, min(y0, H - height)))
    x0 = int(max(0, min(x0, W - width)))

    print(f"\nregion of interest  N={n_} W={w_} S={s_} E={e_}")
    print(f"  grid cells inside  {int(inside.sum())}")
    print(f"  index extent       rows [{ry0}:{ry1+1}]  cols [{rx0}:{rx1+1}]  "
          f"({roi_h} x {roi_w} cells, {roi_h*3} x {roi_w*3} km)")
    print(f"  requested halo     {halo} cells ({halo*3} km) per side")
    print(f"  size needed        {want_h} x {want_w}  ->  legal "
          f"{height} x {width}")
    # Report the halo ACTUALLY achieved on each side. Clamping at a grid edge can
    # silently shrink it, and a thin side is exactly where the crop error lives.
    halos = {"south": ry0 - y0, "north": (y0 + height - 1) - ry1,
             "west": rx0 - x0, "east": (x0 + width - 1) - rx1}
    print("  halo achieved      " + "  ".join(
        f"{k}={v}" + ("!" if v < halo else "") for k, v in halos.items()))
    thin = {k: v for k, v in halos.items() if v < halo}
    if thin:
        print(f"  NOTE: halo is thinner than requested on {', '.join(thin)} "
              "(clamped at the grid edge). Error is largest near a crop edge, so "
              "treat that side's output with more caution.")

    meta = dict(y0=y0, x0=x0, height=height, width=width,
                full_height=int(H), full_width=int(W),
                requested_bbox=bbox, requested_halo=halo,
                sw_lat=float(lats[y0, x0]), sw_lon=float(lons[y0, x0]),
                ne_lat=float(lats[y0 + height - 1, x0 + width - 1]),
                ne_lon=float(lons[y0 + height - 1, x0 + width - 1]))
    return y0, x0, height, width, meta


def crop_npz(src: str, dst: str, y0: int, x0: int, h: int, w: int) -> dict:
    """Crop every spatial array in an npz, pass metadata through unchanged.

    Arrays are identified by shape rather than by name so that a new field added to
    make_ics or make_bcs is cropped automatically instead of being silently passed
    through at full size, which would produce a broken file rather than an error.
    """
    z = np.load(src, allow_pickle=True)
    if "lats" not in z:
        raise ValueError(f"{src} has no 'lats'; not a preprocessed HRRRCast input")
    H, W = z["lats"].shape
    if y0 + h > H or x0 + w > W:
        raise ValueError(f"crop y[{y0}:{y0+h}] x[{x0}:{x0+w}] exceeds the {H}x{W} grid")

    out, report = {}, {"cropped": [], "passthrough": []}
    for key in z.files:
        a = z[key]
        if a.ndim >= 3 and a.shape[1:3] == (H, W):
            # (n, H, W, C): model_input for both ICs and BCs
            out[key] = a[:, y0:y0 + h, x0:x0 + w, ...]
            report["cropped"].append(f"{key}{a.shape}->{out[key].shape}")
        elif a.ndim == 2 and a.shape == (H, W):
            # lats, lons, LAND_raw, OROG_raw
            out[key] = a[y0:y0 + h, x0:x0 + w]
            report["cropped"].append(f"{key}{a.shape}->{out[key].shape}")
        elif a.ndim >= 2 and H in a.shape and W in a.shape:
            raise ValueError(
                f"{key} has shape {a.shape}, which contains the grid dims in an "
                "unexpected position. Refusing to guess; extend crop_npz().")
        else:
            out[key] = a
            report["passthrough"].append(key)

    # Metadata that describes the grid must follow the data, or fcst.py and the
    # NetCDF writer will disagree with the arrays they were handed.
    if "grid_height" in out:
        out["grid_height"] = np.array(h)
    if "grid_width" in out:
        out["grid_width"] = np.array(w)

    np.savez_compressed(dst, **out)
    return report


def main():
    p = argparse.ArgumentParser(
        description="Crop preprocessed HRRRCast inputs to a subdomain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[-1])
    p.add_argument("--in-dir", required=True, help="directory holding hrrr_*.npz and gfs_*.npz")
    p.add_argument("--out-dir", required=True, help="where to write the cropped npz")
    p.add_argument("--init-time", required=True, help="YYYY-MM-DDTHH, used for the filenames")
    p.add_argument("--height", type=int, default=531, help="subdomain rows (must be %%8==3) [531]")
    p.add_argument("--width", type=int, default=903, help="subdomain cols (must be %%8==7) [903]")
    p.add_argument("--y0", type=int, help="south edge row index [centred]")
    p.add_argument("--x0", type=int, help="west edge column index [centred]")
    p.add_argument("--centre-lat", type=float, help="place the box on this latitude instead of --y0")
    p.add_argument("--centre-lon", type=float, help="place the box on this longitude instead of --x0")
    p.add_argument("--bbox", help="region of interest as N,W,S,E in degrees. The box is "
                                  "sized and placed to contain it plus --halo on every "
                                  "side, overriding --height/--width/--y0/--x0.")
    p.add_argument("--halo", type=int, default=40,
                   help="cells of halo around --bbox on every side [40]. Measured: "
                        "error is elevated within the first 10-40 cells of the crop "
                        "edge and flat beyond, so 40 (120 km) captures nearly all the "
                        "available improvement. Treat the halo as scratch, not product.")
    p.add_argument("--dry-run", action="store_true", help="validate and report, write nothing")
    a = p.parse_args()

    # --bbox sizes the box itself, so validate only when the size was given directly.
    if not a.bbox:
        ok, msg = valid_size(a.height, a.width)
        if not ok:
            print(f"ERROR: invalid subdomain size.\n  {msg}\n\n"
                  "  The model's reflect-padding is fixed at [[3,2],[1,0]] and cannot adapt;\n"
                  "  see the module docstring in src/crop_domain.py.", file=sys.stderr)
            sys.exit(1)

    stamp = a.init_time.replace("-", "").replace("T", "_")
    ic_src = os.path.join(a.in_dir, f"hrrr_{stamp}.npz")
    bc_src = os.path.join(a.in_dir, f"gfs_{stamp}.npz")
    for f in (ic_src, bc_src):
        if not os.path.exists(f):
            print(f"ERROR: {f} not found. Run the input stages first.", file=sys.stderr)
            sys.exit(1)

    # Locate the box.
    z = np.load(ic_src, allow_pickle=True)
    lats, lons = z["lats"], z["lons"]
    H, W = lats.shape

    if a.bbox:
        y0, x0, a.height, a.width, _bbox_meta = size_and_place_bbox(lats, lons, a.bbox, a.halo)
    elif a.centre_lat is not None and a.centre_lon is not None:
        d = (lats - a.centre_lat) ** 2 + (lons - a.centre_lon) ** 2
        cy, cx = np.unravel_index(np.argmin(d), d.shape)
        y0, x0 = int(cy) - a.height // 2, int(cx) - a.width // 2
        print(f"centre ({a.centre_lat}, {a.centre_lon}) -> grid ({cy}, {cx})")
    else:
        y0 = a.y0 if a.y0 is not None else (H - a.height) // 2
        x0 = a.x0 if a.x0 is not None else (W - a.width) // 2
    # Keep the box inside the grid without changing its size, so the divisibility
    # constraint still holds after clamping. int() because the --bbox path derives
    # these from np.where, giving numpy int64, which json.dump cannot serialize.
    y0 = int(max(0, min(y0, H - a.height)))
    x0 = int(max(0, min(x0, W - a.width)))

    frac = a.height * a.width / (H * W)
    print(f"\nfull grid      {H} x {W}")
    print(f"subdomain      {a.height} x {a.width}   rows [{y0}:{y0+a.height}]  "
          f"cols [{x0}:{x0+a.width}]")
    print(f"area fraction  {frac:.1%}")
    print(f"corner lat/lon SW ({lats[y0, x0]:.2f}, {lons[y0, x0]:.2f})  "
          f"NE ({lats[y0+a.height-1, x0+a.width-1]:.2f}, {lons[y0+a.height-1, x0+a.width-1]:.2f})")
    print(f"padded to      {a.height + PAD_H} x {a.width + PAD_W}  "
          f"(/{DOWNSAMPLE} = {(a.height+PAD_H)//DOWNSAMPLE} x {(a.width+PAD_W)//DOWNSAMPLE})  ok")

    if a.dry_run:
        print("\ndry run: nothing written")
        return

    os.makedirs(a.out_dir, exist_ok=True)
    for src in (ic_src, bc_src):
        dst = os.path.join(a.out_dir, os.path.basename(src))
        rep = crop_npz(src, dst, y0, x0, a.height, a.width)
        print(f"\n{os.path.basename(src)} -> {dst}")
        for line in rep["cropped"]:
            print(f"  cropped      {line}")
        print(f"  passthrough  {', '.join(rep['passthrough'])}")
        print(f"  size         {os.path.getsize(src)/1e6:.0f} MB -> "
              f"{os.path.getsize(dst)/1e6:.0f} MB")

    # Record the box so compare_domains.py can align the two runs without guessing.
    meta = dict(y0=y0, x0=x0, height=a.height, width=a.width,
                full_height=int(H), full_width=int(W), init_time=a.init_time,
                area_fraction=frac,
                # Provenance: which region was asked for, and how much halo it really
                # got. Without this, a consumer of the output cannot tell which part
                # of the box is product and which is halo that should be discarded.
                requested_bbox=a.bbox, requested_halo=(a.halo if a.bbox else None),
                sw_lat=float(lats[y0, x0]), sw_lon=float(lons[y0, x0]),
                ne_lat=float(lats[y0 + a.height - 1, x0 + a.width - 1]),
                ne_lon=float(lons[y0 + a.height - 1, x0 + a.width - 1]))
    with open(os.path.join(a.out_dir, "subdomain.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote {a.out_dir}/subdomain.json")


if __name__ == "__main__":
    main()
