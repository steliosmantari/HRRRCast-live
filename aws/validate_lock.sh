#!/usr/bin/env bash
#
# aws/validate_lock.sh — sanity-check aws/conda-linux-64.lock before it is used.
#
# Exists because two lock defects have already reached a live instance, both of
# which installed cleanly and failed (or would have failed) only later:
#
#   1. tensorflow resolved to a cpu_* build. conda-lock does not honour
#      CONDA_OVERRIDE_CUDA, so the GPU variant was not selected. Nothing errors:
#      the forecast just runs on CPU at ~1,700 s per lead hour instead of 35 s,
#      turning a 24-hour forecast into an 11-hour one.
#   2. mpich resolved to `external_0` from pkgs/main, a stub that expects the host
#      to supply MPI and ships no libmpi.so.12. The env built fine and the cycle
#      died in make_ics with "libmpi.so.12: cannot open shared object file".
#
# Both are cheap to detect by inspecting the lock, and expensive to detect by
# launching a GPU instance. run_on_ec2.sh calls this before uploading the code.
#
# Usage: aws/validate_lock.sh [path-to-lock]   (exit 0 = usable, 1 = do not use)
set -uo pipefail

LOCK="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/conda-linux-64.lock}"
FAIL=0
ok()   { printf '  \033[1;32mok\033[0m    %s\n' "$*"; }
bad()  { printf '  \033[1;31mFAIL\033[0m  %s\n' "$*"; FAIL=1; }
warn() { printf '  \033[1;33mwarn\033[0m  %s\n' "$*"; }

[ -f "$LOCK" ] || { printf 'No lock file at %s (the unpinned fallback will be used)\n' "$LOCK"; exit 0; }

printf 'Validating %s\n' "$LOCK"

N=$(grep -c '^https' "$LOCK")
[ "$N" -gt 100 ] && ok "${N} packages" || bad "only ${N} package URLs; lock looks truncated"

# 1. GPU TensorFlow, not CPU.
TF=$(grep -oE 'tensorflow-[0-9][^#]*' "$LOCK" | grep -vE 'base|estimator' | sed 's/\.conda$//' | head -1)
case "$TF" in
    *cuda*) ok "tensorflow is a CUDA build: ${TF}" ;;
    *cpu_*) bad "tensorflow is a CPU build: ${TF} -- inference would be ~50x slower. Re-lock with --virtual-package-spec aws/virtual-packages.yaml" ;;
    "")     bad "no tensorflow package found in the lock" ;;
    *)      warn "cannot classify tensorflow build: ${TF}" ;;
esac

# 2. No stub/external builds that expect host-provided libraries.
EXT=$(grep -oE '/[a-z0-9._-]*external_[^#]*' "$LOCK" | sed 's|^/||' | tr '\n' ' ')
[ -z "$EXT" ] && ok "no external_* stub builds" \
              || bad "external_* stub build(s) present: ${EXT}-- these ship no runtime libraries"

# 3. pkgs/main presence is informational, NOT a failure. It was a proxy for the
# real defect in check 2: pkgs/main served the mpich `external_0` stub. But a
# known-good environment proven by three successful forecasts draws 6 ordinary
# libraries (blosc, brotli-python, libxml2, cffi, libllvm22, tifffile) from
# pkgs/main, so failing on the channel alone would reject a working lock.
MAIN=$(grep -c 'repo.anaconda.com' "$LOCK")
if [ "$MAIN" -eq 0 ]; then
    ok "all packages from conda-forge"
else
    warn "${MAIN} package(s) from pkgs/main -- fine in itself; check 2 catches the stub builds that made this channel a problem"
fi

# 4. If an MPI-flavoured ESMF was chosen, a real MPI runtime must accompany it.
ESMF=$(grep -oE 'esmf-[0-9][^#]*' "$LOCK" | sed 's/\.conda$//' | head -1)
case "$ESMF" in
    *nompi*) ok "esmf is an MPI-free build: ${ESMF}" ;;
    *mpi_mpich*)
        if grep -qE '/mpich-[0-9][^#]*' "$LOCK" && ! grep -q 'mpich-[0-9.]*-external' "$LOCK"; then
            ok "esmf needs mpich and a real mpich build is present: $(grep -oE 'mpich-[0-9][^#]*' "$LOCK" | sed 's/\.conda$//' | head -1)"
        else
            bad "esmf build ${ESMF} needs MPI but no usable mpich is in the lock"
        fi ;;
    *mpi_openmpi*)
        grep -qE '/openmpi-[0-9][^#]*' "$LOCK" \
            && ok "esmf needs openmpi and it is present" \
            || bad "esmf build ${ESMF} needs openmpi but it is not in the lock" ;;
    "")  warn "no esmf in the lock (xesmf regridding would fail)" ;;
    *)   warn "cannot classify esmf build: ${ESMF}" ;;
esac

# 5. Packages the pipeline imports directly, so a truncated solve is caught here.
for p in python numpy xarray netcdf4 eccodes pygrib grib2io esmpy cartopy scikit-image boto3 wgrib2; do
    grep -qE "/${p}-[0-9]" "$LOCK" || bad "required package missing from lock: ${p}"
done
ok "all directly-imported packages present"

echo
if [ "$FAIL" -eq 0 ]; then
    printf '\033[1;32mLock looks usable.\033[0m\n'
else
    printf '\033[1;31mLock is NOT safe to use.\033[0m Fix it, or set USE_LOCK=NO to solve from environment.aws.yaml.\n'
fi
exit "$FAIL"
