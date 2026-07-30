# HRRRCast on AWS: on-demand single-member forecasts

This is the deployment path for **on-demand** runs: you ask for a cycle, one GPU
instance appears, runs the forecast, writes NetCDF to S3, and terminates. There
is no scheduler, no always-on infrastructure, and nothing to babysit.

Current shape:

| | |
|---|---|
| trigger | manual (`aws/run_on_ec2.sh`) |
| members | 1 |
| forecast length | 24 h on AWS (2 h for local Mac testing) |
| instance | `g6e.2xlarge` (L40S 48 GB); 24 GB GPUs OOM |
| measured throughput | ~35.5 s per lead hour steady state |
| a 24 h forecast | ~28 min, ~$1.06, ~10.3 GB of NetCDF |
| deliverable | NetCDF to S3, one file per lead hour |
| GRIB2 | off by default everywhere; `NO_GRIB2=NO` (or `--grib2`) opts in |
| plots | separate job, off the critical path (`aws/run_plots.sh`) |
| notifications | optional email on completion via SNS (`--notify-topic`) |
| retention | 12-day S3 expiry; the staged model never expires |
| region | us-east-1, co-located with the public NOAA HRRR/GFS buckets |

For the original manual walkthrough, which is still the best way to debug a new
account interactively, see [README_aws_mvp.md](README_aws_mvp.md).

## Account status (account 334566771276, us-east-1)

Provisioned 2026-07-28. Re-verify at any time, for free:

```bash
aws/run_on_ec2.sh --bucket mantari-cast1 --preflight-only
```

| item | status |
|---|---|
| credentials | working (`user/stelios`) |
| DL Base GPU AMI lookup | working, via SSM public parameter |
| output bucket | `mantari-cast1`, public access blocked, SSE-S3 default |
| model staged | `s3://mantari-cast1/hrrrcast/model.keras` (194 MiB) |
| instance profile | `hrrrcast-runner` (created; see below) |
| G/VT on-demand vCPU quota | **32** (case 178525039300546 granted; confirmed by preflight 2026-07-29) |

**The quota was 8 until 2026-07-29 and is now 32**, so four `g6e.2xlarge` (8 vCPU
each) can run concurrently. The constraint that remains is capacity, not quota:

- A **shutting-down** instance still holds its vCPUs. Preflight refuses in that
  state rather than letting `run-instances` fail with `VcpuLimitExceeded`; wait for
  `terminated`. Observed taking about 4 minutes.
- **`g6e.2xlarge` capacity is genuinely scarce.** On 2026-07-29 it returned
  `InsufficientInstanceCapacity` in *every* us-east-1 AZ on two separate occasions,
  and one launch succeeded only on its fifth placement attempt (EC2-chosen, 1d, 1b,
  1a all refused; 1c worked). Always pass `--wait-for-capacity`.
- `g6e.xlarge` (4 vCPU, same 48 GB L40S) had no capacity in any of the four AZs
  offering it, so it is not a reliable way to create headroom.

`--preflight-only` checks both the quota and in-flight instances and refuses with a
clear message rather than letting `run-instances` fail opaquely. With 32 vCPU on `L-DB2E81BA` there is now room to fall back to a larger type when
capacity is short, which given the scarcity above is the more useful freedom.

Do not reuse the pre-existing `hrrrcast-ssm` profile: it carries only
`AmazonSSMManagedInstanceCore`, so a run using it could neither fetch the model
nor terminate itself.

### The `hrrrcast-runner` role

Policies are checked in under [iam/](iam/) so they can be reviewed and reapplied:
[trust-policy.json](iam/trust-policy.json) and
[hrrrcast-runner-policy.json](iam/hrrrcast-runner-policy.json). Scoped to:

- `s3:GetObject` on the model and the `hrrrcast/code/*` prefix only
- `s3:PutObject` on `hrrrcast/out/*`, `hrrrcast/logs/*`, `hrrrcast/plots/*` and
  `hrrrcast/domain-test/*` only
- `s3:ListBucket` restricted by condition to the `hrrrcast/*` prefix
- `ec2:TerminateInstances` restricted by condition to instances tagged
  `Project=hrrrcast`, which the launcher sets. The instance can end its own run
  and nothing else.
- `AmazonSSMManagedInstanceCore` attached, so you can get a shell via Session
  Manager without opening SSH.

The public NOAA input buckets need no credentials.

Recreating this from scratch in another account:

```bash
aws s3api create-bucket --bucket <name> --region us-east-1
aws s3api put-public-access-block --bucket <name> \
  --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
aws iam create-role --role-name hrrrcast-runner \
  --assume-role-policy-document file://aws/iam/trust-policy.json
aws iam put-role-policy --role-name hrrrcast-runner \
  --policy-name hrrrcast-runner-s3-and-selfterminate \
  --policy-document file://aws/iam/hrrrcast-runner-policy.json   # edit the bucket ARNs first
aws iam attach-role-policy --role-name hrrrcast-runner \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam create-instance-profile --instance-profile-name hrrrcast-runner
aws iam add-role-to-instance-profile --instance-profile-name hrrrcast-runner --role-name hrrrcast-runner
aws s3 cp net-diffusion/model.keras s3://<name>/hrrrcast/model.keras
```

### Retention

A **12-day expiry** is applied to the bucket, defined in
[s3-lifecycle.json](s3-lifecycle.json) and applied with:

```bash
aws s3api put-bucket-lifecycle-configuration --bucket mantari-cast1 \
  --lifecycle-configuration file://aws/s3-lifecycle.json
```

| prefix | retention |
|---|---|
| `hrrrcast/out/` (NetCDF) | 12 days |
| `hrrrcast/plots/` | 12 days |
| `hrrrcast/logs/` | 12 days |
| `hrrrcast/code/` (build tarballs) | 12 days |
| `hrrrcast/domain-test/` | 12 days |
| `hrrrcast/experiments/` | **never expires** (curated results) |
| `hrrrcast/model.keras` | **never expires** |
| incomplete multipart uploads | aborted after 7 days |

`hrrrcast/domain-test/` holds raw subdomain-experiment output, three 24 h forecasts per
run at roughly **20 GB per cycle**, 94% of it the two full-domain runs. Two experiments
had accumulated 43.4 GiB before the 12-day rule was added.

The rule is safe to apply because the **analysis subset is promoted out of the way
first**. `compare_domains.py` needs only the leads actually compared, so for each
experiment those files plus the crop definition are copied into
`hrrrcast/experiments/<name>/netcdf/`, which no rule matches:

```
hrrrcast/experiments/<name>/netcdf/
    full/hrrrcast_m00_f{01,06,12,24}.nc      <- reference
    sub/hrrrcast_m00_f{01,06,12,24}.nc       <- the crop
    full-m1/hrrrcast_m01_f{01,06,12,24}.nc   <- the noise yardstick
    subdomain.json                           <- required to align the crop
```

That is ~3.5 GiB per experiment, and it keeps every published number independently
re-verifiable after the raw prefix expires. What is lost is the ability to analyse a
*different* lead hour later; that costs a fresh run (~$2.30 to $2.80 per cycle), which
is a deliberate trade against ~$1/month of storage per two experiments.

**Promote before you expire.** Run this for a new experiment, verify it, and only then
rely on the rule:

```bash
SRC=s3://mantari-cast1/hrrrcast/domain-test/<cycle>
DST=s3://mantari-cast1/hrrrcast/experiments/<name>/netcdf
for L in 01 06 12 24; do
  aws s3 cp $SRC/full/<YYYYMMDD>/<HH>/hrrrcast_m00_f$L.nc    $DST/full/hrrrcast_m00_f$L.nc
  aws s3 cp $SRC/sub/<YYYYMMDD>/<HH>/hrrrcast_m00_f$L.nc     $DST/sub/hrrrcast_m00_f$L.nc
  aws s3 cp $SRC/full-m1/<YYYYMMDD>/<HH>/hrrrcast_m01_f$L.nc $DST/full-m1/hrrrcast_m01_f$L.nc
done
aws s3 cp $SRC/subdomain.json $DST/subdomain.json
```

Copies are server-side, so nothing transits your machine. Verify with `head-object` on
each pair and compare **size and ETag** before trusting the rule to delete the source.
To reclaim the space immediately rather than in 12 days:

```bash
aws s3 rm --recursive s3://mantari-cast1/hrrrcast/domain-test/
```

**The rules are prefix-scoped specifically so `model.keras` survives.** A
whole-bucket expiry would delete it, and every future run stages the model from
there, so the deployment would break silently 12 days later. If you add rules,
keep them scoped and re-check that the model is not matched.

The abort-multipart rule matters because per-hour NetCDF uploads are multipart
(~24-42 parts each): a run killed mid-upload can leave parts that are billed but
invisible to `s3 ls`.

Each 24 h single-member forecast delivers about **10.3 GB** at the current defaults
(`NC_COMPLEVEL=1`, `NC_LSD=2`), so 12 days of daily forecasts is a steady state of
roughly **125 GB** (~$2.90/month in S3 Standard) rather than unbounded growth.

## Running a forecast

```bash
# 24-hour forecast for a specific cycle
aws/run_on_ec2.sh --bucket <your-bucket> --init-time 2026-07-20T00

# latest safely-available cycle, default 24 h
aws/run_on_ec2.sh --bucket <your-bucket>

# see exactly what would happen, launch nothing
aws/run_on_ec2.sh --bucket <your-bucket> --dry-run

# verify the AWS account is set up correctly, launch nothing
aws/run_on_ec2.sh --bucket <your-bucket> --preflight-only
```

### Options worth knowing

| option | why it exists |
|---|---|
| `--preflight-only` | account checks only; launches nothing, costs nothing |
| `--dry-run` | renders the bootstrap and prints the plan without touching AWS |
| `--wait-for-capacity M` | retry for M minutes when no AZ has GPU capacity. Not optional in practice: **4 of 6 launches on 2026-07-28 needed retries**, and one needed three attempts |
| `--notify-topic ARN` | email a run summary via SNS when the run ends, success or failure (see [Notifications](#notifications)) |
| `--instance-type` | defaults to `g6e.2xlarge`; anything with under 48 GB VRAM will OOM |
| `--nc-lsd ""` | opt back out of lossy quantization for a lossless run |
| `--no-terminate` | leave the box up for inspection; you must then terminate it yourself |
| `--s3-output URI` | override the default `hrrrcast/out/` -- give this an isolated prefix for any run that might reuse a cycle another experiment already touched (see "Output paths don't encode the domain" under Hindcast, below) |

Two safe modes, and they check different things:

- `--dry-run` renders the instance bootstrap and prints the full plan **without
  touching AWS**. Use it whenever you change any of these scripts.
- `--preflight-only` makes the live account checks (credentials, bucket, model,
  instance profile, AMI, GPU vCPU quota) and stops before launching anything. Use
  it to confirm the account is ready, or after changing IAM.

Neither costs anything.

The launcher packages **your working tree** (not `git archive HEAD`) to a tarball
in S3 and the instance unpacks that. This is deliberate: it runs exactly the code
you tested including uncommitted changes, needs no GitHub credentials for a
private repo, and sidesteps Git LFS. The plan output labels the code `<sha>` or
`<sha>-dirty` so you can tell what actually ran.

### Watching a run

Logs land in S3 only when the run **ends**, success or failure:

```bash
aws s3 ls --recursive s3://<your-bucket>/hrrrcast/out/ --human-readable
aws s3 cp s3://<your-bucket>/hrrrcast/logs/<instance-id>-status.txt -
aws s3 cp s3://<your-bucket>/hrrrcast/logs/<instance-id>-bootstrap.log -
```

To follow along live, launch with `--key-name <key> --no-terminate` and tail
`/var/log/hrrrcast-run.log` over SSH. Remember to terminate it yourself
afterwards; `--no-terminate` disables the self-cleanup that normally guarantees
you are not paying for an idle GPU.

### Instance sizing: 48 GB VRAM is a hard floor at full domain

(At full domain. On a crop it is untested; see [Subdomain inference](#subdomain-inference-cheaper-tested).)

Default is `g6e.2xlarge` (1x **L40S 48 GB**, 8 vCPU, 64 GiB host RAM, 450 GB NVMe). This is
measured, not assumed.

**24 GB GPUs cannot run this model at full domain** (they can on a crop; see
[Subdomain inference](#subdomain-inference-cheaper-tested)). A `g5.2xlarge` (A10G 24 GB) exhausted all
20,795 MB usable and died in the forecast stage:

```
Allocator (GPU_0_bfc) ran out of memory trying to allocate 1009.60MiB
RESOURCE_EXHAUSTED: Out of memory while trying to allocate 1058646016 bytes
  layer 'hrrr_preprocessor_recomp', conv f32[1,136,1064,1800]
```

That allocation is exactly one `1059 x 1799 x 139` float32 activation, at
`batch_size 1` with memory growth enabled. There is nothing left to tune. This is
consistent with the model's origins: `jobs/job-fcst.sh` asks for an **H100 (80 GB)**
and 160 GB host RAM. Rules out `g5.*`, `g6.*`, and `g4dn.*` at the xlarge tiers.

Measured on the L40S, 2-hour cycle, single member:

| stage | time |
|---|---|
| boot + conda env build + GPU/model verify | ~12 min |
| get-ics, get-bcs, make-ics, make-bcs | ~5 min |
| forecast, hour 1 (includes one-time XLA compile) | 125.5 s |
| forecast, hour 2 onward (steady state) | **36.4 s** |
| NetCDF write, complevel=1 | 22-26 s (background thread) |
| S3 upload, 879 MB | 1.8-2.4 s (368-501 MB/s) |

For comparison the Apple Silicon CPU path runs ~1,720 s per lead hour, so the L40S
is roughly **47x faster**. Extrapolated 24 h forecast: 125 + 23 x 36.4 = ~16 min of
inference, ~40 min of total instance time.

### Where the wall clock goes (24 lead hours, measured)

| phase | original | current | note |
|---|---|---|---|
| boot + code fetch + env + model stage | 11.2 min | **1.7 min** | GPU verify skipped; env installed from the lock (~35 s) instead of solved (~3 min) |
| get-ics, get-bcs, make-ics | ~3 min | **0** | hidden behind the forecast's model load (`OVERLAP_FCST=YES`) |
| **make_bcs** | ~42 min | **4.5 min** at 24 leads, **0** at 6 | partly hidden while the model loads; see [the README performance notes](../README.md#performance-notes-on-make_bcs) |
| forecast | ~31 min | ~25 min at 24 leads | 36 s per lead hour, unchanged by any of this work |
| **6 h forecast** | -- | **12.5 min, $0.47** | measured |
| **24 h forecast** | **79.5 min, $2.97** | **28.4 min, $1.06** | measured, from a clean clone of the commit |

Timings are launch-to-termination, which is the billing basis. Quoting
launch-to-last-file instead understates a 24-hour run by 6-7 min.

**How much of the input phase the overlap hides depends on forecast length**, and it
is worth being precise because the two cases point in opposite directions. The
overlap hides `min(model load, input phase)`:

| lead hours | input phase | model load | result |
|---|---|---|---|
| 6 | ~2.7 min | ~3.8 min | inputs finished **69 s before** the model loaded: the input phase is free, and further `make_bcs` work buys nothing |
| 24 | ~6 min | ~3.7 min | the forecast **waited 206 s** for inputs: `make_bcs` is on the critical path, so savings there still convert to wall clock |

So at the production length of 24 lead hours, `make_bcs` optimization is still worth
doing. An earlier version of this file claimed the opposite, generalizing from the
6-hour case.

The forecast phase decomposes as 2.1 min TF import plus npz load, 3.4 min
`load_model`, 2.4 min forecaster init, 2.6 min for hour 1 (including ~2.0 min of
XLA compilation), then 13.6 min of actual inference at 35.6 s per lead hour. So
about a third of it is fixed cost that does not scale with forecast length.

Done (both measured on AWS):

- **Overlap the forecast's startup with the stages before it** -- `OVERLAP_FCST=YES`,
  enabled by `user_data.sh`. Inputs became ready 69 s before the model finished
  loading, so the input phase now costs nothing. See
  [the README](../README.md#overlapping-the-forecasts-startup-with-input-preparation).
- **Install the conda env from a lock** -- ~3 min of solving became ~35 s. See
  [Environment reproducibility](#environment-reproducibility).

Remaining, in rough order of value:
1. **Bypass `RecomputeSubModel` at inference.** `src/resnet.py` wraps submodels in
   `tf.recompute_grad` (gradient checkpointing, which has no purpose without
   backprop) inside a `@tf.function(jit_compile=False)` that **opts a large
   convolution submodel out of XLA**, overriding the global `set_jit(True)` in
   `fcst.py`. Returning `self.submodel(inputs)` directly when `training` is falsy
   needs no retraining, since Keras rebuilds the layer from this class definition.
   Unquantified -- profile one lead hour before and after. Note XLA fusion can
   reorder floating-point work, so output may shift slightly and needs checking.
2. **Persist the XLA compilation cache** (~2 min). Hour 1 costs 156 s against
   35.6 s steady state.
3. **Mixed precision (bf16)** (~6 min, estimated). Needs numerical validation: 25
   diffusion steps compounding over 24 autoregressive hours is not a place to
   assume 8 mantissa bits are enough.
4. **Fewer diffusion steps.** `NUM_INFERENCE_STEPS = 25`, so a 24 h forecast is 600
   network evaluations at 1.42 s each, and cost is linear in steps: 15 steps would
   save ~5.5 min, 10 steps ~8 min. The sampler is already DPM-Solver++(2M), which
   is designed for low step counts, so this may be close to free in quality terms.
   That is a forecast-quality decision, not an engineering one.

Other sizing notes:

- **Not `g5.xlarge`/`g6e` with 16 GiB host RAM.** `make_bcs` has been observed to
  OOM there; `MAKE_BCS_WORKERS` defaults to 2 for the same reason.
- **Writer-thread margin is thin at scale.** NetCDF writes take 22-26 s in a single
  background thread against 36 s of inference, so ~10 s of slack per lead hour.
  Raising `NC_COMPLEVEL` inverts this and makes compression the bottleneck.
  `NC_LSD=2` (the default) reduces write time and widens the margin.
- **Capacity is a real constraint.** `g6e.xlarge` (4 vCPU, same 48 GB L40S, which
  would leave quota headroom) returned `InsufficientInstanceCapacity` in **all
  four** AZs that offer it. The launcher walks AZs automatically. If every AZ is
  out, wait and retry.
- The model runs **fp32**; nothing sets mixed precision. bf16 would roughly halve
  activation memory and might fit a 24 GB card, but it needs numerical validation
  before you would trust the output.

## Monitoring a run

```bash
aws/status.sh          # live instances, delivered output, recent outcomes
aws/status.sh --live   # ALSO: what command each instance is executing right now
aws/status.sh --logs   # ALSO: tail of the newest run log from S3
```

`--live` uses SSM Run Command, so it needs no SSH key and no inbound
security-group rule. It matters because **the log tail alone is misleading**: the
serial npz save at the end of `make_bcs` writes nothing for ~8 minutes, and a slow
conda solve looks identical to a hung one. The process table distinguishes them.

For an interactive shell:

```bash
aws ssm start-session --target <instance-id>
tail -f /var/log/hrrrcast-run.log
```

Note that S3 logs only appear when a run **ends**. A run killed externally with
`terminate-instances` never runs its exit trap, so it ships no logs and no status
file: pull anything you need over SSM before killing a run.

## Notifications

Optional email on completion, via SNS. Off unless a topic is given.

One-time setup:

```bash
aws sns create-topic --name hrrrcast-runs --region us-east-1
aws sns subscribe --region us-east-1 \
  --topic-arn arn:aws:sns:us-east-1:<account>:hrrrcast-runs \
  --protocol email --notification-endpoint you@example.com
# then confirm via the email AWS sends
aws iam put-role-policy --role-name hrrrcast-runner \
  --policy-name hrrrcast-runner-s3-and-selfterminate \
  --policy-document file://aws/iam/hrrrcast-runner-policy.json
```

Then either pass `--notify-topic <arn>` or export `HRRRCAST_SNS_TOPIC` once.

SNS rather than SES because it needs no verified sending identity and no SMTP
credentials on the instance, just `sns:Publish` on one topic ARN. The publish
happens inside the exit trap after outputs are already in S3, and is wrapped so a
notification failure cannot affect the run or prevent termination.

The email reports status, cycle, lead hours, instance type, wall clock, delivered
file count and size, NetCDF settings, `make_bcs` worker count and finish time,
steady-state seconds per lead hour, code provenance, and log locations.

## Environment reproducibility

**The environment is pinned**, by [conda-linux-64.lock](conda-linux-64.lock): 353
packages at exact builds, taken verbatim from a run that succeeded. `setup_gpu.sh`
installs from it with **no solving** (~35 s, versus ~3 min to solve), and
`run_on_ec2.sh` validates it before spending anything. Without the file it falls back
to solving `environment.aws.yaml` and warns.

To refresh it after changing dependencies: run a forecast, confirm it succeeded, then
adopt that run's package list.

```bash
aws s3 cp s3://<bucket>/hrrrcast/logs/<instance>-stages/conda-installed.txt \
          aws/conda-linux-64.lock
aws/validate_lock.sh
```

### One file the repo deliberately does not carry

`gfs_to_hrrr_weights.nc` (116 MB of xESMF bilinear weights) is not committed, so
`make_bcs` regenerates it serially at the start of a run. **Measured cost: 33 s**, and
because instances are ephemeral every run pays it. Staging it to S3 alongside
`model.keras` would remove that, but 33 s did not seem worth a 116 MB binary in git.
`run_on_ec2.sh` warns when the file is absent from the checkout being packaged.

## Environment reproducibility, continued

Every run writes `conda list --explicit` to
`logs/<instance>-stages/conda-installed.txt`, which is already in conda's
`@EXPLICIT` format, so the lock is always a copy of something proven rather than a
prediction. Two details from the current lock worth knowing:

- **`esmf-8.9.1-nompi_h8d4c64c_0`, and no `mpich` at all.** The real solve picks the
  MPI-free ESMF build, which is why the pipeline never needed MPI and why the
  conda-lock attempt below failed.
- **6 of 353 packages come from `pkgs/main`** (`blosc`, `brotli-python`, `libxml2`,
  `cffi`, `libllvm22`, `tifffile`). Ordinary builds, proven in service.
  `validate_lock.sh` reports this as a warning, not a failure: an early version
  rejected the channel outright and would have refused this working lock. The real
  defect detector is the `external_*` stub check.

**Do not generate the lock with conda-lock.** It was tried and reverted twice:

1. conda-lock ignores `CONDA_OVERRIDE_CUDA`, so it resolved `tensorflow-2.15.0-cpu_*`
   with no CUDA packages. Silent failure mode: everything installs and runs, on CPU,
   at ~1,700 s per lead hour instead of 35 s.
2. With `defaults` in the channel list it selected `mpich=4.3.2=external_0` from
   pkgs/main, a stub that expects host-provided MPI and ships no `libmpi.so.12`. The
   env installed cleanly and the cycle died in `make_ics`.

Removing `defaults` fixed (2) but made the solve intractable rather than slower
(14+ min on-instance, 28+ min locally with no result), so it was reverted. See the
comment block in [environment.aws.yaml](environment.aws.yaml).
`aws/validate_lock.sh` now catches both defects, and `setup_gpu.sh` fails hard on a
`cpu_*` TensorFlow build.

## Plots (separate job)

Plotting is CPU work that generated ~5,000 PNGs plus animations for one 12 h
single-member cycle, most of the disk footprint of a run. Running it inline would
keep a GPU instance busy doing matplotlib and would force the forecast to retain
every NetCDF locally. So the forecast streams NetCDF to S3 and deletes as it
goes, and plots are made afterwards, from the bucket, on whatever machine you
like:

```bash
# surface fields only
aws/run_plots.sh \
  --s3-input  s3://<your-bucket>/hrrrcast/out \
  --s3-output s3://<your-bucket>/hrrrcast/plots \
  --init-time 2026-07-28T11 --lead-hours 24 --members 0 --variables surface

# everything, plus animations
aws/run_plots.sh --s3-input s3://<your-bucket>/hrrrcast/out \
  --init-time 2026-07-28T11 --lead-hours 24 --members 0 --animate
```

This runs anywhere the `hrrrcast` env exists, including your laptop, and can be
re-run for any cycle still in the bucket. If a forecast died partway it plots the
hours that are there and warns, rather than failing.

`--variables` selects which groups to render (`all`, `surface`, `pressure`).
Measured per lead hour:

| group | figures | note |
|---|---|---|
| surface | 52 | the `sfc_vars` list |
| pressure | 120 | 6 variables x 20 levels |
| summary | 1 | mixes both, so only produced with `all` |
| **all** | **173** | |

So `--variables surface` is about 30% of the output and correspondingly faster.
Over 24 lead hours, `all` is ~4,150 figures, surface-only ~1,250.

Note: `plot.py` warns about `MXUPHL_*`, `MNUPHL_*`, `MAXUVV_*`, and `MAXDVV_*` not
being found. That is pre-existing and unrelated to the AWS path: its surface list
includes diagnostics this model does not output. Harmless.

## Output size

Measured on a real f01 file from a 12 h run (1059x1799, 62 variables):

| setting | size/hour | ratio | write time | error |
|---|---|---|---|---|
| uncompressed (`NC_COMPLEVEL=0`) | 1356 MB | 1.00x | 0.8 s | lossless |
| `NC_COMPLEVEL=1` **(default)** | 879 MB | 1.54x | 11.5 s | lossless |
| `NC_COMPLEVEL=6` | 871 MB | 1.56x | 18.8 s | lossless |
| `NC_LSD=3` | 460 MB | 2.95x | 8.3 s | max abs 0.00049 |
| `NC_LSD=2` | 355 MB | 3.82x | 6.7 s | max abs 0.0039 |

The table above was measured on the previous 148-channel network. The current
upstream network takes 329 input channels, and a lead hour now writes about
**419 MB** at `NC_COMPLEVEL=1, NC_LSD=2` (f00 is ~226 MB). The ratios are
unchanged; only the absolute sizes grew ~20%.

Two things to take from this:

- **Lossless deflate barely helps** on float32 continuous fields, and level 1
  captures essentially all of the benefit. There is no reason to go above 1.
- **Quantization is the actual lever.** `NC_LSD=2` gives 3.8x for a maximum
  absolute error of 0.0039 in each variable's native units (K, dBZ, mm, ...).

**The AWS default is `NC_COMPLEVEL=1` with `NC_LSD=2`**, so a 24 h single-member
forecast delivers about **10.3 GB** rather than ~25 GB lossless. `NC_LSD=2` is a
deliberate, lossy choice: 0.0039 in native units is below the useful precision of
the forecast, and it also shortens the write, which widens the thin writer-thread
margin described under [Instance sizing](#instance-sizing-48-gb-vram-is-a-hard-floor).

For a lossless run, pass `--nc-lsd ""`. Note the local defaults in `run_cycle.sh`
are unchanged (uncompressed, no quantization) so Mac and HPC behavior is exactly
as before; the quantization default lives only in the AWS launcher.

Compression runs in a background thread overlapping GPU compute, so its cost is
not on the critical path until it exceeds the per-hour inference time. That is why
`NC_COMPLEVEL` is a knob rather than hardcoded: if the writer thread ever becomes
the bottleneck, drop it to 0.

## Hourly operation (EventBridge Scheduler → Lambda)

```
EventBridge Scheduler  cron(5 * * * ? *)  UTC
    └─> Lambda  hrrrcast-hourly-launcher        (aws/lambda/handler.py)
          ├─ reads s3://BUCKET/hrrrcast/scheduler/{launch-params.json,user_data.template}
          ├─ refuses if a run is in flight, the cycle is already produced,
          │  the inputs are incomplete, or the vCPU quota is full
          └─> RunInstances: one g6e.2xlarge, self-terminating
```

### Why the pipeline needs a GFS lag to run hourly

Measured publication latency on the public buckets: the HRRR analysis for hour H
appears at about **H+0:51**, but a GFS cycle does not finish publishing its
`f000`–`f036` block until about **cycle+4:00** (one cycle measured directly at
+3:41). So the newest GFS cycle does not exist when the HRRR analysis lands.

Enumerated over a full day, with the default rule `cycle = floor(H/6)*6`:

| | blocked hours | zero-margin hours | GFS age | steps used |
|---|---|---|---|---|
| `GFS_MIN_LAG=0` (default) | **12 of 24** | 4 more (03/09/15/21Z) | 1–6 h | f001–f029 |
| `GFS_MIN_LAG=4` | **0** | 0 (min margin +1 h) | 4–10 h | f005–f033 |

`--gfs-min-lag 4` shifts cycle selection back before rounding down, trading GFS
freshness for a launch slot in every hour. It keeps the initial condition at a
uniform ~1 h old, which is what a rapid-refresh product is judged on.

**This is a real distribution shift.** f005–f033 extends past the f001–f029 range
the network trained on, and the top of that range is reached on three of every six
hours. It has not been measured against a matched `lag=0` run. Do that before
treating hourly output as equivalent to on-demand output.

The default stays `0` everywhere except `pick_cycle.sh`, because on-demand and
retrospective runs have all the data published and should use the freshest cycle.
A consequence worth stating plainly: **a retrospective run is not a valid
verification proxy for the hourly product**, since it gets systematically better
forcing. Pass `--gfs-min-lag 4` for anything meant to characterize hourly skill.

### Deploying it

The schedule is created **DISABLED**, and the Lambda starts in **dry-run**, so
nothing spends money until you say so twice.

```bash
# 1. pin the code and a user-data template to S3 (the deploy step)
GFS_MIN_LAG=4 aws/run_on_ec2.sh --bucket mantari-cast1 --stage-scheduler

# 2. create role, Lambda and schedule; disabled, dry-run
aws/deploy_scheduler.sh --bucket mantari-cast1

# 3. one manual invocation, see what it would pick
aws/deploy_scheduler.sh --bucket mantari-cast1 --invoke-once

# 4. hourly, still dry-run: read a day of logs and check the cycle choices
aws/deploy_scheduler.sh --bucket mantari-cast1 --enable

# 5. actually launch instances
aws/deploy_scheduler.sh --bucket mantari-cast1 --enable --live

# stop
aws/deploy_scheduler.sh --bucket mantari-cast1 --disable
```

Re-running step 1 then step 2 is how a code change rolls out; the Lambda reads
both S3 objects on every invocation. `CodeRef` is tagged on each instance and
appears in the notification email, so a stale deploy is visible.

Why `HH:05`: the analysis lands at about `H+0:51`, so a tick at `H+1:05` leaves
roughly 14 minutes of slack for a late file before the Lambda falls back an hour.

### Cost, 1 member, on-demand

| lead | min/run | $/run | $/month | storage at 12-day retention |
|---|---|---|---|---|
| 24 h | 32 | $1.20 | **$875** | 2.4 TB, ~$56/mo |
| 12 h | 19 | $0.72 | $523 | 1.2 TB, ~$28/mo |
| 6 h | 16 | $0.61 | $442 | 0.6 TB, ~$15/mo |

Spot at the usual ~65% discount takes the 24 h case to roughly **$306/month**,
and the ~28 min of slack per hour absorbs one interruption and retry. That is a
larger lever than any remaining code optimization.

Two decisions to make before enabling: whether every hourly cycle needs 12 days
of retention or only the synoptic ones, and whether to wire the SNS topic, since
under hourly operation a silently skipped hour is invisible without it.

### What is deliberately not there

No Step Functions, no Batch, no spot handling, no ensemble fan-out. The Lambda
launches one instance and forgets it; the instance always self-terminates via its
EXIT trap. If an hour fails, the next hour simply runs. `MaximumRetryAttempts` is
0 on the schedule target for the same reason: a retry would land on the same
missing data, and `pick_cycle`'s lookback already self-heals a missed tick.

## Hindcast (serverless, resumable backfill)

Running many cycles over a past date range -- a hindcast -- has a different
shape than either an on-demand run or hourly production: the cycles are all
already known up front, none of them race a publication clock, and a run
spanning days needs to survive being interrupted (a redeploy, a laptop closing,
an AWS hiccup) without either re-billing finished work or silently dropping a
cycle. The design:

```
EventBridge Scheduler  rate(10 minutes)  UTC
    └─> Lambda  hrrrcast-hindcast-<run-id>     (aws/lambda/hindcast_handler.py)
          ├─ lists s3://BUCKET/hrrrcast/hindcast/<run-id>/status/  (done cycles)
          ├─ refuses if an instance tagged HindcastRun=<run-id> is in flight,
          │  or the vCPU quota is full
          └─> RunInstances for the earliest cycle without a status marker
                └─> aws/run_hindcast_cycle.sh: run_cycle.sh, then ALWAYS write
                    one marker (ok / failed(rc=N)) to that S3 prefix
```

**The only state this trusts is in S3 and on the EC2 instances themselves** --
which cycles have a status marker, and whether one is currently in flight.
There is no separate "job queue" or database to go stale. Redeploying,
re-running `deploy_hindcast.sh` with the same `--run-id`, or just letting the
schedule sit disabled for a week and re-enabling it, all resume from exactly
where the markers say the hindcast left off. A cycle that fails still gets a
marker, so it is never retried and never blocks the cycles after it -- and its
instance's own `--notify-topic` failure email fires exactly as it would for a
single on-demand run, with no separate per-cycle notification code needed.
Once every cycle has a marker, the Lambda sends one completion summary and
disables its own schedule.

### Deploying it

Same disabled-by-default convention as the hourly scheduler:

```bash
# 1. stage code, write config.json, create the Lambda + schedule (DISABLED, dry-run)
aws/deploy_hindcast.sh --bucket mantari-cast1 --run-id socal-june2026 \
    --start 2026-06-01 --end 2026-06-30 --init-hours 00,06,12,18 \
    --lead-hours 24 --output-hours 0:3:24 \
    --bbox 35.0,-118.77,33.25,-117.0 --halo 80 \
    --notify-topic arn:aws:sns:us-east-1:334566771276:hrrrcast-runs

# 2. one manual tick, see what it would launch
aws/deploy_hindcast.sh --bucket mantari-cast1 --run-id socal-june2026 --invoke-once

# 3. go live, then turn the schedule on
aws/deploy_hindcast.sh --bucket mantari-cast1 --run-id socal-june2026 --live --enable

# progress at any time (out of however many cycles the date range enumerates)
aws s3 ls s3://mantari-cast1/hrrrcast/hindcast/socal-june2026/status/ | wc -l

# stop / resume later / remove the launcher only (S3 state and outputs are kept)
aws/deploy_hindcast.sh --bucket mantari-cast1 --run-id socal-june2026 --disable
aws/deploy_hindcast.sh --bucket mantari-cast1 --run-id socal-june2026 --enable
aws/deploy_hindcast.sh --bucket mantari-cast1 --run-id socal-june2026 --delete
```

`--output-hours "start:step:end"` (e.g. `0:3:24`) is a new `fcst.py` option: the
autoregressive rollout still computes every lead hour (the state depends on it),
but only the listed hours are built, written and uploaded. Useful for a hindcast
where 3-hourly output is enough and 8x fewer NetCDF files is worth having.

### `run_hindcast.sh`: the non-serverless alternative

`aws/run_hindcast.sh` loops cycles on ONE already-booted instance instead of one
instance per cycle (`aws/launch_hindcast.sh` launches it). It skips a failed
cycle and keeps going, same as the serverless path, but trades the per-cycle
instance-boot overhead for depending on that one instance staying up for the
whole run.

**It does not currently save the model-load cost either.** Each cycle shells
out to a fresh `python3 src/fcst.py` process (`run_cycle.sh`'s `run_stage
fcst`), so the model reloads once per cycle even though it is the same
GPU/process lineage the whole time -- the ~5 min TF-import + model-load cost
(see `OVERLAP_FCST`'s doc comment in `run_cycle.sh`) repeats N times across N
cycles for no reason. On a 120-cycle month that is roughly 10 h of pure
overhead. Fixing it means restructuring `fcst.py`'s entry point into a
persistent driver that loads the model once and loops internally, which is a
real change, not a small patch -- not done yet. Until it is, the serverless
path above pays the same per-cycle model-load cost but at least parallelizes
none of it away either, so there is currently no efficiency reason to prefer
`run_hindcast.sh` -- use it only when you specifically want everything on one
box (e.g. for interactive debugging across cycles).

### Verified 2026-07-30

A real 2-cycle SoCal run (`2026-07-29T00Z` and `06Z`, halo 80, `--output-hours
0:3:24`) confirmed the design end to end: instance `i-05e08c52ae92cf0f3` (cycle
1) and `i-0e810e895385ebc6c` (cycle 2) both completed successfully, each writing
exactly `f00,f03,f06,...,f24` to S3, and the Lambda correctly picked cycle 2
only after cycle 1's marker existed. Two real bugs surfaced and were fixed
during that test:

- **`hrrrcast-runner`'s own IAM role had no `s3:PutObject` on
  `hrrrcast/hindcast/*`.** The Lambda's role was extended for this feature, but
  the *instance* role that actually writes the per-cycle status marker was not
  -- so cycle 1 ran and delivered correct output, but its marker upload failed
  with `AccessDenied` and only a warning was logged. Fixed in
  [iam/hrrrcast-runner-policy.json](iam/hrrrcast-runner-policy.json); applied
  live via `aws iam put-role-policy`. Backfilled the missing marker by hand for
  that one cycle rather than re-running an already-successful 24 h forecast.
- **Self-disable needed `iam:PassRole` on its own role**, scoped to
  `scheduler.amazonaws.com`, which was missing; and the completion check
  short-circuited before retrying a failed disable on a later tick. Both fixed
  in [iam/lambda-launcher-policy.json](iam/lambda-launcher-policy.json) and
  [lambda/hindcast_handler.py](lambda/hindcast_handler.py).

Two pre-existing gaps, not introduced by this feature, found along the way --
both since resolved:

- **The SNS topic `hrrrcast-runs` did not exist in this account.** Every
  `--notify-topic` reference in this codebase (per-instance failure emails, the
  hindcast completion summary) pointed at an ARN that `aws sns list-topics`
  did not list; every publish attempt failed silently with only a logged
  warning. **Fixed 2026-07-30**: the topic now exists
  (`arn:aws:sns:us-east-1:334566771276:hrrrcast-runs`) with a confirmed email
  subscription. Verify with `aws sns list-subscriptions-by-topic --topic-arn
  arn:aws:sns:us-east-1:334566771276:hrrrcast-runs` -- a real `SubscriptionArn`
  means it is live, `"PendingConfirmation"` means the email link has not been
  clicked yet.
- **Output paths didn't encode the domain.** `s3://BUCKET/hrrrcast/out/<date>/<hour>/`
  is keyed only by init time, so two different experiments (different bbox,
  different halo) that happen to use the same cycle silently overwrite each
  other's files there. Observed directly: this test's `f19.nc` at
  `20260729/00/` was a leftover from an earlier, unrelated SoCal experiment that
  used the same cycle, sitting next to this run's own `f00,f03,...,f24`.
  **Fixed 2026-07-30** for the hindcast path specifically: `deploy_hindcast.sh`
  now passes `run_on_ec2.sh`'s new `--s3-output` override (default unchanged)
  so every hindcast writes to its own `hrrrcast/hindcast/<run-id>/out/`, never
  the shared prefix. Re-verified live the same day with a fresh single-cycle
  run (`i-0120791132589ba7e`, run-id `socal-test-output-fix`,
  `2026-07-29T12`): its 9 files landed under
  `hrrrcast/hindcast/socal-test-output-fix/out/20260729/12/`, and the shared
  `hrrrcast/out/20260729/12/` still held only its original, unrelated
  full-domain files -- nothing leaked across. On-demand and hourly runs still
  use the shared `hrrrcast/out/` by default -- give those `--s3-output` too if
  a run might reuse a cycle another experiment already touched.

**Not yet fixed:** `run_hindcast.sh` reloads the model once per cycle instead
of once for the whole run -- see the note under "the non-serverless
alternative" above.

## Subdomain inference (cheaper, tested)

Cropping the domain cuts cost, because the cost of a step scales with `H*W`. Measured
on 2026-07-29 17Z at 24 h lead, a 25.2% crop (531x903) against a matched full-domain
run on the same instance with bit-identical inputs:

| | full | sub (25.2%) | ratio |
|---|---|---|---|
| wall clock, 24 h | 1308 s | **543 s** | 2.4x |
| inference only (less ~210 s model load) | 45.8 s/h | **13.9 s/h** | **3.3x** |
| NetCDF write, mean | 20.0 s | 4.75 s | 4.2x |
| total output | 10.9 GB | 2.77 GB | 3.9x |

**Fidelity holds.** Crop/noise ratio median 1.02, max 1.49, i.e. cropping costs about
as much as swapping ensemble members. Surface pressure, 10 m wind, reflectivity and
precipitation are indistinguishable from the model's own spread. **2 m temperature
carries a systematic cold bias of about 0.5 to 1.1 K**, most plausibly from the 39
squeeze-excitation blocks whose `GlobalAveragePooling2D` gating changes when the
domain mean changes. It is a bias, not lost skill, so calibration can remove it.

A second test on a much smaller, real box (Southern California, 155x151, **1.2%** of
CONUS) confirms this and revises one point. Ratio median 1.10, max 1.95, so a 20x
smaller box costs only slightly more. But the T2M bias did **not** grow with the
smaller box: it fell to -0.11 to +0.20 K and changed sign. **The bias tracks where the
box sits relative to the CONUS mean, not how small it is**, so it has to be measured
per box rather than predicted from area. On that box the variable to watch is 10 m wind
at f24 instead: ratio 1.50, bias -1.22 m/s, with the crop-versus-full difference as
large as the field's own spatial spread. Inference scaled 9.8x on an 80x area
reduction, sublinear because per-call overhead starts to dominate at that size.

Two things to know before using this:

- **The size is constrained**: `H % 8 == 3` and `W % 8 == 7`, because the model bakes a
  fixed reflect-padding. A wrong size misaligns the UNet skips and may fail silently
  rather than loudly. `512x512` is not legal.
- **A crop runs on a 24 GB A10G. Measured, not assumed.** A 155x151 crop completed a
  24 h forecast on `g5.2xlarge`, status `success`, zero OOM, inside the 20,795 MB
  TensorFlow reports usable. Ignore the 43953 MiB peak the runs report: every run
  reports it, full domain and 1.2% crop alike, median equal to max, because
  `fcst.py` sets `set_jit(True)` and XLA claims a large fixed pool regardless of the
  work. **Do not size hardware from that number or from `nvidia-smi`.**

`g5.2xlarge` costs $1.21/h against $2.24/h, so a ~19 min subdomain run is about $0.38
rather than $0.49, and hourly 24 h operation drops from about $875/month to roughly
$150-200/month before spot, with storage from 2.4 TB to 0.6 TB. There is an
availability argument too: `g6e.2xlarge` returned `InsufficientInstanceCapacity` in
every us-east-1 AZ twice during this work, and one launch only succeeded on its fifth
placement attempt, while both `g5.2xlarge` launches succeeded first try. `g5` pools are
much deeper.

Note that `run_on_ec2.sh` still defaults to `g6e.2xlarge`. A subdomain run needs
`--instance-type g5.2xlarge` passed explicitly, or it silently pays the L40S rate.

Full instructions, including how to place the box and how much halo to leave, are in
**[../docs/subdomain.md](../docs/subdomain.md)**. To re-verify a different box or
season:

```bash
aws/run_domain_test.sh --bucket mantari-cast1 --lead-hours 24   # ~75 min, ~$2.80
```

That runs three forecasts on one instance: full domain, the crop, and a second
full-domain run with a different member ID. The third is not padding. Diffusion noise
is drawn with `stateless_normal(shape, seed=[member, hour])`, so the draw depends on
the tensor SHAPE and a cropped run gets a different realization no matter what.
Without the second full-domain run as a noise floor, a full-vs-crop difference cannot
be attributed to cropping at all.

### How to run a subdomain forecast

`run_on_ec2.sh` takes **`--bbox N,W,S,E`** and **`--halo N`** directly, and passes them
through to `run_cycle.sh` on the instance, which crops between the input stages and the
forecast:

```bash
aws/run_on_ec2.sh --bucket mantari-cast1 --init-time 2026-07-29T00 --lead-hours 24 \
    --instance-type g5.2xlarge \
    --bbox 35.0,-118.77,33.25,-117.0 --halo 40
```

Omit `--bbox` and the run is full-domain, exactly as before. The launcher summary and
the instance's own summary both print the domain, so a log always says which one ran.

This works for the **hourly schedule** too, with no Lambda change: `--stage-scheduler`
bakes the bbox into the staged user-data template, and the Lambda only substitutes the
init time, lead hours and GFS lag.

```bash
aws/run_on_ec2.sh --bucket mantari-cast1 --lead-hours 24 \
    --bbox 35.0,-118.77,33.25,-117.0 --halo 40 --stage-scheduler
```

Re-run that to change the box; the schedule picks up the new template on its next tick.

**The older path**, [run_subdomain_forecast.sh](run_subdomain_forecast.sh) via
`--run-cmd`, still works and is what the fidelity experiments used. Prefer the flags
above for anything new: it is one code path shared with full-domain runs, and it is the
only one the scheduler can drive.

For the older `--run-cmd` form, the positional arguments to
`run_subdomain_forecast.sh` are `INIT_TIME LEAD_HOURS [HEIGHT] [WIDTH] [BBOX] [HALO]`,
and they must be kept consistent with `--init-time`/`--lead-hours` on the launcher by
hand. That duplication is the reason the flags above exist.

`run_cycle.sh` also accepts the knobs directly, for a local or on-instance run:
`SUB_BBOX="N,W,S,E"`, `SUB_HALO`, or `SUB_HEIGHT`/`SUB_WIDTH` for a fixed grid-centred
box. `SUB_HEIGHT` and `SUB_WIDTH` must be set together; half a request is rejected
rather than silently running the full domain.

Everything else in `run_on_ec2.sh` behaves normally, including `--instance-type`,
`--dry-run`, `--preflight-only`, `--no-terminate` and `--notify-topic`. The bootstrap
exports `DATAROOT`, `S3_OUTPUT`, `NC_COMPLEVEL`, `NC_LSD`, `GFS_MIN_LAG` and
`PURGE_LOCAL`, and the script reads all of them, so a subdomain run ships to S3 and
self-terminates like any other.

Check the box before paying for an instance:

```bash
python3 src/crop_domain.py --in-dir . --out-dir . --init-time 2026-07-29T17 \
    --bbox 35.0,-118.77,33.25,-117.0 --halo 40 --dry-run
```

That prints the chosen size, its origin in the full grid, the achieved halo on each
side, and the fraction of CONUS area. For the Southern California box above it
returns 155x151 at `y0=359, x0=202`, 1.2% of CONUS, halo 40/40/42/42.

**How much halo.** From the measured error-versus-distance profile, T2M RMSE against
the full-domain run falls from 1.038 K in the outermost 5 cells to 0.707 K in the deep
interior at f01 (2.540 to 1.458 K at f06), and is flat past roughly 20 to 40 cells.
So **20 to 40 cells (60 to 120 km) of halo** captures nearly all of the available
improvement; 40 is the default. More halo buys very little, and it cannot remove the
T2M cold bias at all, because that comes from the changed domain mean feeding 39
squeeze-excitation blocks rather than from the boundary.

Two limits of this path as it stands:

- **GRIB2 output works on a crop** as of 2026-07-30. `src/nc2grib.py` previously built
  Section 3 from a hardcoded `nx=1799, ny=1059` and a hardcoded first grid point, so
  grib2io rejected cropped data on a shape mismatch and wrote nothing. Section 3 is now
  derived from the dataset's own coordinates, verified by decoding the output back and
  comparing against the NetCDF. A GRIB2 conversion that fails now fails the run instead
  of logging a warning nobody reads.
- **The input stages always run at full domain.** The regridding weights are fixed at
  1059x1799, so `get_*`/`make_*` cost the same as a full run (about 417 s measured).
  At a small crop that fixed cost, plus a ~210 s model load, dominates the wall clock,
  and inference is only a few percent of it. Cropping the input stage, or a baked AMI,
  is where the next saving is.

## What this does not do

Deliberately absent: AWS Batch or Step Functions, spot instances with
checkpointing, ensemble fan-out, and a baked AMI. Each becomes worth adding at a
specific point:

- **More members** is the trigger for Batch (fan-out) and spot (cost).
- **Frequent runs** is the trigger for a baked AMI; every launch currently
  rebuilds the conda env from the lock file, which takes ~1.7 min.

## Related files

| file | role |
|---|---|
| [run_on_ec2.sh](run_on_ec2.sh) | launcher; packages code, launches, reports. `--stage-scheduler` is the deploy step for hourly |
| [pick_cycle.sh](pick_cycle.sh) | picks a cycle, probes every input file, checks idempotency; exit 3 = nothing to do |
| [deploy_scheduler.sh](deploy_scheduler.sh) | creates/updates the role, Lambda and EventBridge schedule (disabled + dry-run by default) |
| [lambda/handler.py](lambda/handler.py) | the hourly launcher itself |
| [iam/lambda-launcher-policy.json](iam/lambda-launcher-policy.json) | launcher permissions; `RunInstances` split across resource types on purpose |
| [../src/gfs_cycle.py](../src/gfs_cycle.py) | GFS cycle rule + input manifest, stdlib-only so Lambda can import it |
| [../src/crop_domain.py](../src/crop_domain.py) | crops IC/BC npz to a subdomain; enforces the `H%8==3, W%8==7` rule |
| [../src/compare_domains.py](../src/compare_domains.py) | full vs subdomain fidelity, against a stochastic-noise yardstick |
| [run_subdomain_forecast.sh](run_subdomain_forecast.sh) | one production forecast on a cropped box; drive it with `run_on_ec2.sh --run-cmd` |
| [domain_test.sh](domain_test.sh) | three-forecast subdomain experiment, run on the instance |
| [run_domain_test.sh](run_domain_test.sh) | launcher for the subdomain experiment |
| [../docs/subdomain.md](../docs/subdomain.md) | how to run on a subdomain, and what it costs |
| [user_data.sh](user_data.sh) | instance bootstrap template; always self-terminates |
| [setup_gpu.sh](setup_gpu.sh) | provisions the conda env, stages the model, verifies GPU |
| [run_plots.sh](run_plots.sh) | independent plot job reading NetCDF from S3 |
| [deploy_hindcast.sh](deploy_hindcast.sh) | serverless hindcast: stages code, writes `config.json`, creates/updates the Lambda + schedule |
| [lambda/hindcast_handler.py](lambda/hindcast_handler.py) | hindcast driver: resumes from S3 status markers, launches one cycle at a time |
| [run_hindcast_cycle.sh](run_hindcast_cycle.sh) | one cycle + its S3 status marker; what each serverless-hindcast instance runs |
| [run_hindcast.sh](run_hindcast.sh) | non-serverless alternative: loops cycles on one already-booted instance |
| [launch_hindcast.sh](launch_hindcast.sh) | launcher for `run_hindcast.sh` |
| [environment.aws.yaml](environment.aws.yaml) | GPU conda env (TF 2.15 + boto3) |
| [../src/s3io.py](../src/s3io.py) | per-hour upload with retries and local purge |
| [../run_cycle.sh](../run_cycle.sh) | the cycle itself; see its header for all knobs |
