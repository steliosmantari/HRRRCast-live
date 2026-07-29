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
| a 24 h forecast | ~35 min, ~$1.32, ~9 GB of NetCDF |
| deliverable | NetCDF to S3, one file per lead hour |
| GRIB2 | disabled (`NO_GRIB2=YES`) |
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
| G/VT on-demand vCPU quota | 8; **increase to 32 requested** 2026-07-28 (case 178525039300546) |

**The vCPU quota of 8 is a live constraint**, not a footnote. The working instance
type, `g6e.2xlarge`, needs exactly 8 vCPU, so at the current quota you can run
**one forecast at a time and nothing else**. Two consequences seen in practice:

- A **shutting-down** instance still holds its vCPUs, so relaunching immediately
  after a run fails with `VcpuLimitExceeded`. Wait for `terminated`.
- The obvious workaround, `g6e.xlarge` (4 vCPU, same 48 GB L40S), had
  `InsufficientInstanceCapacity` in **all four** AZs offering it, so you cannot
  rely on it to create headroom.

`--preflight-only` checks both the quota and in-flight instances and refuses with a
clear message rather than letting `run-instances` fail opaquely. An increase to 32
on `L-DB2E81BA` is pending (case 178525039300546); once granted it buys both
concurrency and the freedom to fall back to larger types when capacity is short.

Do not reuse the pre-existing `hrrrcast-ssm` profile: it carries only
`AmazonSSMManagedInstanceCore`, so a run using it could neither fetch the model
nor terminate itself.

### The `hrrrcast-runner` role

Policies are checked in under [iam/](iam/) so they can be reviewed and reapplied:
[trust-policy.json](iam/trust-policy.json) and
[hrrrcast-runner-policy.json](iam/hrrrcast-runner-policy.json). Scoped to:

- `s3:GetObject` on the model and the `hrrrcast/code/*` prefix only
- `s3:PutObject` on `hrrrcast/out/*`, `hrrrcast/logs/*`, `hrrrcast/plots/*` only
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
| `hrrrcast/model.keras` | **never expires** |
| incomplete multipart uploads | aborted after 7 days |

**The rules are prefix-scoped specifically so `model.keras` survives.** A
whole-bucket expiry would delete it, and every future run stages the model from
there, so the deployment would break silently 12 days later. If you add rules,
keep them scoped and re-check that the model is not matched.

The abort-multipart rule matters because per-hour NetCDF uploads are multipart
(~24-42 parts each): a run killed mid-upload can leave parts that are billed but
invisible to `s3 ls`.

Each 24 h single-member forecast delivers about **8.5 GB** at the current defaults
(`NC_COMPLEVEL=1`, `NC_LSD=2`), so 12 days of daily forecasts is a steady state of
roughly **100 GB** (~$2.30/month in S3 Standard) rather than unbounded growth.

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

### Instance sizing: 48 GB VRAM is a hard floor

Default is `g6e.2xlarge` (1x **L40S 48 GB**, 8 vCPU, 64 GiB host RAM, 450 GB NVMe). This is
measured, not assumed.

**24 GB GPUs cannot run this model.** A `g5.2xlarge` (A10G 24 GB) exhausted all
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

Two things to take from this:

- **Lossless deflate barely helps** on float32 continuous fields, and level 1
  captures essentially all of the benefit. There is no reason to go above 1.
- **Quantization is the actual lever.** `NC_LSD=2` gives 3.8x for a maximum
  absolute error of 0.0039 in each variable's native units (K, dBZ, mm, ...).

**The AWS default is `NC_COMPLEVEL=1` with `NC_LSD=2`**, so a 24 h single-member
forecast delivers about **8.5 GB** rather than ~21 GB lossless. `NC_LSD=2` is a
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

## What this does not do

Deliberately absent, because on-demand single-member runs do not need it:
scheduled triggers, AWS Batch or Step Functions, spot instances with
checkpointing, ensemble fan-out, and a baked AMI. Each becomes worth adding at a
specific point:

- **More members** is the trigger for Batch (fan-out) and spot (cost).
- **Frequent runs** is the trigger for a baked AMI; every launch currently
  rebuilds the conda env from `environment.aws.yaml`, which takes minutes. The
  env is also unpinned apart from TensorFlow, so a rebuild months from now may
  not reproduce.
- **Recurring cadence** is the trigger for EventBridge.

## Related files

| file | role |
|---|---|
| [run_on_ec2.sh](run_on_ec2.sh) | launcher; packages code, launches, reports |
| [user_data.sh](user_data.sh) | instance bootstrap template; always self-terminates |
| [setup_gpu.sh](setup_gpu.sh) | provisions the conda env, stages the model, verifies GPU |
| [run_plots.sh](run_plots.sh) | independent plot job reading NetCDF from S3 |
| [environment.aws.yaml](environment.aws.yaml) | GPU conda env (TF 2.15 + boto3) |
| [../src/s3io.py](../src/s3io.py) | per-hour upload with retries and local purge |
| [../run_cycle.sh](../run_cycle.sh) | the cycle itself; see its header for all knobs |
