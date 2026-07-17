# HRRRCast on AWS: MVP (single GPU EC2)

Goal of this MVP: run one real forecast on a GPU instance to confirm GPU parity
and get true per-hour timing, before committing to any orchestration
(ParallelCluster or Batch). It is intentionally manual.

Run everything in **us-east-1**: the input data (HRRR `noaa-hrrr-bdp-pds`, GFS
`noaa-gfs-bdp-pds`) are public S3 buckets there, so downloads are fast and
egress-free.

## 1. Choose an instance

The model is small (about 50M parameters); the cost is diffusion sampling steps
at the full 1059x1799 grid. A single modern GPU is enough for one member.

| Instance | GPU | Use |
|----------|-----|-----|
| `g5.xlarge` | 1x A10G 24GB | cheapest starting point; validate parity/timing |
| `g6.xlarge` | 1x L4 24GB | similar tier |
| `p4d.24xlarge` | 8x A100 40GB | when you need speed or many members |
| `p5.48xlarge` | 8x H100 | matches the validation hardware; overkill for MVP |

Start with `g5.xlarge`. Scale up only after the MVP timing tells you to.

## 2. Launch

- **AMI**: "AWS Deep Learning Base GPU AMI (Ubuntu)". It ships the NVIDIA driver
  and CUDA, so `nvidia-smi` works out of the box. (Any Ubuntu + NVIDIA driver
  also works; the Base GPU AMI is the least setup.)
- **Storage**: attach at least a 200 GB gp3 EBS volume. One cycle writes large
  intermediates (about 1 GB HRRR npz, 0.5 GB GFS npz) plus GRIB2/NetCDF outputs.
- **IAM role**: attach an instance profile allowing:
  - read of the model bucket (if using `MODEL_S3`) and write of your output bucket,
  - the public NOAA buckets need no credentials.
- **Security group**: outbound HTTPS only; SSH inbound from your IP.

Example (adjust ids; requires your AWS CLI configured locally):

```bash
aws ec2 run-instances --region us-east-1 \
  --image-id <dl-base-gpu-ami-id> \
  --instance-type g5.xlarge \
  --key-name <your-key> \
  --iam-instance-profile Name=<your-instance-profile> \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}}]' \
  --security-group-ids <sg-id> --subnet-id <subnet-id> \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=hrrrcast-mvp}]'
```

## 3. Get the model onto the instance

Two options.

**Option A — from your GitHub repo via Git LFS (no S3 needed).** `setup_gpu.sh`
does this automatically when `MODEL_S3` is unset: it fetches from `origin` (your
fork), and if the fork does not serve the blob it falls back to the canonical
upstream that does. Requirements:
- The instance can authenticate to GitHub for your private fork (the same SSH
  key or token the `git clone` uses).
- Note: GitHub forks do not store their own LFS copies; the blob is served
  through the fork network (rooted at `NOAA-GSL/HRRRCast-live`). The fallback
  handles this. Override the fallback with `MODEL_LFS_UPSTREAM=<repo-url>` if
  your upstream differs.

**Option B — stage to S3 once (more robust for repeated/automated launches).**
From this repo with the model resolved locally:

```bash
aws s3 cp net-diffusion/model.keras s3://<your-bucket>/hrrrcast/model.keras
```

S3 is faster and needs no GitHub auth on the box, which matters for the later
ParallelCluster/Batch paths. For a one-off MVP, Option A is fine.

## 4. Set up the instance

SSH in, get the code, run the setup script:

```bash
git clone git@github.com:steliosmantari/HRRRCast-live.git
cd HRRRCast-live
git checkout simplified_run_script

# Option A (model from your GitHub repo via Git LFS, no S3):
./aws/setup_gpu.sh

# Option B (model from S3):
# MODEL_S3=s3://<your-bucket>/hrrrcast/model.keras ./aws/setup_gpu.sh
```

`setup_gpu.sh` verifies `nvidia-smi`, installs Miniforge if needed, builds the
`hrrrcast` env from `aws/environment.aws.yaml` (GPU TensorFlow 2.15 via
`CONDA_OVERRIDE_CUDA`), stages the model, and confirms TensorFlow sees the GPU
and the model loads. It is fatal on any failure.

## 5. Run a forecast cycle

```bash
conda activate hrrrcast
./run_cycle.sh 2024-05-06T23 6 1
```

This runs the same stages as the HPC `submit_all.sh`, sequentially. Outputs land
under `<DATAROOT>/<YYYYMMDD>/<HH>/` (NetCDF + GRIB2), logs under
`<DATAROOT>/logs/`. See the forecast-option env vars documented in
[../README.mac.md](../README.mac.md#quick-start); they apply here unchanged.

## 6. Collect outputs and stop

```bash
aws s3 cp --recursive 20240506/23 s3://<your-bucket>/hrrrcast/out/20240506/23/
```

Then **stop or terminate the instance** so it stops billing. GPU instances are
the dominant cost; do not leave one idle.

## What this MVP proves (and what it does not)

Proves: the pipeline runs end-to-end on an AWS GPU, the model loads and infers
on GPU, and you have real per-hour wall-clock timing to size production.

Does not cover (deliberately, these are the next step): scheduling, retries,
elastic scaling, multi-member fan-out, output delivery, and cost controls.
Those belong to the ParallelCluster or AWS Batch paths.

## Notes

- No code changes are needed for GPU beyond what is already on this branch. The
  `src/resnet.py` (negative gather index) and `src/plot.py` (spawn-safe worker
  logger) fixes are correct and harmless on Linux/GPU.
- `run_cycle.sh` sources `etc/env_mac.sh`, which only activates the `hrrrcast`
  env; the `TF_USE_LEGACY_KERAS=1` it sets is a no-op under TF 2.15 (already
  Keras 2). No separate AWS env shim is required for the MVP.
