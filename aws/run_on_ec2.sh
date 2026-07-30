#!/usr/bin/env bash
#
# aws/run_on_ec2.sh — launch one on-demand HRRRCast forecast on a GPU EC2
# instance and walk away. Run this from your workstation.
#
# It renders aws/user_data.sh (atparse, the same @[VAR] convention as the
# jobs/job-*.sh SLURM templates), launches one instance with that user-data, and
# returns. The instance provisions itself, runs the cycle, streams NetCDF to S3
# per lead hour, uploads its logs, and terminates itself.
#
# This is deliberately not an orchestrator. On-demand single-member runs do not
# need Batch or Step Functions; they need one instance that reliably cleans up
# after itself. Revisit that when the member count grows.
#
# Usage:
#   aws/run_on_ec2.sh --bucket my-bucket [options]
#
# Required:
#   --bucket NAME          S3 bucket for outputs and logs (must already exist)
#
# Common options (defaults in brackets):
#   --init-time TIME       cycle init, YYYY-MM-DDTHH   [latest complete HRRR cycle]
#   --lead-hours N         forecast length              [24]
#   --members N            ensemble members             [1]
#   --instance-type TYPE   EC2 instance type            [g6e.2xlarge; needs >=48 GB VRAM]
#   --ami-id ID            AMI; must have NVIDIA drivers [latest DL Base GPU AMI, looked up]
#   --key-name NAME        EC2 key pair for SSH         [none: no SSH access]
#   --subnet-id ID         subnet                       [account default for the AZ]
#   --security-group-ids   security group(s)            [account default]
#   --instance-profile     IAM instance profile name    [hrrrcast-runner]
#   --volume-size GB       root EBS size                [200]
#   --region REGION        AWS region                   [us-east-1]
#   --code-s3 URI          override where the code tarball is staged
#                          [s3://BUCKET/hrrrcast/code/<sha>-<timestamp>.tar.gz]
#   --stage-scheduler      pin the code tarball and a user-data template to S3 for
#                          the hourly Lambda, then exit without launching. This is
#                          the deploy step for unattended operation; re-run it to
#                          roll out a code change. See aws/deploy_scheduler.sh.
#   --model-s3 URI         model.keras location         [s3://BUCKET/hrrrcast/model.keras]
#   --nc-complevel N       NetCDF zlib level, 0-9       [1]
#   --nc-lsd N             LOSSY quantization digits    [2; pass "" for lossless]
#   --make-bcs-workers N   GFS regrid worker cap        [2]
#   --notify-topic ARN     SNS topic to email a run summary to when the run ends,
#                          success or failure [$HRRRCAST_SNS_TOPIC, else none]
#   --wait-for-capacity M  keep retrying for M minutes if no AZ has GPU capacity [0]
#   --preflight-only       run the account checks (creds, bucket, model, IAM, AMI,
#                          GPU quota) and stop; launches nothing, costs nothing
#   --no-terminate         leave the instance up afterwards (for debugging)
#   --dry-run              print the plan and rendered user-data; launch nothing
#
# Region note: us-east-1 co-locates with the public NOAA HRRR/GFS buckets
# (noaa-hrrr-bdp-pds / noaa-gfs-bdp-pds), so input pulls are fast and egress-free.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- defaults --------------------------------------------------------------
BUCKET=""
INIT_TIME=""
LEAD_HOUR=24
N_ENSEMBLES=1
# g6e.2xlarge (1x L40S 48 GB, 8 vCPU, 64 GiB RAM, 450 GB NVMe), NOT a g5. Measured, not assumed:
# g5.2xlarge (A10G 24 GB) OOMs in the forecast, exhausting all 20795 MB usable
# while allocating a single 1059x1799x139 float32 activation (1058646016 bytes) at
# batch_size 1 with memory growth on. There is nothing to tune; 24 GB is too small.
# The L40S runs it in 36.4 s per lead hour steady state. Note g6e.2xlarge consumes
# 8 vCPU, the entire default G/VT quota, so only one run at a time until the quota
# rises; g6e.xlarge (4 vCPU) would leave headroom but had no capacity in any
# us-east-1 AZ when tested.
INSTANCE_TYPE="g6e.2xlarge"
AMI_ID=""
KEY_NAME=""
SUBNET_ID=""
SECURITY_GROUP_IDS=""
INSTANCE_PROFILE="hrrrcast-runner"
# 200 GB, not 100. Measured on the first smoke test: the Deep Learning Base GPU
# AMI plus the conda env already occupies ~63 GB, leaving only ~34 GB on a 100 GB
# root volume. A 24-hour cycle needs roughly 20 GB of inputs and intermediates
# (GFS is ~522 MB per lead hour, so ~12.5 GB at 24 h, plus ~0.7 GB HRRR, ~4 GB
# npz, and ~2 GB of in-flight NetCDF before purge). That fits in 34 GB but with
# little margin, and gp3 for a few hours costs cents.
VOLUME_SIZE=200
REGION="us-east-1"
MODEL_S3=""
CODE_S3_OVERRIDE=""
NC_COMPLEVEL=1
# NC_LSD=2 quantizes NetCDF to 2 decimal digits before compression. Chosen
# deliberately: measured on a real f01 file it gives 3.82x (879 MB -> 355 MB, so
# ~21 GB -> ~8.5 GB per 24 h forecast) for a maximum absolute error of 0.0039 in
# each variable's native units (K, dBZ, mm). It is LOSSY. It also cuts write time,
# which matters because the single NetCDF writer thread has only ~10 s of margin
# per lead hour against 36 s of L40S inference. Pass --nc-lsd "" for lossless.
NC_LSD=2
# "auto" lets the instance size its own make_bcs pool from measured RAM (see
# user_data.sh). A fixed number was only ever right for one instance type.
MAKE_BCS_WORKERS=auto
# Empty = no notification. Set to an SNS topic ARN (or export HRRRCAST_SNS_TOPIC)
# to get an email summary when a run ends, success or failure.
SNS_TOPIC="${HRRRCAST_SNS_TOPIC:-}"
# How far back to shift GFS cycle selection before rounding to 00/06/12/18Z.
# 0 keeps the newest cycle, matching every run so far. It is the right default for
# an on-demand run of a past cycle, where all the data is long since published.
# aws/pick_cycle.sh emits GFS_MIN_LAG=4 for real-time hourly operation, where the
# newest cycle has not finished publishing yet. See src/get_bcs.py gfs_cycle_for().
GFS_MIN_LAG="${GFS_MIN_LAG:-0}"
TERMINATE="YES"
DRY_RUN="NO"
# GPU capacity comes and goes: g6e.2xlarge launched fine in us-east-1b and was
# refused in every AZ 40 minutes later. 0 means try once and give up.
WAIT_CAPACITY_MIN=0
WAIT_CAPACITY_INTERVAL=120
PREFLIGHT_ONLY="NO"
# Deploy-time staging for the hourly scheduler (see the --stage-scheduler block).
STAGE_SCHEDULER="NO"
# Command the instance runs once its env is ready. Default reproduces the original
# behavior; --run-cmd swaps in an experiment (see aws/run_domain_test.sh). Rendered
# into user_data.sh at @[RUN_CMD], so it must be a single valid shell statement and
# must not contain @[...] sequences of its own.
RUN_CMD=""

log() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --bucket)              BUCKET="$2"; shift 2 ;;
        --init-time)           INIT_TIME="$2"; shift 2 ;;
        --lead-hours)          LEAD_HOUR="$2"; shift 2 ;;
        --members)             N_ENSEMBLES="$2"; shift 2 ;;
        --instance-type)       INSTANCE_TYPE="$2"; shift 2 ;;
        --ami-id)              AMI_ID="$2"; shift 2 ;;
        --key-name)            KEY_NAME="$2"; shift 2 ;;
        --subnet-id)           SUBNET_ID="$2"; shift 2 ;;
        --security-group-ids)  SECURITY_GROUP_IDS="$2"; shift 2 ;;
        --instance-profile)    INSTANCE_PROFILE="$2"; shift 2 ;;
        --volume-size)         VOLUME_SIZE="$2"; shift 2 ;;
        --region)              REGION="$2"; shift 2 ;;
        --code-s3)             CODE_S3_OVERRIDE="$2"; shift 2 ;;
        --model-s3)            MODEL_S3="$2"; shift 2 ;;
        --nc-complevel)        NC_COMPLEVEL="$2"; shift 2 ;;
        --nc-lsd)              NC_LSD="$2"; shift 2 ;;
        --gfs-min-lag)         GFS_MIN_LAG="$2"; shift 2 ;;
        --make-bcs-workers)    MAKE_BCS_WORKERS="$2"; shift 2 ;;
        --notify-topic)        SNS_TOPIC="$2"; shift 2 ;;
        --wait-for-capacity)   WAIT_CAPACITY_MIN="$2"; shift 2 ;;
        --preflight-only)      PREFLIGHT_ONLY="YES"; shift ;;
        --stage-scheduler)     STAGE_SCHEDULER="YES"; shift ;;
        --run-cmd)             RUN_CMD="$2"; shift 2 ;;
        --no-terminate)        TERMINATE="NO"; shift ;;
        --dry-run)             DRY_RUN="YES"; shift ;;
        -h|--help)             sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d'; exit 0 ;;
        *)                     die "Unknown option: $1 (try --help)" ;;
    esac
done

[ -n "$BUCKET" ] || die "--bucket is required (try --help)"
command -v aws >/dev/null 2>&1 || die "aws CLI not found."
export AWS_DEFAULT_REGION="$REGION"
export AWS_PAGER=""

# --- derive defaults -------------------------------------------------------
# Default init time: the most recent cycle old enough that the HRRR analysis has
# certainly landed. HRRR posts roughly 50-90 minutes after cycle time, so back
# off 3 hours and truncate to the hour.
if [ -z "$INIT_TIME" ]; then
    INIT_TIME="$(date -u -v-3H +%Y-%m-%dT%H 2>/dev/null || date -u -d '3 hours ago' +%Y-%m-%dT%H)"
    log "No --init-time given; using latest safely-available cycle: ${INIT_TIME}"
fi
[[ "$INIT_TIME" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}$ ]] \
    || die "--init-time must be YYYY-MM-DDTHH (got '${INIT_TIME}')"

[ -z "$MODEL_S3" ] && MODEL_S3="s3://${BUCKET}/hrrrcast/model.keras"

# Provenance string for the code the instance will run. Records the commit plus
# whether the tree was dirty, since the tarball ships the working tree, not HEAD.
GIT_SHA="$(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo nogit)"
if [ -n "$(git -C "$REPO_DIR" status --porcelain --untracked-files=no 2>/dev/null)" ]; then
    CODE_REF="${GIT_SHA}-dirty"
else
    CODE_REF="${GIT_SHA}"
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CODE_S3="${CODE_S3_OVERRIDE:-s3://${BUCKET}/hrrrcast/code/hrrrcast-${CODE_REF}-${STAMP}.tar.gz}"

S3_OUTPUT="s3://${BUCKET}/hrrrcast/out"
S3_LOGS="s3://${BUCKET}/hrrrcast/logs"

# Latest AWS Deep Learning Base GPU AMI (Ubuntu 22.04) from SSM public parameters.
if [ -z "$AMI_ID" ]; then
    log "Looking up the latest Deep Learning Base GPU AMI"
    AMI_ID="$(aws ssm get-parameter \
        --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id \
        --query Parameter.Value --output text 2>/dev/null || true)"
    [ -n "$AMI_ID" ] && [ "$AMI_ID" != "None" ] \
        || die "AMI lookup failed. Pass --ami-id explicitly (needs NVIDIA drivers preinstalled)."
fi

# --- preflight -------------------------------------------------------------
if [ "$DRY_RUN" == "NO" ]; then
    log "Preflight checks"
    CALLER="$(aws sts get-caller-identity --query Arn --output text 2>/dev/null)" \
        || die "No usable AWS credentials. Authenticate first."
    echo "  identity: ${CALLER}"
    aws s3api head-bucket --bucket "$BUCKET" >/dev/null 2>&1 \
        || die "Bucket '${BUCKET}' is not reachable. Create it (in ${REGION}) first."
    aws s3 ls "$MODEL_S3" >/dev/null 2>&1 \
        || die "Model not found at ${MODEL_S3}. Stage it once:
    aws s3 cp net-diffusion/model.keras ${MODEL_S3}"
    aws iam get-instance-profile --instance-profile-name "$INSTANCE_PROFILE" >/dev/null 2>&1 \
        || die "Instance profile '${INSTANCE_PROFILE}' not found. It needs read on ${MODEL_S3},
    write on s3://${BUCKET}/hrrrcast/, and ec2:TerminateInstances on itself."
    echo "  instance profile: ${INSTANCE_PROFILE}"

    # GPU vCPU quota. A g5.2xlarge needs 8 vCPUs of "Running On-Demand G and VT
    # instances"; the account default is often exactly 8, so a larger type or a
    # second concurrent run fails at run-instances with an opaque error. Check
    # here where the message can be useful.
    VCPU_NEED="$(aws ec2 describe-instance-types --instance-types "$INSTANCE_TYPE" \
        --query 'InstanceTypes[0].VCpuInfo.DefaultVCpus' --output text 2>/dev/null || echo unknown)"
    VCPU_QUOTA="$(aws service-quotas get-service-quota --service-code ec2 \
        --quota-code L-DB2E81BA --query 'Quota.Value' --output text 2>/dev/null || echo unknown)"
    echo "  ${INSTANCE_TYPE} needs ${VCPU_NEED} vCPU; G/VT on-demand quota is ${VCPU_QUOTA}"
    if [ "$VCPU_NEED" != "unknown" ] && [ "$VCPU_QUOTA" != "unknown" ]; then
        awk -v n="$VCPU_NEED" -v q="$VCPU_QUOTA" 'BEGIN{exit !(n>q)}' \
            && die "${INSTANCE_TYPE} needs ${VCPU_NEED} vCPU but the G/VT on-demand quota is ${VCPU_QUOTA}.
    Request an increase for 'Running On-Demand G and VT instances' (quota L-DB2E81BA) in ${REGION},
    or choose a smaller type."
    fi

    # In-flight runs hold their vCPUs until they reach `terminated`, including
    # while `shutting-down`. With a quota of exactly one g5.2xlarge, launching
    # while a previous run winds down fails with a bare VcpuLimitExceeded, which
    # does not hint that the fix is simply to wait. Say so here instead.
    INFLIGHT="$(aws ec2 describe-instances \
        --filters "Name=tag:Project,Values=hrrrcast" \
                  "Name=instance-state-name,Values=pending,running,shutting-down,stopping" \
        --query 'Reservations[].Instances[].[InstanceId,State.Name]' --output text 2>/dev/null || true)"
    if [ -n "$INFLIGHT" ]; then
        printf '\033[1;33mWARNING:\033[0m other hrrrcast instances are still holding vCPU quota:\n%s\n' "$INFLIGHT"
        die "Wait for these to reach 'terminated', then relaunch. A shutting-down instance still counts
    against the G/VT vCPU quota (currently ${VCPU_QUOTA}), so this launch would fail with
    VcpuLimitExceeded. Watch with:
      aws ec2 describe-instances --filters Name=tag:Project,Values=hrrrcast \\
        --query 'Reservations[].Instances[].[InstanceId,State.Name]' --output text"
    fi

    if [ "$PREFLIGHT_ONLY" == "YES" ]; then
        log "Preflight passed. Nothing launched (--preflight-only)."
        exit 0
    fi
fi
if [ "$PREFLIGHT_ONLY" == "YES" ] && [ "$DRY_RUN" == "YES" ]; then
    die "--preflight-only and --dry-run are mutually exclusive: preflight needs live AWS calls."
fi

# --- package and stage the code -------------------------------------------
# Ship the working tree, not `git archive HEAD`: HEAD would omit uncommitted work
# and would ship the Git LFS pointer instead of model.keras. Excludes: .git, the
# 203 MB model (staged separately, pulled by setup_gpu.sh), per-cycle output
# directories (tens of GB), and local logs/caches.
CODE_TARBALL="$(mktemp -t hrrrcast-code).tar.gz"
log "Packaging the working tree (${CODE_REF})"
tar -czf "$CODE_TARBALL" -C "$REPO_DIR" \
    --exclude='.git' \
    --exclude='.DS_Store' \
    --exclude='__pycache__' \
    --exclude='net-diffusion/model.keras' \
    --exclude='logs' \
    --exclude='[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]' \
    . || die "packaging failed"
TAR_MB=$(( $(wc -c < "$CODE_TARBALL") / 1000000 ))
log "Code tarball: ${TAR_MB} MB"

# Sanity-check that the tarball actually carries what the run needs. A silent
# exclude bug here surfaces as a confusing failure 10 minutes into boot.
# conda-linux-64.lock is deliberately NOT in this list: it is optional, and
# setup_gpu.sh falls back to solving environment.aws.yaml without it. It is
# validated separately below when present.
for required in run_cycle.sh src/fcst.py src/s3io.py src/make_bcs.py aws/setup_gpu.sh aws/environment.aws.yaml etc/env_mac.sh; do
    tar -tzf "$CODE_TARBALL" "./${required}" >/dev/null 2>&1 \
        || die "packaging bug: ${required} missing from the tarball"
done
# Validate the conda lock before spending money on an instance. Two lock defects
# have already reached a live GPU box: a CPU TensorFlow build (silent 50x
# slowdown) and an external_* mpich stub (crash in make_ics). Both are visible by
# inspecting the lock, so check here rather than 4 minutes into a run.
if [ -f "${REPO_DIR}/aws/conda-linux-64.lock" ]; then
    "${REPO_DIR}/aws/validate_lock.sh" "${REPO_DIR}/aws/conda-linux-64.lock" \
        || die "aws/conda-linux-64.lock failed validation (see above). Fix the lock, delete it to
    fall back to solving environment.aws.yaml, or pass USE_LOCK=NO through to the instance."
else
    printf '\033[1;33mWARNING:\033[0m no aws/conda-linux-64.lock; the instance will solve environment.aws.yaml (unpinned, slower).\n'
fi

if ! tar -tzf "$CODE_TARBALL" ./gfs_to_hrrr_weights.nc >/dev/null 2>&1; then
    printf '\033[1;33mnote:\033[0m gfs_to_hrrr_weights.nc is not in this checkout; make_bcs regenerates it on the instance (measured: 33 s).\n'
fi

if [ "$DRY_RUN" == "NO" ]; then
    log "Uploading code to ${CODE_S3}"
    aws s3 cp "$CODE_TARBALL" "$CODE_S3" || die "code upload failed"
fi
rm -f "$CODE_TARBALL"

# --- render user-data ------------------------------------------------------
# atparse aborts on undefined @[VAR], so every placeholder must be set here.
RUN_USER="ubuntu"
if [ -z "$RUN_CMD" ]; then
    RUN_CMD="./run_cycle.sh '${INIT_TIME}' '${LEAD_HOUR}' '${N_ENSEMBLES}' 1 \"\$WORKDIR\" \"\$WORKDIR\" NO"
fi
case "$RUN_CMD" in *'@['*) die "--run-cmd must not contain @[...]; atparse would try to expand it." ;; esac
USER_DATA_FILE="$(mktemp -t hrrrcast-userdata)"
# shellcheck disable=SC1091
source "${REPO_DIR}/atparse.bash"
atparse \
    INIT_TIME="$INIT_TIME" LEAD_HOUR="$LEAD_HOUR" N_ENSEMBLES="$N_ENSEMBLES" \
    S3_OUTPUT="$S3_OUTPUT" S3_LOGS="$S3_LOGS" \
    CODE_S3="$CODE_S3" CODE_REF="$CODE_REF" MODEL_S3="$MODEL_S3" \
    RUN_USER="$RUN_USER" NC_COMPLEVEL="$NC_COMPLEVEL" NC_LSD="$NC_LSD" \
    MAKE_BCS_WORKERS="$MAKE_BCS_WORKERS" TERMINATE="$TERMINATE" \
    SNS_TOPIC="$SNS_TOPIC" GFS_MIN_LAG="$GFS_MIN_LAG" RUN_CMD="$RUN_CMD" \
    < "${REPO_DIR}/aws/user_data.sh" > "$USER_DATA_FILE"

bash -n "$USER_DATA_FILE" || die "Rendered user-data is not valid bash (template bug)."

cat <<EOF

  init time        ${INIT_TIME}
  lead hours       ${LEAD_HOUR}
  members          ${N_ENSEMBLES}
  instance         ${INSTANCE_TYPE}  (${REGION})
  ami              ${AMI_ID}
  root volume      ${VOLUME_SIZE} GB gp3
  code             ${CODE_REF} (working tree) -> ${CODE_S3}
  model            ${MODEL_S3}
  outputs          ${S3_OUTPUT}/<YYYYMMDD>/<HH>/
  logs             ${S3_LOGS}/
  netcdf           complevel=${NC_COMPLEVEL} lsd=${NC_LSD:-off}
  gfs cycle lag    ${GFS_MIN_LAG} h
  grib2            disabled
  plots            separate job (aws/run_plots.sh)
  self-terminate   ${TERMINATE}
  notify           ${SNS_TOPIC:-off}
EOF

if [ "$DRY_RUN" == "YES" ]; then
    log "Dry run: rendered user-data follows; nothing was launched"
    cat "$USER_DATA_FILE"
    rm -f "$USER_DATA_FILE"
    exit 0
fi

# --- deploy-time staging for the hourly scheduler --------------------------
# --stage-scheduler makes this script the DEPLOY half of the unattended path,
# and stops before launching anything.
#
# The problem it solves: this script packages the local working tree on every
# invocation, which is right for on-demand development and wrong for a scheduler.
# An hourly Lambda has no checkout, and even if it did, shipping "whatever is in
# the tree" 24 times a day is not a thing you want. So the code tarball and the
# rendered user-data are pinned ONCE here, and the Lambda only substitutes the
# three values that vary per run.
#
# Those three are rendered as __INIT_TIME__, __LEAD_HOUR__ and __GFS_MIN_LAG__
# rather than left as @[VAR], because atparse aborts on any placeholder it was
# not given a value for. Passing sentinels through atparse keeps the template
# path identical to the interactive one, including the `bash -n` check below, so
# a template bug cannot reach the scheduler without also breaking normal runs.
if [ "$STAGE_SCHEDULER" == "YES" ]; then
    STAGE_PREFIX="s3://${BUCKET}/hrrrcast/scheduler"
    TEMPLATE_FILE="$(mktemp -t hrrrcast-udtemplate)"
    atparse \
        INIT_TIME="__INIT_TIME__" LEAD_HOUR="__LEAD_HOUR__" N_ENSEMBLES="$N_ENSEMBLES" \
        S3_OUTPUT="$S3_OUTPUT" S3_LOGS="$S3_LOGS" \
        CODE_S3="$CODE_S3" CODE_REF="$CODE_REF" MODEL_S3="$MODEL_S3" \
        RUN_USER="$RUN_USER" NC_COMPLEVEL="$NC_COMPLEVEL" NC_LSD="$NC_LSD" \
        MAKE_BCS_WORKERS="$MAKE_BCS_WORKERS" TERMINATE="$TERMINATE" \
        SNS_TOPIC="$SNS_TOPIC" GFS_MIN_LAG="__GFS_MIN_LAG__" RUN_CMD="$RUN_CMD" \
        < "${REPO_DIR}/aws/user_data.sh" > "$TEMPLATE_FILE"

    # The template still has to be valid bash with the sentinels in place, since
    # the Lambda only does string substitution and never re-validates.
    bash -n "$TEMPLATE_FILE" || die "Rendered user-data template is not valid bash (template bug)."
    for sentinel in __INIT_TIME__ __LEAD_HOUR__ __GFS_MIN_LAG__; do
        grep -q "$sentinel" "$TEMPLATE_FILE" \
            || die "sentinel ${sentinel} is absent from the template; user_data.sh no longer references it."
    done
    grep -q '@\[' "$TEMPLATE_FILE" \
        && die "template still contains unexpanded @[...] placeholders."

    # Launch parameters the Lambda would otherwise have to hardcode. Resolved here,
    # where the AMI lookup and subnet discovery already happened.
    PARAMS_FILE="$(mktemp -t hrrrcast-params)"
    cat > "$PARAMS_FILE" <<EOF
{
  "region": "${REGION}",
  "image_id": "${AMI_ID}",
  "instance_type": "${INSTANCE_TYPE}",
  "instance_profile": "${INSTANCE_PROFILE}",
  "volume_size": ${VOLUME_SIZE},
  "security_group_ids": "${SECURITY_GROUP_IDS}",
  "s3_output": "${S3_OUTPUT}",
  "s3_logs": "${S3_LOGS}",
  "code_ref": "${CODE_REF}",
  "code_s3": "${CODE_S3}",
  "lead_hours": ${LEAD_HOUR},
  "gfs_min_lag": ${GFS_MIN_LAG},
  "n_ensembles": ${N_ENSEMBLES},
  "user_data_template": "${STAGE_PREFIX}/user_data.template",
  "staged_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$PARAMS_FILE" \
        || die "generated launch-params JSON is invalid"

    log "Staging scheduler artifacts to ${STAGE_PREFIX}/"
    aws s3 cp "$TEMPLATE_FILE" "${STAGE_PREFIX}/user_data.template" --quiet \
        || die "user-data template upload failed"
    aws s3 cp "$PARAMS_FILE" "${STAGE_PREFIX}/launch-params.json" --quiet \
        || die "launch-params upload failed"

    cat <<EOF

  Staged for the hourly scheduler:
    code         ${CODE_S3}
    code ref     ${CODE_REF}
    template     ${STAGE_PREFIX}/user_data.template
    params       ${STAGE_PREFIX}/launch-params.json
    lead hours   ${LEAD_HOUR}
    gfs min lag  ${GFS_MIN_LAG}

  The Lambda reads both objects at every invocation, so re-running
  --stage-scheduler is how you roll out a code change. Nothing was launched.
EOF
    rm -f "$TEMPLATE_FILE" "$PARAMS_FILE" "$USER_DATA_FILE"
    exit 0
fi

# --- launch ----------------------------------------------------------------
log "Launching"
RUN_ARGS=(
    --region "$REGION"
    --image-id "$AMI_ID"
    --instance-type "$INSTANCE_TYPE"
    --iam-instance-profile "Name=${INSTANCE_PROFILE}"
    --instance-initiated-shutdown-behavior terminate
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${VOLUME_SIZE},\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]"
    --user-data "file://${USER_DATA_FILE}"
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=hrrrcast-${INIT_TIME}},{Key=Project,Value=hrrrcast},{Key=InitTime,Value=${INIT_TIME}},{Key=LeadHours,Value=${LEAD_HOUR}}]"
    --count 1
)
[ -n "$KEY_NAME" ]           && RUN_ARGS+=(--key-name "$KEY_NAME")
[ -n "$SECURITY_GROUP_IDS" ] && RUN_ARGS+=(--security-group-ids "$SECURITY_GROUP_IDS")

# Try each candidate subnet until one has capacity. InsufficientInstanceCapacity
# is routine for on-demand GPU types (observed immediately on g6e.xlarge), and it
# is per-AZ, so the fix is to ask a different AZ rather than to give up. If the
# caller pinned --subnet-id we respect it and try only that one.
if [ -n "$SUBNET_ID" ]; then
    CANDIDATE_SUBNETS="$SUBNET_ID"
else
    # First candidate is the empty string, meaning "send no --subnet-id at all".
    # EC2 then picks the AZ itself and can place the instance anywhere with room,
    # which succeeds in cases where every explicit per-AZ request is refused.
    # AWS's own capacity error recommends exactly this ("by not specifying an
    # Availability Zone"), and pinning a subnet silently gives that up.
    CANDIDATE_SUBNETS="AUTO"
    # Default subnets, restricted to AZs that actually offer this instance type.
    OFFERED_AZS="$(aws ec2 describe-instance-type-offerings --location-type availability-zone \
        --filters "Name=instance-type,Values=${INSTANCE_TYPE}" \
        --query 'InstanceTypeOfferings[].Location' --output text 2>/dev/null | tr '\t' '\n' | sort)"
    [ -n "$OFFERED_AZS" ] \
        || die "${INSTANCE_TYPE} is not offered in any ${REGION} availability zone."
    for az in $OFFERED_AZS; do
        sn="$(aws ec2 describe-subnets \
            --filters "Name=default-for-az,Values=true" "Name=availability-zone,Values=${az}" \
            --query 'Subnets[0].SubnetId' --output text 2>/dev/null)"
        [ -n "$sn" ] && [ "$sn" != "None" ] && CANDIDATE_SUBNETS="${CANDIDATE_SUBNETS} ${sn}:${az}"
    done
    [ -n "$CANDIDATE_SUBNETS" ] \
        || die "No default subnet found in any AZ offering ${INSTANCE_TYPE}. Pass --subnet-id."
fi

INSTANCE_ID=""
LAUNCH_ERR=""
LAUNCH_AZ=""
DEADLINE=$(( $(date +%s) + WAIT_CAPACITY_MIN * 60 ))
ATTEMPT=0
while : ; do
    ATTEMPT=$(( ATTEMPT + 1 ))
    for entry in $CANDIDATE_SUBNETS; do
        if [ "$entry" == "AUTO" ]; then
            sn=""; az="EC2-chosen"
        else
            sn="${entry%%:*}"; az="${entry##*:}"
            [ "$sn" == "$az" ] && az="(pinned)"
        fi
        printf '    trying %-12s %s ... ' "$az" "${sn:-any AZ}"
        if [ -z "$sn" ]; then
            OUT="$(aws ec2 run-instances "${RUN_ARGS[@]}" \
                    --query 'Instances[0].InstanceId' --output text 2>&1)" && RC=0 || RC=$?
        else
            OUT="$(aws ec2 run-instances "${RUN_ARGS[@]}" --subnet-id "$sn" \
                    --query 'Instances[0].InstanceId' --output text 2>&1)" && RC=0 || RC=$?
        fi
        if [ $RC -eq 0 ]; then
            INSTANCE_ID="$OUT"; LAUNCH_AZ="$az"
            printf 'launched\n'
            break
        fi
        LAUNCH_ERR="$OUT"
        case "$OUT" in
            *InsufficientInstanceCapacity*) printf 'no capacity\n' ;;
            *VcpuLimitExceeded*)            printf 'vCPU limit\n' ;;
            *)                              printf 'failed\n' ;;
        esac
    done
    [ -n "$INSTANCE_ID" ] && break

    # Only capacity is worth waiting on. A vCPU limit or a bad parameter will not
    # fix itself, so fail fast on anything else.
    case "$LAUNCH_ERR" in
        *InsufficientInstanceCapacity*) ;;
        *) break ;;
    esac
    NOW=$(date +%s)
    [ "$NOW" -ge "$DEADLINE" ] && break
    LEFT=$(( (DEADLINE - NOW) / 60 ))
    log "No ${INSTANCE_TYPE} capacity on attempt ${ATTEMPT}; retrying in ${WAIT_CAPACITY_INTERVAL}s (${LEFT} min left of --wait-for-capacity)"
    sleep "$WAIT_CAPACITY_INTERVAL"
done

if [ -z "$INSTANCE_ID" ]; then
    printf '\n%s\n' "$LAUNCH_ERR" >&2
    die "run-instances failed in every candidate AZ for ${INSTANCE_TYPE}.
    If this is InsufficientInstanceCapacity, the type is momentarily unavailable in
    ${REGION}: retry shortly, or try another type with enough VRAM."
fi
rm -f "$USER_DATA_FILE"

log "Launched ${INSTANCE_ID}"
cat <<EOF
The instance provisions itself, runs the cycle, and terminates. Nothing to babysit.

  watch outputs land:   aws s3 ls --recursive ${S3_OUTPUT}/ --human-readable
  read the run log:     aws s3 cp ${S3_LOGS}/${INSTANCE_ID}-bootstrap.log -
  final status:         aws s3 cp ${S3_LOGS}/${INSTANCE_ID}-status.txt -
  still alive?          aws ec2 describe-instances --instance-ids ${INSTANCE_ID} \\
                          --query 'Reservations[].Instances[].State.Name' --output text
  kill it now:          aws ec2 terminate-instances --instance-ids ${INSTANCE_ID}

Logs only appear in S3 once the run ends (success or failure). To follow along
live, relaunch with --key-name and --no-terminate and tail
/var/log/hrrrcast-run.log over SSH.
EOF
