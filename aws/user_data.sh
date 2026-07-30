#!/usr/bin/env bash
#
# aws/user_data.sh — EC2 user-data template for one on-demand HRRRCast forecast.
#
# Rendered by aws/run_on_ec2.sh via atparse (the same at-bracket placeholder
# convention the jobs/job-*.sh SLURM templates use) and passed as instance
# user-data. Runs once at first boot as root, drives the cycle as the
# unprivileged user, ships logs and outputs to S3, then terminates the instance.
#
# Note: every at-bracket placeholder in this file, including ones in comments,
# is substituted by atparse, which aborts on an undefined name. Do not mention
# placeholder syntax here except for names run_on_ec2.sh actually provides.
#
# Design notes:
#   - The instance always terminates, success or failure. A forgotten GPU box is
#     the single most expensive failure mode here, so the trap comes first.
#   - Logs are uploaded before termination. Without that, a failed run leaves no
#     evidence: the instance is gone and so is the console output.
#   - Outputs stream to S3 per lead hour (--s3_output/--purge_local in fcst.py),
#     so a run that dies at hour 20 has still delivered hours 0-19 and the EBS
#     volume never holds the whole forecast.
#
# Placeholder names (all required, injected by run_on_ec2.sh; written without
# their bracket syntax here so this comment is not itself substituted):
#   INIT_TIME, LEAD_HOUR, N_ENSEMBLES, S3_OUTPUT, S3_LOGS, CODE_S3, CODE_REF,
#   MODEL_S3, RUN_USER, NC_COMPLEVEL, NC_LSD, MAKE_BCS_WORKERS, TERMINATE, SNS_TOPIC
set -uo pipefail

RUN_USER="@[RUN_USER]"
WORKDIR="/home/${RUN_USER}/hrrrcast"
BOOTLOG="/var/log/hrrrcast-run.log"
TERMINATE="@[TERMINATE]"

exec > >(tee -a "$BOOTLOG") 2>&1
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
STARTED_EPOCH="$(date -u +%s)"
echo "=== HRRRCast on-demand run starting ${STARTED_AT} ==="

REGION="$(curl -fsS -m 5 http://169.254.169.254/latest/meta-data/placement/region || echo us-east-1)"
INSTANCE_ID="$(curl -fsS -m 5 http://169.254.169.254/latest/meta-data/instance-id || echo unknown)"
export AWS_DEFAULT_REGION="$REGION"
echo "instance=${INSTANCE_ID} region=${REGION}"

# --- always clean up, whatever happens -------------------------------------
STATUS="unknown"
finish() {
    local rc=$?
    [ "$STATUS" == "unknown" ] && STATUS=$([ $rc -eq 0 ] && echo success || echo "failed(rc=$rc)")
    echo "=== run finished: ${STATUS} at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

    # Ship logs before the box disappears; this is the only postmortem material.
    if [ -n "@[S3_LOGS]" ]; then
        aws s3 cp "$BOOTLOG" "@[S3_LOGS]/${INSTANCE_ID}-bootstrap.log" || true
        if [ -d "${WORKDIR}/logs" ]; then
            aws s3 cp --recursive "${WORKDIR}/logs" "@[S3_LOGS]/${INSTANCE_ID}-stages/" || true
        fi
        printf '%s\n' "$STATUS" > /tmp/hrrrcast.status
        aws s3 cp /tmp/hrrrcast.status "@[S3_LOGS]/${INSTANCE_ID}-status.txt" || true
    fi

    # --- email notification (optional) -------------------------------------
    # Publishes a run summary to SNS, which fans out to whatever email addresses
    # are subscribed to the topic. Chosen over SES because it needs no verified
    # sending identity and no SMTP credentials on the box -- just sns:Publish on
    # one topic ARN, which is what the hrrrcast-runner role grants.
    #
    # Everything here is wrapped so a notification problem can never affect the
    # run or prevent termination: this executes inside the EXIT trap, after the
    # outputs are already in S3.
    if [ -n "@[SNS_TOPIC]" ]; then
        {
            ENDED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            ELAPSED_MIN=$(( ( $(date -u +%s) - STARTED_EPOCH ) / 60 ))
            ITYPE="$(curl -fsS -m 5 http://169.254.169.254/latest/meta-data/instance-type || echo unknown)"

            # Delivered outputs, read back from S3 rather than from local disk,
            # since --purge_local has already deleted the local copies.
            OUT_PREFIX="@[S3_OUTPUT]/$(echo '@[INIT_TIME]' | cut -c1-4,6-7,9-10)/$(echo '@[INIT_TIME]' | cut -c12-13)"
            OUT_LINE="$(aws s3 ls --recursive "${OUT_PREFIX}/" 2>/dev/null \
                | awk '/\.nc$/ {n++; s+=$3} END {printf "%d files, %.2f GB", n+0, s/1e9}')"

            # Stage timings, pulled from the logs the run just wrote.
            BCS_T="$(grep -a 'Preprocessing complete' "${WORKDIR}/logs/make-bcs.out" 2>/dev/null | tail -1 | awk '{print $1" "$2}')"
            PREDICTS="$(grep -aoE 'predict took [0-9.]+s' "${WORKDIR}/logs/fcst.out" 2>/dev/null \
                | awk '{gsub("s","",$3); n++; t+=$3; if(n>1){m++; u+=$3}} END {if(n) printf "%d hours, %.1f s/hour steady-state", n, (m? u/m : t/n)}')"
            WORKERS="$(grep -a 'worker process(es)' "${WORKDIR}/logs/make-bcs.out" 2>/dev/null | tail -1 | grep -oE 'Using [0-9]+' | grep -oE '[0-9]+')"

            case "$STATUS" in
                success) SUBJ="HRRRCast OK: @[INIT_TIME] +@[LEAD_HOUR]h (${ELAPSED_MIN} min)" ;;
                *)       SUBJ="HRRRCast FAILED: @[INIT_TIME] +@[LEAD_HOUR]h" ;;
            esac

            BODY="$(cat <<EOF
HRRRCast forecast run ${STATUS}

Cycle            @[INIT_TIME]  (+@[LEAD_HOUR] h, @[N_ENSEMBLES] member)
Instance         ${INSTANCE_ID}  ${ITYPE}  ${REGION}
Started          ${STARTED_AT}
Ended            ${ENDED}
Wall clock       ${ELAPSED_MIN} min

Outputs          ${OUT_LINE:-none}
                 ${OUT_PREFIX}/
NetCDF settings  complevel=@[NC_COMPLEVEL], least_significant_digit=@[NC_LSD] (GRIB2 off, the default)
GFS cycle lag    @[GFS_MIN_LAG] h (0 = newest cycle)
Domain           @[SUB_DESC]

make_bcs         finished ${BCS_T:-n/a}, ${WORKERS:-?} worker process(es)
forecast         ${PREDICTS:-did not run}

Code             @[CODE_REF]
Logs             @[S3_LOGS]/${INSTANCE_ID}-bootstrap.log
                 @[S3_LOGS]/${INSTANCE_ID}-stages/
Env installed    @[S3_LOGS]/${INSTANCE_ID}-stages/conda-installed.txt

Plots are a separate job:
  aws/run_plots.sh --s3-input @[S3_OUTPUT] --init-time @[INIT_TIME] \\
      --lead-hours @[LEAD_HOUR] --members 0 --variables surface
EOF
)"
            aws sns publish --region "$REGION" --topic-arn "@[SNS_TOPIC]" \
                --subject "$(printf '%.99s' "$SUBJ")" --message "$BODY" >/dev/null \
                && echo "Notification sent to @[SNS_TOPIC]" \
                || echo "WARNING: SNS publish failed; run outcome is unaffected"
        } || echo "WARNING: could not build the notification; run outcome is unaffected"
    fi

    if [ "$TERMINATE" == "YES" ]; then
        echo "Terminating ${INSTANCE_ID}"
        aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION" || shutdown -h now
    else
        echo "TERMINATE=NO; leaving the instance running for inspection. Remember to stop it."
    fi
}
trap finish EXIT

fail() { STATUS="failed: $*"; echo "ERROR: $*" >&2; exit 1; }

# --- get the code ----------------------------------------------------------
# A tarball of the operator's working tree, staged to S3 by run_on_ec2.sh, rather
# than a git clone. Three reasons: the instance needs no GitHub credentials for a
# private repo, it runs exactly the tree that was tested (a clone of HEAD would
# silently omit uncommitted changes), and it sidesteps Git LFS entirely.
# Provenance is recorded below rather than implied by a ref.
echo "code provenance: @[CODE_REF]"
mkdir -p "$WORKDIR"
chown "${RUN_USER}:${RUN_USER}" "$WORKDIR"
aws s3 cp "@[CODE_S3]" /tmp/hrrrcast-code.tar.gz || fail "could not fetch code from @[CODE_S3]"
tar -xzf /tmp/hrrrcast-code.tar.gz -C "$WORKDIR" || fail "could not extract code tarball"
chown -R "${RUN_USER}:${RUN_USER}" "$WORKDIR"
rm -f /tmp/hrrrcast-code.tar.gz

# --- provision the env and stage the model ---------------------------------
# setup_gpu.sh is fatal on any failure and verifies GPU TF plus a model load.
# Note: the env assignment goes inside the shell, not between sudo and the
# command; `sudo -u user VAR=val cmd` is a syntax error, not an assignment.
sudo -u "$RUN_USER" -H bash -lc "
    set -euo pipefail
    cd '$WORKDIR'
    export MODEL_S3='@[MODEL_S3]'
    # Skip the 3.5-minute TF-import-and-load-model check: fcst.py repeats it
    # verbatim minutes later and fails loudly, so it is duplicated cost here.
    export VERIFY_GPU=NO
    ./aws/setup_gpu.sh
" || fail "setup_gpu.sh failed"

# --- size the make_bcs worker pool to this instance -------------------------
# A hardcoded worker count is wrong on every instance except the one it was tuned
# on, and getting it wrong costs either wall-clock (too few) or an OOM-killed
# worker (too many). Derive it from actual RAM instead.
#
# Two consumers, both measured on a 24-lead-hour cycle:
#   - each pool worker holds the GFS fields plus the regridded HRRR-grid arrays.
#     MEASURED peak 1.89 GB (96 samples over a full make_bcs, top five 1.70-1.89),
#     so 4 GB is budgeted: better than 2x headroom. The earlier 15 GB figure was
#     wrong -- it came from sampling the process tree before make_bcs stored float32
#     and before it stopped building the raw arrays that every caller discarded, so
#     it conflated worker memory with the parent's float64 accumulation.
#   - the PARENT accumulates the normalized arrays for every lead hour and holds
#     them until the npz is written. It holds two copies at peak (the accumulated
#     pressure+surface arrays, and the assembled model_input), which at 42 channels
#     on a 1059x1799 grid in float32 is 2 x 320 MB = ~0.64 GB per lead hour, plus
#     roughly 2 GB of interpreter and library overhead. This is what actually caps
#     the pool, and it was missed when the cap was first set.
# Peak is therefore parent + workers x 4 GB. Measured parent peak was 14.58 GB at
# 24 lead hours (it was 29.5 GB before the float32 change), against ~17.6 GB from
# this arithmetic, so the parent estimate is also conservative.
#
# With workers this light, the binding constraint is cores rather than memory, so
# the pool is capped one below nproc: the parent stays responsive, and it leaves a
# core free if the forecast's model load is ever overlapped with this stage.
# Set MAKE_BCS_WORKERS explicitly to override.
MAKE_BCS_WORKERS="@[MAKE_BCS_WORKERS]"
if [ "$MAKE_BCS_WORKERS" == "auto" ]; then
    TOTAL_GB="$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo)"
    NPROC="$(nproc)"
    LEADS="@[LEAD_HOUR]"
    PER_WORKER_GB=4
    PARENT_GB=$(( 2 + (LEADS * 65 + 99) / 100 ))
    W=$(( (TOTAL_GB - PARENT_GB) / PER_WORKER_GB ))
    CORE_CAP=$(( NPROC > 1 ? NPROC - 1 : 1 ))
    [ "$W" -lt 1 ] && W=1
    [ "$W" -gt "$CORE_CAP" ] && W="$CORE_CAP"
    [ "$W" -gt "$LEADS" ] && W="$LEADS"
    MAKE_BCS_WORKERS="$W"
    echo "make_bcs workers: ${MAKE_BCS_WORKERS} (auto: ${TOTAL_GB} GB RAM, ${NPROC} cores," \
         "${LEADS} lead hours, ~${PARENT_GB} GB parent + ${MAKE_BCS_WORKERS}x${PER_WORKER_GB} GB workers," \
         "core cap ${CORE_CAP})"
else
    echo "make_bcs workers: ${MAKE_BCS_WORKERS} (explicit)"
fi

# --- run the cycle ---------------------------------------------------------
# RUNPLOT=NO: plotting is a separate job (aws/run_plots.sh) reading the NetCDF
# back from S3. It is CPU work and does not belong on GPU-instance time.
# etc/env_mac.sh requires conda on PATH, and setup_gpu.sh installs Miniforge
# without running `conda init`, so a fresh login shell does not have it. Put conda
# on PATH here explicitly rather than relying on shell rc files.
sudo -u "$RUN_USER" -H bash -lc "
    set -euo pipefail
    cd '$WORKDIR'
    if [ -f \"\$HOME/miniforge3/etc/profile.d/conda.sh\" ]; then
        source \"\$HOME/miniforge3/etc/profile.d/conda.sh\"
    fi
    command -v conda >/dev/null 2>&1 || { echo 'conda still not on PATH' >&2; exit 1; }
    source etc/env_mac.sh
    # Redundant now that GRIB2 is off by default, kept explicit so an AWS run's
    # output volume cannot change from under it if the default ever moves.
    export NO_GRIB2=YES
    export NC_COMPLEVEL='@[NC_COMPLEVEL]'
    export S3_OUTPUT='@[S3_OUTPUT]'
    export PURGE_LOCAL=YES
    export MAKE_BCS_WORKERS='${MAKE_BCS_WORKERS}'
    # Start the forecast before the input stages so its TF import and model load
    # (~5 min, independent of the input data) overlap with input preparation. The
    # GPU is otherwise idle for that whole phase.
    export OVERLAP_FCST=YES
    export NC_LSD='@[NC_LSD]'
    export GFS_MIN_LAG='@[GFS_MIN_LAG]'
    # Subdomain cropping. Empty SUB_BBOX = full domain, which is the default and
    # what every run before this did. run_cycle.sh does the crop between the input
    # stages and the forecast; see docs/subdomain.md.
    export SUB_BBOX='@[SUB_BBOX]'
    export SUB_HALO='@[SUB_HALO]'
    export DATAROOT='$WORKDIR'
    # RUN_CMD is normally the run_cycle.sh line below. run_on_ec2.sh --run-cmd
    # replaces it so an experiment can reuse this whole bootstrap (code fetch, conda
    # env, logging, log shipping, self-termination) instead of duplicating it. Used by
    # aws/run_domain_test.sh. PURGE_LOCAL is deliberately left YES: any replacement
    # command still streams to S3 and must not fill the root volume.
    @[RUN_CMD]
" || fail "the run command failed"

STATUS="success"
echo "Outputs delivered to @[S3_OUTPUT]"
