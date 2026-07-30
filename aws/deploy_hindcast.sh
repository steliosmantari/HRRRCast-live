#!/usr/bin/env bash
#
# aws/deploy_hindcast.sh — deploy (or update, or resume) a serverless HRRRCast
# hindcast: EventBridge Scheduler -> Lambda (aws/lambda/hindcast_handler.py) ->
# one on-demand GPU instance per cycle, one cycle at a time.
#
#   EventBridge Scheduler (rate, e.g. every 10 min)
#        -> Lambda  aws/lambda/hindcast_handler.py
#             -> reads s3://BUCKET/hrrrcast/hindcast/<run-id>/{config.json,scheduler/*}
#             -> lists   s3://BUCKET/hrrrcast/hindcast/<run-id>/status/  (done cycles)
#             -> RunInstances for the earliest cycle not yet done (if none in flight)
#
# Each instance runs aws/run_hindcast_cycle.sh, which is run_cycle.sh plus one
# S3 status marker per cycle -- the ONLY state this whole pipeline trusts. That
# marker is what makes it resumable: re-running this script with the same
# --run-id, at any point, for any reason, picks up from whatever is already
# marked done. Nothing here needs "continue" as a separate mode.
#
# THE SCHEDULE IS CREATED DISABLED. Nothing launches until you pass --enable.
# The intended sequence is:
#
#   1. aws/deploy_hindcast.sh --bucket B --run-id R --start ... --end ... --bbox ...
#        (stages code, writes config.json, creates the function+schedule, DISABLED)
#   2. aws/deploy_hindcast.sh --bucket B --run-id R --invoke-once
#        (one manual tick; check the result and CloudWatch logs)
#   3. aws/deploy_hindcast.sh --bucket B --run-id R --enable
#        (now it runs on its own, one cycle per tick, until the range is done)
#
# Re-running step 1 with the same --run-id is always safe: it re-stages the
# code/template and rewrites config.json, but never touches status/ markers, so
# it is exactly how you both roll out a code change AND resume an interrupted
# hindcast.
#
# Usage:
#   aws/deploy_hindcast.sh --bucket NAME --run-id ID --bbox N,W,S,E \
#       --start YYYY-MM-DD --end YYYY-MM-DD [options]
#
# Required (on first deploy; omit --start/--end/--bbox on a later
# --invoke-once/--enable/--disable/--delete against an existing run-id):
#   --bucket NAME          S3 bucket for outputs, logs and hindcast state
#   --run-id ID            identifies this hindcast's S3 prefix, Lambda,
#                          schedule and EC2 tag (e.g. "socal-june2026")
#   --bbox N,W,S,E         region to crop the forecast to
#   --start YYYY-MM-DD     first cycle date (UTC), inclusive
#   --end YYYY-MM-DD       last cycle date (UTC), inclusive
#
# Options (defaults in brackets):
#   --init-hours LIST      comma list of UTC init hours              [00,06,12,18]
#   --lead-hours N         forecast length per cycle                 [24]
#   --output-hours SPEC    only keep these lead hours ("start:step:end" or a
#                          comma list, e.g. "0:3:24")                 [all hours]
#   --halo N               halo cells around --bbox                  [40]
#   --gfs-min-lag N        GFS cycle lag; 0 is correct for a hindcast [0]
#   --instance-type TYPE   EC2 instance type per cycle          [g6e.2xlarge]
#   --rate EXPR            EventBridge rate/cron expression  [rate(10 minutes)]
#   --notify-topic ARN     SNS topic: per-cycle failure emails (via the
#                          instance's own bootstrap) AND the one hindcast-
#                          complete summary from the Lambda [$HRRRCAST_SNS_TOPIC]
#   --enable               enable the schedule (default: created/left DISABLED)
#   --disable              disable the schedule and exit
#   --live                 actually launch (default: create/update, dry-run
#                          Lambda decisions only, same convention as
#                          deploy_scheduler.sh)
#   --invoke-once          invoke the Lambda once now and print the result
#   --delete               remove the schedule and function (NOT the S3 state
#                          or the shared IAM role -- other things may use it)
#   --skip-role            do not touch IAM
#   --region REGION        AWS region                                [us-east-1]
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUCKET=""
RUN_ID=""
BBOX=""
START_DATE=""
END_DATE=""
INIT_HOURS="00,06,12,18"
LEAD_HOUR=24
OUTPUT_HOURS=""
HALO=40
GFS_MIN_LAG_OPT=0
INSTANCE_TYPE="g6e.2xlarge"
RATE="rate(10 minutes)"
SNS_TOPIC="${HRRRCAST_SNS_TOPIC:-}"
STATE=""
LIVE="NO"
INVOKE_ONCE="NO"
DELETE="NO"
SKIP_ROLE="NO"
REGION="us-east-1"
ROLE="hrrrcast-launcher"          # shared with the hourly scheduler, by design
TIMEOUT=60
MEMORY=256
RUNTIME="python3.12"

log() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --bucket)        BUCKET="$2"; shift 2 ;;
        --run-id)        RUN_ID="$2"; shift 2 ;;
        --bbox)          BBOX="$2"; shift 2 ;;
        --start)         START_DATE="$2"; shift 2 ;;
        --end)           END_DATE="$2"; shift 2 ;;
        --init-hours)    INIT_HOURS="$2"; shift 2 ;;
        --lead-hours)    LEAD_HOUR="$2"; shift 2 ;;
        --output-hours)  OUTPUT_HOURS="$2"; shift 2 ;;
        --halo)          HALO="$2"; shift 2 ;;
        --gfs-min-lag)   GFS_MIN_LAG_OPT="$2"; shift 2 ;;
        --instance-type) INSTANCE_TYPE="$2"; shift 2 ;;
        --rate)          RATE="$2"; shift 2 ;;
        --notify-topic)  SNS_TOPIC="$2"; shift 2 ;;
        --enable)        STATE="ENABLED"; shift ;;
        --disable)       STATE="DISABLED"; shift ;;
        --live)          LIVE="YES"; shift ;;
        --invoke-once)   INVOKE_ONCE="YES"; shift ;;
        --delete)        DELETE="YES"; shift ;;
        --skip-role)     SKIP_ROLE="YES"; shift ;;
        --region)        REGION="$2"; shift 2 ;;
        -h|--help)       sed -n '2,58p' "${BASH_SOURCE[0]}" | sed 's/^#\{0,1\} \{0,1\}//'; exit 0 ;;
        *) die "Unknown option: $1 (try --help)" ;;
    esac
done

[ -n "$BUCKET" ] || die "--bucket is required"
[ -n "$RUN_ID" ] || die "--run-id is required"
[[ "$RUN_ID" =~ ^[a-z0-9-]+$ ]] || die "--run-id must be lowercase letters, digits, hyphens (used in a Lambda/schedule name and an S3 prefix)"
command -v aws >/dev/null 2>&1 || die "aws CLI not found."
export AWS_DEFAULT_REGION="$REGION"
export AWS_PAGER=""

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)" || die "no valid AWS credentials"
FUNCTION="hrrrcast-hindcast-${RUN_ID}"
SCHEDULE="hrrrcast-hindcast-${RUN_ID}"
ROLE_ARN="arn:aws:iam::${ACCOUNT}:role/${ROLE}"
FUNC_ARN="arn:aws:lambda:${REGION}:${ACCOUNT}:function:${FUNCTION}"
HINDCAST_PREFIX="hrrrcast/hindcast/${RUN_ID}"
STAGE_PREFIX="${HINDCAST_PREFIX}/scheduler"

# --- delete ------------------------------------------------------------------
# Deliberately does NOT touch the shared IAM role (the hourly scheduler may use
# it too) or any S3 state (status markers and outputs are the hindcast's actual
# record; deleting the launcher is not the same decision as deleting that).
if [ "$DELETE" == "YES" ]; then
    log "Deleting schedule and function for run-id '${RUN_ID}' (S3 state and the shared IAM role are left alone)"
    aws scheduler delete-schedule --name "$SCHEDULE" --region "$REGION" 2>/dev/null \
        && echo "  schedule deleted" || echo "  schedule absent"
    aws lambda delete-function --function-name "$FUNCTION" --region "$REGION" 2>/dev/null \
        && echo "  function deleted" || echo "  function absent"
    exit 0
fi

# --- first-deploy / redeploy requires the full config ------------------------
NEEDS_FULL_CONFIG="NO"
aws lambda get-function --function-name "$FUNCTION" --region "$REGION" >/dev/null 2>&1 || NEEDS_FULL_CONFIG="YES"
if [ "$NEEDS_FULL_CONFIG" == "YES" ] || [ -n "$START_DATE$END_DATE$BBOX" ]; then
    [ -n "$BBOX" ]       || die "--bbox N,W,S,E is required"
    [ -n "$START_DATE" ] || die "--start YYYY-MM-DD is required"
    [ -n "$END_DATE" ]   || die "--end YYYY-MM-DD is required"
fi

# --- validate and enumerate cycles -------------------------------------------
if [ -n "$START_DATE$END_DATE" ]; then
    [[ "$START_DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || die "--start must be YYYY-MM-DD"
    [[ "$END_DATE"   =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || die "--end must be YYYY-MM-DD"
    N_CYCLES="$(python3 -c '
import sys, datetime as d
s = d.datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
e = d.datetime.strptime(sys.argv[2], "%Y-%m-%d").date()
hours = sys.argv[3].split(",")
if e < s:
    sys.exit("--end is before --start")
for h in hours:
    if not (len(h) == 2 and h.isdigit() and 0 <= int(h) <= 23):
        sys.exit(f"invalid --init-hours entry {h!r}")
print((e - s).days + 1)
' "$START_DATE" "$END_DATE" "$INIT_HOURS")" || die "$N_CYCLES"
    IFS=',' read -ra _H <<< "$INIT_HOURS"
    N_CYCLES=$(( N_CYCLES * ${#_H[@]} ))
fi
if [ -n "$BBOX" ]; then
    python3 - "$BBOX" <<'PY' || die "invalid --bbox (see above)"
import sys
parts = sys.argv[1].split(",")
if len(parts) != 4:
    sys.exit(f"--bbox must be 'N,W,S,E', got {sys.argv[1]!r}")
n, w, s, e = (float(p) for p in parts)
if not (s < n and w < e):
    sys.exit(f"--bbox 'N,W,S,E'={sys.argv[1]!r} is not a valid box (need S<N and W<E)")
PY
fi

# --- stage code + template (only meaningful on a config-bearing deploy) -----
if [ -n "$START_DATE$END_DATE$BBOX" ]; then
    FIRST_HOUR="${INIT_HOURS%%,*}"
    STAGE_INIT_TIME="${START_DATE}T${FIRST_HOUR}"   # cosmetic placeholder; see run_on_ec2.sh's
                                                      # --stage-scheduler sentinel substitution
    STATUS_S3="s3://${BUCKET}/${HINDCAST_PREFIX}/status"
    RUN_CMD="./aws/run_hindcast_cycle.sh '__INIT_TIME__' '${LEAD_HOUR}' '${OUTPUT_HOURS}' '${STATUS_S3}'"
    # Isolated from the shared s3://BUCKET/hrrrcast/out/ default: that prefix is
    # keyed only by init time, not domain, so two experiments sharing a cycle
    # (a real thing that happened during testing) silently collide there. Every
    # hindcast gets its own out/ under its own run-id prefix instead.
    HINDCAST_S3_OUTPUT="s3://${BUCKET}/${HINDCAST_PREFIX}/out"

    log "Staging code + user-data template to s3://${BUCKET}/${STAGE_PREFIX}/"
    log "Output prefix: ${HINDCAST_S3_OUTPUT}/ (isolated per run-id, not the shared hrrrcast/out/)"
    "${REPO_DIR}/aws/run_on_ec2.sh" \
        --bucket "$BUCKET" \
        --init-time "$STAGE_INIT_TIME" \
        --lead-hours "$LEAD_HOUR" \
        --members 1 \
        --bbox "$BBOX" \
        --halo "$HALO" \
        --gfs-min-lag "$GFS_MIN_LAG_OPT" \
        --instance-type "$INSTANCE_TYPE" \
        --region "$REGION" \
        --s3-output "$HINDCAST_S3_OUTPUT" \
        ${SNS_TOPIC:+--notify-topic "$SNS_TOPIC"} \
        --run-cmd "$RUN_CMD" \
        --stage-scheduler --stage-prefix "$STAGE_PREFIX" \
        || die "staging failed"

    log "Writing config to s3://${BUCKET}/${HINDCAST_PREFIX}/config.json"
    CONFIG_FILE="$(mktemp)"
    python3 -c '
import json, sys
cfg = dict(
    run_id=sys.argv[1], start_date=sys.argv[2], end_date=sys.argv[3],
    init_hours=sys.argv[4].split(","), lead_hours=int(sys.argv[5]),
    output_hours=sys.argv[6], bbox=sys.argv[7], halo=int(sys.argv[8]),
    gfs_min_lag=int(sys.argv[9]), n_cycles=int(sys.argv[10]),
)
json.dump(cfg, open(sys.argv[11], "w"), indent=2)
' "$RUN_ID" "$START_DATE" "$END_DATE" "$INIT_HOURS" "$LEAD_HOUR" "$OUTPUT_HOURS" \
    "$BBOX" "$HALO" "$GFS_MIN_LAG_OPT" "${N_CYCLES:-0}" "$CONFIG_FILE"
    aws s3 cp "$CONFIG_FILE" "s3://${BUCKET}/${HINDCAST_PREFIX}/config.json" --quiet \
        || die "config upload failed"
    rm -f "$CONFIG_FILE"
fi

# --- IAM role: extend the shared launcher role -------------------------------
if [ "$SKIP_ROLE" == "NO" ]; then
    log "IAM role ${ROLE} (shared with the hourly scheduler)"
    if aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
        echo "  exists; updating inline policy"
    else
        aws iam create-role --role-name "$ROLE" \
            --assume-role-policy-document "file://${REPO_DIR}/aws/iam/lambda-launcher-trust.json" \
            --description "HRRRCast launcher (hourly scheduler + hindcast drivers)" \
            >/dev/null || die "create-role failed"
        echo "  created"
    fi
    aws iam put-role-policy --role-name "$ROLE" --policy-name hrrrcast-launcher \
        --policy-document "file://${REPO_DIR}/aws/iam/lambda-launcher-policy.json" \
        || die "put-role-policy failed (check the bucket/account ARNs in the policy)"
    aws iam attach-role-policy --role-name "$ROLE" \
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole \
        2>/dev/null || true
    echo "  policies applied"
fi

# --- Lambda function ----------------------------------------------------------
# update-function-configuration REPLACES the whole environment, not merges. A
# bare --enable/--disable/--invoke-once (no --notify-topic repeated) must not
# silently wipe out a topic set on an earlier deploy.
if [ -z "$SNS_TOPIC" ]; then
    EXISTING_TOPIC="$(aws lambda get-function-configuration --function-name "$FUNCTION" \
        --region "$REGION" --query 'Environment.Variables.HRRRCAST_SNS_TOPIC' \
        --output text 2>/dev/null || true)"
    [ -n "$EXISTING_TOPIC" ] && [ "$EXISTING_TOPIC" != "None" ] && SNS_TOPIC="$EXISTING_TOPIC"
fi

log "Building deployment package"
ZIP="$(mktemp -d)/function.zip"
STAGE="$(mktemp -d)"
cp "${REPO_DIR}/aws/lambda/hindcast_handler.py" "$STAGE/"
python3 -c "import py_compile; py_compile.compile('$STAGE/hindcast_handler.py', doraise=True); print('  compiles')" \
    || die "hindcast_handler.py does not compile"
( cd "$STAGE" && zip -q -r "$ZIP" hindcast_handler.py )
echo "  $(du -h "$ZIP" | cut -f1) $ZIP"

ENV_VARS="Variables={HRRRCAST_BUCKET=${BUCKET},HRRRCAST_RUN_ID=${RUN_ID},HRRRCAST_STAGE_PREFIX=${STAGE_PREFIX},HRRRCAST_SCHEDULE_NAME=${SCHEDULE},HRRRCAST_SNS_TOPIC=${SNS_TOPIC},HRRRCAST_DRY_RUN=$([ "$LIVE" == "YES" ] && echo NO || echo YES)}"
log "Lambda ${FUNCTION} (dry_run=$([ "$LIVE" == "YES" ] && echo NO || echo YES))"
if aws lambda get-function --function-name "$FUNCTION" --region "$REGION" >/dev/null 2>&1; then
    aws lambda update-function-code --function-name "$FUNCTION" --region "$REGION" \
        --zip-file "fileb://${ZIP}" >/dev/null || die "update-function-code failed"
    aws lambda wait function-updated --function-name "$FUNCTION" --region "$REGION"
    aws lambda update-function-configuration --function-name "$FUNCTION" --region "$REGION" \
        --timeout "$TIMEOUT" --memory-size "$MEMORY" --handler hindcast_handler.lambda_handler \
        --environment "$ENV_VARS" >/dev/null || die "update-function-configuration failed"
    aws lambda wait function-updated --function-name "$FUNCTION" --region "$REGION"
    echo "  updated"
else
    for attempt in 1 2 3 4 5 6; do
        if aws lambda create-function --function-name "$FUNCTION" --region "$REGION" \
            --runtime "$RUNTIME" --role "$ROLE_ARN" --handler hindcast_handler.lambda_handler \
            --zip-file "fileb://${ZIP}" --timeout "$TIMEOUT" --memory-size "$MEMORY" \
            --environment "$ENV_VARS" \
            --description "HRRRCast hindcast driver (${RUN_ID})" >/dev/null 2>&1; then
            echo "  created"; break
        fi
        [ "$attempt" -eq 6 ] && die "create-function failed (role propagation? check IAM)"
        echo "  waiting for role propagation (attempt ${attempt})"; sleep 10
    done
    aws lambda wait function-active --function-name "$FUNCTION" --region "$REGION"
fi

# --- EventBridge Scheduler ----------------------------------------------------
log "EventBridge schedule ${SCHEDULE}"
aws iam put-role-policy --role-name "$ROLE" --policy-name "hrrrcast-invoke-${FUNCTION}" \
    --policy-document "{
      \"Version\": \"2012-10-17\",
      \"Statement\": [{
        \"Sid\": \"InvokeTheHindcastDriver\",
        \"Effect\": \"Allow\",
        \"Action\": \"lambda:InvokeFunction\",
        \"Resource\": \"${FUNC_ARN}\"
      }]
    }" >/dev/null 2>&1 || printf '  note: could not add invoke policy (--skip-role?)\n'

TARGET="{\"Arn\":\"${FUNC_ARN}\",\"RoleArn\":\"${ROLE_ARN}\",\"RetryPolicy\":{\"MaximumRetryAttempts\":0}}"
DESC="HRRRCast hindcast ${RUN_ID}: ${START_DATE:-<unchanged>}..${END_DATE:-<unchanged>} lead=${LEAD_HOUR}h"

if aws scheduler get-schedule --name "$SCHEDULE" --region "$REGION" >/dev/null 2>&1; then
    CUR_STATE="$(aws scheduler get-schedule --name "$SCHEDULE" --region "$REGION" --query State --output text)"
    NEW_STATE="${STATE:-$CUR_STATE}"
    aws scheduler update-schedule --name "$SCHEDULE" --region "$REGION" \
        --schedule-expression "$RATE" --schedule-expression-timezone UTC \
        --flexible-time-window '{"Mode":"OFF"}' \
        --target "$TARGET" --state "$NEW_STATE" --description "$DESC" >/dev/null \
        || die "update-schedule failed"
    echo "  updated, state ${CUR_STATE} -> ${NEW_STATE}"
else
    NEW_STATE="${STATE:-DISABLED}"
    for attempt in 1 2 3 4 5 6; do
        if aws scheduler create-schedule --name "$SCHEDULE" --region "$REGION" \
            --schedule-expression "$RATE" --schedule-expression-timezone UTC \
            --flexible-time-window '{"Mode":"OFF"}' \
            --target "$TARGET" --state "$NEW_STATE" --description "$DESC" \
            >/dev/null 2>&1; then
            echo "  created, state ${NEW_STATE}"; break
        fi
        if [ "$attempt" -eq 6 ]; then
            aws scheduler create-schedule --name "$SCHEDULE" --region "$REGION" \
                --schedule-expression "$RATE" --schedule-expression-timezone UTC \
                --flexible-time-window '{"Mode":"OFF"}' \
                --target "$TARGET" --state "$NEW_STATE" --description "$DESC" || true
            die "create-schedule failed after 6 attempts"
        fi
        echo "  waiting for role propagation (attempt ${attempt})"; sleep 10
    done
fi

# --- optional single invocation -----------------------------------------------
if [ "$INVOKE_ONCE" == "YES" ]; then
    log "Invoking once"
    OUT="$(mktemp)"
    aws lambda invoke --function-name "$FUNCTION" --region "$REGION" \
        --cli-binary-format raw-in-base64-out --payload '{}' "$OUT" >/dev/null \
        || die "invoke failed"
    echo "  result: $(cat "$OUT")"
    rm -f "$OUT"
    echo "  logs:   aws logs tail /aws/lambda/${FUNCTION} --region ${REGION} --since 5m"
fi

FINAL_STATE="$(aws scheduler get-schedule --name "$SCHEDULE" --region "$REGION" --query State --output text)"
cat <<EOF

  hindcast run-id  ${RUN_ID}
  cycles           ${N_CYCLES:-<unchanged>}
  schedule         ${SCHEDULE}   ${RATE}  UTC   state=${FINAL_STATE}
  function         ${FUNCTION}   dry_run=$([ "$LIVE" == "YES" ] && echo NO || echo YES)
  role             ${ROLE} (shared)
  state (S3)       s3://${BUCKET}/${HINDCAST_PREFIX}/status/
  outputs (S3)     s3://${BUCKET}/${HINDCAST_PREFIX}/out/<YYYYMMDD>/<HH>/  (isolated, not the shared hrrrcast/out/)

  logs             aws logs tail /aws/lambda/${FUNCTION} --region ${REGION} --follow
  invoke by hand    aws lambda invoke --function-name ${FUNCTION} --region ${REGION} \\
                      --payload '{}' /dev/stdout
  progress          aws s3 ls s3://${BUCKET}/${HINDCAST_PREFIX}/status/ | wc -l
  resume / update   re-run this exact command (status markers are untouched)
  enable            aws/deploy_hindcast.sh --bucket ${BUCKET} --run-id ${RUN_ID} --enable
  stop the schedule aws/deploy_hindcast.sh --bucket ${BUCKET} --run-id ${RUN_ID} --disable
  remove entirely   aws/deploy_hindcast.sh --bucket ${BUCKET} --run-id ${RUN_ID} --delete
EOF

if [ "$FINAL_STATE" == "ENABLED" ] && [ "$LIVE" == "YES" ]; then
    printf '\n\033[1;33mThis is now live: one instance launches whenever a tick finds none in flight and cycles remain.\033[0m\n'
fi
