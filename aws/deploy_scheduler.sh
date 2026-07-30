#!/usr/bin/env bash
#
# aws/deploy_scheduler.sh — create or update the hourly HRRRCast launcher:
# an EventBridge Scheduler rule that invokes a Lambda, which picks a cycle and
# launches one GPU instance.
#
#   EventBridge Scheduler (cron, hourly)
#        -> Lambda  aws/lambda/handler.py
#             -> reads s3://BUCKET/hrrrcast/scheduler/{launch-params.json,user_data.template}
#             -> RunInstances (one g6e.2xlarge, self-terminating)
#
# THE SCHEDULE IS CREATED DISABLED AND IN DRY-RUN MODE. That is deliberate. An
# hourly 24 h forecast costs roughly $875/month on demand, so nothing here starts
# spending money until you pass --enable, and even then the Lambda only logs its
# decisions until you pass --live. The intended sequence is:
#
#   1. aws/run_on_ec2.sh --bucket B --stage-scheduler     # pin code + template
#   2. aws/deploy_scheduler.sh --bucket B                 # create, disabled, dry-run
#   3. aws/deploy_scheduler.sh --bucket B --invoke-once   # one manual dry-run
#   4. aws/deploy_scheduler.sh --bucket B --enable        # hourly, still dry-run
#      ... read a day of CloudWatch logs, confirm the cycle choices ...
#   5. aws/deploy_scheduler.sh --bucket B --enable --live # actually launch
#
# Re-running is safe and is how you roll out changes: everything is create-or-update.
#
# Usage:
#   aws/deploy_scheduler.sh --bucket NAME [options]
#
# Options (defaults in brackets):
#   --bucket NAME        S3 bucket holding the staged artifacts        [required]
#   --region REGION      AWS region                                    [us-east-1]
#   --function NAME      Lambda function name              [hrrrcast-hourly-launcher]
#   --role NAME          Lambda execution role name        [hrrrcast-launcher]
#   --schedule NAME      EventBridge schedule name         [hrrrcast-hourly]
#   --cron EXPR          schedule expression         [cron(5 * * * ? *)] i.e. HH:05
#   --enable             enable the schedule (default: created/left DISABLED)
#   --disable            disable the schedule and exit
#   --live               Lambda actually launches instances (default: dry-run logs)
#   --lead-hours N       recorded in the schedule description only; the real value
#                        comes from launch-params.json                 [from params]
#   --invoke-once        invoke the Lambda once now and print the result
#   --delete             remove the schedule, function and role, then exit
#   --skip-role          do not touch IAM (use when the role already exists and you
#                        lack iam: permissions)
#
# Why HH:05 and not HH:00. The HRRR analysis for hour H lands at about H+0:51, so a
# tick at H+1:05 gives roughly 14 minutes of slack for a late analysis before the
# Lambda would have to fall back to an older hour. Firing exactly on the hour buys
# nothing and collides with every other cron on earth.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUCKET=""
REGION="us-east-1"
FUNCTION="hrrrcast-hourly-launcher"
ROLE="hrrrcast-launcher"
SCHEDULE="hrrrcast-hourly"
CRON="cron(5 * * * ? *)"
STATE=""            # empty = leave as-is on update, DISABLED on create
LIVE="NO"
INVOKE_ONCE="NO"
DELETE="NO"
SKIP_ROLE="NO"
# Lambda's own timeout must exceed the worst-case probe time. Measured: 25 files in
# about 4 s serial from a laptop; the handler probes in parallel with 16 workers, but
# a lookback of 6 hours times 3 GFS cycles is up to 18 manifests. 120 s is generous.
TIMEOUT=120
MEMORY=256
RUNTIME="python3.12"

log() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --bucket)      BUCKET="$2"; shift 2 ;;
        --region)      REGION="$2"; shift 2 ;;
        --function)    FUNCTION="$2"; shift 2 ;;
        --role)        ROLE="$2"; shift 2 ;;
        --schedule)    SCHEDULE="$2"; shift 2 ;;
        --cron)        CRON="$2"; shift 2 ;;
        --enable)      STATE="ENABLED"; shift ;;
        --disable)     STATE="DISABLED"; shift ;;
        --live)        LIVE="YES"; shift ;;
        --invoke-once) INVOKE_ONCE="YES"; shift ;;
        --delete)      DELETE="YES"; shift ;;
        --skip-role)   SKIP_ROLE="YES"; shift ;;
        --lead-hours)  shift 2 ;;   # accepted for symmetry; params.json is authoritative
        -h|--help)     sed -n '2,50p' "${BASH_SOURCE[0]}" | sed 's/^#\{0,1\} \{0,1\}//'; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

[ -n "$BUCKET" ] || die "--bucket is required"
command -v aws >/dev/null 2>&1 || die "aws CLI not found"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)" \
    || die "no valid AWS credentials"
ROLE_ARN="arn:aws:iam::${ACCOUNT}:role/${ROLE}"
FUNC_ARN="arn:aws:lambda:${REGION}:${ACCOUNT}:function:${FUNCTION}"

# --- delete ----------------------------------------------------------------
if [ "$DELETE" == "YES" ]; then
    log "Deleting schedule, function and role"
    aws scheduler delete-schedule --name "$SCHEDULE" --region "$REGION" 2>/dev/null \
        && echo "  schedule deleted" || echo "  schedule absent"
    aws lambda delete-function --function-name "$FUNCTION" --region "$REGION" 2>/dev/null \
        && echo "  function deleted" || echo "  function absent"
    if [ "$SKIP_ROLE" == "NO" ]; then
        aws iam delete-role-policy --role-name "$ROLE" --policy-name hrrrcast-launcher 2>/dev/null || true
        aws iam detach-role-policy --role-name "$ROLE" \
            --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole 2>/dev/null || true
        aws iam delete-role --role-name "$ROLE" 2>/dev/null \
            && echo "  role deleted" || echo "  role absent"
    fi
    exit 0
fi

# --- preflight: the staged artifacts must exist ----------------------------
# Creating a schedule that points at a Lambda that will fail on its first
# invocation is worse than refusing, because the failure only shows up in
# CloudWatch an hour later.
log "Checking staged artifacts"
for key in hrrrcast/scheduler/launch-params.json hrrrcast/scheduler/user_data.template; do
    aws s3api head-object --bucket "$BUCKET" --key "$key" >/dev/null 2>&1 \
        || die "s3://${BUCKET}/${key} is missing.
    Run first:  aws/run_on_ec2.sh --bucket ${BUCKET} --stage-scheduler"
    echo "  ok  s3://${BUCKET}/${key}"
done
PARAMS_JSON="$(aws s3 cp "s3://${BUCKET}/hrrrcast/scheduler/launch-params.json" - 2>/dev/null)"
STAGED_LEAD="$(printf '%s' "$PARAMS_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["lead_hours"])')"
STAGED_LAG="$(printf '%s' "$PARAMS_JSON"  | python3 -c 'import json,sys; print(json.load(sys.stdin)["gfs_min_lag"])')"
STAGED_REF="$(printf '%s' "$PARAMS_JSON"  | python3 -c 'import json,sys; print(json.load(sys.stdin)["code_ref"])')"
echo "  staged: lead=${STAGED_LEAD}h gfs_min_lag=${STAGED_LAG}h code=${STAGED_REF}"

if [ "$STAGED_LAG" == "0" ]; then
    printf '\n\033[1;33mWARNING:\033[0m the staged gfs_min_lag is 0, which cannot launch 12 of 24 hours
    (GFS needs ~4 h to publish; the HRRR analysis lands in ~51 min). Re-stage with
    GFS_MIN_LAG=4 for hourly operation:
      GFS_MIN_LAG=4 aws/run_on_ec2.sh --bucket %s --stage-scheduler\n' "$BUCKET"
fi

# --- IAM role --------------------------------------------------------------
if [ "$SKIP_ROLE" == "NO" ]; then
    log "IAM role ${ROLE}"
    if aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
        echo "  exists; updating inline policy"
    else
        aws iam create-role --role-name "$ROLE" \
            --assume-role-policy-document "file://${REPO_DIR}/aws/iam/lambda-launcher-trust.json" \
            --description "HRRRCast hourly launcher (EventBridge -> Lambda -> RunInstances)" \
            >/dev/null || die "create-role failed"
        echo "  created"
    fi
    aws iam put-role-policy --role-name "$ROLE" --policy-name hrrrcast-launcher \
        --policy-document "file://${REPO_DIR}/aws/iam/lambda-launcher-policy.json" \
        || die "put-role-policy failed (check the bucket/account ARNs in the policy)"
    aws iam attach-role-policy --role-name "$ROLE" \
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole \
        || die "attach AWSLambdaBasicExecutionRole failed"
    echo "  policies applied"
fi

# --- build the deployment package ------------------------------------------
# Two files only: the handler and the shared cycle-selection module. boto3 is
# already present in the Lambda runtime, and gfs_cycle.py imports nothing outside
# the standard library, which is exactly why it exists as a separate module.
log "Building deployment package"
ZIP="$(mktemp -d)/function.zip"
STAGE="$(mktemp -d)"
cp "${REPO_DIR}/aws/lambda/handler.py" "$STAGE/"
cp "${REPO_DIR}/src/gfs_cycle.py" "$STAGE/"
python3 -c "
import py_compile, sys
for f in ('handler.py', 'gfs_cycle.py'):
    py_compile.compile('$STAGE/' + f, doraise=True)
print('  both modules compile')
" || die "the Lambda sources do not compile"
# Reject a gfs_cycle.py that has grown a third-party import, which would fail at
# Lambda cold start rather than here.
BAD="$(grep -E '^\s*(import|from) ' "$STAGE/gfs_cycle.py" \
        | grep -vE '(datetime|typing)' || true)"
[ -z "$BAD" ] || die "gfs_cycle.py gained non-stdlib imports, which Lambda cannot resolve:
${BAD}"
( cd "$STAGE" && zip -q -r "$ZIP" handler.py gfs_cycle.py )
echo "  $(du -h "$ZIP" | cut -f1) $ZIP"

# --- Lambda function -------------------------------------------------------
ENV_VARS="Variables={HRRRCAST_BUCKET=${BUCKET},HRRRCAST_DRY_RUN=$([ "$LIVE" == "YES" ] && echo NO || echo YES)}"
log "Lambda ${FUNCTION} (dry_run=$([ "$LIVE" == "YES" ] && echo NO || echo YES))"
if aws lambda get-function --function-name "$FUNCTION" --region "$REGION" >/dev/null 2>&1; then
    aws lambda update-function-code --function-name "$FUNCTION" --region "$REGION" \
        --zip-file "fileb://${ZIP}" >/dev/null || die "update-function-code failed"
    aws lambda wait function-updated --function-name "$FUNCTION" --region "$REGION"
    aws lambda update-function-configuration --function-name "$FUNCTION" --region "$REGION" \
        --timeout "$TIMEOUT" --memory-size "$MEMORY" --environment "$ENV_VARS" >/dev/null \
        || die "update-function-configuration failed"
    aws lambda wait function-updated --function-name "$FUNCTION" --region "$REGION"
    echo "  updated"
else
    # A freshly created role is not immediately usable by Lambda; retry briefly.
    for attempt in 1 2 3 4 5 6; do
        if aws lambda create-function --function-name "$FUNCTION" --region "$REGION" \
            --runtime "$RUNTIME" --role "$ROLE_ARN" --handler handler.lambda_handler \
            --zip-file "fileb://${ZIP}" --timeout "$TIMEOUT" --memory-size "$MEMORY" \
            --environment "$ENV_VARS" \
            --description "HRRRCast hourly launcher" >/dev/null 2>&1; then
            echo "  created"; break
        fi
        [ "$attempt" -eq 6 ] && die "create-function failed (role propagation? check IAM)"
        echo "  waiting for role propagation (attempt ${attempt})"; sleep 10
    done
    aws lambda wait function-active --function-name "$FUNCTION" --region "$REGION"
fi

# --- EventBridge Scheduler -------------------------------------------------
# The schedule assumes the SAME role as the Lambda for simplicity; it needs only
# lambda:InvokeFunction, granted below via the role's inline policy addition.
log "EventBridge schedule ${SCHEDULE}"
aws iam put-role-policy --role-name "$ROLE" --policy-name hrrrcast-invoke-launcher \
    --policy-document "{
      \"Version\": \"2012-10-17\",
      \"Statement\": [{
        \"Sid\": \"InvokeTheLauncher\",
        \"Effect\": \"Allow\",
        \"Action\": \"lambda:InvokeFunction\",
        \"Resource\": \"${FUNC_ARN}\"
      }]
    }" >/dev/null 2>&1 || printf '  note: could not add invoke policy (--skip-role?)\n'

# Re-apply the trust policy from the file rather than an inline document. Both
# lambda.amazonaws.com and scheduler.amazonaws.com live in
# aws/iam/lambda-launcher-trust.json, so create-role already sets both; this call
# only matters for a role created by an older version of this script. Errors are
# reported, not swallowed: an earlier version sent them to /dev/null, and when
# create-schedule then failed with "must allow AWS EventBridge Scheduler to assume
# the role" there was nothing in the output to say whether the trust update had run.
if [ "$SKIP_ROLE" == "NO" ]; then
    aws iam update-assume-role-policy --role-name "$ROLE" \
        --policy-document "file://${REPO_DIR}/aws/iam/lambda-launcher-trust.json" \
        || die "could not update the role trust policy"
fi

TARGET="{\"Arn\":\"${FUNC_ARN}\",\"RoleArn\":\"${ROLE_ARN}\",\"RetryPolicy\":{\"MaximumRetryAttempts\":0}}"
DESC="HRRRCast hourly: lead=${STAGED_LEAD}h gfs_min_lag=${STAGED_LAG}h code=${STAGED_REF}"

if aws scheduler get-schedule --name "$SCHEDULE" --region "$REGION" >/dev/null 2>&1; then
    CUR_STATE="$(aws scheduler get-schedule --name "$SCHEDULE" --region "$REGION" \
                  --query State --output text)"
    NEW_STATE="${STATE:-$CUR_STATE}"
    aws scheduler update-schedule --name "$SCHEDULE" --region "$REGION" \
        --schedule-expression "$CRON" --schedule-expression-timezone UTC \
        --flexible-time-window '{"Mode":"OFF"}' \
        --target "$TARGET" --state "$NEW_STATE" --description "$DESC" >/dev/null \
        || die "update-schedule failed"
    echo "  updated, state ${CUR_STATE} -> ${NEW_STATE}"
else
    NEW_STATE="${STATE:-DISABLED}"
    # Retry on role propagation, exactly as create-function above. EventBridge
    # Scheduler validates that the role trusts scheduler.amazonaws.com at
    # create time, and a trust policy written seconds earlier is not yet visible
    # to it: observed failing with "The execution role you provide must allow AWS
    # EventBridge Scheduler to assume the role" against a role whose trust policy
    # already listed that principal. The Lambda path had this retry and this one
    # did not, which is why the first real deploy got this far and then stopped.
    for attempt in 1 2 3 4 5 6; do
        if aws scheduler create-schedule --name "$SCHEDULE" --region "$REGION" \
            --schedule-expression "$CRON" --schedule-expression-timezone UTC \
            --flexible-time-window '{"Mode":"OFF"}' \
            --target "$TARGET" --state "$NEW_STATE" --description "$DESC" \
            >/dev/null 2>&1; then
            echo "  created, state ${NEW_STATE}"; break
        fi
        if [ "$attempt" -eq 6 ]; then
            # Surface the real error rather than the swallowed one.
            aws scheduler create-schedule --name "$SCHEDULE" --region "$REGION" \
                --schedule-expression "$CRON" --schedule-expression-timezone UTC \
                --flexible-time-window '{"Mode":"OFF"}' \
                --target "$TARGET" --state "$NEW_STATE" --description "$DESC" || true
            die "create-schedule failed after 6 attempts"
        fi
        echo "  waiting for role propagation (attempt ${attempt})"; sleep 10
    done
fi

# --- optional single invocation -------------------------------------------
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

  schedule   ${SCHEDULE}   ${CRON}  UTC   state=${FINAL_STATE}
  function   ${FUNCTION}   dry_run=$([ "$LIVE" == "YES" ] && echo NO || echo YES)
  role       ${ROLE}
  staged     lead=${STAGED_LEAD}h gfs_min_lag=${STAGED_LAG}h code=${STAGED_REF}

  logs             aws logs tail /aws/lambda/${FUNCTION} --region ${REGION} --follow
  invoke by hand   aws lambda invoke --function-name ${FUNCTION} --region ${REGION} \\
                     --payload '{}' /dev/stdout
  what is running  aws/status.sh
  stop everything  aws/deploy_scheduler.sh --bucket ${BUCKET} --disable
EOF

if [ "$FINAL_STATE" == "ENABLED" ] && [ "$LIVE" == "YES" ]; then
    printf '\n\033[1;33mThis is now live: one GPU instance per hour, about $875/month at 24 h lead.\033[0m\n'
fi
