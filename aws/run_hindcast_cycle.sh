#!/usr/bin/env bash
#
# aws/run_hindcast_cycle.sh — run ONE hindcast cycle and record its outcome.
#
# Runs through the serverless hindcast driver (aws/lambda/hindcast_handler.py +
# aws/deploy_hindcast.sh): one Lambda tick launches one instance per cycle, via
# run_on_ec2.sh --run-cmd, with this as the command. Everything DATAROOT,
# S3_OUTPUT, SUB_BBOX, SUB_HALO, GFS_MIN_LAG, NC_COMPLEVEL, NC_LSD, PURGE_LOCAL,
# MAKE_BCS_WORKERS needs is already exported by aws/user_data.sh before this
# runs (see run_cycle.sh's own env-var docs).
#
# What this adds over calling run_cycle.sh directly: ONE status marker object
# written to S3 per cycle, unconditionally (success or failure). That marker is
# the entire resume mechanism -- the Lambda driver treats "no marker yet" as
# "not done" and "a marker exists" as "don't run this again", so the whole
# hindcast picks up wherever it left off after any interruption (a redeploy, a
# manual stop, a stuck cycle) with no other state to go stale.
#
# This exits with run_cycle.sh's own exit code, unchanged, so the existing
# per-instance SNS failure notification in aws/user_data.sh (--notify-topic)
# fires exactly as it would for a single on-demand run -- no new notification
# code needed for "tell me when a cycle fails".
#
# Usage:
#   aws/run_hindcast_cycle.sh INIT_TIME LEAD_HOURS OUTPUT_HOURS STATUS_S3_PREFIX
#
#   STATUS_S3_PREFIX   s3://bucket/hrrrcast/hindcast/<run-id>/status
set -uo pipefail

INIT_TIME="${1:?INIT_TIME (YYYY-MM-DDTHH) required}"
LEAD_HOUR="${2:?LEAD_HOURS required}"
OUTPUT_HOURS="${3:-}"
STATUS_PREFIX="${4:?STATUS_S3_PREFIX required, e.g. s3://bucket/hrrrcast/hindcast/RUNID/status}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATAROOT="${DATAROOT:-$(pwd)}"
STATUS_KEY="${STATUS_PREFIX%/}/${INIT_TIME}.txt"

export OUTPUT_HOURS

echo "hindcast cycle ${INIT_TIME}Z  lead=${LEAD_HOUR}h  output_hours=${OUTPUT_HOURS:-all}  status->${STATUS_KEY}"

"${REPO_DIR}/run_cycle.sh" "$INIT_TIME" "$LEAD_HOUR" 1 1 "$REPO_DIR" "$DATAROOT" NO ""
RC=$?

MARKER="ok"
[ "$RC" -eq 0 ] || MARKER="failed(rc=${RC})"
printf '%s\n' "$MARKER" | aws s3 cp - "$STATUS_KEY" \
    || echo "WARNING: could not write status marker to ${STATUS_KEY}; this cycle may be re-attempted." >&2

exit "$RC"
