#!/usr/bin/env bash
#
# aws/status.sh — what is HRRRCast doing on AWS right now?
#
# Shows live instances, how much of the current forecast has been delivered, and
# how recent runs ended. Read-only: makes no changes and costs nothing.
#
# Usage:
#   aws/status.sh [--bucket NAME] [--region REGION] [--logs]
#
#   --bucket NAME   output bucket to inspect            [mantari-cast1]
#   --region REGION AWS region                          [us-east-1]
#   --logs          also print the tail of the newest run log
#   --live          for each running instance, show over SSM what command it is
#                   actually executing right now, plus stage, RSS and GPU state
#
# Instances are found by the Project=hrrrcast tag that run_on_ec2.sh sets, so
# unrelated instances in the account are never shown.
set -euo pipefail

BUCKET="mantari-cast1"
REGION="us-east-1"
SHOW_LOGS="NO"
SHOW_LIVE="NO"

while [ $# -gt 0 ]; do
    case "$1" in
        --bucket) BUCKET="$2"; shift 2 ;;
        --region) REGION="$2"; shift 2 ;;
        --logs)   SHOW_LOGS="YES"; shift ;;
        --live)   SHOW_LIVE="YES"; shift ;;
        -h|--help) sed -n '2,18p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 1 ;;
    esac
done

export AWS_DEFAULT_REGION="$REGION"
export AWS_PAGER=""
hdr() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }

hdr "Live instances (tag Project=hrrrcast)"
LIVE="$(aws ec2 describe-instances \
    --filters "Name=tag:Project,Values=hrrrcast" \
              "Name=instance-state-name,Values=pending,running,shutting-down,stopping,stopped" \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,Placement.AvailabilityZone,LaunchTime,Tags[?Key==`InitTime`]|[0].Value,Tags[?Key==`LeadHours`]|[0].Value,Tags[?Key==`HindcastRun`]|[0].Value]' \
    --output text 2>/dev/null || true)"

if [ -z "$LIVE" ]; then
    echo "  none running"
else
    printf '  %-21s %-13s %-14s %-11s %-17s %-6s %s\n' INSTANCE TYPE STATE AZ CYCLE LEADS HINDCAST
    while IFS=$'\t' read -r id type state az launched init leads hindcast; do
        [ -z "$id" ] && continue
        printf '  %-21s %-13s %-14s %-11s %-17s %-6s %s\n' "$id" "$type" "$state" "$az" "${init:-?}" "${leads:-?}" "${hindcast:-}"
        # Elapsed wall clock, which is what you are billed for.
        if command -v python3 >/dev/null 2>&1; then
            python3 - "$launched" <<'PY' 2>/dev/null || true
import sys, datetime
t = datetime.datetime.fromisoformat(sys.argv[1].replace("Z", "+00:00"))
mins = (datetime.datetime.now(datetime.timezone.utc) - t).total_seconds() / 60
print(f"      up {mins:.0f} min (since {t:%H:%M:%SZ})")
PY
        fi
    done <<< "$LIVE"
    echo
    echo "  shell in:   aws ssm start-session --target <INSTANCE>"
    echo "  kill:       aws ec2 terminate-instances --instance-ids <INSTANCE>"
fi

# --live: ask each running instance what it is doing. Uses SSM Run Command, so it
# needs no SSH key and no inbound security-group rule -- the hrrrcast-runner role
# carries AmazonSSMManagedInstanceCore. This answers "which command is actually
# running", which the log tail alone does not: a stage can be mid-step for many
# minutes with nothing new written.
if [ "$SHOW_LIVE" == "YES" ] && [ -n "$LIVE" ]; then
    while IFS=$'\t' read -r id _ state _ _ _ _ _; do
        [ -z "$id" ] && continue
        [ "$state" == "running" ] || continue
        hdr "What ${id} is executing now (via SSM)"
        PARAMS=$(mktemp)
        cat > "$PARAMS" <<'JSON'
{"commands":[
 "echo \"host $(date -u +%H:%M:%SZ)\"",
 "ps -eo etime,rss,args --sort=-rss | awk 'NR==1 || /src\\/[a-z_]*\\.py|run_cycle|conda (env )?create|wgrib2/' | grep -v ' awk ' | head -6",
 "echo stage: $(grep -aoE '==> Stage: [a-z-]+' /var/log/hrrrcast-run.log 2>/dev/null | tail -1)",
 "tail -c 240 /var/log/hrrrcast-run.log 2>/dev/null | tr -d '\\r' | tr -s '\\n' | tail -1",
 "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
]}
JSON
        CID=$(aws ssm send-command --instance-ids "$id" --document-name AWS-RunShellScript \
              --parameters "file://$PARAMS" --query 'Command.CommandId' --output text 2>/dev/null)
        if [ -z "$CID" ]; then
            echo "  (SSM not available yet -- the instance may still be booting)"
        else
            for _ in $(seq 1 12); do
                S=$(aws ssm get-command-invocation --command-id "$CID" --instance-id "$id" \
                    --query Status --output text 2>/dev/null)
                case "$S" in Success|Failed) break ;; esac
                sleep 5
            done
            aws ssm get-command-invocation --command-id "$CID" --instance-id "$id" \
                --query 'StandardOutputContent' --output text 2>/dev/null | sed 's/^/  /'
        fi
        rm -f "$PARAMS"
    done <<< "$LIVE"
fi

hdr "Delivered forecast output (s3://${BUCKET}/hrrrcast/out/ -- on-demand + hourly only; hindcasts write elsewhere, see below)"
OUT="$(aws s3 ls --recursive "s3://${BUCKET}/hrrrcast/out/" 2>/dev/null | grep '\.nc$' || true)"
if [ -z "$OUT" ]; then
    echo "  nothing yet"
else
    # Group by cycle directory so multiple cycles in the bucket stay legible.
    printf '%s\n' "$OUT" | awk '
        { path=$4; n=split(path, p, "/"); cycle=p[3]"/"p[4]; bytes[cycle]+=$3; files[cycle]++;
          if (path > latest[cycle]) latest[cycle]=path }
        END { for (c in files) printf "  %-14s %3d files  %7.2f GB  newest: %s\n",
                                      c, files[c], bytes[c]/1e9, latest[c] }' | sort
fi

# Hindcasts (aws/deploy_hindcast.sh) live entirely under their own
# hrrrcast/hindcast/<run-id>/ prefix -- config, status markers AND output, kept
# separate from the shared hrrrcast/out/ above precisely so two runs sharing a
# cycle cannot collide (see aws/README_aws.md, "Output paths don't encode the
# domain"). Each run-id is a subdirectory here, so this section is where their
# progress actually lives.
hdr "Hindcast runs (s3://${BUCKET}/hrrrcast/hindcast/)"
RUN_IDS="$(aws s3 ls "s3://${BUCKET}/hrrrcast/hindcast/" 2>/dev/null \
           | awk '/PRE/ {gsub("/",""); print $2}' || true)"
if [ -z "$RUN_IDS" ]; then
    echo "  none"
else
    for run_id in $RUN_IDS; do
        PREFIX="hrrrcast/hindcast/${run_id}"
        CFG="$(aws s3 cp "s3://${BUCKET}/${PREFIX}/config.json" - 2>/dev/null || true)"
        N_TOTAL="?"; DATES="?"; DOMAIN="?"
        if [ -n "$CFG" ]; then
            N_TOTAL="$(printf '%s' "$CFG" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("n_cycles","?"))' 2>/dev/null || echo "?")"
            DATES="$(printf '%s' "$CFG" | python3 -c "import json,sys
c=json.load(sys.stdin)
print(f\"{c['start_date']}..{c['end_date']} @ {','.join(c['init_hours'])}Z\")" 2>/dev/null || echo "?")"
            DOMAIN="$(printf '%s' "$CFG" | python3 -c "import json,sys
c=json.load(sys.stdin)
print(f\"bbox {c['bbox']} +{c['halo']}\")" 2>/dev/null || echo "?")"
        fi

        STATUS_LIST="$(aws s3 ls "s3://${BUCKET}/${PREFIX}/status/" 2>/dev/null | awk '{print $4}')"
        # _complete.txt is the Lambda's own "already notified" marker, not a
        # cycle -- excluded from the count, same as hindcast_handler.py's own
        # done_cycles() does.
        N_DONE="$(printf '%s\n' "$STATUS_LIST" | grep -c '\.txt$' || true)"
        printf '%s\n' "$STATUS_LIST" | grep -q '^_complete\.txt$' \
            && { N_DONE=$((N_DONE - 1)); COMPLETE="yes"; } || COMPLETE="no"

        SCHED_STATE="$(aws scheduler get-schedule --name "hrrrcast-hindcast-${run_id}" \
                       --region "$REGION" --query State --output text 2>/dev/null || echo "(deleted)")"

        OUT_STATS="$(aws s3 ls --recursive "s3://${BUCKET}/${PREFIX}/out/" 2>/dev/null \
                     | grep '\.nc$' | awk '{n++; b+=$3} END {printf "%d files, %.2f GB", n+0, b/1e9}' || true)"

        printf '  %-28s %s\n' "$run_id" "$DATES"
        printf '    domain     %s\n' "$DOMAIN"
        printf '    cycles     %s/%s have a status marker (ok+failed; see per-cycle logs for the split)   complete=%-4s schedule=%s\n' \
               "${N_DONE:-0}" "$N_TOTAL" "$COMPLETE" "$SCHED_STATE"
        printf '    output     %s   s3://%s/%s/out/\n' "${OUT_STATS:-0 files}" "$BUCKET" "$PREFIX"
    done
fi

hdr "Recent run outcomes"
STATUSES="$(aws s3 ls --recursive "s3://${BUCKET}/hrrrcast/logs/" 2>/dev/null \
            | grep 'status.txt$' | sort -r | head -8 || true)"
if [ -z "$STATUSES" ]; then
    echo "  no completed runs"
else
    while read -r date time size key; do
        [ -z "${key:-}" ] && continue
        iid="$(basename "$key")"; iid="${iid%-status.txt}"
        st="$(aws s3 cp "s3://${BUCKET}/${key}" - 2>/dev/null | tr -d '\n' || echo '?')"
        case "$st" in
            success) mark=$'\033[1;32mok     \033[0m' ;;
            *)       mark=$'\033[1;31mFAILED \033[0m' ;;
        esac
        printf '  %b %-21s %s %s  %s\n' "$mark" "$iid" "$date" "$time" "$st"
    done <<< "$STATUSES"
fi

if [ "$SHOW_LOGS" == "YES" ]; then
    NEWEST="$(aws s3 ls --recursive "s3://${BUCKET}/hrrrcast/logs/" 2>/dev/null \
              | grep 'bootstrap.log$' | sort -r | head -1 | awk '{print $4}' || true)"
    if [ -n "$NEWEST" ]; then
        hdr "Tail of newest run log (${NEWEST##*/})"
        aws s3 cp "s3://${BUCKET}/${NEWEST}" - 2>/dev/null \
            | sed -e 's/\x1b\[[0-9;]*[A-Za-z]//g' \
            | grep -aE 'Stage:|predict took|Wrote NetCDF|Uploaded s3|ERROR|run finished' \
            | tail -20
    fi
fi

echo
