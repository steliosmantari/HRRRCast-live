#!/usr/bin/env python3
"""Hindcast launcher, for AWS Lambda behind an EventBridge Scheduler rule.

One invocation launches AT MOST one GPU instance, for the earliest cycle in the
configured date range that does not yet have a status marker in S3. This is
the serverless counterpart to aws/run_hindcast.sh (which loops cycles on one
long-lived instance): here, each cycle gets its own on-demand instance
(aws/run_on_ec2.sh's proven bootstrap), launched one at a time by whatever tick
of the schedule finds nothing currently in flight.

RESUME, FOR FREE. The only state this function trusts is:
  1. EC2 instance state (tag HindcastRun=<run-id>, pending/running) -- "is a
     cycle in flight right now".
  2. S3 objects under .../hindcast/<run-id>/status/<init_time>.txt -- "is this
     cycle done" (written by aws/run_hindcast_cycle.sh on the instance, always,
     success or failure).
There is no third place state can go stale: redeploying this function, editing
the config, or just letting the schedule sit idle for a week all resume
correctly, because "what's left to do" is recomputed from those two facts on
every single tick.

WHY A LAMBDA AND NOT ONE LONG-LIVED INSTANCE
    See aws/run_hindcast.sh for that alternative. This one trades a slightly
    slower per-cycle turnaround (one instance boot per cycle instead of a
    shared already-booted one) for never depending on a box staying up for the
    whole hindcast, and for the failure-notification and log-shipping paths
    being exactly the ones already proven for single on-demand runs.

FAILURE HANDLING
    A cycle that fails still gets a status marker (failed(rc=N)), so it is
    never retried and never blocks the cycles after it. Its instance's own
    --notify-topic email fires unchanged (aws/user_data.sh). This function
    only sends ITS OWN notification once, when every cycle has a marker --a
    summary, not a per-cycle alert.

Environment variables (set by aws/deploy_hindcast.sh):
    HRRRCAST_BUCKET          required. Bucket holding the staged artifacts.
    HRRRCAST_RUN_ID          required. Identifies this hindcast's S3 prefix and
                              the EC2 tag used for the in-flight check.
    HRRRCAST_STAGE_PREFIX    where the template/params were staged
                              [hrrrcast/hindcast/<run-id>/scheduler]
    HRRRCAST_SCHEDULE_NAME   this function's own EventBridge schedule name, so
                              it can self-disable once every cycle is done.
                              Optional: skipped (logged) if unset.
    HRRRCAST_SNS_TOPIC       optional SNS topic ARN for the one completion
                              summary notification.
    HRRRCAST_DRY_RUN         "YES" to log the decision without launching.
"""
import datetime as dt
import json
import logging
import os
import re

import boto3
from botocore.exceptions import ClientError

log = logging.getLogger()
log.setLevel(logging.INFO)

BUCKET = os.environ.get("HRRRCAST_BUCKET", "")
RUN_ID = os.environ.get("HRRRCAST_RUN_ID", "")
STAGE_PREFIX = os.environ.get("HRRRCAST_STAGE_PREFIX", f"hrrrcast/hindcast/{RUN_ID}/scheduler")
SCHEDULE_NAME = os.environ.get("HRRRCAST_SCHEDULE_NAME", "")
SNS_TOPIC = os.environ.get("HRRRCAST_SNS_TOPIC", "")
DRY_RUN = os.environ.get("HRRRCAST_DRY_RUN", "NO").upper() == "YES"

HINDCAST_PREFIX = f"hrrrcast/hindcast/{RUN_ID}"
STATUS_PREFIX = f"{HINDCAST_PREFIX}/status/"
COMPLETE_KEY = f"{STATUS_PREFIX}_complete.txt"
CONFIG_KEY = f"{HINDCAST_PREFIX}/config.json"

PROJECT_TAG = "hrrrcast"
RUN_TAG = "HindcastRun"


class NothingToDo(Exception):
    """Not an error: the whole hindcast is already done."""


class NoCapacity(Exception):
    """EC2 had no capacity for the instance type in any attempted AZ."""


# --- config and staged artifacts --------------------------------------------
def _load_config(s3):
    try:
        cfg = json.loads(s3.get_object(Bucket=BUCKET, Key=CONFIG_KEY)["Body"].read())
    except ClientError as e:
        raise RuntimeError(
            f"Config missing at s3://{BUCKET}/{CONFIG_KEY}. Run aws/deploy_hindcast.sh."
        ) from e
    for key in ("start_date", "end_date", "init_hours", "lead_hours", "gfs_min_lag"):
        if key not in cfg:
            raise RuntimeError(f"config.json missing required key {key!r}")
    return cfg


def _load_staged(s3):
    try:
        params = json.loads(s3.get_object(
            Bucket=BUCKET, Key=f"{STAGE_PREFIX}/launch-params.json"
        )["Body"].read())
        template = s3.get_object(
            Bucket=BUCKET, Key=f"{STAGE_PREFIX}/user_data.template"
        )["Body"].read().decode()
    except ClientError as e:
        raise RuntimeError(
            f"Staged artifacts missing under s3://{BUCKET}/{STAGE_PREFIX}/. "
            "Run aws/deploy_hindcast.sh (or aws/run_on_ec2.sh --stage-scheduler "
            f"--stage-prefix {STAGE_PREFIX})."
        ) from e
    for sentinel in ("__INIT_TIME__", "__LEAD_HOUR__", "__GFS_MIN_LAG__"):
        if sentinel not in template:
            raise RuntimeError(
                f"user_data.template has no {sentinel}; it was staged from an "
                "incompatible run_on_ec2.sh. Re-stage.")
    return params, template


# --- cycle bookkeeping -------------------------------------------------------
def all_cycles(cfg):
    """Every (start_date..end_date) x init_hours cycle, chronological."""
    start = dt.datetime.strptime(cfg["start_date"], "%Y-%m-%d").date()
    end = dt.datetime.strptime(cfg["end_date"], "%Y-%m-%d").date()
    hours = cfg["init_hours"] if isinstance(cfg["init_hours"], list) else cfg["init_hours"].split(",")
    out = []
    day = start
    while day <= end:
        for h in hours:
            out.append(f"{day:%Y-%m-%d}T{h}")
        day += dt.timedelta(days=1)
    return out


def done_cycles(s3):
    """init_time strings that already have a status marker (ok or failed)."""
    done = set()
    token = None
    while True:
        kwargs = dict(Bucket=BUCKET, Prefix=STATUS_PREFIX)
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for o in resp.get("Contents", []):
            name = o["Key"].rsplit("/", 1)[-1]
            if name.endswith(".txt") and name != "_complete.txt":
                done.add(name[:-len(".txt")])
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return done


def _in_flight(ec2):
    resp = ec2.describe_instances(Filters=[
        {"Name": f"tag:{RUN_TAG}", "Values": [RUN_ID]},
        {"Name": "instance-state-name", "Values": ["pending", "running"]},
    ])
    return [i["InstanceId"] for r in resp["Reservations"] for i in r["Instances"]]


def _vcpu_headroom(session, region: str, instance_type: str) -> bool:
    try:
        need = session.client("ec2", region_name=region).describe_instance_types(
            InstanceTypes=[instance_type]
        )["InstanceTypes"][0]["VCpuInfo"]["DefaultVCpus"]
        quota = session.client("service-quotas", region_name=region).get_service_quota(
            ServiceCode="ec2", QuotaCode="L-DB2E81BA"
        )["Quota"]["Value"]
    except (ClientError, KeyError, IndexError) as e:
        log.warning("could not check vCPU quota (%s); proceeding", e)
        return True
    log.info("%s needs %d vCPU; G/VT on-demand quota is %g", instance_type, need, quota)
    return need <= quota


def _candidate_subnets(ec2, instance_type: str):
    try:
        azs = {o["Location"] for o in ec2.describe_instance_type_offerings(
            LocationType="availability-zone",
            Filters=[{"Name": "instance-type", "Values": [instance_type]}],
        )["InstanceTypeOfferings"]}
        subnets = ec2.describe_subnets(
            Filters=[{"Name": "default-for-az", "Values": ["true"]}])["Subnets"]
    except ClientError as e:
        log.warning("could not enumerate subnets for the AZ walk (%s); "
                    "falling back to the EC2-chosen attempt only", e)
        return []
    return [(s["SubnetId"], s["AvailabilityZone"])
            for s in subnets if s["AvailabilityZone"] in azs]


# --- launch ------------------------------------------------------------------
def launch(ec2, params, template, init_time: str, lead_hours, gfs_min_lag):
    user_data = (template
                 .replace("__INIT_TIME__", init_time)
                 .replace("__LEAD_HOUR__", str(lead_hours))
                 .replace("__GFS_MIN_LAG__", str(gfs_min_lag)))
    leftovers = sorted(set(re.findall(r"__[A-Z][A-Z0-9_]*__", user_data)))
    if leftovers:
        raise RuntimeError(
            f"user-data substitution incomplete, unresolved sentinels: {leftovers}. "
            "Re-stage with a matching run_on_ec2.sh, or teach launch() the new value.")

    args = dict(
        ImageId=params["image_id"],
        InstanceType=params["instance_type"],
        MinCount=1, MaxCount=1,
        IamInstanceProfile={"Name": params["instance_profile"]},
        InstanceInitiatedShutdownBehavior="terminate",
        BlockDeviceMappings=[{
            "DeviceName": "/dev/sda1",
            "Ebs": {"VolumeSize": int(params["volume_size"]), "VolumeType": "gp3",
                    "DeleteOnTermination": True},
        }],
        UserData=user_data,
        TagSpecifications=[
            {"ResourceType": rt, "Tags": [
                {"Key": "Name", "Value": f"hrrrcast-hindcast-{RUN_ID}-{init_time}"},
                {"Key": "Project", "Value": PROJECT_TAG},
                {"Key": RUN_TAG, "Value": RUN_ID},
                {"Key": "InitTime", "Value": init_time},
                {"Key": "LeadHours", "Value": str(lead_hours)},
                {"Key": "GfsMinLag", "Value": str(gfs_min_lag)},
                {"Key": "CodeRef", "Value": str(params.get("code_ref", "unknown"))},
                {"Key": "LaunchedBy", "Value": "hindcast-lambda"},
            ]}
            for rt in ("instance", "volume")
        ],
    )
    if params.get("security_group_ids"):
        args["SecurityGroupIds"] = params["security_group_ids"].split()

    attempts = [("EC2-chosen", None)]
    for subnet_id, az in _candidate_subnets(ec2, params["instance_type"]):
        attempts.append((az, subnet_id))

    last_err = None
    for az, subnet_id in attempts:
        call = dict(args)
        if subnet_id:
            call["SubnetId"] = subnet_id
        try:
            iid = ec2.run_instances(**call)["Instances"][0]["InstanceId"]
            log.info("launched in %s%s", az, f" ({subnet_id})" if subnet_id else "")
            return iid
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code not in ("InsufficientInstanceCapacity", "Unsupported",
                            "InsufficientHostCapacity"):
                raise
            log.warning("no capacity in %s (%s)", az, code)
            last_err = e

    raise NoCapacity(
        f"no capacity for {params['instance_type']} in any of "
        f"{len(attempts)} placement attempts: {last_err}")


# --- completion --------------------------------------------------------------
def _finish_up(s3, session, region, total, done):
    """Every cycle has a marker. Notify once; always retry the self-disable.

    The two are independent on purpose: the notification (and the S3 marker
    that guards it) should fire exactly once, but a schedule that failed to
    disable on an earlier tick (a transient error, a permission fix landing
    late) must keep getting retried on every later tick, not just the first.
    Otherwise "already notified" silently becomes "stuck enabled forever".
    """
    already_notified = True
    try:
        s3.head_object(Bucket=BUCKET, Key=COMPLETE_KEY)
    except ClientError:
        already_notified = False

    if not already_notified:
        msg = (f"HRRRCast hindcast '{RUN_ID}' complete: {len(done)}/{total} cycles have a "
               f"status marker.\nDetail: s3://{BUCKET}/{STATUS_PREFIX}\n"
               f"(This counts cycles that ran, success or failure; a per-cycle failure "
               f"already sent its own notification if --notify-topic was staged.)")
        log.info(msg)
        s3.put_object(Bucket=BUCKET, Key=COMPLETE_KEY, Body=msg.encode())

        if SNS_TOPIC:
            try:
                session.client("sns", region_name=region).publish(
                    TopicArn=SNS_TOPIC, Subject=f"HRRRCast hindcast complete: {RUN_ID}"[:99],
                    Message=msg)
            except ClientError as e:
                log.warning("SNS publish failed (%s); completion is still recorded in S3", e)
    else:
        log.info("hindcast %s already marked complete; retrying schedule self-disable only", RUN_ID)

    if SCHEDULE_NAME:
        try:
            sched = session.client("scheduler", region_name=region)
            cur = sched.get_schedule(Name=SCHEDULE_NAME)
            if cur["State"] == "DISABLED":
                log.info("schedule %s already disabled", SCHEDULE_NAME)
            else:
                sched.update_schedule(
                    Name=SCHEDULE_NAME,
                    ScheduleExpression=cur["ScheduleExpression"],
                    ScheduleExpressionTimezone=cur.get("ScheduleExpressionTimezone", "UTC"),
                    FlexibleTimeWindow=cur["FlexibleTimeWindow"],
                    Target=cur["Target"],
                    State="DISABLED",
                )
                log.info("disabled schedule %s", SCHEDULE_NAME)
        except ClientError as e:
            log.warning("could not self-disable schedule %s (%s); disable it by hand",
                        SCHEDULE_NAME, e)
    else:
        log.info("HRRRCAST_SCHEDULE_NAME not set; leaving the schedule as-is")


# --- entry point -------------------------------------------------------------
def lambda_handler(event, context):
    if not BUCKET:
        raise RuntimeError("HRRRCAST_BUCKET is not set")
    if not RUN_ID:
        raise RuntimeError("HRRRCAST_RUN_ID is not set")

    session = boto3.Session()
    s3 = session.client("s3")
    cfg = _load_config(s3)
    params, template = _load_staged(s3)
    region = params["region"]
    ec2 = session.client("ec2", region_name=region)

    cycles = all_cycles(cfg)
    done = done_cycles(s3)
    remaining = [c for c in cycles if c not in done]
    log.info("hindcast %s: %d/%d cycles done, %d remaining",
              RUN_ID, len(done), len(cycles), len(remaining))

    if not remaining:
        _finish_up(s3, session, region, len(cycles), done)
        return {"status": "complete", "total": len(cycles), "done": len(done)}

    busy = _in_flight(ec2)
    if busy:
        msg = f"{len(busy)} hindcast instance(s) still in flight: {busy}; skipping"
        log.warning(msg)
        return {"status": "skipped", "reason": msg, "remaining": len(remaining)}

    if not _vcpu_headroom(session, region, params["instance_type"]):
        msg = "vCPU quota cannot fit another instance; skipping"
        log.warning(msg)
        return {"status": "skipped", "reason": msg, "remaining": len(remaining)}

    init_time = remaining[0]
    decision = {
        "init_time": init_time,
        "lead_hours": cfg["lead_hours"],
        "gfs_min_lag": cfg["gfs_min_lag"],
        "remaining_after_this": len(remaining) - 1,
    }

    if DRY_RUN:
        log.info("DRY RUN, would launch: %s", json.dumps(decision))
        return {"status": "dry_run", **decision}

    try:
        iid = launch(ec2, params, template, init_time, cfg["lead_hours"], cfg["gfs_min_lag"])
    except NoCapacity as e:
        log.warning("skipping %s: %s", init_time, e)
        return {"status": "no_capacity", "reason": str(e), **decision}
    log.info("launched %s for %s", iid, init_time)
    return {"status": "launched", "instance_id": iid, **decision}
