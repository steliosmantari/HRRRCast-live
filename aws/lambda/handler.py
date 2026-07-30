#!/usr/bin/env python3
"""Hourly HRRRCast launcher, for AWS Lambda behind an EventBridge Scheduler rule.

One invocation decides whether there is a forecast worth running right now and, if
so, launches exactly one GPU instance to run it. Everything the instance needs was
pinned to S3 at deploy time by `aws/run_on_ec2.sh --stage-scheduler`, so this
function never packages code and never renders a template from scratch: it
substitutes three values and calls RunInstances.

WHY A LAMBDA AND NOT CRON ON A BOX
    A g6e.2xlarge costs about $2.24/h and an hourly schedule launches 24 a day. The
    launcher is the one component that must not fail silently, and the cheapest way
    to get that is to not own a host. This function has no state, no disk, and
    nothing to patch.

WHAT IT REFUSES TO DO
    It will not launch when the inputs are not fully published, when the cycle has
    already been produced, when a previous run is still going, or when the vCPU
    quota cannot fit another instance. Each of those checks costs milliseconds to
    seconds; a wasted launch costs a dollar and, worse, can produce a truncated
    forecast rather than a clean failure.

DELIBERATELY SHARED, NOT REIMPLEMENTED
    Cycle selection and the input file manifest come from src/gfs_cycle.py, which is
    bundled into the deployment zip. That module is dependency-free precisely so
    this function can import it; get_bcs.py cannot be imported here because it pulls
    in `requests` and `dateutil` through utils. Two copies of "which GFS cycle backs
    this forecast" would eventually disagree, and the failure would be a forecast
    quietly built from the wrong cycle.

Environment variables (set by aws/deploy_scheduler.sh):
    HRRRCAST_BUCKET     required. Bucket holding the staged artifacts and outputs.
    HRRRCAST_DRY_RUN    "YES" to do every check and log the decision without
                        launching. Safe way to watch a new schedule for a day.
    HRRRCAST_MAX_LOOKBACK       hours back to consider            [6]
    HRRRCAST_MAX_EXTRA_CYCLES   extra 6 h GFS cycle steps back    [2]
    HRRRCAST_PROBE_WORKERS      parallel availability probes      [16]
"""
import concurrent.futures
import datetime as dt
import json
import logging
import os
import re
import urllib.error
import urllib.request

import boto3
from botocore.exceptions import ClientError

from gfs_cycle import get_gfs_urls, get_hrrr_urls, gfs_cycle_for

log = logging.getLogger()
log.setLevel(logging.INFO)

BUCKET = os.environ.get("HRRRCAST_BUCKET", "")
DRY_RUN = os.environ.get("HRRRCAST_DRY_RUN", "NO").upper() == "YES"
MAX_LOOKBACK = int(os.environ.get("HRRRCAST_MAX_LOOKBACK", "6"))
MAX_EXTRA_CYCLES = int(os.environ.get("HRRRCAST_MAX_EXTRA_CYCLES", "2"))
PROBE_WORKERS = int(os.environ.get("HRRRCAST_PROBE_WORKERS", "16"))

STAGE_PREFIX = "hrrrcast/scheduler"

# Instances are tagged Project=hrrrcast by run_on_ec2.sh and by this function, so
# the in-flight check can find them regardless of which launched them.
PROJECT_TAG = "hrrrcast"


class NothingToDo(Exception):
    """Not an error: no cycle is ready, or the work is already done."""


class NoCapacity(Exception):
    """EC2 had no capacity for the instance type in any attempted AZ.

    Separate from NothingToDo because the cause is entirely on AWS's side and the
    hour is genuinely lost, which is worth seeing distinctly in the logs.
    """


# --- availability probing --------------------------------------------------
def _exists(url: str, timeout: int = 20) -> bool:
    """True if the object is readable.

    A one-byte ranged GET rather than a HEAD. Both are cheap, but some S3 fronting
    layers answer HEAD from a cache that can lag by minutes, and a file that looks
    present while still being written is exactly the failure this guards against.
    """
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status in (200, 206)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
        return False


def _all_exist(urls, label):
    """Probe every URL in parallel. Returns (n_missing, first_missing_or_None)."""
    missing = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=PROBE_WORKERS) as ex:
        for url, ok in zip(urls, ex.map(_exists, urls)):
            if not ok:
                missing.append(url)
    if missing:
        # ThreadPoolExecutor.map yields in input order, so missing[0] is the earliest
        # step in the manifest, which is the useful diagnostic.
        log.info("%s: %d/%d MISSING, first %s", label, len(missing), len(urls),
                 missing[0].rsplit("/", 1)[-1])
        return len(missing), missing[0]
    log.info("%s: all %d files present", label, len(urls))
    return 0, None


# --- deploy-time artifacts -------------------------------------------------
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
            "Run: aws/run_on_ec2.sh --bucket <bucket> --stage-scheduler"
        ) from e
    for sentinel in ("__INIT_TIME__", "__LEAD_HOUR__", "__GFS_MIN_LAG__"):
        if sentinel not in template:
            raise RuntimeError(
                f"user_data.template has no {sentinel}. It was staged from an "
                "incompatible run_on_ec2.sh; re-stage."
            )
    return params, template


# --- guards ----------------------------------------------------------------
def _already_produced(s3, s3_output: str, init: dt.datetime) -> bool:
    """True if this cycle already has NetCDF output.

    Guards against a duplicate EventBridge tick and against re-running a cycle
    someone produced by hand. Without it a retry storm bills once per attempt for an
    identical product.
    """
    bucket, _, prefix = s3_output.replace("s3://", "").partition("/")
    key = f"{prefix.rstrip('/')}/{init:%Y%m%d}/{init:%H}/"
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=200)
    n = sum(1 for o in resp.get("Contents", []) if o["Key"].endswith(".nc"))
    if n:
        log.info("cycle %s already has %d .nc under s3://%s/%s", init, n, bucket, key)
    return n > 0


def _in_flight(ec2):
    """Instance IDs of hrrrcast runs already pending or running.

    A 24 h forecast takes about 28 minutes, comfortably inside an hourly slot, but a
    slow or stuck run must not have a second instance stacked on top of it: the vCPU
    quota would reject it, or worse, two instances would write the same output
    prefix.
    """
    resp = ec2.describe_instances(Filters=[
        {"Name": "tag:Project", "Values": [PROJECT_TAG]},
        {"Name": "instance-state-name", "Values": ["pending", "running"]},
    ])
    return [i["InstanceId"]
            for r in resp["Reservations"] for i in r["Instances"]]


def _vcpu_headroom(session, region: str, instance_type: str) -> bool:
    """True if the G/VT on-demand quota can fit one more instance.

    Cheap insurance: RunInstances would fail anyway, but failing here is one API
    call and produces a log line that says why, instead of a VcpuLimitExceeded
    buried in an exception.
    """
    try:
        need = session.client("ec2", region_name=region).describe_instance_types(
            InstanceTypes=[instance_type]
        )["InstanceTypes"][0]["VCpuInfo"]["DefaultVCpus"]
        quota = session.client("service-quotas", region_name=region).get_service_quota(
            ServiceCode="ec2", QuotaCode="L-DB2E81BA"
        )["Quota"]["Value"]
    except (ClientError, KeyError, IndexError) as e:
        # Not fatal. A missing service-quotas permission should not stop forecasts.
        log.warning("could not check vCPU quota (%s); proceeding", e)
        return True
    log.info("%s needs %d vCPU; G/VT on-demand quota is %g", instance_type, need, quota)
    return need <= quota


# --- cycle selection -------------------------------------------------------
def choose_cycle(s3, params, now: dt.datetime):
    """Pick (init_time, gfs_min_lag) or raise NothingToDo.

    Two nested searches, and the nesting order matters. The HRRR hour is the outer
    loop, freshest first, because it is the initial condition and the thing the
    product is judged on. The GFS cycle is the inner loop, stepping back a whole 6 h
    at a time. Holding the HRRR hour and stepping the cycle gives up the right thing
    when data is late: four of every six consecutive hours share a GFS cycle, so
    stepping the hour instead would discard initial-condition freshness while
    re-probing the same missing file.
    """
    lead = int(params["lead_hours"])
    base_lag = int(params["gfs_min_lag"])
    top_of_hour = now.replace(minute=0, second=0, microsecond=0)

    for back in range(1, MAX_LOOKBACK + 1):
        init = top_of_hour - dt.timedelta(hours=back)
        log.info("candidate -%dh  %s", back, init.strftime("%Y-%m-%dT%HZ"))

        hrrr = get_hrrr_urls(init)
        n_missing, _ = _all_exist([u for u, _ in hrrr], "  hrrr (4 files)")
        if n_missing:
            continue

        if _already_produced(s3, params["s3_output"], init):
            continue

        for k in range(MAX_EXTRA_CYCLES + 1):
            lag = base_lag + 6 * k
            if lag > 23:      # gfs_cycle_for's documented ceiling
                break
            cyc = gfs_cycle_for(init, lag)
            urls = [u for u, _ in get_gfs_urls(
                f"{init:%Y}", f"{init:%m}", f"{init:%d}", f"{init:%H}", lead, lag)]
            label = f"  GFS {cyc:%Y%m%d %H}Z ({int((init - cyc).total_seconds() // 3600)}h old, lag={lag})"
            n_missing, _ = _all_exist(urls, label)
            if n_missing == 0:
                if k:
                    # Not a silent fallback: forecast hours move further outside the
                    # f001-f029 range the network trained on with each step back.
                    log.warning("using a GFS cycle %d step(s) older than intended "
                                "(lag=%d); forcing is further out of the trained range",
                                k, lag)
                return init, lag, cyc, len(urls)

    raise NothingToDo(
        f"no cycle with complete inputs in the last {MAX_LOOKBACK} h")


# --- launch ----------------------------------------------------------------
def launch(ec2, params, template, init: dt.datetime, lag: int):
    user_data = (template
                 .replace("__INIT_TIME__", init.strftime("%Y-%m-%dT%H"))
                 .replace("__LEAD_HOUR__", str(params["lead_hours"]))
                 .replace("__GFS_MIN_LAG__", str(lag)))
    # A sentinel surviving substitution would reach the instance as a literal and
    # break the run in a way that is tedious to diagnose from bootstrap logs. Match
    # the __UPPER_CASE__ shape generically rather than the three known names: if
    # user_data.sh gains a fourth per-run value and this function is not updated,
    # that must fail here rather than silently ship "__NEW_THING__" to a $2.24/h box.
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
        # Volumes are tagged as well as the instance, and not only for cost
        # attribution: the launcher's IAM policy restricts RunInstances on both
        # instance/* and volume/* with a Project=hrrrcast request-tag condition, so
        # an untagged volume makes the whole call fail with an opaque
        # UnauthorizedOperation. See aws/iam/lambda-launcher-policy.json.
        TagSpecifications=[
            {"ResourceType": rt, "Tags": [
                {"Key": "Name", "Value": f"hrrrcast-{init:%Y-%m-%dT%H}"},
                {"Key": "Project", "Value": PROJECT_TAG},
                {"Key": "InitTime", "Value": init.strftime("%Y-%m-%dT%H")},
                {"Key": "LeadHours", "Value": str(params["lead_hours"])},
                {"Key": "GfsMinLag", "Value": str(lag)},
                {"Key": "CodeRef", "Value": str(params.get("code_ref", "unknown"))},
                {"Key": "LaunchedBy", "Value": "eventbridge-lambda"},
            ]}
            for rt in ("instance", "volume")
        ],
    )
    if params.get("security_group_ids"):
        args["SecurityGroupIds"] = params["security_group_ids"].split()

    # Attempt 1: no subnet, so EC2 picks an AZ itself. This is the best single
    # attempt and is what run_on_ec2.sh does first.
    #
    # Then walk AZs on InsufficientInstanceCapacity. That error is routine for
    # on-demand GPU types and is PER-AZ: the first real launch through this Lambda
    # hit it, and earlier in the project g6e.2xlarge launched fine in one AZ and was
    # refused in every AZ forty minutes later. EC2's own choice is not exhaustive,
    # so an explicit walk finds capacity the implicit attempt misses. This mirrors
    # run_on_ec2.sh, which had the walk while this function did not.
    #
    # Note boto3 already retried the first call internally ("reached max retries: 4")
    # without changing AZ, which is why that retry does not help here.
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
                raise          # a real error: authorization, bad AMI, quota
            log.warning("no capacity in %s (%s)", az, code)
            last_err = e

    raise NoCapacity(
        f"no capacity for {params['instance_type']} in any of "
        f"{len(attempts)} placement attempts: {last_err}")


def _candidate_subnets(ec2, instance_type: str):
    """Default subnets in AZs that actually offer this instance type.

    Filtering by the offering list matters: this account has 6 default subnets but
    only 4 AZs offer g6e.2xlarge, so two of them would fail with `Unsupported`
    rather than a capacity error and waste attempts.
    """
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


# --- entry point -----------------------------------------------------------
def lambda_handler(event, context):
    if not BUCKET:
        raise RuntimeError("HRRRCAST_BUCKET is not set")

    session = boto3.Session()
    s3 = session.client("s3")
    params, template = _load_staged(s3)
    region = params["region"]
    ec2 = session.client("ec2", region_name=region)

    log.info("staged code %s (%s), lead=%sh, base lag=%sh",
             params.get("code_ref"), params.get("staged_at"),
             params["lead_hours"], params["gfs_min_lag"])

    busy = _in_flight(ec2)
    if busy:
        # Checked before cycle selection: if a run is already going there is nothing
        # to decide, and the probes would be wasted work.
        msg = f"{len(busy)} hrrrcast instance(s) still in flight: {busy}; skipping"
        log.warning(msg)
        return {"status": "skipped", "reason": msg}

    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    try:
        init, lag, cyc, n_gfs = choose_cycle(s3, params, now)
    except NothingToDo as e:
        log.info("nothing to do: %s", e)
        return {"status": "nothing_to_do", "reason": str(e)}

    if not _vcpu_headroom(session, region, params["instance_type"]):
        msg = "vCPU quota cannot fit another instance; skipping"
        log.warning(msg)
        return {"status": "skipped", "reason": msg}

    decision = {
        "init_time": init.strftime("%Y-%m-%dT%H"),
        "gfs_cycle": cyc.strftime("%Y-%m-%dT%H"),
        "gfs_min_lag": lag,
        "gfs_files": n_gfs,
        "lead_hours": params["lead_hours"],
        "code_ref": params.get("code_ref"),
    }

    if DRY_RUN:
        log.info("DRY RUN, would launch: %s", json.dumps(decision))
        return {"status": "dry_run", **decision}

    try:
        iid = launch(ec2, params, template, init, lag)
    except NoCapacity as e:
        # Not a failure of this function. The next hourly tick will try again, and
        # pick_cycle-style lookback means the missed hour can still be produced then
        # if it is within the lookback window.
        log.warning("skipping %s: %s", decision["init_time"], e)
        return {"status": "no_capacity", "reason": str(e), **decision}
    log.info("launched %s for %s (GFS %s, lag=%d)", iid, decision["init_time"],
             decision["gfs_cycle"], lag)
    return {"status": "launched", "instance_id": iid, **decision}
