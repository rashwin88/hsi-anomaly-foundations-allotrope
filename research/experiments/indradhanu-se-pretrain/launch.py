"""
Launch one run of the SpectralEarth pretraining experiment on a g6e (one L40S).

    python launch.py E003-cross-attention c002970 [--wait-shards]

Uploads <run>/config.json to the run's S3 folder, then starts the instance with bootstrap.sh
as user-data (RUN and SHA substituted). With --wait-shards it first polls the prod box until the
sharding run has finished, and refuses to launch unless the manifest and stats.json are complete.
Every launch is approved by a human first (feedback: GPU starts are gated).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import boto3

HERE = Path(__file__).resolve().parent
REGION, PROFILE = "ap-south-1", "Allotrope-Prod"
BUCKET = "allotrope-training-data-prod-india"
PROD_BOX = "i-07f79c11c54120f63"
AMI = "ami-0eb5a4f0d81f671e3"                 # DLAMI OSS Nvidia, PyTorch 2.7, Ubuntu 22.04, 2026-04-27
# On 2026-10-14 no zone had g6e capacity, so the launcher works down a list: the L40S sizes
# first, then A10G (~half the L40S speed for C) and L4 (~a quarter). The run config - batch 4 x 8
# accumulation, 200 shards - fits every one: 22 GB cards hold 4 patches of C in bf16, and the
# smallest local disk is 250 GB. g6e.4xlarge needs the G quota raised to 16 vCPU.
INSTANCE_TYPES = [
    "g6e.xlarge",    # L40S 48 GB   $2.24/h
    "g6e.2xlarge",   # L40S 48 GB   $2.69/h
    "g6e.4xlarge",   # L40S 48 GB   $3.61/h
    "g5.2xlarge",    # A10G 24 GB   $1.46/h
    "g6.2xlarge",    # L4   24 GB   $1.17/h
]
SUBNETS = {"ap-south-1b": "subnet-0c7adf2426dc24f75", "ap-south-1a": "subnet-02f12cb4d1db68dc4"}
SG, PROFILE_NAME = "sg-00ebc962aa99f6810", "allotrope-gpu-train"
EXPECTED = {"train": 3419, "test": 180}       # ceil(512,849 / 150), ceil(26,906 / 150)


def wait_for_shards(session) -> None:
    ssm = session.client("ssm")
    while True:
        cid = ssm.send_command(InstanceIds=[PROD_BOX], DocumentName="AWS-RunShellScript",
                               Parameters={"commands": ["cat /var/log/se-shard/DONE 2>/dev/null || echo running"]})["Command"]["CommandId"]
        time.sleep(10)
        out = ssm.get_command_invocation(CommandId=cid, InstanceId=PROD_BOX)["StandardOutputContent"].strip()
        print(time.strftime("%H:%M"), "sharding:", out.splitlines()[0] if out else "?", flush=True)
        if out.startswith("ok"):
            return
        if out.startswith("failed"):
            sys.exit("sharding finalize failed - not launching")
        time.sleep(110)


def check_manifest(s3) -> None:
    prefix = "patches/spectralearth/se-v1"
    m = json.loads(s3.get_object(Bucket=BUCKET, Key=f"{prefix}/manifest.json")["Body"].read())
    got = {k: len(v["shards"]) for k, v in m["splits"].items()}
    stats = json.loads(s3.get_object(Bucket=BUCKET, Key=f"{prefix}/stats.json")["Body"].read())
    print("manifest:", got, "| patches:", {k: v["patches"] for k, v in m["splits"].items()},
          "| stats bands:", len(stats["mean"]))
    if got != EXPECTED or len(stats["mean"]) != 202:
        sys.exit(f"manifest/stats incomplete (expected {EXPECTED}, 202 bands) - not launching")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("sha")
    ap.add_argument("--wait-shards", action="store_true")
    ap.add_argument("--spot", action="store_true", help="one-time spot; terminates on interruption, relaunch resumes from S3")
    ap.add_argument("--zones", default=",".join(SUBNETS), help="comma list, tried in order")
    ap.add_argument("--types", default=",".join(INSTANCE_TYPES), help="comma list, tried in order")
    args = ap.parse_args()
    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    s3, ec2 = session.client("s3"), session.client("ec2")
    if args.wait_shards:
        wait_for_shards(session)
    check_manifest(s3)
    s3.upload_file(str(HERE / args.run / "config.json"), BUCKET,
                   f"experiments/indradhanu-se-pretrain/{args.run}/config.json")
    user_data = (HERE / "bootstrap.sh").read_text(encoding="utf-8").replace("__RUN__", args.run).replace("__SHA__", args.sha)
    root = ec2.describe_images(ImageIds=[AMI])["Images"][0]["RootDeviceName"]
    attempts = [(it, az, SUBNETS[az]) for it in args.types.split(",") for az in args.zones.split(",")]
    for itype, az, subnet in attempts:
        try:
            r = launch(ec2, itype, subnet, root, user_data, args)
            break
        except ec2.exceptions.ClientError as e:
            # No capacity, or (4xlarge) over the vCPU quota: try the next option.
            if not any(s in str(e) for s in ("InsufficientInstanceCapacity", "VcpuLimitExceeded", "MaxSpotInstanceCountExceeded", "SpotMaxPriceTooLow")):
                raise
            print(f"{itype} {az}: {e.response['Error']['Code']}", flush=True)
    else:
        sys.exit("no capacity in any zone - not launched")
    iid = r["Instances"][0]["InstanceId"]
    print(f"LAUNCHED {iid} ({itype}) for {args.run} in {az} at {time.strftime('%H:%M')}", flush=True)


def launch(ec2, itype, subnet, root, user_data, args):
    return ec2.run_instances(
        ImageId=AMI, InstanceType=itype, MinCount=1, MaxCount=1, SubnetId=subnet,
        SecurityGroupIds=[SG], IamInstanceProfile={"Name": PROFILE_NAME}, UserData=user_data,
        InstanceInitiatedShutdownBehavior="terminate" if args.spot else "stop",
        **({"InstanceMarketOptions": {"MarketType": "spot", "SpotOptions": {
            "SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate"}}} if args.spot else {}),
        BlockDeviceMappings=[{"DeviceName": root, "Ebs": {"VolumeSize": 100, "VolumeType": "gp3", "DeleteOnTermination": True}}],
        MetadataOptions={"HttpTokens": "required", "HttpPutResponseHopLimit": 2},
        TagSpecifications=[{"ResourceType": "instance", "Tags": [
            {"Key": "Name", "Value": f"allotrope-{args.run}"}, {"Key": "project", "Value": "indradhanu-se-pretrain"},
            {"Key": "run", "Value": args.run}, {"Key": "code", "Value": args.sha},
            {"Key": "market", "Value": "spot" if args.spot else "on-demand"}]}])


if __name__ == "__main__":
    main()
