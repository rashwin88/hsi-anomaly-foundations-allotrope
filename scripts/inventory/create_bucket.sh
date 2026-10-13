#!/usr/bin/env bash
# Create the raw-dataset inventory bucket in the Prod account. Idempotent.
# Usage: ./scripts/inventory/create_bucket.sh
set -euo pipefail

bucket="allotrope-raw-inventory-india"
aws=(aws --profile Allotrope-Prod --region ap-south-1)

if ! "${aws[@]}" s3api head-bucket --bucket "$bucket" 2>/dev/null; then
    "${aws[@]}" s3api create-bucket --bucket "$bucket" \
        --create-bucket-configuration LocationConstraint=ap-south-1
fi
"${aws[@]}" s3api put-bucket-versioning --bucket "$bucket" \
    --versioning-configuration Status=Enabled
"${aws[@]}" s3api put-public-access-block --bucket "$bucket" \
    --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
"${aws[@]}" s3api put-bucket-encryption --bucket "$bucket" \
    --server-side-encryption-configuration \
    '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"},"BucketKeyEnabled":true}]}'
# Everything goes to Intelligent-Tiering at once; abandoned multipart uploads
# (presigned uploads a client never finalised) are cleaned up after 7 days.
"${aws[@]}" s3api put-bucket-lifecycle-configuration --bucket "$bucket" \
    --lifecycle-configuration '{"Rules":[{"ID":"tiering-and-mpu-cleanup",
      "Status":"Enabled","Filter":{},
      "Transitions":[{"Days":0,"StorageClass":"INTELLIGENT_TIERING"}],
      "AbortIncompleteMultipartUpload":{"DaysAfterInitiation":7}}]}'
echo "ok: s3://$bucket"
