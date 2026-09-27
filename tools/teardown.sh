#!/bin/sh
# Remove every Blindspot resource. Touches only the stack named Blindspot and
# the bucket it created, using the dedicated profile.
set -eu
PROFILE="${AWS_PROFILE:-blindspot}"
BUCKET=$(aws --profile "$PROFILE" cloudformation describe-stacks --stack-name Blindspot \
  --query "Stacks[0].Outputs[?OutputKey=='BucketName'].OutputValue" --output text)
[ -n "$BUCKET" ] && aws --profile "$PROFILE" s3 rm "s3://$BUCKET" --recursive
cd "$(dirname "$0")/../infra" && npx cdk destroy --profile "$PROFILE" --force Blindspot
