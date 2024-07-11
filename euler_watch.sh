#!/usr/bin/env sh

set -e

output=$(./euler_run.sh $1)
job_id=$(echo $output | grep -oP 'Submitted batch job \K[0-9]+')

echo "Waiting for job $job_id to run..."
while true; do
  status=$(ssh euler.ethz.ch "squeue -j $job_id -h -o %T")
  if [ "$status" == "RUNNING" ]; then
    echo "Job $job_id is now running."
    break
  fi
  sleep 2
done

ssh euler.ethz.ch -t "cd formoniq; tail -f formoniq.stdout"
