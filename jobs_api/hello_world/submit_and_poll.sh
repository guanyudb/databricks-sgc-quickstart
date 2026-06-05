#!/usr/bin/env bash
# Submit a Databricks notebook to SGC via the Jobs API runs/submit endpoint,
# poll until done, then read the notebook output.
#
# Usage:
#   ./submit_and_poll.sh                       # uses $DATABRICKS_CONFIG_PROFILE or DEFAULT
#   DATABRICKS_PROFILE=MY_PROFILE ./submit_and_poll.sh
#
# Customize the compute by editing submit.json:
#   "hardware_accelerator": "GPU_1xA10"   # or GPU_1xH100 (Beta) / GPU_8xH100
#   "spec": {"environment_version": "5"}  # Standard v5 (no torch preinstalled — the notebook installs it)
#                                          # or {"base_environment": "databricks_ai_v5"} for the bundled ML stack

set -euo pipefail

PROFILE="${DATABRICKS_PROFILE:-${DATABRICKS_CONFIG_PROFILE:-DEFAULT}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ">>> Using Databricks profile: $PROFILE"

# 1) Resolve the current user's email — used to namespace the workspace notebook upload path
EMAIL="$(databricks -p "$PROFILE" current-user me --output json | jq -r '.userName')"
WORKSPACE_DIR="/Workspace/Users/$EMAIL/sgc_hello_world_jobs_api"
echo ">>> Logged in as $EMAIL"
echo ">>> Workspace notebook path: $WORKSPACE_DIR/gpu_notebook"

# 2) Upload the notebook (idempotent — --overwrite re-uploads if it exists)
databricks -p "$PROFILE" workspace import \
  "$WORKSPACE_DIR/gpu_notebook" \
  --file gpu_notebook.py \
  --language PYTHON \
  --format SOURCE \
  --overwrite
echo ">>> Notebook uploaded"

# 3) Expand <YOUR_EMAIL> in submit.json and submit
sed "s|<YOUR_EMAIL>|$EMAIL|g" submit.json > /tmp/sgc_hello_world_submit.json
RUN_RESPONSE=$(databricks -p "$PROFILE" api post /api/2.1/jobs/runs/submit \
  --json @/tmp/sgc_hello_world_submit.json)
RUN_ID=$(echo "$RUN_RESPONSE" | jq -r '.run_id')
echo ">>> Submitted run_id=$RUN_ID"
echo ">>> Run page: $(databricks -p $PROFILE auth describe --output json 2>/dev/null | jq -r '.host')#job/runs/$RUN_ID"

# 4) Poll until terminal state
echo ">>> Polling (provisioning can take 1-4 min for SGC)..."
STATE_JSON=""
while :; do
  STATE_JSON=$(databricks -p "$PROFILE" api get /api/2.1/jobs/runs/get \
    --json "{\"run_id\": $RUN_ID}")
  LIFE=$(echo "$STATE_JSON" | jq -r '.state.life_cycle_state // "UNKNOWN"')
  RESULT=$(echo "$STATE_JSON" | jq -r '.state.result_state // empty')
  echo "    [$(date +%H:%M:%S)] life_cycle_state=$LIFE  result_state=${RESULT:-<pending>}"
  case "$LIFE" in
    TERMINATED|INTERNAL_ERROR|SKIPPED) break ;;
  esac
  sleep 15
done

# 5) Read the notebook's exit() payload
#    IMPORTANT: get-output requires the TASK run_id (tasks[0].run_id), NOT the top-level job run_id.
TASK_RUN_ID=$(echo "$STATE_JSON" | jq -r '.tasks[0].run_id')
echo ">>> Task run_id=$TASK_RUN_ID (used for get-output)"

databricks -p "$PROFILE" api get /api/2.1/jobs/runs/get-output \
  --json "{\"run_id\": $TASK_RUN_ID}" \
  | jq '{
      notebook_output: .notebook_output,
      error: .error,
      error_trace: .error_trace
    }'

# 6) Exit code reflects job success
FINAL_RESULT=$(echo "$STATE_JSON" | jq -r '.state.result_state // empty')
case "$FINAL_RESULT" in
  SUCCESS) echo ">>> ✓ Run succeeded"; exit 0 ;;
  *)       echo ">>> ✗ Run did not succeed (result_state=$FINAL_RESULT). Check the run page for driver logs."; exit 1 ;;
esac
