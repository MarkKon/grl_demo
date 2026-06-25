#!/usr/bin/env bash
set -euo pipefail

MODE="plan"
SUBMIT=0
EPOCHS=20
BATCH_SIZE=32
EVAL_BATCH_SIZE=256
SEED=0
RANDOM_REPEATS=1
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RESULTS_ROOT="results/ablations"
SLURM_HEADER=""
DEVICE=""

usage() {
  cat <<'EOF'
Usage: ./scripts/run_ablations.sh [options]

Modes:
  --mode plan      Write readable task tables only. Default.
  --mode local     Run one config job at a time on this machine.
  --mode slurm     Write one Slurm array with one job per config.
  --submit         With --mode slurm, submit config and report jobs with sbatch.

Options:
  --run-id ID
  --results-root DIR
  --epochs N
  --batch-size N
  --eval-batch-size N
  --seed N
  --random-repeats N
  --device DEVICE
  --slurm-header FILE   Optional cluster-specific #SBATCH settings for config jobs.

Output layout:
  results/ablations/<run-id>/
    README.md
    config_tasks.tsv
    slurm/config_array.slurm
    slurm/report.slurm
    logs/
    metrics/
    tables/
    plots/
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --submit)
      SUBMIT=1
      shift
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --results-root)
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --eval-batch-size)
      EVAL_BATCH_SIZE="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --random-repeats)
      RANDOM_REPEATS="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --slurm-header)
      SLURM_HEADER="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$MODE" != "plan" && "$MODE" != "local" && "$MODE" != "slurm" ]]; then
  echo "--mode must be one of: plan, local, slurm" >&2
  exit 2
fi
if [[ "$SUBMIT" -eq 1 && "$MODE" != "slurm" ]]; then
  echo "--submit only makes sense with --mode slurm" >&2
  exit 2
fi

RUN_DIR="${RESULTS_ROOT}/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
METRIC_DIR="${RUN_DIR}/metrics"
TABLE_DIR="${RUN_DIR}/tables"
PLOT_DIR="${RUN_DIR}/plots"
SLURM_DIR="${RUN_DIR}/slurm"
TMP_DIR="${RUN_DIR}/tmp"
README="${RUN_DIR}/README.md"
CONFIG_TASKS="${RUN_DIR}/config_tasks.tsv"

mkdir -p "$LOG_DIR" "$METRIC_DIR" "$TABLE_DIR" "$PLOT_DIR" "$SLURM_DIR" "$TMP_DIR" checkpoints/ablations
: > "$CONFIG_TASKS"
printf 'array_index\tmethod\tmetric_csv\tcommand\n' >> "$CONFIG_TASKS"

{
  echo "# GRL Ablation Run"
  echo
  echo "- mode: \`$MODE\`"
  echo "- run directory: \`$RUN_DIR\`"
  echo "- epochs: \`$EPOCHS\`"
  echo "- batch size: \`$BATCH_SIZE\`"
  echo "- eval batch size: \`$EVAL_BATCH_SIZE\`"
  echo "- seed: \`$SEED\`"
  echo "- random repeats: \`$RANDOM_REPEATS\`"
  if [[ -n "$DEVICE" ]]; then
    echo "- device: \`$DEVICE\`"
  fi
  echo
  echo "## Config Array"
  echo
  echo "| index | method | metric CSV |"
  echo "| --- | --- | --- |"
} > "$README"

shell_join() {
  printf '%q ' "$@"
}

run_local_task() {
  local method="$1"
  shift
  local stdout_path="${LOG_DIR}/${method}.out"
  local stderr_path="${LOG_DIR}/${method}.err"
  echo "+ [$method] $(shell_join "$@")"
  "$@" > "$stdout_path" 2> "$stderr_path"
  sed -n '1,8p' "$stdout_path"
}

append_config_task() {
  local index="$1"
  local method="$2"
  local metric_csv="$3"
  shift 3
  local command
  command="$(shell_join "$@")"
  printf '%s\t%s\t%s\t%s\n' "$index" "$method" "$metric_csv" "$command" >> "$CONFIG_TASKS"
  printf '| %s | `%s` | `%s` |\n' "$index" "$method" "$metric_csv" >> "$README"

  if [[ "$MODE" == "local" ]]; then
    run_local_task "$method" "$@"
  fi
}

METHODS=()
while IFS= read -r method; do
  METHODS+=("$method")
done < <(uv run python scripts/run_ablation_config.py --list-methods)
INDEX=0
for method in "${METHODS[@]}"; do
  INDEX=$((INDEX + 1))
  metric_csv="${METRIC_DIR}/${method}.csv"
  command=(
    uv run python scripts/run_ablation_config.py "$method"
      --output "$metric_csv"
      --epochs "$EPOCHS"
      --batch-size "$BATCH_SIZE"
      --eval-batch-size "$EVAL_BATCH_SIZE"
      --seed "$SEED"
  )
  if [[ "$method" == "random" ]]; then
    command+=(--random-repeats "$RANDOM_REPEATS")
  fi
  if [[ -n "$DEVICE" ]]; then
    command+=(--device "$DEVICE")
  fi
  append_config_task "$INDEX" "$method" "$metric_csv" "${command[@]}"
done

write_array_script() {
  local script_path="${SLURM_DIR}/config_array.slurm"
  {
    echo "#!/usr/bin/env bash"
    echo "#SBATCH --job-name=grl_${RUN_ID}_config"
    echo "#SBATCH --array=1-${INDEX}"
    echo "#SBATCH --output=${LOG_DIR}/config_%A_%a.out"
    echo "#SBATCH --error=${LOG_DIR}/config_%A_%a.err"
    echo "#SBATCH --chdir=$(pwd)"
    if [[ -n "$SLURM_HEADER" ]]; then
      echo
      cat "$SLURM_HEADER"
    fi
    echo
    echo "set -euo pipefail"
    echo "echo \"=========================================\""
    echo "echo \"Job ID: \${SLURM_JOB_ID}\""
    echo "echo \"Array Task ID: \${SLURM_ARRAY_TASK_ID}\""
    echo "echo \"Job Name: \${SLURM_JOB_NAME}\""
    echo "echo \"Node: \${SLURMD_NODENAME}\""
    echo "echo \"=========================================\""
    echo "export TMPDIR='${TMP_DIR}'"
    echo "mkdir -p \"\$TMPDIR\""
    echo "TASK_FILE='${CONFIG_TASKS}'"
    echo "CMD=\$(awk -F '\\t' -v i=\"\$SLURM_ARRAY_TASK_ID\" 'NR > 1 && \$1 == i {print \$NF}' \"\$TASK_FILE\")"
    echo "if [[ -z \"\$CMD\" ]]; then echo \"no command for array index \$SLURM_ARRAY_TASK_ID\" >&2; exit 2; fi"
    echo "echo \"+ \$CMD\""
    echo "eval \"\$CMD\""
  } > "$script_path"
  chmod +x "$script_path"
}

write_report_script() {
  local script_path="${SLURM_DIR}/report.slurm"
  {
    echo "#!/usr/bin/env bash"
    echo "#SBATCH --job-name=grl_${RUN_ID}_report"
    echo "#SBATCH --output=${LOG_DIR}/report_%j.out"
    echo "#SBATCH --error=${LOG_DIR}/report_%j.err"
    echo "#SBATCH --chdir=$(pwd)"
    echo "#SBATCH --nodes=1"
    echo "#SBATCH --ntasks-per-node=1"
    echo "#SBATCH --cpus-per-task=1"
    echo "#SBATCH --mem=4G"
    echo "#SBATCH --time=00:30:00"
    echo
    echo "set -euo pipefail"
    echo "echo \"=========================================\""
    echo "echo \"Job ID: \${SLURM_JOB_ID}\""
    echo "echo \"Job Name: \${SLURM_JOB_NAME}\""
    echo "echo \"Node: \${SLURMD_NODENAME}\""
    echo "echo \"=========================================\""
    echo "export TMPDIR='${TMP_DIR}'"
    echo "mkdir -p \"\$TMPDIR\""
    shell_join uv run python scripts/summarize_ablations.py "$RUN_DIR"
    echo
  } > "$script_path"
  chmod +x "$script_path"
}

if [[ "$MODE" == "local" ]]; then
  uv run python scripts/summarize_ablations.py "$RUN_DIR"
fi

if [[ "$MODE" == "slurm" ]]; then
  write_array_script
  write_report_script
  if [[ "$SUBMIT" -eq 1 ]]; then
    config_id="$(sbatch --parsable "${SLURM_DIR}/config_array.slurm")"
    report_id="$(sbatch --parsable --dependency=afterok:"$config_id" "${SLURM_DIR}/report.slurm")"
    echo "submitted config array: $config_id"
    echo "submitted report job:   $report_id"
  fi
fi

{
  echo
  echo "## Output Files"
  echo
  echo "- config task table: \`$CONFIG_TASKS\`"
  echo "- logs: \`$LOG_DIR\`"
  echo "- per-config metric CSV files: \`$METRIC_DIR\`"
  echo "- generated tables: \`$TABLE_DIR\`"
  echo "- generated plots: \`$PLOT_DIR\`"
  if [[ "$MODE" == "slurm" ]]; then
    echo "- Slurm scripts: \`$SLURM_DIR\`"
  fi
} >> "$README"

echo
echo "run directory: $RUN_DIR"
echo "readable overview: $README"
echo "config array tasks: $CONFIG_TASKS"
if [[ "$MODE" == "slurm" ]]; then
  echo "slurm scripts:      $SLURM_DIR"
  if [[ "$SUBMIT" -eq 0 ]]; then
    echo "not submitted; rerun with --mode slurm --submit after cluster settings are set"
  fi
fi
