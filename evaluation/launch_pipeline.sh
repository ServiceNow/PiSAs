#!/bin/bash
# Launch pipeline runs across systems/models/privacy/memory combos.
# For local models (e.g. gpt-oss), spins up an SGLang server on a GPU cluster first.
# The --local block targets a generic cluster job scheduler; override these env vars
# (or edit them) for your environment:
#   SCHED_CLI        scheduler CLI            (default: CHANGE_ME-sched)
#   SCHED_ACCOUNT    compute account
#   SCHED_DATA_MOUNT data volume to mount at /mnt/home
#   SGLANG_IMAGE    container image with SGLang
#   PYTHON          local python            (default: python3)
#   REMOTE_PYTHON   python inside the container (default: python)
#
# Usage:
#   bash launch_pipeline.sh --model <MODEL> --systems decentralized,centralized,single
#     [--scenarios-folder PATH]                (default: data)
#     [--privacy-levels None,Medium,High]      (default: High)
#     [--memory-modes no,shared,private,both]  (default: no)
#     [--run-index 0,1,2]                      (default: 0)
#     [--api-key KEY]                          (default: $OPENROUTER_API_KEY)
#     [--coordinator-thinking]                 (TC only)
#     [--local] [--port N] [--gpu N]
#
# Examples:
#   bash launch_pipeline.sh --model anthropic/claude-sonnet-4-6 \
#     --systems centralized --privacy-levels High --memory-modes no,shared --run-index 0,1,2
#
#   bash launch_pipeline.sh --model openai/gpt-oss-120b --local --gpu 2 \
#     --systems decentralized,centralized --privacy-levels None,High --run-index 0
#
# Run from: the repository root (where this script lives).

set -e

MODEL=""
PORT=8003
GPU=8
RUN_INDICES="0"
SYSTEMS=""
PRIVACY_LEVELS="High"
MEMORY_MODES="no"
IS_LOCAL=false
API_KEY=""
COORDINATOR_THINKING=false
WORKERS=32
DATA_DIR="data"
RESULTS_DIR="results/paper/jira"
SHARED_MEMORY_WRITER="all"
MEMORY_CLEANUP=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)                MODEL="$2";                  shift 2 ;;
        --port)                 PORT="$2";                   shift 2 ;;
        --gpu)                  GPU="$2";                    shift 2 ;;
        --run-index)            RUN_INDICES="$2";            shift 2 ;;
        --memory-modes)         MEMORY_MODES="$2";           shift 2 ;;
        --local)                IS_LOCAL=true;               shift ;;
        --systems)              SYSTEMS="$2";                shift 2 ;;
        --privacy-levels)       PRIVACY_LEVELS="$2";         shift 2 ;;
        --api-key)              API_KEY="$2";                shift 2 ;;
        --coordinator-thinking) COORDINATOR_THINKING=true;   shift ;;
        --scenarios-folder)     DATA_DIR="$2";               shift 2 ;;
        --results-dir)          RESULTS_DIR="$2";            shift 2 ;;
        --workers)              WORKERS="$2";                shift 2 ;;
        --shared-memory-writer) SHARED_MEMORY_WRITER="$2";  shift 2 ;;
        --memory-cleanup)       MEMORY_CLEANUP=true;         shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "ERROR: --model is required"; exit 1
fi
if [[ -z "$SYSTEMS" ]]; then
    echo "ERROR: --systems is required"; exit 1
fi

# ── Config ─────────────────────────────────────────────────────────────────────
SCHED="${SCHED_CLI:-CHANGE_ME-sched}"
ACCOUNT="${SCHED_ACCOUNT:-CHANGE_ME.account}"
SGLANG_PORT=$PORT
PYTHON="${PYTHON:-python3}"
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | tr '.-' '__')

if $IS_LOCAL; then
    AGENT_LLM="local/$(echo "$MODEL" | sed 's|.*/||')"
else
    AGENT_LLM="$MODEL"
fi

log()         { echo "[$(date '+%H:%M:%S')] $*"; }
log_section() { echo ""; echo "════════════════════════════════════════════════"; echo "  $*"; echo "════════════════════════════════════════════════"; }

LAUNCH_START=$(date +%s)

# ── SGLang setup (local models only) ──────────────────────────────────────────
if $IS_LOCAL; then
    REASONING_PARSER_FLAG=""
    [[ "$MODEL" == *"gpt-oss"* ]] && REASONING_PARSER_FLAG="--reasoning-parser gpt-oss"

    log_section "STEP 1/4 — Submitting SGLang server job (${GPU}x H100)"
    JOB_ID=$($SCHED job new \
        --account "$ACCOUNT" \
        --name "sglang_p${SGLANG_PORT}_$(date +%Y%m%d_%H%M%S)" \
        --cpu 32 --gpu $GPU --mem 640 --gpu-mem 80 --gpu-model-filter H100 \
        --image "${SGLANG_IMAGE:-CHANGE_ME/sglang-image}" \
        --data "${SCHED_DATA_MOUNT:-CHANGE_ME.data_mount}:/mnt/home:rw" \
        --env HOME=/mnt/home --env HF_HOME=/mnt/home/.cache/huggingface \
        --preemptable --restartable --tunnel \
        --fields id --no-header \
        -- bash -c "
            export HF_HOME=/mnt/home/.cache/huggingface
            exec ${REMOTE_PYTHON:-python} -m sglang.launch_server \
                --model-path $MODEL --host 0.0.0.0 --port $SGLANG_PORT \
                --tp $GPU --mem-fraction-static 0.85 $REASONING_PARSER_FLAG \
                --disable-cuda-graph --trust-remote-code
        ")
    log "Job submitted — ID: $JOB_ID"

    log_section "STEP 2/4 — Waiting for RUNNING state"
    for i in $(seq 1 60); do
        STATE=$($SCHED job get "$JOB_ID" --fields state --no-header 2>/dev/null)
        log "[$i/60] Job state: $STATE"
        [ "$STATE" = "RUNNING" ] && break
        sleep 15
    done
    [ "$STATE" != "RUNNING" ] && { log "ERROR: job not RUNNING after 15 min."; exit 1; }

    log_section "STEP 3/4 — Port-forward localhost:$SGLANG_PORT → job:$SGLANG_PORT"
    $SCHED job port-forward "$JOB_ID" "$SGLANG_PORT" &
    PF_PID=$!
    trap "kill $PF_PID 2>/dev/null; $SCHED job kill $JOB_ID > /dev/null 2>&1 || true" EXIT
    sleep 3

    log "Waiting for SGLang to be ready…"
    for i in $(seq 1 120); do
        curl -s "http://localhost:$SGLANG_PORT/v1/models" > /dev/null 2>&1 && break
        log "[$i/120] not ready — waiting 10s…"; sleep 10
    done
    curl -s "http://localhost:$SGLANG_PORT/v1/models" > /dev/null 2>&1 || { log "ERROR: SGLang not ready."; exit 1; }

    export LOCAL_MODEL_API_BASE="http://localhost:$SGLANG_PORT/v1"
    export LOCAL_MODEL_API_KEY="sk-local"
fi

# ── Launch pipeline grid ───────────────────────────────────────────────────────
log_section "STEP 4/4 — Launching pipeline grid | systems=$SYSTEMS | privacy=$PRIVACY_LEVELS | memory=$MEMORY_MODES | runs=$RUN_INDICES"

cd "$BASE_DIR"
IFS=',' read -ra SYSTEM_LIST  <<< "$SYSTEMS"
IFS=',' read -ra PRIVACY_LIST <<< "$PRIVACY_LEVELS"
IFS=',' read -ra MEMORY_LIST  <<< "$MEMORY_MODES"

mkdir -p results/logs

PIDS=()
COMBO_LABELS=()

for SYSTEM in "${SYSTEM_LIST[@]}"; do
    for PRIVACY_LEVEL in "${PRIVACY_LIST[@]}"; do
        for MEMORY_MODE in "${MEMORY_LIST[@]}"; do
            PRIVACY_LOWER=$(echo "$PRIVACY_LEVEL" | tr '[:upper:]' '[:lower:]')
            case "$MEMORY_MODE" in
                no)      MEMORY_TAG="memory_no";      MEMORY_FLAGS="" ;;
                shared)  MEMORY_TAG="memory_shared";  MEMORY_FLAGS="--shared-memory" ;;
                private) MEMORY_TAG="memory_private"; MEMORY_FLAGS="--private-memory" ;;
                both)    MEMORY_TAG="memory_both";    MEMORY_FLAGS="--shared-memory --private-memory" ;;
                *) echo "Unknown memory mode: $MEMORY_MODE"; exit 1 ;;
            esac
            # Encode write access and cleanup in folder name
            case "$SHARED_MEMORY_WRITER" in
                all)      WRITER_TAG="all" ;;
                executor) WRITER_TAG="exe" ;;
                *)        WRITER_TAG="${SHARED_MEMORY_WRITER}" ;;
            esac
            CLEANUP_TAG="nocl"
            $MEMORY_CLEANUP && CLEANUP_TAG="clean"
            OUTPUT_DIR="${RESULTS_DIR}/${SYSTEM}_${MODEL_SHORT}_${PRIVACY_LOWER}_${MEMORY_TAG}_${WRITER_TAG}_${CLEANUP_TAG}"
            LABEL="${SYSTEM}|${PRIVACY_LEVEL}|${MEMORY_MODE}|writer=${SHARED_MEMORY_WRITER}|cleanup=${MEMORY_CLEANUP}"
            COMBO_LOG="${BASE_DIR}/results/logs/pipeline_${MODEL_SHORT}_${SYSTEM}_${PRIVACY_LOWER}_${MEMORY_TAG}_${WRITER_TAG}_${CLEANUP_TAG}.log"
            log "→ $LABEL  →  $OUTPUT_DIR"

            API_KEY_FLAG=""
            [[ -n "$API_KEY" ]] && API_KEY_FLAG="--api-key $API_KEY"
            COORD_THINKING_FLAG=""
            $COORDINATOR_THINKING && COORD_THINKING_FLAG="--coordinator-thinking"
            CLEANUP_FLAG=""
            $MEMORY_CLEANUP && CLEANUP_FLAG="--memory-cleanup"

            $PYTHON run_pipeline.py \
                --scenarios-folder "$DATA_DIR" \
                --results-path "$OUTPUT_DIR" \
                -s $SYSTEM \
                --agent-llm $AGENT_LLM \
                $MEMORY_FLAGS \
                --run-index "$RUN_INDICES" \
                --privacy-level $PRIVACY_LEVEL \
                --max-rounds 30 \
                --workers "$WORKERS" \
                --shared-memory-writer "$SHARED_MEMORY_WRITER" \
                $CLEANUP_FLAG \
                $API_KEY_FLAG \
                $COORD_THINKING_FLAG \
                > "$COMBO_LOG" 2>&1 &

            PIDS+=($!)
            COMBO_LABELS+=("$LABEL")
        done
    done
done

log "Waiting for ${#PIDS[@]} combo(s) to finish…"
FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        log "✓ ${COMBO_LABELS[$i]}"
    else
        log "✗ FAILED: ${COMBO_LABELS[$i]}"
        FAILED=$(( FAILED + 1 ))
    fi
done

[[ $FAILED -gt 0 ]] && log "WARNING: $FAILED combo(s) failed"

if $IS_LOCAL; then
    log "Killing SGLang job $JOB_ID…"
    $SCHED job kill "$JOB_ID" > /dev/null 2>&1 && log "Job killed." || log "WARNING: could not kill job $JOB_ID"
fi

ELAPSED=$(( $(date +%s) - LAUNCH_START ))
log_section "ALL DONE — $FAILED failed | $(( ELAPSED/3600 ))h $(( (ELAPSED%3600)/60 ))m $(( ELAPSED%60 ))s"
