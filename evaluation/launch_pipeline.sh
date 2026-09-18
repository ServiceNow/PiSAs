#!/bin/bash
# Launch pipeline runs across systems/models/privacy/memory combos.
#   PYTHON          python interpreter to use (default: python3)
#
# Usage:
#   bash launch_pipeline.sh --model <MODEL> --systems decentralized,centralized,single
#     [--scenarios-folder PATH]                (default: data)
#     [--privacy-levels None,Medium,High]      (default: High)
#     [--memory-modes no,shared,private,both]  (default: no)
#     [--run-index 0,1,2]                      (default: 0)
#     [--api-key KEY]                          (default: $OPENROUTER_API_KEY)
#     [--coordinator-thinking]                 (TC only)
#
# Examples:
#   bash launch_pipeline.sh --model anthropic/claude-sonnet-4-6 \
#     --systems centralized --privacy-levels High --memory-modes no,shared --run-index 0,1,2
#
# Run from: the repository root (where this script lives).

set -e

MODEL=""
RUN_INDICES="0"
SYSTEMS=""
PRIVACY_LEVELS="High"
MEMORY_MODES="no"
API_KEY=""
COORDINATOR_THINKING=false
WORKERS=32
DATA_DIR="data"
RESULTS_DIR="results/PiSAs"
SHARED_MEMORY_WRITER="all"
MEMORY_CLEANUP=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)                MODEL="$2";                  shift 2 ;;
        --run-index)            RUN_INDICES="$2";            shift 2 ;;
        --memory-modes)         MEMORY_MODES="$2";           shift 2 ;;
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
PYTHON="${PYTHON:-python3}"
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | tr '.-' '__')

AGENT_LLM="$MODEL"

log()         { echo "[$(date '+%H:%M:%S')] $*"; }
log_section() { echo ""; echo "════════════════════════════════════════════════"; echo "  $*"; echo "════════════════════════════════════════════════"; }

LAUNCH_START=$(date +%s)

# ── Launch pipeline grid ───────────────────────────────────────────────────────
log_section "Launching pipeline grid | systems=$SYSTEMS | privacy=$PRIVACY_LEVELS | memory=$MEMORY_MODES | runs=$RUN_INDICES"

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

ELAPSED=$(( $(date +%s) - LAUNCH_START ))
log_section "ALL DONE — $FAILED failed | $(( ELAPSED/3600 ))h $(( (ELAPSED%3600)/60 ))m $(( ELAPSED%60 ))s"
