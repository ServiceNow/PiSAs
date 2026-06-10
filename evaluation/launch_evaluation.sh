#!/bin/bash
# Launch evaluation across systems/models/privacy/memory combos.
# Derives the results folder from the same inputs used in launch_pipeline.sh.
# Judge and verifier LLMs have sensible defaults — no need to set them every time.
#
# Usage:
#   bash launch_evaluation.sh --model <MODEL> --systems decentralized,centralized,single
#     [--privacy-levels None,Medium,High]   (default: None)
#     [--memory-modes no,shared,private,both]  (default: no)
#     [--shared-memory-writer all|executor]    must match the pipeline run (default: all)
#     [--memory-cleanup]                        must match the pipeline run
#     [--run-index 0,1,2]   evaluate only these runs (default: all runs)
#     [--agent-audit]       enable per-agent knowledge audit
#     [--memory-audit]      enable memory violation checks
#     [--judge-llm MODEL]   override judge (default: google/gemini-2.5-pro)
#     [--api-key KEY]       OpenRouter key (default: $OPENROUTER_API_KEY)
#
# Examples:
#   bash launch_evaluation.sh --model anthropic/claude-sonnet-4-6 \
#     --systems centralized --privacy-levels High --memory-modes no,shared
#
#   bash launch_evaluation.sh --model anthropic/claude-sonnet-4-6 \
#     --systems centralized --privacy-levels High --memory-modes no --agent-audit
#
# Run from: the repository root (where this script lives).

set -e

MODEL=""
SYSTEMS=""
PRIVACY_LEVELS="None"
MEMORY_MODES="no"
RUN_INDEX="None"
AGENT_AUDIT=false
MEMORY_AUDIT=false
SHARED_MEMORY_WRITER="all"
MEMORY_CLEANUP=false
WORKERS=32
JUDGE_LLM="google/gemini-2.5-pro"
MEMORY_JUDGE_LLM="google/gemini-2.5-flash"
VERIFIER_1="anthropic/claude-haiku-4-5"
VERIFIER_2="openai/gpt-4o-mini"
VERIFIER_3="google/gemini-2.5-flash"
API_KEY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)            MODEL="$2";            shift 2 ;;
        --systems)          SYSTEMS="$2";          shift 2 ;;
        --privacy-levels)   PRIVACY_LEVELS="$2";   shift 2 ;;
        --memory-modes)     MEMORY_MODES="$2";     shift 2 ;;
        --run-index)        RUN_INDEX="$2";        shift 2 ;;
        --agent-audit)      AGENT_AUDIT=true;      shift ;;
        --memory-audit)     MEMORY_AUDIT=true;     shift ;;
        --shared-memory-writer) SHARED_MEMORY_WRITER="$2"; shift 2 ;;
        --memory-cleanup)   MEMORY_CLEANUP=true;   shift ;;
        --workers)          WORKERS="$2";          shift 2 ;;
        --judge-llm)        JUDGE_LLM="$2";        shift 2 ;;
        --memory-judge-llm) MEMORY_JUDGE_LLM="$2"; shift 2 ;;
        --api-key)          API_KEY="$2";          shift 2 ;;
        --results-dir)      RESULTS_DIR="$2";      shift 2 ;;
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
DATA_DIR="data"
RESULTS_DIR="${RESULTS_DIR:-results/paper/jira}"   # honor --results-dir if passed

MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | tr '.-' '__')

log()         { echo "[$(date '+%H:%M:%S')] $*"; }
log_section() { echo ""; echo "════════════════════════════════════════════════"; echo "  $*"; echo "════════════════════════════════════════════════"; }

LAUNCH_START=$(date +%s)
cd "$BASE_DIR"

IFS=',' read -ra SYSTEM_LIST  <<< "$SYSTEMS"
IFS=',' read -ra PRIVACY_LIST <<< "$PRIVACY_LEVELS"
IFS=',' read -ra MEMORY_LIST  <<< "$MEMORY_MODES"

mkdir -p results/logs

# Build optional flags
AUDIT_FLAGS=""
$AGENT_AUDIT  && AUDIT_FLAGS="$AUDIT_FLAGS --agent-audit"
$MEMORY_AUDIT && AUDIT_FLAGS="$AUDIT_FLAGS --memory-audit"

API_KEY_FLAG=""
[[ -n "$API_KEY" ]] && API_KEY_FLAG="--api-key $API_KEY"

RUN_INDEX_FLAG=""
[[ "$RUN_INDEX" != "None" ]] && RUN_INDEX_FLAG="--run-index $RUN_INDEX"

log_section "Launching evaluation grid | systems=$SYSTEMS | privacy=$PRIVACY_LEVELS | memory=$MEMORY_MODES | run-index=$RUN_INDEX"
log "Judge: $JUDGE_LLM | Agent audit: $AGENT_AUDIT | Memory audit: $MEMORY_AUDIT"

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
            # Mirror launch_pipeline.sh's folder naming so we locate the pipeline output:
            # the writer/cleanup tags must match what the pipeline run wrote.
            case "$SHARED_MEMORY_WRITER" in
                all)      WRITER_TAG="all" ;;
                executor) WRITER_TAG="exe" ;;
                *)        WRITER_TAG="${SHARED_MEMORY_WRITER}" ;;
            esac
            CLEANUP_TAG="nocl"
            $MEMORY_CLEANUP && CLEANUP_TAG="clean"
            OUTPUT_DIR="${RESULTS_DIR}/${SYSTEM}_${MODEL_SHORT}_${PRIVACY_LOWER}_${MEMORY_TAG}_${WRITER_TAG}_${CLEANUP_TAG}"
            LABEL="${SYSTEM}|${PRIVACY_LEVEL}|${MEMORY_MODE}"
            COMBO_LOG="${BASE_DIR}/results/logs/eval_${MODEL_SHORT}_${SYSTEM}_${PRIVACY_LOWER}_${MEMORY_TAG}.log"
            log "→ $LABEL  →  $OUTPUT_DIR"

            $PYTHON run_evaluation.py \
                --scenarios-folder "$DATA_DIR" \
                --results-path "$OUTPUT_DIR" \
                --judge-llm "$JUDGE_LLM" \
                --memory-judge-llm "$MEMORY_JUDGE_LLM" \
                --workers "$WORKERS" \
                --verifier-llm-1 "$VERIFIER_1" \
                --verifier-llm-2 "$VERIFIER_2" \
                --verifier-llm-3 "$VERIFIER_3" \
                $AUDIT_FLAGS \
                $API_KEY_FLAG \
                $MEMORY_FLAGS \
                $RUN_INDEX_FLAG \
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

ELAPSED=$(( $(date +%s) - LAUNCH_START ))
log_section "ALL DONE — $FAILED failed | $(( ELAPSED/3600 ))h $(( (ELAPSED%3600)/60 ))m $(( ELAPSED%60 ))s"
