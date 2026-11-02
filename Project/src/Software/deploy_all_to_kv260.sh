#!/bin/bash
# =============================================================================
# Deployment Script: Sync Entire 4-Phase SW-HW Evaluation Suite to Kria KV260
# =============================================================================
set -e

JUMP_HOST="ktmt@100.124.168.26"
TARGET_BOARD="ubuntu@192.168.1.179"
REMOTE_DIR="/home/ubuntu/mha_eval"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "================================================================================"
echo "    DEPLOYING 4-PHASE EVALUATION SUITE TO STANDALONE KRIA KV260"
echo "================================================================================"
echo "[INFO] Jump Host   : $JUMP_HOST"
echo "[INFO] Target Board: $TARGET_BOARD"
echo "[INFO] Remote Dir  : $REMOTE_DIR"
echo "--------------------------------------------------------------------------------"

# 1. Ensure remote target directory exists
echo "[Step 1/2] Creating remote directory $REMOTE_DIR on KV260..."
ssh -J "$JUMP_HOST" -o StrictHostKeyChecking=accept-new "$TARGET_BOARD" "mkdir -p $REMOTE_DIR"

# 2. Synchronize directories using rsync or scp
echo "[Step 2/2] Synchronizing software modules..."
if command -v rsync > /dev/null 2>&1; then
    rsync -avz -e "ssh -J $JUMP_HOST" \
        --exclude="__pycache__" \
        --exclude="*.git*" \
        "$SCRIPT_DIR/" "$TARGET_BOARD:$REMOTE_DIR/"
else
    # SCP recursive fallback
    scp -J "$JUMP_HOST" -r \
        "$SCRIPT_DIR/00_common" \
        "$SCRIPT_DIR/01_baseline_mobilevit" \
        "$SCRIPT_DIR/02_ivit_sw_mobilevit" \
        "$SCRIPT_DIR/03_ivit_hw_accelerated" \
        "$SCRIPT_DIR/04_all_9_transformers_hw" \
        "$SCRIPT_DIR/markdown" \
        "$SCRIPT_DIR/run_all_phases_kv260.sh" \
        "$TARGET_BOARD:$REMOTE_DIR/"
fi

# 3. Ensure executable permissions on all shell scripts on remote board
ssh -J "$JUMP_HOST" "$TARGET_BOARD" "chmod +x $REMOTE_DIR/*.sh $REMOTE_DIR/*/*.sh 2>/dev/null || true"

echo ""
echo "================================================================================"
echo "[SUCCESS] All files deployed! To run the complete 4-phase benchmark on KV260:"
echo ""
echo "  1. SSH to KV260:"
echo "     ssh -J $JUMP_HOST $TARGET_BOARD"
echo ""
echo "  2. Execute Master Benchmark:"
echo "     cd $REMOTE_DIR"
echo "     ./run_all_phases_kv260.sh"
echo "================================================================================"
