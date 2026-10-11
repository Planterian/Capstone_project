// ============================================================================
// AUTO-GENERATED HARDWARE SCALE PARAMETERS FOR KRIA KV260 ATTENTION CORE
// Generated from MobileViT + I-ViT QAT Model (Accuracy: 67.28%)
// Remediation Step 1: Bit-Exact Alignment with PyTorch I-ViT
// ============================================================================

`ifndef HW_SCALE_PARAMS_SVH
`define HW_SCALE_PARAMS_SVH

// STAGE2: Head Dim = 16, Effective Scale = 0.006590
`define STAGE2_HEAD_DIM       16'd16
`define STAGE2_SCALE_M        16'd27640
`define STAGE2_SCALE_E        5'd22

// STAGE3: Head Dim = 20, Effective Scale = 0.005230
`define STAGE3_HEAD_DIM       16'd20
`define STAGE3_SCALE_M        16'd21936
`define STAGE3_SCALE_E        5'd22

// STAGE4: Head Dim = 24, Effective Scale = 0.004485
`define STAGE4_HEAD_DIM       16'd24
`define STAGE4_SCALE_M        16'd18810
`define STAGE4_SCALE_E        5'd22

// DEFAULT SETTINGS (MobileViT-XXS Stage 2)
`define DEFAULT_SEQ_LEN       16'd256
`define DEFAULT_HEAD_DIM      16'd16
`define DEFAULT_NUM_HEADS     8'd4
`define DEFAULT_SCALE_M       16'd27640
`define DEFAULT_SCALE_E       5'd22

`endif // HW_SCALE_PARAMS_SVH
