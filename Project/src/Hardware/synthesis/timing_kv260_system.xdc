# ==============================================================================
# File:        timing_kv260_system.xdc
# Description: System-Level Timing Constraints for MHA Accelerator (Variant 2A)
# Target:      AMD Kria KV260 Vision AI Starter Kit (xck26-sfvc784-2LV-c)
# Clock:       Generated automatically by Zynq UltraScale+ MPSoC pl_clk0 (200 MHz)
# ==============================================================================

# ==============================================================================
# Timing Exceptions: Quasi-Static Control Registers (Configured over AXI-Lite)
# These registers are programmed once by ARM PS driver before inference begins and
# remain completely static throughout active computation.
# ==============================================================================
set_false_path -from [get_cells -hierarchical *slv_seq_len_reg*]
set_false_path -from [get_cells -hierarchical *slv_head_dim_reg*]
set_false_path -from [get_cells -hierarchical *slv_scale_m_reg*]
set_false_path -from [get_cells -hierarchical *slv_scale_e_reg*]

