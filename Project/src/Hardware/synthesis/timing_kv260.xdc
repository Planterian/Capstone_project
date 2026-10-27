# ==============================================================================
# File:        timing_kv260.xdc
# Description: Timing Constraints for MobileViT-XXS Multi-Stage Attention Core
# Target:      AMD Kria KV260 Vision AI Starter Kit (xck26-sfvc784-2LV-c)
# Clock Spec:  Target Freq = 200 MHz (Clock Period = 5.000 ns)
# ==============================================================================

# Define primary clock at 200 MHz
create_clock -period 5.000 -name clk -waveform {0.000 2.500} [get_ports clk]

# Clock Uncertainty / Jitter Margin
set_clock_uncertainty 0.100 [get_clocks clk]

# Out-Of-Context (OOC) Mode: No board I/O pin delays needed as this is an IP core module
