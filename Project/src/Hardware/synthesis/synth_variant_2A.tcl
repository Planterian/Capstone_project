# ==============================================================================
# Script:      synth_variant_2A.tcl
# Description: Vivado Synthesis & Implementation Script for Variant 2A
# Target:      AMD Kria KV260 (xck26-sfvc784-2LV-c)
# Clock:       200 MHz (5.0 ns period)
# Mode:        Out-of-Context (OOC) IP Synthesis
# ==============================================================================
set_param general.maxThreads 4

set TOP_NAME      attention_core_top_2A
set PART          xck26-sfvc784-2LV-c
set RTL_DIR       "../rtl_ver02_2A"
set OUT_DIR       "./reports_2A"

file mkdir $OUT_DIR

# Read SystemVerilog RTL Sources
read_verilog -sv "$RTL_DIR/axi_lite_regs.sv"
read_verilog -sv "$RTL_DIR/axis_adapter.sv"
read_verilog -sv "$RTL_DIR/ping_pong_bram_buffer.sv"
read_verilog -sv "$RTL_DIR/pe_slice_dsp_packing.sv"
read_verilog -sv "$RTL_DIR/dsp_packed_adder_tree.sv"
read_verilog -sv "$RTL_DIR/scale_unit.sv"
read_verilog -sv "$RTL_DIR/softmax_lut.sv"
read_verilog -sv "$RTL_DIR/score_v_gemm.sv"
read_verilog -sv "$RTL_DIR/attention_core_top_2A.sv"

# Read Timing Constraints
read_xdc "./timing_kv260.xdc"

# Out-of-Context Synthesis
puts "================================================================="
puts "  STARTING SYNTHESIS FOR VARIANT 2A (DSP PACKING)"
puts "================================================================="
synth_design -top $TOP_NAME -part $PART -mode out_of_context

# Write Post-Synthesis Reports
report_utilization -file "$OUT_DIR/post_synth_utilization.rpt"
report_timing_summary -file "$OUT_DIR/post_synth_timing.rpt"

# Implementation: Place and Route
puts "================================================================="
puts "  STARTING IMPLEMENTATION (PLACE & ROUTE) FOR VARIANT 2A"
puts "================================================================="
opt_design
place_design
phys_opt_design
route_design

# Write Post-Implementation Reports
report_utilization -hierarchical -file "$OUT_DIR/post_route_utilization_hier.rpt"
report_utilization -file "$OUT_DIR/post_route_utilization.rpt"
report_timing_summary -max_paths 10 -file "$OUT_DIR/post_route_timing.rpt"
report_power -file "$OUT_DIR/post_route_power.rpt"
report_drc -file "$OUT_DIR/post_route_drc.rpt"

puts "================================================================="
puts "  VARIANT 2A SYNTHESIS & IMPLEMENTATION COMPLETED SUCCESSFULLY!"
puts "================================================================="
exit
