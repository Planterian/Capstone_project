# ==============================================================================
# Script:      build_kv260_mha_2A.tcl
# Description: Automated Vivado SoC Project Generator & Bitstream Builder
#              for Variant 2A MHA Accelerator on AMD Kria KV260
# Target Part: xck26-sfvc784-2LV-c
# Vivado Tool: AMD Vivado 2026.1 / 2024.x / 2022.2+ (Batch Mode)
# ==============================================================================

# Use 4 threads to prevent laptop thermal throttling or freezing
set_param general.maxThreads 2

set PROJ_NAME   "mha_kv260_deploy"
set PROJ_DIR    "./vivado_prj_2A"
set PART_NAME   "xck26-sfvc784-2LV-c"
set RTL_DIR     "../rtl_ver02_2A"
set SYNTH_DIR   "."
set OUT_DIR     "../../Software"

# Helper procedure: Query active Vivado IP catalog for latest IP version
proc get_latest_ip_vlnv {ip_name default_vlnv} {
    set ip_list [get_ipdefs -all -filter "NAME == $ip_name"]
    if {[llength $ip_list] > 0} {
        return [lindex $ip_list end]
    }
    return $default_vlnv
}

puts "================================================================="
puts "  ==> STEP 1/7: INITIALIZING VIVADO PROJECT: $PROJ_NAME"
puts "================================================================="
file mkdir $OUT_DIR

# Close any currently opened project to prevent 'Project already open' error
catch { close_project }

create_project -force $PROJ_NAME $PROJ_DIR -part $PART_NAME

puts "================================================================="
puts "  ==> STEP 2/7: IMPORTING SYSTEMVERILOG RTL SOURCES & EXP ROM HEX"
puts "================================================================="
# Separate synthesizable design RTL and simulation testbenches
set all_sv   [glob -nocomplain "$RTL_DIR/*.sv"]
set all_svh  [glob -nocomplain "$RTL_DIR/*.svh"]
set synth_sv {}
set sim_sv   {}

foreach f $all_sv {
    if {[string match "*tb_*" [file tail $f]]} {
        lappend sim_sv $f
    } else {
        lappend synth_sv $f
    }
}

if {[llength $synth_sv] > 0} {
    add_files -fileset sources_1 $synth_sv
    puts "  -> Added [llength $synth_sv] SystemVerilog design files to sources_1"
}
if {[llength $all_svh] > 0} {
    add_files -fileset sources_1 $all_svh
    puts "  -> Added [llength $all_svh] header files to sources_1"
}
if {[llength $sim_sv] > 0} {
    add_files -fileset sim_1 $sim_sv
    puts "  -> Added [llength $sim_sv] testbench files to sim_1"
}

# Softmax LUT memory initialization file
if {[file exists "$RTL_DIR/exp_rom_all_stages.hex"]} {
    add_files -fileset sources_1 "$RTL_DIR/exp_rom_all_stages.hex"
    set_property file_type "Memory Initialization Files" [get_files "$RTL_DIR/exp_rom_all_stages.hex"]
    file copy -force "$RTL_DIR/exp_rom_all_stages.hex" "$PROJ_DIR/exp_rom_all_stages.hex"
    puts "  -> Successfully registered and staged exp_rom_all_stages.hex"
} else {
    puts "*** ERROR: exp_rom_all_stages.hex NOT found in $RTL_DIR!"
    exit 1
}

update_compile_order -fileset sources_1

puts "================================================================="
puts "  ==> STEP 3/7: BUILDING BLOCK DESIGN: mha_system"
puts "================================================================="
create_bd_design "mha_system"

# Query dynamic VLNVs for IP compatibility across Vivado versions
set vlnv_zynq   [get_latest_ip_vlnv "zynq_ultra_ps_e"   "xilinx.com:ip:zynq_ultra_ps_e:3.5"]
set vlnv_reset  [get_latest_ip_vlnv "proc_sys_reset"    "xilinx.com:ip:proc_sys_reset:5.0"]
set vlnv_dma    [get_latest_ip_vlnv "axi_dma"            "xilinx.com:ip:axi_dma:7.1"]
set vlnv_smc    [get_latest_ip_vlnv "smartconnect"       "xilinx.com:ip:smartconnect:1.0"]
set vlnv_ic     [get_latest_ip_vlnv "axi_interconnect"   "xilinx.com:ip:axi_interconnect:2.1"]
set vlnv_concat [get_latest_ip_vlnv "xlconcat"          "xilinx.com:ip:xlconcat:2.1"]

# 1. Zynq UltraScale+ MPSoC IP (150 MHz pl_clk0, HPM0 AXI-Lite, HP0 128-bit)
create_bd_cell -type ip -vlnv $vlnv_zynq zynq_ps
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP2 {0} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} \
    CONFIG.PSU__HIGH_PERF_PORT__ENABLE {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {150.000000} \
    CONFIG.PSU__USE__IRQ0 {1} \
] [get_bd_cells zynq_ps]

# 2. Processor System Reset
create_bd_cell -type ip -vlnv $vlnv_reset proc_sys_reset_0

# 3. AXI Direct Memory Access (AXI DMA) - 128-bit Direct Register Mode
create_bd_cell -type ip -vlnv $vlnv_dma axi_dma_0
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_sg_include_stscntrl_strm {0} \
    CONFIG.c_m_axis_mm2s_tdata_width {128} \
    CONFIG.c_m_axi_mm2s_data_width {128} \
    CONFIG.c_mm2s_burst_size {16} \
    CONFIG.c_s_axis_s2mm_tdata_width {128} \
    CONFIG.c_s_axi_s2mm_data_width {128} \
    CONFIG.c_s2mm_burst_size {16} \
    CONFIG.c_sg_length_width {26} \
] [get_bd_cells axi_dma_0]

# 4. AXI SmartConnect for 128-bit High-Performance Memory Bus
create_bd_cell -type ip -vlnv $vlnv_smc axi_smc_hp
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] [get_bd_cells axi_smc_hp]

# 5. AXI SmartConnect for Control Registers (eliminates 260-char MAX_PATH limit)
create_bd_cell -type ip -vlnv $vlnv_smc axi_smc_ctrl
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2}] [get_bd_cells axi_smc_ctrl]

# 6. Concat IP for Interrupts
create_bd_cell -type ip -vlnv $vlnv_concat irq_concat
set_property -dict [list CONFIG.NUM_PORTS {2}] [get_bd_cells irq_concat]

# 7. Add Attention Core Top (Variant 2A) as Module Reference
create_bd_cell -type module -reference attention_core_top_2A mha_core_0

puts "================================================================="
puts "  ==> STEP 4/7: WIRING BUSES, CLOCKS & INTERRUPTS"
puts "================================================================="
# Clock connections (150 MHz pl_clk0)
set clk_pin [get_bd_pins zynq_ps/pl_clk0]
connect_bd_net $clk_pin [get_bd_pins proc_sys_reset_0/slowest_sync_clk]
connect_bd_net $clk_pin [get_bd_pins zynq_ps/maxihpm0_fpd_aclk]
connect_bd_net $clk_pin [get_bd_pins zynq_ps/saxihp0_fpd_aclk]
connect_bd_net $clk_pin [get_bd_pins axi_smc_ctrl/aclk]
connect_bd_net $clk_pin [get_bd_pins axi_smc_hp/aclk]
connect_bd_net $clk_pin [get_bd_pins axi_dma_0/s_axi_lite_aclk]
connect_bd_net $clk_pin [get_bd_pins axi_dma_0/m_axi_mm2s_aclk]
connect_bd_net $clk_pin [get_bd_pins axi_dma_0/m_axi_s2mm_aclk]
connect_bd_net $clk_pin [get_bd_pins mha_core_0/clk]

# Reset connections
connect_bd_net [get_bd_pins zynq_ps/pl_resetn0] [get_bd_pins proc_sys_reset_0/ext_reset_in]
set rst_peri [get_bd_pins proc_sys_reset_0/peripheral_aresetn]

connect_bd_net $rst_peri [get_bd_pins axi_smc_ctrl/aresetn]
connect_bd_net $rst_peri [get_bd_pins axi_smc_hp/aresetn]
connect_bd_net $rst_peri [get_bd_pins axi_dma_0/axi_resetn]
connect_bd_net $rst_peri [get_bd_pins mha_core_0/rst_n]

# Control Bus Connections (AXI-Lite via SmartConnect)
connect_bd_intf_net [get_bd_intf_pins zynq_ps/M_AXI_HPM0_FPD] [get_bd_intf_pins axi_smc_ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_smc_ctrl/M00_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins axi_smc_ctrl/M01_AXI] [get_bd_intf_pins mha_core_0/s_axi]

# High-Performance Data Mover Bus (AXI4 128-bit)
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_smc_hp/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_smc_hp/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_smc_hp/M00_AXI]   [get_bd_intf_pins zynq_ps/S_AXI_HP0_FPD]

# Streaming Bus (AXI4-Stream 128-bit)
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] [get_bd_intf_pins mha_core_0/s_axis]
connect_bd_intf_net [get_bd_intf_pins mha_core_0/m_axis]    [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]

# Interrupt connections
connect_bd_net [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins irq_concat/In0]
connect_bd_net [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins irq_concat/In1]
connect_bd_net [get_bd_pins irq_concat/dout]        [get_bd_pins zynq_ps/pl_ps_irq0]

# Memory Address Mapping
assign_bd_address

# Robust address assignment (compatible with any default naming pattern)
catch {
    set dma_seg [get_bd_addr_segs -of_objects [get_bd_cells axi_dma_0] -filter {NAME =~ "*Reg*"}]
    if {$dma_seg ne ""} {
        set_property offset 0xA0000000 $dma_seg
        set_property range 64K $dma_seg
    }
}
catch {
    set mha_seg [get_bd_addr_segs -of_objects [get_bd_cells mha_core_0]]
    if {$mha_seg ne ""} {
        set_property offset 0xA0010000 $mha_seg
        set_property range 64K $mha_seg
    }
}

regenerate_bd_layout
validate_bd_design
save_bd_design

puts "================================================================="
puts "  ==> STEP 5/7: GENERATING HDL WRAPPER & CONSTRAINTS"
puts "================================================================="
set bd_file [get_files [get_property FILE_NAME [current_bd_design]]]
set wrapper_path [make_wrapper -files $bd_file -top]

if {[file exists $wrapper_path]} {
    add_files -norecurse $wrapper_path
    puts "  -> Added wrapper: $wrapper_path"
} else {
    # Check standard .gen and .srcs locations (Vivado 2020.2+ uses .gen)
    set gen_wrapper [glob -nocomplain "$PROJ_DIR/$PROJ_NAME.gen/sources_1/bd/mha_system/hdl/mha_system_wrapper.*"]
    set srcs_wrapper [glob -nocomplain "$PROJ_DIR/$PROJ_NAME.srcs/sources_1/bd/mha_system/hdl/mha_system_wrapper.*"]
    if {[llength $gen_wrapper] > 0} {
        add_files -norecurse [lindex $gen_wrapper 0]
        puts "  -> Added wrapper from .gen: [lindex $gen_wrapper 0]"
    } elseif {[llength $srcs_wrapper] > 0} {
        add_files -norecurse [lindex $srcs_wrapper 0]
        puts "  -> Added wrapper from .srcs: [lindex $srcs_wrapper 0]"
    }
}

set_property top mha_system_wrapper [current_fileset]
update_compile_order -fileset sources_1

puts "================================================================="
puts "  ==> STEP 6/7: LAUNCHING SYNTHESIS & IMPLEMENTATION"
puts "================================================================="
launch_runs impl_1 -to_step write_bitstream -jobs 2
wait_on_run impl_1

if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "*** ERROR: Implementation failed! Check run log for details."
    exit 1
}

puts "================================================================="
puts "  ==> STEP 7/7: EXPORTING BITSTREAM & HARDWARE HANDOFF (.HWH / .XSA)"
puts "================================================================="
file copy -force "$PROJ_DIR/$PROJ_NAME.runs/impl_1/mha_system_wrapper.bit" "$OUT_DIR/attention_core_top_2A.bit"
file copy -force "$PROJ_DIR/$PROJ_NAME.runs/impl_1/mha_system_wrapper.bit" "$OUT_DIR/03_ivit_hw_accelerated/overlay/attention_core_top_2A.bit"
file copy -force "$PROJ_DIR/$PROJ_NAME.runs/impl_1/mha_system_wrapper.bit" "$OUT_DIR/04_all_9_transformers_hw/overlay/attention_core_top_2A.bit"

# Export HWH for PYNQ Overlay
set hwh_source [glob -nocomplain "$PROJ_DIR/$PROJ_NAME.gen/sources_1/bd/mha_system/hw_handoff/*.hwh"]
if {[llength $hwh_source] > 0} {
    file copy -force [lindex $hwh_source 0] "$OUT_DIR/attention_core_top_2A.hwh"
    file copy -force [lindex $hwh_source 0] "$OUT_DIR/03_ivit_hw_accelerated/overlay/attention_core_top_2A.hwh"
    file copy -force [lindex $hwh_source 0] "$OUT_DIR/04_all_9_transformers_hw/overlay/attention_core_top_2A.hwh"
    puts "  -> Exported: $OUT_DIR/attention_core_top_2A.hwh"
}

write_hw_platform -fixed -include_bit -force -file "$OUT_DIR/attention_core_top_2A.xsa"

puts "================================================================="
puts "  BUILD COMPLETED SUCCESSFULLY!"
puts "  Bitstream : $OUT_DIR/attention_core_top_2A.bit"
puts "  Handoff   : $OUT_DIR/attention_core_top_2A.hwh"
puts "  Platform  : $OUT_DIR/attention_core_top_2A.xsa"
puts "================================================================="
exit
