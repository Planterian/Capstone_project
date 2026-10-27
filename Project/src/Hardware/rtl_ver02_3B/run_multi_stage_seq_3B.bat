@echo off
REM ============================================================================
REM Script:        run_multi_stage_seq_3B.bat
REM Description:   Sequential Multi-Stage Test Runner for Variant 3B.
REM                Runs Stage 2 -> Stage 3 -> Stage 4 continuously without reset.
REM Target:        AMD Kria KV260 / Icarus Verilog Simulation
REM ============================================================================

set IVERILOG=C:\iverilog\bin\iverilog.exe
set VVP=C:\iverilog\bin\vvp.exe

echo ==========================================================
echo   RUNNING SEQUENTIAL MULTI-STAGE DV SUITE - VARIANT 3B
echo ==========================================================

%IVERILOG% -g2012 -o tb_seq_3b.vvp tb_multi_stage_seq_3B.sv attention_core_top_3B.sv axi_lite_regs.sv axis_adapter.sv ping_pong_bram_buffer.sv pe_slice_dsp_packing.sv systolic_mac_array.sv reorder_buffer_3B.sv scale_unit.sv softmax_lut.sv score_v_gemm.sv requant_residual_add.sv
if %ERRORLEVEL% NEQ 0 goto error

%VVP% tb_seq_3b.vvp
if %ERRORLEVEL% NEQ 0 goto error

goto end

:error
echo.
echo [ERROR] Sequential multi-stage simulation failed with code %ERRORLEVEL%!

:end
