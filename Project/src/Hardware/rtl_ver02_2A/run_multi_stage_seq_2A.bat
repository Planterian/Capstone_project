@echo off
REM ============================================================================
REM Script:        run_multi_stage_seq_2A.bat
REM Description:   Sequential Multi-Stage Test Runner for Variant 2A.
REM                Runs Stage 2 -> Stage 3 -> Stage 4 continuously without reset.
REM Target:        AMD Kria KV260 / Icarus Verilog Simulation
REM ============================================================================

set IVERILOG=C:\iverilog\bin\iverilog.exe
set VVP=C:\iverilog\bin\vvp.exe

echo ==========================================================
echo   RUNNING SEQUENTIAL MULTI-STAGE DV SUITE - VARIANT 2A
echo ==========================================================

%IVERILOG% -g2012 -o tb_seq_2a.vvp tb_multi_stage_seq_2A.sv attention_core_top_2A.sv axi_lite_regs.sv axis_adapter.sv ping_pong_bram_buffer.sv pe_slice_dsp_packing.sv dsp_packed_adder_tree.sv scale_unit.sv softmax_lut.sv score_v_gemm.sv
if %ERRORLEVEL% NEQ 0 goto error

%VVP% tb_seq_2a.vvp
if %ERRORLEVEL% NEQ 0 goto error

goto end

:error
echo.
echo [ERROR] Sequential multi-stage simulation failed with code %ERRORLEVEL%!

:end
