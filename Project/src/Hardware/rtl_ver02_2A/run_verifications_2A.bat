@echo off
REM ============================================================================
REM Script:        run_verifications_2A.bat
REM Description:   Automated Test Runner for Variant 2A (DSP-Packed Streaming Engine).
REM Target:        AMD Kria KV260 / Icarus Verilog Simulation
REM ============================================================================

set IVERILOG=C:\iverilog\bin\iverilog.exe
set VVP=C:\iverilog\bin\vvp.exe

echo ==========================================================
echo   RUNNING HARDWARE VERIFICATION SUITE FOR VARIANT 2A
echo   HIGH-THROUGHPUT DSP-PACKED STREAMING (8 DSP48E2)
echo ==========================================================

echo [1/5] Running Unit Test 1: DSP-Packed Adder Tree (65,536 scores)...
%IVERILOG% -g2012 -o tb_tree_2a.vvp pe_slice_dsp_packing.sv dsp_packed_adder_tree.sv tb_dsp_packed_adder_tree_2A.sv
if %ERRORLEVEL% NEQ 0 goto error
%VVP% tb_tree_2a.vvp
if %ERRORLEVEL% NEQ 0 goto error

echo.
echo [2/5] Running Unit Test 2: Dyadic Rescaler Unit (65,536 scores)...
%IVERILOG% -g2012 -o tb_scale_2a.vvp scale_unit.sv tb_golden_scale_unit.sv
if %ERRORLEVEL% NEQ 0 goto error
%VVP% tb_scale_2a.vvp
if %ERRORLEVEL% NEQ 0 goto error

echo.
echo [3/5] Running Unit Test 3: Hardware IntSoftmax Engine (1,024 probs)...
%IVERILOG% -g2012 -o tb_softmax_2a.vvp softmax_lut.sv tb_golden_softmax_lut.sv
if %ERRORLEVEL% NEQ 0 goto error
%VVP% tb_softmax_2a.vvp
if %ERRORLEVEL% NEQ 0 goto error

echo.
echo [4/5] Running Unit Test 4: Score-V Context GEMM (4,096 context bytes)...
%IVERILOG% -g2012 -o tb_score_v_2a.vvp score_v_gemm.sv tb_score_v_gemm.sv
if %ERRORLEVEL% NEQ 0 goto error
%VVP% tb_score_v_2a.vvp
if %ERRORLEVEL% NEQ 0 goto error

echo.
echo [5/5] Running Full Top Integration: Attention Core 2A Top (4,096 context bytes)...
%IVERILOG% -g2012 -o tb_top_2a.vvp tb_attention_core_2A.sv attention_core_top_2A.sv axi_lite_regs.sv axis_adapter.sv ping_pong_bram_buffer.sv pe_slice_dsp_packing.sv dsp_packed_adder_tree.sv scale_unit.sv softmax_lut.sv score_v_gemm.sv
if %ERRORLEVEL% NEQ 0 goto error
%VVP% tb_top_2a.vvp
if %ERRORLEVEL% NEQ 0 goto error

echo.
echo ==========================================================
echo   ALL 5 VARIANT 2A VERIFICATIONS PASSED 100%% BIT-EXACT!
echo ==========================================================
goto end

:error
echo.
echo [ERROR] Simulation failed with code %ERRORLEVEL%!

:end
