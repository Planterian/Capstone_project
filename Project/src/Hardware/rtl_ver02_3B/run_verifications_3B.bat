@echo off
REM ============================================================================
REM Script:        run_verifications_3B.bat
REM Description:   Automated Test Runner for Variant 3B (2D Systolic MAC Array).
REM Target:        AMD Kria KV260 / Icarus Verilog Simulation
REM ============================================================================

set IVERILOG=C:\iverilog\bin\iverilog.exe
set VVP=C:\iverilog\bin\vvp.exe

echo ==========================================================
echo   RUNNING HARDWARE VERIFICATION SUITE FOR VARIANT 3B
echo   2D SYSTOLIC MAC ARRAY WITH DSP PACKING (16 DSP48E2)
echo ==========================================================

echo [1/5] Running Unit Test 1: Systolic MAC Array & Tile Drain...
%IVERILOG% -g2012 -o tb_systolic_3b.vvp pe_slice_dsp_packing.sv systolic_mac_array.sv tb_systolic_mac_array_3B.sv
if %ERRORLEVEL% NEQ 0 goto error
%VVP% tb_systolic_3b.vvp
if %ERRORLEVEL% NEQ 0 goto error

echo.
echo [2/5] Running Unit Test 2: Dyadic Rescaler Unit (65,536 scores)...
%IVERILOG% -g2012 -o tb_scale_3b.vvp scale_unit.sv tb_golden_scale_unit.sv
if %ERRORLEVEL% NEQ 0 goto error
%VVP% tb_scale_3b.vvp
if %ERRORLEVEL% NEQ 0 goto error

echo.
echo [3/5] Running Unit Test 3: Hardware IntSoftmax Engine (1,024 probs)...
%IVERILOG% -g2012 -o tb_softmax_3b.vvp softmax_lut.sv tb_golden_softmax_lut.sv
if %ERRORLEVEL% NEQ 0 goto error
%VVP% tb_softmax_3b.vvp
if %ERRORLEVEL% NEQ 0 goto error

echo.
echo [4/5] Running Unit Test 4: Score-V Context GEMM (4,096 context bytes)...
%IVERILOG% -g2012 -o tb_score_v_3b.vvp score_v_gemm.sv tb_score_v_gemm.sv requant_residual_add.sv
if %ERRORLEVEL% NEQ 0 goto error
%VVP% tb_score_v_3b.vvp
if %ERRORLEVEL% NEQ 0 goto error

echo.
echo [5/5] Running Full Top Integration: Attention Core 3B Top (4,096 context bytes)...
%IVERILOG% -g2012 -o tb_top_3b.vvp tb_attention_core_3B.sv attention_core_top_3B.sv axi_lite_regs.sv axis_adapter.sv ping_pong_bram_buffer.sv pe_slice_dsp_packing.sv systolic_mac_array.sv reorder_buffer_3B.sv scale_unit.sv softmax_lut.sv score_v_gemm.sv requant_residual_add.sv
if %ERRORLEVEL% NEQ 0 goto error
%VVP% tb_top_3b.vvp
if %ERRORLEVEL% NEQ 0 goto error

echo.
echo ==========================================================
echo   ALL 5 VARIANT 3B VERIFICATIONS PASSED 100%% BIT-EXACT!
echo ==========================================================
goto end

:error
echo.
echo [ERROR] Simulation failed with code %ERRORLEVEL%!

:end
