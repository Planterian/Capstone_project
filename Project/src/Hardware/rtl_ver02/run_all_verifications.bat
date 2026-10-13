@echo off
REM ============================================================================
REM Script:        run_all_verifications.bat
REM Description:   One-click automated test runner for all rtl_ver02 testbenches (CMD).
REM Target:        AMD Kria KV260 / Icarus Verilog Simulation
REM ============================================================================

set IVERILOG=C:\iverilog\bin\iverilog.exe
set VVP=C:\iverilog\bin\vvp.exe

echo ==========================================================
echo   RUNNING FULL HARDWARE VERIFICATION SUITE (rtl_ver02)
echo ==========================================================

echo [1/6] Running tb_golden_scale_unit...
%IVERILOG% -g2012 -o tb_scale.vvp scale_unit.sv tb_golden_scale_unit.sv
%VVP% tb_scale.vvp
if %ERRORLEVEL% NEQ 0 goto error

echo.
echo [2/6] Running tb_golden_softmax_lut...
%IVERILOG% -g2012 -o tb_softmax.vvp softmax_lut.sv tb_golden_softmax_lut.sv
%VVP% tb_softmax.vvp
if %ERRORLEVEL% NEQ 0 goto error

echo.
echo [3/6] Running tb_softmax_lut_reloadable...
%IVERILOG% -g2012 -o tb_reloadable.vvp softmax_lut.sv axi_lite_regs.sv tb_softmax_lut_reloadable.sv
%VVP% tb_reloadable.vvp
if %ERRORLEVEL% NEQ 0 goto error

echo.
echo [4/6] Running tb_systolic_mac_array...
%IVERILOG% -g2012 -o tb_systolic.vvp systolic_mac_array.sv pe_slice_dsp_packing.sv tb_systolic_mac_array.sv
%VVP% tb_systolic.vvp
if %ERRORLEVEL% NEQ 0 goto error

echo.
echo [5/6] Running tb_score_v_gemm...
%IVERILOG% -g2012 -o tb_score_v.vvp score_v_gemm.sv tb_score_v_gemm.sv
%VVP% tb_score_v.vvp
if %ERRORLEVEL% NEQ 0 goto error

echo.
echo [6/6] Running tb_attention_core (Full Top Integration)...
%IVERILOG% -g2012 -o tb_top.vvp tb_attention_core.sv attention_core_top.sv axi_lite_regs.sv axis_adapter.sv ping_pong_bram_buffer.sv systolic_mac_array.sv pe_slice_dsp_packing.sv scale_unit.sv softmax_lut.sv score_v_gemm.sv
%VVP% tb_top.vvp
if %ERRORLEVEL% NEQ 0 goto error

echo.
echo ==========================================================
echo   ALL 6 HARDWARE VERIFICATIONS PASSED 100% BIT-EXACT!
echo ==========================================================
goto end

:error
echo.
echo [ERROR] Simulation failed with code %ERRORLEVEL%!

:end
