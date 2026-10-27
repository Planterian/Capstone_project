@echo off
REM ============================================================================
REM Script:        run_stage_isolated_dv_1A.bat
REM Description:   Automated Isolated Multi-Stage DV Suite for Variant 1A.
REM                Executes separate verification on each individual Stage:
REM                  - Stage 2: dk = 16, 4,096 bytes (1 beat/token)
REM                  - Stage 3: dk = 20, 5,120 bytes (2 beats/token)
REM                  - Stage 4: dk = 24, 6,144 bytes (2 beats/token)
REM Target:        AMD Kria KV260 / Icarus Verilog Simulation
REM ============================================================================

set IVERILOG=C:\iverilog\bin\iverilog.exe
set VVP=C:\iverilog\bin\vvp.exe

echo ==========================================================
echo   ISOLATED MULTI-STAGE DV SUITE - VARIANT 1A (FABRIC TREE)
echo ==========================================================

echo [1/3] Running Dedicated Verification for STAGE 2 (dk=16, 4,096 Bytes)...
%IVERILOG% -g2012 -o tb_top_1a.vvp tb_attention_core_1A.sv attention_core_top_1A.sv axi_lite_regs.sv axis_adapter.sv ping_pong_bram_buffer.sv pipelined_adder_tree.sv scale_unit.sv softmax_lut.sv score_v_gemm.sv
if %ERRORLEVEL% NEQ 0 goto error
%VVP% tb_top_1a.vvp
if %ERRORLEVEL% NEQ 0 goto error

echo.
echo [2/3] Running Dedicated Verification for STAGE 3 (dk=20, 5,120 Bytes)...
%IVERILOG% -g2012 -o tb_stage3_1a.vvp tb_stage3_isolated_1A.sv attention_core_top_1A.sv axi_lite_regs.sv axis_adapter.sv ping_pong_bram_buffer.sv pipelined_adder_tree.sv scale_unit.sv softmax_lut.sv score_v_gemm.sv
if %ERRORLEVEL% NEQ 0 goto error
%VVP% tb_stage3_1a.vvp
if %ERRORLEVEL% NEQ 0 goto error

echo.
echo [3/3] Running Dedicated Verification for STAGE 4 (dk=24, 6,144 Bytes)...
%IVERILOG% -g2012 -o tb_stage4_1a.vvp tb_stage4_isolated_1A.sv attention_core_top_1A.sv axi_lite_regs.sv axis_adapter.sv ping_pong_bram_buffer.sv pipelined_adder_tree.sv scale_unit.sv softmax_lut.sv score_v_gemm.sv
if %ERRORLEVEL% NEQ 0 goto error
%VVP% tb_stage4_1a.vvp
if %ERRORLEVEL% NEQ 0 goto error

echo.
echo ==========================================================
echo   ALL 3 ISOLATED STAGES PASSED 100%% BIT-EXACT!
echo   TOTAL VERIFIED: 15,360 BYTES (0 LSB ERROR)
echo ==========================================================
goto end

:error
echo.
echo [ERROR] Isolated stage verification failed with code %ERRORLEVEL%!

:end
