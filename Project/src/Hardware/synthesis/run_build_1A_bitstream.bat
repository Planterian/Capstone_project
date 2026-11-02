@echo off
REM ==============================================================================
REM Script:      run_build_1A_bitstream.bat
REM Description: One-click Runner to build Variant 1A Vivado SoC Block Design
REM              and generate attention_core_top_1A.bit & .hwh for Kria KV260.
REM Target Part: AMD Kria KV260 (xck26-sfvc784-2LV-c)
REM ==============================================================================

set VIVADO_BIN=C:\AMDDesignTools\2026.1\Vivado\bin\vivado.bat

if not exist "%VIVADO_BIN%" (
    echo [ERROR] Vivado not found at %VIVADO_BIN%!
    echo Please check your Vivado installation path.
    exit /b 1
)

echo =================================================================
echo   STARTING AUTOMATED VIVADO 1A BITSTREAM GENERATION (2 THREADS)
echo =================================================================

call "%VIVADO_BIN%" -mode batch -notrace -source build_kv260_mha_1A.tcl
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Vivado build failed with error code %ERRORLEVEL%!
    exit /b %ERRORLEVEL%
)

echo.
echo =================================================================
echo   BUILD COMPLETED! Artifacts generated in Project/src/Software/
echo     - attention_core_top_1A.bit
echo     - attention_core_top_1A.hwh
echo     - attention_core_top_1A.xsa
echo =================================================================

