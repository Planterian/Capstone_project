@echo off
REM ==============================================================================
REM Script:      run_build_3B_bitstream.bat
REM Description: One-click Runner to build Variant 3B Vivado SoC Block Design
REM              and generate attention_core_top_3B.bit & .hwh for Kria KV260.
REM Target Part: AMD Kria KV260 (xck26-sfvc784-2LV-c)
REM ==============================================================================

set VIVADO_BIN=C:\AMDDesignTools\2026.1\Vivado\bin\vivado.bat

if not exist "%VIVADO_BIN%" (
    echo [ERROR] Vivado not found at %VIVADO_BIN%!
    echo Please check your Vivado installation path.
    exit /b 1
)

echo =================================================================
echo   STARTING AUTOMATED VIVADO 3B BITSTREAM GENERATION (2 THREADS)
echo =================================================================

call "%VIVADO_BIN%" -mode batch -notrace -source build_kv260_mha_3B.tcl
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Vivado build failed with error code %ERRORLEVEL%!
    exit /b %ERRORLEVEL%
)

echo.
echo =================================================================
echo   BUILD COMPLETED! Artifacts generated in Project/src/Software/
echo     - attention_core_top_3B.bit
echo     - attention_core_top_3B.hwh
echo     - attention_core_top_3B.xsa
echo =================================================================

