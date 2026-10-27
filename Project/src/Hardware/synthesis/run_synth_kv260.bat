@echo off
REM ==============================================================================
REM Script:      run_synth_kv260.bat
REM Description: Automated Vivado Synthesis & Implementation Runner for all 3 variants.
REM Target:      AMD Kria KV260 Starter Kit (xck26-sfvc784-2LV-c)
REM Vivado Tool: C:\AMDDesignTools\2026.1\Vivado\bin\vivado.bat
REM ==============================================================================

set VIVADO_BIN=C:\AMDDesignTools\2026.1\Vivado\bin\vivado.bat

if not exist "%VIVADO_BIN%" (
    echo [ERROR] Vivado 2026.1 not found at %VIVADO_BIN%!
    exit /b 1
)

echo =================================================================
echo   STARTING KRIA KV260 VIVADO SYNTHESIS AND IMPLEMENTATION SUITE
echo =================================================================

if "%1"=="" goto run_all
if "%1"=="1A" goto run_1A
if "%1"=="2A" goto run_2A
if "%1"=="3B" goto run_3B

:run_all
echo [1/3] Running Out-of-Context Synthesis and PnR for Variant 1A...
call "%VIVADO_BIN%" -mode batch -notrace -source synth_variant_1A.tcl
if %ERRORLEVEL% NEQ 0 goto error

echo.
echo [2/3] Running Out-of-Context Synthesis and PnR for Variant 2A...
call "%VIVADO_BIN%" -mode batch -notrace -source synth_variant_2A.tcl
if %ERRORLEVEL% NEQ 0 goto error

echo.
echo [3/3] Running Out-of-Context Synthesis and PnR for Variant 3B...
call "%VIVADO_BIN%" -mode batch -notrace -source synth_variant_3B.tcl
if %ERRORLEVEL% NEQ 0 goto error

goto success

:run_1A
echo Running Out-of-Context Synthesis and PnR for Variant 1A...
call "%VIVADO_BIN%" -mode batch -notrace -source synth_variant_1A.tcl
if %ERRORLEVEL% NEQ 0 goto error
goto success

:run_2A
echo Running Out-of-Context Synthesis and PnR for Variant 2A...
call "%VIVADO_BIN%" -mode batch -notrace -source synth_variant_2A.tcl
if %ERRORLEVEL% NEQ 0 goto error
goto success

:run_3B
echo Running Out-of-Context Synthesis and PnR for Variant 3B...
call "%VIVADO_BIN%" -mode batch -notrace -source synth_variant_3B.tcl
if %ERRORLEVEL% NEQ 0 goto error
goto success

:success
echo.
echo =================================================================
echo   VIVADO SYNTHESIS AND IMPLEMENTATION SUITE COMPLETED!
echo   Reports are saved in reports_1A/, reports_2A/, reports_3B/
echo =================================================================
goto end

:error
echo.
echo [ERROR] Vivado process terminated with error code %ERRORLEVEL%!

:end
