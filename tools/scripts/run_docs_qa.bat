@echo off
REM Documentation Quality Assurance Helper Script
REM Usage: run_docs_qa.bat [command] [options]

if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="validate" goto validate
if "%1"=="daily" goto daily
if "%1"=="weekly" goto weekly
if "%1"=="monthly" goto monthly
if "%1"=="quarterly" goto quarterly
if "%1"=="fix-links" goto fix-links

:help
echo.
echo Documentation Quality Assurance Tools
echo ====================================
echo.
echo Usage: run_docs_qa.bat [command] [options]
echo.
echo Commands:
echo   validate     - Run documentation validation
echo   daily        - Run daily maintenance checks
echo   weekly       - Run weekly maintenance checks  
echo   monthly      - Run monthly maintenance checks
echo   quarterly    - Run quarterly maintenance checks
echo   fix-links    - Attempt to fix common link issues
echo   help         - Show this help message
echo.
echo Examples:
echo   run_docs_qa.bat validate
echo   run_docs_qa.bat validate --severity error
echo   run_docs_qa.bat daily
echo   run_docs_qa.bat weekly
echo.
goto end

:validate
echo Running documentation validation...
if "%2"=="" (
    python scripts/validate_documentation.py
) else (
    python scripts/validate_documentation.py %2 %3 %4 %5
)
goto end

:daily
echo Running daily maintenance checks...
python scripts/run_maintenance_checks.py daily
goto end

:weekly
echo Running weekly maintenance checks...
python scripts/run_maintenance_checks.py weekly
goto end

:monthly
echo Running monthly maintenance checks...
python scripts/run_maintenance_checks.py monthly
goto end

:quarterly
echo Running quarterly maintenance checks...
python scripts/run_maintenance_checks.py quarterly
goto end

:fix-links
echo Attempting to fix common link issues...
echo This feature is not yet implemented.
echo Please run validation first to identify specific issues:
echo   run_docs_qa.bat validate --severity error
goto end

:end