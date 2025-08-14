@echo off
REM Quick launcher for transcription system
REM This script redirects to the organized transcription folder

echo ========================================
echo   Enhanced Sanskrit/Hindi Transcription
echo ========================================
echo.
echo The transcription system has been organized into:
echo   transcription/scripts/    - Executable scripts
echo   transcription/docs/       - Documentation
echo   transcription/examples/   - Test scripts
echo   transcription/output/     - Output files
echo.
echo Quick commands:
echo   1. Basic transcription (3-second segments):
echo      cd transcription/scripts
echo      transcribe_hindi.bat "video.mp4" "output_name"
echo.
echo   2. Granular transcription (custom segments):
echo      cd transcription/scripts  
echo      transcribe_granular.bat "video.mp4" "output_name" 2
echo.
echo   3. View documentation:
echo      start transcription/README.md
echo.

choice /c 123 /m "Choose an option (1-3)"

if errorlevel 3 (
    start transcription\README.md
    goto end
)

if errorlevel 2 (
    cd transcription\scripts
    echo.
    echo You are now in the transcription scripts directory.
    echo Run: transcribe_granular.bat "your_video.mp4" "output_name" [segment_length]
    cmd /k
    goto end
)

if errorlevel 1 (
    cd transcription\scripts
    echo.
    echo You are now in the transcription scripts directory.
    echo Run: transcribe_hindi.bat "your_video.mp4" "output_name"
    cmd /k
    goto end
)

:end
echo.
echo For more options, see: transcription/docs/TRANSCRIPTION_GUIDE.md
pause