@echo off
REM Simplified Hindi transcription script
REM Usage: transcribe_hindi.bat "video_file.mp4" [output_name]

set KMP_DUPLICATE_LIB_OK=TRUE

if "%~1"=="" (
    echo Usage: transcribe_hindi.bat "video_file.mp4" [output_name]
    echo.
    echo Examples:
    echo   transcribe_hindi.bat "my_video.mp4"
    echo   transcribe_hindi.bat "my_video.mp4" custom_output
    echo.
    echo This will use:
    echo   - Large model with force override
    echo   - Hindi language detection
    echo   - Comprehensive Sanskrit/Hindi vocabulary
    echo   - Balanced 3-second segmentation
    echo   - Progress bar and romanization
    pause
    exit /b 1
)

set INPUT_FILE=%~1
set OUTPUT_NAME=%~2

if "%OUTPUT_NAME%"=="" (
    for %%f in ("%INPUT_FILE%") do set OUTPUT_NAME=%%~nf
)

set OUTPUT_FILE=..\output\%OUTPUT_NAME%_transcript.json

echo [INFO] Transcribing: %INPUT_FILE%
echo [INFO] Output will be: %OUTPUT_FILE%
echo [INFO] Using large model with comprehensive Sanskrit/Hindi vocabulary...
echo.

python -m ai_video_editor.cli.features transcribe "%INPUT_FILE%" ^
    --backend faster-whisper ^
    --model large ^
    --language hi ^
    --device cpu ^
    --compute-type int8 ^
    --force-model ^
    --preset comprehensive ^
    --vocab-size 150 ^
    --vad ^
    --vad-threshold 0.3 ^
    --min-silence-duration 300 ^
    --segment-length 3 ^
    --word-timestamps ^
    --romanize ^
    --romanize-scheme hk ^
    --progress ^
    --output "%OUTPUT_FILE%"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [SUCCESS] Transcription completed successfully!
    echo Output saved to: %OUTPUT_FILE%
) else (
    echo.
    echo [ERROR] Transcription failed with error code %ERRORLEVEL%
)

pause