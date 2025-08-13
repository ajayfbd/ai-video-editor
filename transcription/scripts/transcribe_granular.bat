@echo off
REM Granular Hindi transcription script - creates small segments (1-2 seconds each)
REM Usage: transcribe_granular.bat "video_file.mp4" [output_name] [segment_length]

set KMP_DUPLICATE_LIB_OK=TRUE

if "%~1"=="" (
    echo Usage: transcribe_granular.bat "video_file.mp4" [output_name] [segment_length]
    echo.
    echo Examples:
    echo   transcribe_granular.bat "my_video.mp4"
    echo   transcribe_granular.bat "my_video.mp4" custom_output
    echo   transcribe_granular.bat "my_video.mp4" custom_output 2
    echo.
    echo This will create small segments for:
    echo   - Subtitle generation
    echo   - Word-by-word analysis
    echo   - Precise timing control
    echo   - Educational content breakdown
    pause
    exit /b 1
)

set INPUT_FILE=%~1
set OUTPUT_NAME=%~2
set SEGMENT_LENGTH=%~3

if "%OUTPUT_NAME%"=="" (
    for %%f in ("%INPUT_FILE%") do set OUTPUT_NAME=%%~nf
)

if "%SEGMENT_LENGTH%"=="" (
    set SEGMENT_LENGTH=2
)

set OUTPUT_FILE=..\output\%OUTPUT_NAME%_granular.json

echo [INFO] Transcribing: %INPUT_FILE%
echo [INFO] Output will be: %OUTPUT_FILE%
echo [INFO] Maximum segment length: %SEGMENT_LENGTH% seconds
echo [INFO] Using granular segmentation settings...
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
    --vad-threshold 0.2 ^
    --min-silence-duration 200 ^
    --segment-length %SEGMENT_LENGTH% ^
    --word-timestamps ^
    --romanize ^
    --romanize-scheme hk ^
    --progress ^
    --output "%OUTPUT_FILE%"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [SUCCESS] Granular transcription completed successfully!
    echo Output saved to: %OUTPUT_FILE%
    echo.
    echo The transcript has been broken into small %SEGMENT_LENGTH%-second segments
    echo Perfect for:
    echo   - Creating precise subtitles
    echo   - Word-by-word analysis
    echo   - Educational content breakdown
    echo   - Timing-sensitive applications
) else (
    echo.
    echo [ERROR] Transcription failed with error code %ERRORLEVEL%
)

pause