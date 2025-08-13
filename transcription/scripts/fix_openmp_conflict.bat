@echo off
REM Fix OpenMP library conflict for faster-whisper
set KMP_DUPLICATE_LIB_OK=TRUE
echo OpenMP conflict workaround enabled
echo You can now run transcription commands with large models
echo.
echo Example:
echo python -m ai_video_editor.cli.features transcribe "video.mp4" --backend faster-whisper --model large --force-model --output transcript.json
pause