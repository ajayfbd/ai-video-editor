# Troubleshooting Guide

> **📚 This document has been consolidated into a comprehensive unified guide.**
> 
> **Please see: [Unified Troubleshooting and Support Guide](troubleshooting-unified.md)**
> 
> The unified guide consolidates all troubleshooting information, error handling patterns, and recovery procedures in one comprehensive document.

---

*Legacy troubleshooting content below - please use the unified guide above for the most complete and up-to-date information.*

## 🚨 Quick Diagnostics

### System Status Check

```bash
# Check overall system status
python -m ai_video_editor.cli.main status

# Test workflow system
python -m ai_video_editor.cli.main test-workflow --mock
```

### Common Quick Fixes

```bash
# Clear cache and restart
rm -rf temp/cache/*
python -m ai_video_editor.cli.main status

# Reset configuration
python -m ai_video_editor.cli.main init --output .env

# Check dependencies
pip install -r requirements.txt --upgrade
```

## 🔧 Installation and Setup Issues

### Python Environment Problems

**Issue**: `ModuleNotFoundError` or import errors

**Solutions**:
```bash
# Verify Python version (3.9+ required)
python --version

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Use virtual environment
python -m venv ai_video_editor_env
source ai_video_editor_env/bin/activate  # Linux/Mac
# ai_video_editor_env\Scripts\activate  # Windows
pip install -r requirements.txt
```

**Issue**: Package conflicts or version mismatches

**Solutions**:
```bash
# Create clean environment
pip freeze > old_requirements.txt
pip uninstall -r old_requirements.txt -y
pip install -r requirements.txt

# Use specific versions
pip install -r requirements.txt --no-deps
pip check  # Verify no conflicts
```

### API Configuration Issues

**Issue**: "API key not found" or authentication errors

**Solutions**:
1. **Verify .env file location**:
   ```bash
   ls -la .env  # Should be in project root
   cat .env     # Check contents
   ```

2. **Check API key format**:
   ```bash
   # Correct format in .env
   AI_VIDEO_EDITOR_GEMINI_API_KEY=AIzaSyD...
   AI_VIDEO_EDITOR_IMAGEN_API_KEY=your_imagen_key
   AI_VIDEO_EDITOR_GOOGLE_CLOUD_PROJECT=your_project_id
   ```

3. **Test API connectivity**:
   ```bash
   python test_gemini_access.py
   ```

**Issue**: API quota exceeded or rate limiting

**Solutions**:
```bash
# Check API usage
python -c "
from ai_video_editor.core.config import get_settings
settings = get_settings()
print(f'Project: {settings.google_cloud_project}')
"

# Reduce API calls with caching
export AI_VIDEO_EDITOR_ENABLE_CACHING=true

# Use lower quality mode
python -m ai_video_editor.cli.main process video.mp4 --mode fast
```

## 💾 Memory and Performance Issues

### Out of Memory Errors

**Issue**: `MemoryError` or system freezing during processing

**Solutions**:

1. **Reduce memory usage**:
   ```bash
   # Lower memory limit
   python -m ai_video_editor.cli.main process video.mp4 \
     --max-memory 4 \
     --mode fast \
     --quality medium
   ```

2. **Process shorter segments**:
   ```bash
   # Split long videos first
   ffmpeg -i long_video.mp4 -t 600 -c copy segment1.mp4
   ffmpeg -i long_video.mp4 -ss 600 -t 600 -c copy segment2.mp4
   ```

3. **Disable parallel processing**:
   ```bash
   python -m ai_video_editor.cli.main process video.mp4 \
     --no-parallel \
     --max-memory 6
   ```

4. **Clear system memory**:
   ```bash
   # Linux/Mac
   sudo sync && sudo sysctl vm.drop_caches=3
   
   # Windows - restart system or close applications
   ```

### Slow Processing Performance

**Issue**: Processing takes much longer than expected

**Solutions**:

1. **Check system resources**:
   ```bash
   # Monitor during processing
   htop  # Linux/Mac
   # Task Manager on Windows
   ```

2. **Optimize processing mode**:
   ```bash
   # Fast mode for testing
   python -m ai_video_editor.cli.main process video.mp4 \
     --mode fast \
     --quality medium
   
   # Balanced for production
   python -m ai_video_editor.cli.main process video.mp4 \
     --mode balanced \
     --parallel
   ```

3. **Enable caching**:
   ```bash
   export AI_VIDEO_EDITOR_ENABLE_CACHING=true
   python -m ai_video_editor.cli.main process video.mp4
   ```

4. **Check internet connection**:
   ```bash
   # Test API connectivity
   curl -I https://generativelanguage.googleapis.com
   ping google.com
   ```

## 🎬 Video Processing Issues

### Video Format Problems

**Issue**: "Unsupported video format" or codec errors

**Solutions**:

1. **Convert to supported format**:
   ```bash
   # Convert to MP4 with H.264
   ffmpeg -i input_video.mov -c:v libx264 -c:a aac output_video.mp4
   
   # Check video info
   ffprobe input_video.mp4
   ```

2. **Supported formats**:
   - **Video**: MP4 (H.264), MOV, AVI, MKV
   - **Audio**: AAC, MP3, WAV
   - **Resolution**: 720p to 4K
   - **Frame rates**: 24, 25, 30, 60 fps

### Audio Processing Issues

**Issue**: Poor audio transcription or filler word detection

**Solutions**:

1. **Improve audio quality**:
   ```bash
   # Enhance audio first
   ffmpeg -i input.mp4 -af "highpass=f=80,lowpass=f=8000,volume=1.5" \
     -c:v copy enhanced.mp4
   ```

2. **Check audio levels**:
   ```bash
   # Analyze audio
   python -m ai_video_editor.cli.main analyze audio.wav
   ```

3. **Use higher quality mode**:
   ```bash
   python -m ai_video_editor.cli.main process video.mp4 \
     --type educational \
     --mode high_quality
   ```

### B-Roll Generation Issues

**Issue**: No B-roll generated or poor quality graphics

**Solutions**:

1. **Verify content analysis**:
   ```bash
   # Check if concepts are detected
   python -m ai_video_editor.cli.main process video.mp4 \
     --type educational \
     --debug
   ```

2. **Improve concept clarity**:
   - Speak clearly about specific concepts
   - Use educational terminology
   - Include visual descriptions in speech

3. **Check B-roll output**:
   ```bash
   # Look for B-roll files
   ls -la output/broll/
   cat output/analytics/ai_decisions.json | grep broll
   ```

## 🖼️ Thumbnail Generation Issues

### Poor Thumbnail Quality

**Issue**: Blurry, low-quality, or inappropriate thumbnails

**Solutions**:

1. **Improve source video quality**:
   - Use 1080p or higher resolution
   - Ensure good lighting
   - Clear facial expressions
   - Stable camera work

2. **Check visual highlights**:
   ```bash
   # Review detected highlights
   cat output/analytics/content_analysis.json | grep visual_highlights
   ```

3. **Adjust thumbnail settings**:
   ```bash
   python -m ai_video_editor.cli.main process video.mp4 \
     --type educational \
     --quality ultra  # Better thumbnail generation
   ```

### Missing Thumbnail Variations

**Issue**: Only one thumbnail generated or missing strategies

**Solutions**:

1. **Check emotional analysis**:
   ```bash
   # Verify emotional peaks detected
   cat output/analytics/content_analysis.json | grep emotional_markers
   ```

2. **Ensure face detection**:
   - Good lighting on speaker's face
   - Clear facial expressions
   - Speaker visible in frame

3. **Review AI Director decisions**:
   ```bash
   cat output/analytics/ai_decisions.json | grep thumbnail
   ```

## 🔄 Workflow and Orchestration Issues

### Workflow Stuck or Hanging

**Issue**: Processing appears to hang at a specific stage

**Solutions**:

1. **Check workflow status**:
   ```bash
   python -m ai_video_editor.cli.main workflow --list
   python -m ai_video_editor.cli.main workflow PROJECT_ID --details
   ```

2. **Increase timeout**:
   ```bash
   python -m ai_video_editor.cli.main process video.mp4 \
     --timeout 3600  # 1 hour timeout
   ```

3. **Restart with recovery**:
   ```bash
   # Check for checkpoint files
   ls -la temp/checkpoints/
   
   # Process with recovery enabled
   python -m ai_video_editor.cli.main process video.mp4 \
     --enable-recovery
   ```

### Stage Failures

**Issue**: Specific processing stage fails consistently

**Solutions**:

1. **Skip problematic stages**:
   ```bash
   # Skip B-roll generation if failing
   python -m ai_video_editor.cli.main process video.mp4 \
     --skip-broll
   ```

2. **Check stage-specific logs**:
   ```bash
   tail -f logs/ai_video_editor.log
   grep ERROR logs/ai_video_editor.log
   ```

3. **Test individual components**:
   ```bash
   # Test thumbnail generation
   python examples/thumbnail_generation_example.py
   
   # Test audio analysis
   python examples/audio_analysis_example.py
   ```

## 🌐 Network and API Issues

### API Connection Problems

**Issue**: Network timeouts or API unavailable

**Solutions**:

1. **Check internet connectivity**:
   ```bash
   ping google.com
   curl -I https://generativelanguage.googleapis.com
   ```

2. **Configure proxy if needed**:
   ```bash
   export HTTP_PROXY=http://proxy.company.com:8080
   export HTTPS_PROXY=http://proxy.company.com:8080
   ```

3. **Retry with backoff**:
   ```bash
   # The system has built-in retry logic
   # Check retry attempts in logs
   grep "retry" logs/ai_video_editor.log
   ```

### API Rate Limiting

**Issue**: "Rate limit exceeded" or quota errors

**Solutions**:

1. **Enable caching**:
   ```bash
   export AI_VIDEO_EDITOR_ENABLE_CACHING=true
   ```

2. **Reduce API calls**:
   ```bash
   python -m ai_video_editor.cli.main process video.mp4 \
     --mode fast  # Fewer API calls
   ```

3. **Batch processing with delays**:
   ```bash
   for video in *.mp4; do
     python -m ai_video_editor.cli.main process "$video"
     sleep 60  # Wait between videos
   done
   ```

## 📁 File and Directory Issues

### Permission Errors

**Issue**: "Permission denied" or file access errors

**Solutions**:

1. **Check file permissions**:
   ```bash
   ls -la video.mp4
   chmod 644 video.mp4  # Make readable
   ```

2. **Check directory permissions**:
   ```bash
   ls -la output/
   mkdir -p output && chmod 755 output/
   ```

3. **Run with appropriate permissions**:
   ```bash
   # Avoid running as root unless necessary
   whoami
   ```

### Disk Space Issues

**Issue**: "No space left on device" or disk full

**Solutions**:

1. **Check available space**:
   ```bash
   df -h
   du -sh output/
   ```

2. **Clean temporary files**:
   ```bash
   rm -rf temp/cache/*
   rm -rf temp/*/
   ```

3. **Use external storage**:
   ```bash
   python -m ai_video_editor.cli.main process video.mp4 \
     --output /external/drive/output
   ```

## 🔍 Debugging and Diagnostics

### Enable Debug Mode

```bash
# Enable detailed logging
python -m ai_video_editor.cli.main --debug process video.mp4

# Check debug logs
tail -f logs/ai_video_editor.log
```

### Collect Diagnostic Information

```bash
# System information
python -c "
import sys, platform, psutil
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'Memory: {psutil.virtual_memory().total / (1024**3):.1f}GB')
print(f'CPU: {psutil.cpu_count()} cores')
"

# Package versions
pip list | grep -E "(torch|opencv|whisper|movis)"

# Configuration status
python -m ai_video_editor.cli.main status
```

### Performance Profiling

```bash
# Profile memory usage
python -m memory_profiler examples/workflow_orchestrator_example.py

# Profile processing time
time python -m ai_video_editor.cli.main process video.mp4
```

## 📞 Getting Additional Help

### Log Analysis

```bash
# Check error patterns
grep -i error logs/ai_video_editor.log | tail -20

# Check warning patterns
grep -i warning logs/ai_video_editor.log | tail -10

# Check performance metrics
grep -i "processing time" logs/ai_video_editor.log
```

### Create Support Report

```bash
# Generate support information
python -c "
from ai_video_editor.core.config import get_settings, validate_environment
import json

# System status
env_status = validate_environment()
print('Environment Status:')
print(json.dumps(env_status, indent=2))

# Configuration (without sensitive data)
settings = get_settings()
print(f'\\nConfiguration:')
print(f'Max Memory: {settings.max_memory_usage_gb}GB')
print(f'Concurrent Processes: {settings.max_concurrent_processes}')
"
```

### Community Resources

- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check latest updates and examples
- **Examples**: Review working code samples
- **Performance Benchmarks**: Compare your results

---

*Resolve issues quickly and get back to creating amazing content*