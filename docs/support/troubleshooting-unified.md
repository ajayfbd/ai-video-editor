# Unified Troubleshooting and Support Guide

Comprehensive troubleshooting guide for the AI Video Editor system, consolidating all support information, error handling patterns, and performance optimization guidance.

## 🧭 Navigation

**New to troubleshooting?** Start with [**Quick Diagnostics**](#quick-diagnostics) below for immediate help.

**Need general help?** Check the [**FAQ**](faq-unified.md) for common questions and answers.

**Performance issues?** Jump to the [**Performance Guide**](performance-unified.md) for optimization tips.

**Looking for error details?** Browse the [**Error Handling Guide**](error-handling-unified.md) for specific error patterns.

## 🚨 Quick Diagnostics

### System Status Check

```bash
# Check overall system status
python -m ai_video_editor.cli.main status

# Test workflow system
python -m ai_video_editor.cli.main test-workflow --mock

# Verify API connectivity
python test_gemini_access.py
```

### Emergency Quick Fixes

```bash
# Clear cache and restart
rm -rf temp/cache/*
python -m ai_video_editor.cli.main status

# Reset configuration
python -m ai_video_editor.cli.main init --output .env

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check for conflicts
pip check
```

## 🔧 Installation and Setup Issues

### Python Environment Problems

**Issue**: `ModuleNotFoundError` or import errors

**Root Causes**:
- Incorrect Python version (requires 3.9+)
- Missing or corrupted dependencies
- Virtual environment issues
- Package conflicts

**Solutions**:
```bash
# Verify Python version (3.9+ required, 3.11+ recommended)
python --version

# Create clean virtual environment
python -m venv ai_video_editor_env
source ai_video_editor_env/bin/activate  # Linux/Mac
# ai_video_editor_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# For package conflicts
pip freeze > old_requirements.txt
pip uninstall -r old_requirements.txt -y
pip install -r requirements.txt --no-deps
```

**Recovery Actions**:
1. Use Python 3.11+ for best compatibility
2. Always use virtual environments
3. Check for system-wide package conflicts
4. Verify all dependencies are correctly installed

### API Configuration Issues

**Issue**: "API key not found" or authentication errors

**Root Causes**:
- Missing or incorrect API keys
- Wrong .env file location
- Incorrect environment variable names
- API quota exceeded

**Solutions**:
1. **Verify .env file location and format**:
   ```bash
   # Check .env file exists in project root
   ls -la .env
   
   # Verify correct format
   cat .env
   # Should contain:
   AI_VIDEO_EDITOR_GEMINI_API_KEY=AIzaSyD...
   AI_VIDEO_EDITOR_IMAGEN_API_KEY=your_imagen_key
   AI_VIDEO_EDITOR_GOOGLE_CLOUD_PROJECT=your_project_id
   ```

2. **Test API connectivity**:
   ```bash
   # Test Gemini API
   python test_gemini_access.py
   
   # Test search capabilities
   python test_gemini_search.py
   ```

3. **Check API quotas and billing**:
   - Visit [Google Cloud Console](https://console.cloud.google.com/)
   - Verify billing is enabled
   - Check API usage limits

**Recovery Actions**:
- Enable caching to reduce API calls: `export AI_VIDEO_EDITOR_ENABLE_CACHING=true`
- Use lower quality modes during testing
- Implement rate limiting for batch processing

## 💾 Memory and Performance Issues

### Out of Memory Errors

**Issue**: `MemoryError`, system freezing, or crashes during processing

**Root Causes**:
- Insufficient system RAM (minimum 8GB required)
- Large video files exceeding memory limits
- Memory leaks in processing pipeline
- Concurrent processing overwhelming system

**Solutions**:

1. **Immediate Memory Relief**:
   ```bash
   # Reduce memory usage
   python -m ai_video_editor.cli.main process video.mp4 \
     --max-memory 4 \
     --mode fast \
     --quality medium \
     --no-parallel
   ```

2. **Process Video Segments**:
   ```bash
   # Split long videos first
   ffmpeg -i long_video.mp4 -t 600 -c copy segment1.mp4
   ffmpeg -i long_video.mp4 -ss 600 -t 600 -c copy segment2.mp4
   ```

3. **System Memory Optimization**:
   ```bash
   # Linux/Mac - Clear system cache
   sudo sync && sudo sysctl vm.drop_caches=3
   
   # Windows - Close applications or restart
   # Use Task Manager to identify memory-heavy processes
   ```

4. **Configure Memory Limits by System**:
   ```bash
   # For 8GB systems
   python -m ai_video_editor.cli.main process video.mp4 \
     --max-memory 6 --mode fast --quality medium
   
   # For 16GB systems
   python -m ai_video_editor.cli.main process video.mp4 \
     --max-memory 12 --parallel --quality high
   
   # For 32GB+ systems
   python -m ai_video_editor.cli.main process video.mp4 \
     --max-memory 24 --parallel --quality ultra
   ```

**Recovery Actions**:
- Monitor memory usage during processing
- Use progressive quality settings based on available resources
- Enable aggressive caching to reduce memory pressure
- Process shorter video segments for large files

### Slow Processing Performance

**Issue**: Processing takes much longer than expected

**Performance Targets**:
- Educational content (15 min): <10 minutes processing
- Music videos (5 min): <5 minutes processing
- General content (10 min): <7 minutes processing

**Root Causes**:
- Insufficient system resources
- Slow internet connection affecting API calls
- Suboptimal processing settings
- Cache misses increasing processing time

**Solutions**:

1. **System Resource Optimization**:
   ```bash
   # Monitor system resources
   htop  # Linux/Mac
   # Task Manager on Windows
   
   # Check CPU and memory usage patterns
   ```

2. **Processing Mode Optimization**:
   ```bash
   # Fast mode (2-3x faster)
   python -m ai_video_editor.cli.main process video.mp4 \
     --mode fast --quality medium
   
   # Balanced mode (recommended)
   python -m ai_video_editor.cli.main process video.mp4 \
     --mode balanced --quality high --parallel
   
   # High quality mode (slower but best results)
   python -m ai_video_editor.cli.main process video.mp4 \
     --mode high_quality --quality ultra
   ```

3. **Enable Comprehensive Caching**:
   ```bash
   # Enable all caching systems
   export AI_VIDEO_EDITOR_ENABLE_CACHING=true
   export AI_VIDEO_EDITOR_CACHE_SIZE_GB=2
   export AI_VIDEO_EDITOR_CACHE_AGGRESSIVE=true
   ```

4. **Network Optimization**:
   ```bash
   # Test API connectivity
   curl -I https://generativelanguage.googleapis.com
   ping google.com
   
   # Configure proxy if needed
   export HTTP_PROXY=http://proxy.company.com:8080
   export HTTPS_PROXY=http://proxy.company.com:8080
   ```

**Recovery Actions**:
- Use SSD storage for temporary files
- Close unnecessary applications
- Enable parallel processing on multi-core systems
- Use appropriate quality settings for your use case

## 🎬 Video Processing Issues

### Video Format Problems

**Issue**: "Unsupported video format" or codec errors

**Supported Formats**:
- **Video**: MP4 (H.264), MOV, AVI, MKV, WebM
- **Audio**: AAC, MP3, WAV, FLAC
- **Resolution**: 720p to 4K
- **Frame rates**: 24, 25, 30, 60 fps

**Solutions**:

1. **Convert to Supported Format**:
   ```bash
   # Convert to MP4 with H.264 (recommended)
   ffmpeg -i input_video.mov -c:v libx264 -c:a aac output_video.mp4
   
   # Check video information
   ffprobe input_video.mp4
   ```

2. **Optimize Video Properties**:
   ```bash
   # Standardize frame rate
   ffmpeg -i input.mp4 -r 30 -c:v libx264 -c:a aac output.mp4
   
   # Reduce resolution if needed
   ffmpeg -i input.mp4 -vf scale=1920:1080 -c:v libx264 -c:a aac output.mp4
   ```

**Recovery Actions**:
- Always use MP4 with H.264 video and AAC audio for best compatibility
- Check video properties before processing
- Convert problematic formats using FFmpeg

### Audio Processing Issues

**Issue**: Poor audio transcription, filler word detection, or audio quality

**Root Causes**:
- Poor audio quality in source video
- Background noise or music interference
- Incorrect audio levels
- Unsupported audio format

**Solutions**:

1. **Improve Audio Quality**:
   ```bash
   # Enhance audio before processing
   ffmpeg -i input.mp4 -af "highpass=f=80,lowpass=f=8000,volume=1.5" \
     -c:v copy enhanced.mp4
   ```

2. **Audio Analysis and Optimization**:
   ```bash
   # Analyze audio levels
   python -m ai_video_editor.cli.main analyze audio.wav
   
   # Use higher quality mode for better transcription
   python -m ai_video_editor.cli.main process video.mp4 \
     --type educational --mode high_quality
   ```

3. **Content-Specific Audio Settings**:
   ```bash
   # Educational content (optimize for speech)
   python -m ai_video_editor.cli.main process lecture.mp4 \
     --type educational --enable-filler-detection
   
   # Music content (preserve audio quality)
   python -m ai_video_editor.cli.main process music.mp4 \
     --type music --disable-filler-detection
   ```

**Recovery Actions**:
- Ensure clear, consistent audio in source videos
- Use appropriate content type for audio processing
- Consider audio preprocessing for poor quality sources

### B-Roll Generation Issues

**Issue**: No B-roll generated, poor quality graphics, or irrelevant content

**Root Causes**:
- Unclear concept detection in content analysis
- Insufficient visual context in speech
- API failures affecting content understanding
- Poor source video quality

**Solutions**:

1. **Verify Content Analysis**:
   ```bash
   # Check if concepts are detected
   python -m ai_video_editor.cli.main process video.mp4 \
     --type educational --debug
   
   # Review analysis results
   cat output/analytics/content_analysis.json | grep key_concepts
   ```

2. **Improve Content Clarity**:
   - Speak clearly about specific concepts
   - Use educational terminology
   - Include visual descriptions in speech
   - Structure content with clear sections

3. **Check B-Roll Output**:
   ```bash
   # Look for B-roll files
   ls -la output/broll/
   
   # Review AI Director decisions
   cat output/analytics/ai_decisions.json | grep broll
   ```

**Recovery Actions**:
- Use educational content type for concept-heavy videos
- Ensure clear concept explanations in speech
- Review AI Director decisions for content understanding

## 🖼️ Thumbnail Generation Issues

### Poor Thumbnail Quality

**Issue**: Blurry, low-quality, or inappropriate thumbnails

**Root Causes**:
- Poor source video quality
- Insufficient lighting on speaker
- Missing emotional peaks in content
- API failures affecting generation

**Solutions**:

1. **Improve Source Video Quality**:
   - Use 1080p or higher resolution
   - Ensure good lighting on speaker's face
   - Clear facial expressions and gestures
   - Stable camera work

2. **Verify Analysis Results**:
   ```bash
   # Check visual highlights detection
   cat output/analytics/content_analysis.json | grep visual_highlights
   
   # Review emotional analysis
   cat output/analytics/content_analysis.json | grep emotional_markers
   ```

3. **Optimize Thumbnail Settings**:
   ```bash
   # Use higher quality for better thumbnails
   python -m ai_video_editor.cli.main process video.mp4 \
     --type educational --quality ultra
   ```

**Recovery Actions**:
- Ensure speaker visibility and good lighting
- Use high or ultra quality settings
- Review emotional analysis results for peak detection

### Missing Thumbnail Variations

**Issue**: Only one thumbnail generated or missing strategies

**Expected Output**: 3-5 different strategies with 2-3 variations each (6-15 total thumbnails)

**Solutions**:

1. **Check Emotional Analysis**:
   ```bash
   # Verify emotional peaks detected
   cat output/analytics/content_analysis.json | grep emotional_markers
   ```

2. **Ensure Face Detection**:
   - Good lighting on speaker's face
   - Clear facial expressions
   - Speaker visible and centered in frame

3. **Review AI Director Decisions**:
   ```bash
   # Check thumbnail strategy decisions
   cat output/analytics/ai_decisions.json | grep thumbnail
   ```

**Recovery Actions**:
- Use educational or general content types for multiple strategies
- Ensure clear emotional content in video
- Verify face detection is working properly

## 🔄 Workflow and Orchestration Issues

### Workflow Stuck or Hanging

**Issue**: Processing appears to hang at a specific stage

**Root Causes**:
- Network timeouts during API calls
- Resource exhaustion
- Deadlocks in parallel processing
- Corrupted temporary files

**Solutions**:

1. **Check Workflow Status**:
   ```bash
   # List active workflows
   python -m ai_video_editor.cli.main workflow --list
   
   # Get detailed status
   python -m ai_video_editor.cli.main workflow PROJECT_ID --details
   ```

2. **Increase Timeouts**:
   ```bash
   # Increase processing timeout
   python -m ai_video_editor.cli.main process video.mp4 \
     --timeout 3600  # 1 hour timeout
   ```

3. **Enable Recovery Mode**:
   ```bash
   # Check for checkpoint files
   ls -la temp/checkpoints/
   
   # Process with recovery enabled
   python -m ai_video_editor.cli.main process video.mp4 \
     --enable-recovery
   ```

**Recovery Actions**:
- Monitor processing logs for stuck stages
- Use checkpoint-based recovery
- Restart with appropriate timeout settings

### Stage Failures

**Issue**: Specific processing stage fails consistently

**Common Failing Stages**:
- Audio analysis (Whisper API issues)
- Content analysis (Gemini API issues)
- Thumbnail generation (Imagen API issues)
- B-roll generation (concept detection issues)

**Solutions**:

1. **Skip Problematic Stages**:
   ```bash
   # Skip B-roll generation if failing
   python -m ai_video_editor.cli.main process video.mp4 \
     --skip-broll
   
   # Skip thumbnail generation
   python -m ai_video_editor.cli.main process video.mp4 \
     --skip-thumbnails
   ```

2. **Check Stage-Specific Logs**:
   ```bash
   # Monitor processing logs
   tail -f logs/ai_video_editor.log
   
   # Check for specific errors
   grep ERROR logs/ai_video_editor.log
   grep "stage_name" logs/ai_video_editor.log
   ```

3. **Test Individual Components**:
   ```bash
   # Test thumbnail generation
   python examples/thumbnail_generation_example.py
   
   # Test audio analysis
   python examples/audio_analysis_example.py
   
   # Test content analysis
   python examples/content_analyzer_example.py
   ```

**Recovery Actions**:
- Use graceful degradation to continue processing
- Enable fallback methods for failed stages
- Test individual components to isolate issues

## 🌐 Network and API Issues

### API Connection Problems

**Issue**: Network timeouts, API unavailable, or connection errors

**Root Causes**:
- Internet connectivity issues
- API service outages
- Firewall or proxy blocking requests
- DNS resolution problems

**Solutions**:

1. **Check Internet Connectivity**:
   ```bash
   # Test basic connectivity
   ping google.com
   
   # Test API endpoints
   curl -I https://generativelanguage.googleapis.com
   curl -I https://imagen.googleapis.com
   ```

2. **Configure Proxy Settings**:
   ```bash
   # Set proxy environment variables
   export HTTP_PROXY=http://proxy.company.com:8080
   export HTTPS_PROXY=http://proxy.company.com:8080
   export NO_PROXY=localhost,127.0.0.1
   ```

3. **DNS and Network Troubleshooting**:
   ```bash
   # Check DNS resolution
   nslookup generativelanguage.googleapis.com
   
   # Test with different DNS servers
   export DNS_SERVER=8.8.8.8
   ```

**Recovery Actions**:
- Enable caching to reduce API dependency
- Use offline processing modes when available
- Implement retry logic with exponential backoff

### API Rate Limiting

**Issue**: "Rate limit exceeded", quota errors, or throttling

**Root Causes**:
- Exceeding API rate limits
- Insufficient API quotas
- Batch processing overwhelming APIs
- Concurrent requests from multiple processes

**Solutions**:

1. **Enable Comprehensive Caching**:
   ```bash
   # Reduce API calls through caching
   export AI_VIDEO_EDITOR_ENABLE_CACHING=true
   export AI_VIDEO_EDITOR_CACHE_AGGRESSIVE=true
   ```

2. **Reduce API Call Frequency**:
   ```bash
   # Use fast mode for fewer API calls
   python -m ai_video_editor.cli.main process video.mp4 \
     --mode fast --quality medium
   ```

3. **Batch Processing with Delays**:
   ```bash
   # Process videos with delays
   for video in *.mp4; do
     python -m ai_video_editor.cli.main process "$video"
     sleep 60  # Wait between videos
   done
   ```

4. **Configure Rate Limiting**:
   ```bash
   # Set conservative rate limits
   export AI_VIDEO_EDITOR_API_RATE_LIMIT=10  # requests per minute
   export AI_VIDEO_EDITOR_API_RETRY_DELAY=5  # seconds between retries
   ```

**Recovery Actions**:
- Monitor API usage and quotas
- Implement intelligent batching
- Use fallback methods when APIs are unavailable

## 📁 File and Directory Issues

### Permission Errors

**Issue**: "Permission denied" or file access errors

**Root Causes**:
- Insufficient file permissions
- Directory access restrictions
- Running as wrong user
- File system limitations

**Solutions**:

1. **Check and Fix Permissions**:
   ```bash
   # Check file permissions
   ls -la video.mp4
   
   # Make file readable
   chmod 644 video.mp4
   
   # Check directory permissions
   ls -la output/
   mkdir -p output && chmod 755 output/
   ```

2. **User and Ownership Issues**:
   ```bash
   # Check current user
   whoami
   
   # Change file ownership if needed
   sudo chown $USER:$USER video.mp4
   
   # Avoid running as root unless necessary
   ```

**Recovery Actions**:
- Ensure proper file and directory permissions
- Run with appropriate user privileges
- Use dedicated directories for processing

### Disk Space Issues

**Issue**: "No space left on device" or disk full errors

**Root Causes**:
- Insufficient disk space for temporary files
- Large video files consuming storage
- Cache files accumulating over time
- Output files not being cleaned up

**Solutions**:

1. **Check Available Space**:
   ```bash
   # Check disk usage
   df -h
   du -sh output/
   du -sh temp/
   ```

2. **Clean Temporary Files**:
   ```bash
   # Clean cache directories
   rm -rf temp/cache/*
   rm -rf temp/*/
   
   # Clean old output files
   find output/ -name "*.mp4" -mtime +7 -delete
   ```

3. **Use External Storage**:
   ```bash
   # Process to external drive
   python -m ai_video_editor.cli.main process video.mp4 \
     --output /external/drive/output \
     --temp-dir /external/drive/temp
   ```

**Recovery Actions**:
- Monitor disk usage during processing
- Use external storage for large projects
- Implement automatic cleanup of temporary files

## 🔍 Error Handling and Recovery

### ContentContext Error Recovery

The AI Video Editor uses a ContentContext-driven architecture with built-in error recovery mechanisms.

**Error Categories**:

1. **ContentContext Integrity Errors**
   - Data corruption in processing pipeline
   - Invalid state transitions
   - Missing required data

2. **API Integration Errors**
   - Gemini API failures
   - Imagen API failures
   - Network connectivity issues

3. **Resource Constraint Errors**
   - Memory exhaustion
   - Processing timeouts
   - Disk space issues

**Recovery Strategies**:

1. **Checkpoint-Based Recovery**:
   ```bash
   # Check for recovery checkpoints
   ls -la temp/checkpoints/
   
   # Resume from checkpoint
   python -m ai_video_editor.cli.main process video.mp4 \
     --resume-from-checkpoint PROJECT_ID_stage_name
   ```

2. **Graceful Degradation**:
   - API failures → Use cached data or fallback methods
   - Memory constraints → Reduce quality settings automatically
   - Timeout issues → Process in smaller segments

3. **Partial Processing Recovery**:
   ```bash
   # Continue processing from failed stage
   python -m ai_video_editor.cli.main process video.mp4 \
     --continue-from audio_analysis
   ```

### Debug Mode and Logging

**Enable Comprehensive Debugging**:

```bash
# Enable debug logging
python -m ai_video_editor.cli.main --debug process video.mp4

# Check debug logs
tail -f logs/ai_video_editor.log

# Filter specific log levels
grep ERROR logs/ai_video_editor.log
grep WARNING logs/ai_video_editor.log
grep INFO logs/ai_video_editor.log
```

**Log Analysis Patterns**:

```bash
# Check error patterns
grep -i error logs/ai_video_editor.log | tail -20

# Check performance metrics
grep -i "processing time" logs/ai_video_editor.log

# Check API call patterns
grep -i "api call" logs/ai_video_editor.log
```

## 📊 Performance Optimization

### System Requirements and Recommendations

**Minimum System (Budget)**:
- CPU: 4-core processor (Intel i5/AMD Ryzen 5)
- RAM: 8GB (12GB recommended)
- Storage: 100GB free space (SSD preferred)
- Network: Stable broadband connection

**Recommended System (Balanced)**:
- CPU: 8-core processor (Intel i7/AMD Ryzen 7)
- RAM: 16GB
- Storage: 200GB free SSD space
- Network: High-speed broadband (50+ Mbps)

**High-Performance System (Professional)**:
- CPU: 12+ core processor (Intel i9/AMD Ryzen 9)
- RAM: 32GB+
- Storage: 500GB+ NVMe SSD
- GPU: Dedicated GPU for OpenCV acceleration
- Network: Fiber connection (100+ Mbps)

### Performance Optimization Strategies

**Memory Optimization**:
```bash
# Configure memory limits by system
# 8GB systems
python -m ai_video_editor.cli.main process video.mp4 \
  --max-memory 6 --mode fast --quality medium

# 16GB systems
python -m ai_video_editor.cli.main process video.mp4 \
  --max-memory 12 --parallel --quality high

# 32GB+ systems
python -m ai_video_editor.cli.main process video.mp4 \
  --max-memory 24 --parallel --quality ultra
```

**Processing Speed Optimization**:
```bash
# Fast mode (2-3x faster)
python -m ai_video_editor.cli.main process video.mp4 \
  --mode fast --quality medium

# Balanced mode (recommended)
python -m ai_video_editor.cli.main process video.mp4 \
  --mode balanced --quality high --parallel

# Enable comprehensive caching
export AI_VIDEO_EDITOR_ENABLE_CACHING=true
export AI_VIDEO_EDITOR_CACHE_SIZE_GB=2
```

**Content-Specific Optimization**:
```bash
# Educational content
python -m ai_video_editor.cli.main process lecture.mp4 \
  --type educational --quality high --enable-concept-detection

# Music videos
python -m ai_video_editor.cli.main process music.mp4 \
  --type music --quality ultra --disable-filler-detection

# General content
python -m ai_video_editor.cli.main process video.mp4 \
  --type general --quality high --adaptive-processing
```

## 🆘 Emergency Procedures

### System Recovery Checklist

When everything goes wrong, follow this systematic recovery process:

1. **Stop All Processing**:
   ```bash
   # Kill any running processes
   pkill -f "ai_video_editor"
   
   # Clear any locks
   rm -f temp/*.lock
   ```

2. **Check System Resources**:
   ```bash
   # Check memory usage
   free -h
   
   # Check disk space
   df -h
   
   # Check CPU usage
   top
   ```

3. **Clean Temporary Files**:
   ```bash
   # Clean all temporary files
   rm -rf temp/cache/*
   rm -rf temp/*/
   
   # Reset configuration
   python -m ai_video_editor.cli.main init --output .env
   ```

4. **Verify Installation**:
   ```bash
   # Check Python environment
   python --version
   pip check
   
   # Reinstall if needed
   pip install -r requirements.txt --force-reinstall
   ```

5. **Test Basic Functionality**:
   ```bash
   # Test system status
   python -m ai_video_editor.cli.main status
   
   # Test with minimal settings
   python -m ai_video_editor.cli.main process small_test_video.mp4 \
     --mode fast --quality low --max-memory 4
   ```

### Support Information Collection

When seeking help, collect this diagnostic information:

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

# Recent error logs
tail -50 logs/ai_video_editor.log | grep -E "(ERROR|CRITICAL)"
```

## 📞 Getting Additional Help

### Community Resources

- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check latest updates and examples
- **Performance Benchmarks**: Compare your results
- **Example Code**: Review working implementations

### Before Seeking Help

1. **Check this troubleshooting guide** for your specific issue
2. **Review the FAQ** for common questions
3. **Check recent logs** for error messages
4. **Try basic recovery steps** listed above
5. **Collect diagnostic information** as shown above

### When Reporting Issues

Include this information in your report:
- System specifications (OS, RAM, CPU)
- Python version and package versions
- Complete error messages and logs
- Steps to reproduce the issue
- Video file characteristics (format, size, duration)
- Processing settings used

---

*This unified guide consolidates all troubleshooting information to help you resolve issues quickly and get back to creating amazing content.*