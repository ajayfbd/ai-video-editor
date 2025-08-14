# AI Video Editor - Project Workspace

## 📁 Folder Structure

```
video_projects/
├── input_clips/     ← PUT YOUR VIDEO CLIPS HERE
├── transcripts/     ← Generated transcripts (auto-created)
├── ai_plans/        ← AI Director plans (auto-created)
├── timelines/       ← Execution timelines (auto-created)
├── outputs/         ← FINISHED VIDEOS (auto-created)
└── batch_process.py ← Automated processing script
```

## 🚀 Quick Start

1. **Add your video clips** to `input_clips/` folder
2. **Run the batch processor**: `python batch_process.py`
3. **Get finished videos** from `outputs/` folder

## 📋 Supported Formats

- **Video**: MP4, AVI, MOV, MKV
- **Audio**: MP3, WAV, M4A (for audio-only content)

## ⚙️ Processing Options

The batch processor will automatically:
- ✅ Transcribe audio using Whisper Large model
- ✅ Analyze content with AI Director
- ✅ Generate intelligent editing plans
- ✅ Create B-roll and effects
- ✅ Render final videos with movis

## 🎯 Content Types

- **Devotional/Religious**: Automatically detects Hindi/Sanskrit content
- **Educational**: Optimizes for learning and engagement
- **Music**: Focuses on rhythm and emotional peaks
- **General**: Balanced approach for any content

## 📊 Output Quality

- **Resolution**: 1920x1080 (Full HD)
- **Frame Rate**: 30 FPS
- **Quality**: High (configurable)
- **Audio**: 48kHz stereo

## 🔧 Advanced Usage

For custom processing, use individual commands:

```bash
# Step 1: Transcribe
ai-ve transcribe video.mp4 --output transcript.json

# Step 2: Generate AI plan
ai-ve generate-plan video.mp4 --transcript transcript.json --ai-director --output plan.json

# Step 3: Execute plan
ai-ve plan-execute --ai-plan plan.json --output timeline.json

# Step 4: Render video
ai-ve render --timeline timeline.json --videos video.mp4 --output final.mp4
```

## 📈 Performance Tips

- **Large files**: Process in segments for better performance
- **Multiple files**: Use batch processing for efficiency
- **Quality vs Speed**: Adjust quality settings in batch_process.py
- **Memory**: Close other applications for large video processing