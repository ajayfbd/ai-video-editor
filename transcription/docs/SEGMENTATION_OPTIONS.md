# Transcription Segmentation Options

## 🎯 Segmentation Control

The enhanced transcription system now provides precise control over how audio is broken into segments, from natural speech patterns to ultra-granular word-by-word breakdown.

## 📊 Segmentation Levels

### 1. Natural Segmentation (Default)
```bash
# Large segments based on natural speech pauses
python -m ai_video_editor.cli.features transcribe "video.mp4" \
  --no-vad --output ../transcription/output/natural_segments.json
```
- **Result**: 1-3 large segments for entire audio
- **Best for**: General transcription, content analysis
- **Segment length**: Variable (5-30+ seconds)

### 2. Balanced Segmentation
```bash
# 3-5 second segments - good balance
python -m ai_video_editor.cli.features transcribe "video.mp4" \
  --vad --vad-threshold 0.3 --min-silence-duration 300 \
  --segment-length 3 --output ../transcription/output/balanced_segments.json
```
- **Result**: ~10-15 segments for 30-second audio
- **Best for**: Subtitles, general use
- **Segment length**: 2-5 seconds each

### 3. Granular Segmentation
```bash
# 1-2 second segments - precise timing
python -m ai_video_editor.cli.features transcribe "video.mp4" \
  --vad --vad-threshold 0.2 --min-silence-duration 200 \
  --segment-length 2 --word-timestamps --output ../transcription/output/granular_segments.json
```
- **Result**: ~15-20 segments for 30-second audio
- **Best for**: Educational content, precise subtitles
- **Segment length**: 1-2 seconds each

### 4. Ultra-Granular Segmentation
```bash
# Word-by-word breakdown
python -m ai_video_editor.cli.features transcribe "video.mp4" \
  --vad --vad-threshold 0.1 --min-silence-duration 100 \
  --segment-length 1 --word-timestamps --output ../transcription/output/ultra_granular.json
```
- **Result**: 40-50+ segments for 30-second audio
- **Best for**: Word-level analysis, karaoke, language learning
- **Segment length**: 0.5-1 second each

## 🛠️ Parameter Explanation

### VAD (Voice Activity Detection)
- `--vad` / `--no-vad`: Enable/disable voice activity detection
- `--vad-threshold`: Sensitivity (0.0-1.0)
  - **0.1**: Very sensitive, more segments
  - **0.3**: Balanced (recommended)
  - **0.5**: Less sensitive, fewer segments

### Silence Detection
- `--min-silence-duration`: Minimum silence in ms to create segment break
  - **100ms**: Ultra-sensitive, maximum segments
  - **300ms**: Balanced segmentation
  - **500ms**: Conservative, fewer segments
  - **1000ms**: Natural speech pauses only

### Forced Segmentation
- `--segment-length`: Maximum segment length in seconds
  - **0**: Natural segmentation (no forced splits)
  - **1**: Ultra-granular (word-level)
  - **2-3**: Granular (phrase-level)
  - **5+**: Sentence-level

### Word Timestamps
- `--word-timestamps`: Enable word-level timing information
  - Improves segmentation accuracy
  - Slightly slower processing
  - Better for granular segmentation

## 📋 Quick Reference Scripts

### Ultra-Granular (1-second segments)
```bash
cd transcription/scripts
transcribe_granular.bat "video.mp4" "output" 1
```

### Balanced (3-second segments)
```bash
cd transcription/scripts
transcribe_granular.bat "video.mp4" "output" 3
```

### Custom Granular
```bash
python -m ai_video_editor.cli.features transcribe "video.mp4" \
  --preset comprehensive \
  --vad --vad-threshold 0.2 --min-silence-duration 200 \
  --segment-length 2 --word-timestamps \
  --output ../transcription/output/custom_granular.json
```

## 🎬 Use Case Examples

### Subtitle Creation (SRT/VTT)
```bash
# Create granular transcript
python -m ai_video_editor.cli.features transcribe "video.mp4" \
  --segment-length 3 --output ../transcription/output/subtitles.json

# Convert to SRT
python -m ai_video_editor.cli.features to-srt ../transcription/output/subtitles.json \
  --output ../transcription/output/subtitles.srt --max-line-len 40
```

### Language Learning (Word-by-Word)
```bash
# Ultra-granular for word study
python -m ai_video_editor.cli.features transcribe "lesson.mp4" \
  --segment-length 1 --word-timestamps \
  --preset comprehensive --output ../transcription/output/word_study.json
```

### Content Analysis (Natural Flow)
```bash
# Natural segments for content understanding
python -m ai_video_editor.cli.features transcribe "lecture.mp4" \
  --no-vad --preset comprehensive --output ../transcription/output/content_analysis.json
```

### Karaoke/Sing-Along
```bash
# Ultra-precise timing for music
python -m ai_video_editor.cli.features transcribe "song.mp4" \
  --vad-threshold 0.1 --min-silence-duration 50 \
  --segment-length 1 --word-timestamps --output ../transcription/output/karaoke.json
```

## 📊 Comparison Results

| Segmentation Type | 30s Audio | Segments | Avg Length | Best For |
|-------------------|-----------|----------|------------|----------|
| Natural | 1-3 | 10-30s | Content analysis |
| Balanced | 10-15 | 2-3s | General subtitles |
| Granular | 15-20 | 1-2s | Precise subtitles |
| Ultra-Granular | 40-50+ | 0.5-1s | Word-level study |

## ⚡ Performance Impact

| Granularity | Processing Time | Memory Usage | Accuracy |
|-------------|----------------|--------------|----------|
| Natural | Fastest | Lowest | Good |
| Balanced | Fast | Low | Very Good |
| Granular | Medium | Medium | Excellent |
| Ultra-Granular | Slower | Higher | Excellent |

## 🎯 Recommendations

### For Subtitles
- Use **Balanced** (3-second segments)
- Enable VAD with 0.3 threshold
- Use comprehensive vocabulary preset

### For Educational Content
- Use **Granular** (2-second segments)
- Enable word timestamps
- Use appropriate vocabulary preset

### For Language Learning
- Use **Ultra-Granular** (1-second segments)
- Enable word timestamps
- Use comprehensive vocabulary

### For Content Analysis
- Use **Natural** segmentation
- Disable VAD for full context
- Use comprehensive vocabulary

The system now provides complete control over segmentation granularity, from natural speech flow to word-by-word breakdown, perfect for any use case!