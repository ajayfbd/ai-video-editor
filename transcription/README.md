# Enhanced Sanskrit/Hindi Transcription System

A comprehensive transcription system with built-in Sanskrit and Hindi vocabulary support, granular segmentation control, and professional output formatting.

## 🚀 Quick Start

```bash
# Navigate to scripts directory
cd transcription/scripts

# Basic transcription (balanced 3-second segments)
transcribe_hindi.bat "your_video.mp4" "output_name"

# Granular transcription (custom segment length)
transcribe_granular.bat "your_video.mp4" "output_name" 2
```

## 📁 Folder Structure

```
transcription/
├── scripts/                    # Executable scripts
│   ├── transcribe_hindi.bat   # Main transcription script
│   ├── transcribe_granular.bat # Granular segmentation script
│   └── fix_openmp_conflict.bat # Fix OpenMP library conflicts
├── docs/                       # Documentation
│   ├── TRANSCRIPTION_GUIDE.md  # Complete usage guide
│   └── SEGMENTATION_OPTIONS.md # Segmentation control guide
├── examples/                   # Test and example scripts
│   ├── test_vocabulary.py      # Vocabulary system demo
│   └── compare_vocabulary_results.py # Results comparison
├── output/                     # Transcription outputs
│   ├── example_granular_segments.json
│   ├── example_ultra_granular_1sec.json
│   └── example_comprehensive_vocab.json
└── README.md                   # This file
```

## ✨ Key Features

### 🎯 **Comprehensive Vocabulary System**
- **580+ Sanskrit/Hindi terms** built-in
- **8 specialized categories**: Religious, Deities, Classical, Scriptures, Rituals, Mythological, Philosophical, Common Hindi
- **No external files needed** - everything is built-in
- **Context-aware selection** - automatically chooses relevant terms

### 📊 **Granular Segmentation Control**
- **Natural**: 1-3 large segments (5-30+ seconds)
- **Balanced**: 10-15 segments (2-5 seconds) - recommended for subtitles
- **Granular**: 15-20 segments (1-2 seconds) - precise timing
- **Ultra-Granular**: 40-50+ segments (0.5-1 second) - word-level analysis

### 🎬 **Professional Output**
- **Dual format**: Original Devanagari + Romanized (Hinglish)
- **Precise timing**: Word-level timestamps
- **Multiple formats**: JSON, SRT, VTT subtitles
- **Processing metrics**: Model used, processing time, accuracy stats

## 🎯 Vocabulary Categories

| Category | Terms | Examples |
|----------|-------|----------|
| Religious | 64 | भगवान, प्रभु, भक्ति, पूजा |
| Deities | 118 | राम, कृष्ण, शिव, हनुमान, प्रह्लाद, हिरण्यकशिपु |
| Classical | 90 | योग, ध्यान, वेदांत, संस्कृत |
| Scriptures | 54 | गीता, रामायण, वेद, उपनिषद |
| Rituals | 75 | आरती, यज्ञ, पूजा, मंत्र |
| Mythological | 78 | प्रह्लाद, रावण, भीम, अर्जुन |
| Philosophical | 61 | आत्मा, ब्रह्म, मोक्ष, चेतना |
| Common Hindi | 80 | करना, होना, घर, परिवार |
| **Total Unique** | **580** | **All categories combined** |

## 🛠️ Usage Examples

### Religious/Devotional Content
```bash
python -m ai_video_editor.cli.features transcribe "bhajan.mp4" \
  --preset hindi-religious --vocab-size 100 \
  --output ../transcription/output/bhajan_transcript.json
```

### Mythological Stories
```bash
python -m ai_video_editor.cli.features transcribe "story.mp4" \
  --preset mythological --vocab-size 150 \
  --output ../transcription/output/story_transcript.json
```

### Granular Segmentation for Subtitles
```bash
python -m ai_video_editor.cli.features transcribe "video.mp4" \
  --preset comprehensive --segment-length 3 --word-timestamps \
  --output ../transcription/output/subtitles.json

# Convert to SRT
python -m ai_video_editor.cli.features to-srt ../transcription/output/subtitles.json \
  --output ../transcription/output/subtitles.srt
```

### Ultra-Granular for Language Learning
```bash
python -m ai_video_editor.cli.features transcribe "lesson.mp4" \
  --preset comprehensive --segment-length 1 --word-timestamps \
  --vad-threshold 0.1 --min-silence-duration 100 \
  --output ../transcription/output/word_study.json
```

## 📊 Performance Comparison

| Approach | Model | Accuracy | Speed | Vocabulary | Segments |
|----------|-------|----------|-------|------------|----------|
| Basic | base (auto) | ⭐⭐ | ⭐⭐⭐⭐ | Limited | Few |
| Large Model | large-v3 | ⭐⭐⭐⭐ | ⭐⭐ | Limited | Few |
| **Enhanced** | large-v3 | ⭐⭐⭐⭐⭐ | ⭐⭐ | **580+ terms** | Configurable |

## 🚨 Troubleshooting

### OpenMP Library Conflict
```bash
cd transcription/scripts
fix_openmp_conflict.bat
```

### Memory Issues
- Use `--vocab-size 50` for smaller vocabulary
- Use `--model medium` instead of large
- Close other applications

### Slow Processing
- Use `--device cuda` if GPU available
- Reduce `--vocab-size` to 75
- Use `--model base` for fastest processing

## 📚 Documentation

- **[TRANSCRIPTION_GUIDE.md](docs/TRANSCRIPTION_GUIDE.md)**: Complete usage guide with all options
- **[SEGMENTATION_OPTIONS.md](docs/SEGMENTATION_OPTIONS.md)**: Detailed segmentation control guide

## 🎯 Perfect For

- 🎵 **Bhajans and devotional songs**
- 📖 **Religious discourse and lectures**
- 🎭 **Mythological stories and epics**
- 🧘 **Yoga and meditation content**
- 📚 **Classical Sanskrit texts**
- 🎬 **General Hindi content with religious references**
- 📝 **Subtitle creation**
- 🎓 **Language learning applications**
- 🎤 **Karaoke and sing-along content**

## 🚀 Advanced Features

- **Smart model selection** with automatic fallbacks
- **Progress tracking** with real-time ETA
- **Context-aware vocabulary** selection
- **Professional romanization** with multiple schemes
- **Batch processing** capabilities
- **Error recovery** and graceful degradation
- **Memory optimization** for large files
- **Multi-format export** (JSON, SRT, VTT)

The system provides professional-grade transcription with comprehensive Sanskrit/Hindi support, perfect for any content involving religious, mythological, or classical themes!