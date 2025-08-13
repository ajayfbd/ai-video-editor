# Enhanced Sanskrit/Hindi Transcription Guide

## 🚀 Quick Start

### Simplified Command (Recommended)
```bash
# Use the simplified script
cd transcription/scripts
transcribe_hindi.bat "your_video.mp4" "output_name"
```

### Manual Command with All Features
```bash
python -m ai_video_editor.cli.features transcribe "video.mp4" \
  --backend faster-whisper \
  --model large \
  --force-model \
  --preset comprehensive \
  --vocab-size 150 \
  --progress \
  --romanize \
  --output ../transcription/output/transcript.json
```

## 📚 Vocabulary Presets

### `--preset comprehensive` (Recommended)
- **580+ terms** from all categories
- Best for mixed religious/mythological content
- Includes deities, scriptures, rituals, philosophy

### `--preset hindi-religious`
- **~200 terms** focused on devotional content
- Best for bhajans, prayers, religious discourse
- Terms: भगवान, प्रभु, भक्ति, पूजा, आरती, etc.

### `--preset sanskrit-classical`
- **~200 terms** for classical texts
- Best for philosophical content, yoga, Ayurveda
- Terms: योग, ध्यान, वेदांत, उपनिषद, etc.

### `--preset mythological`
- **~200 terms** for epic stories
- Best for Ramayana, Mahabharata content
- Terms: प्रह्लाद, हिरण्यकशिपु, रावण, etc.

## 🎯 Vocabulary Statistics

| Category | Terms | Examples |
|----------|-------|----------|
| Religious | 64 | भगवान, प्रभु, भक्ति, पूजा |
| Deities | 118 | राम, कृष्ण, शिव, हनुमान |
| Classical | 90 | योग, ध्यान, वेदांत, संस्कृत |
| Scriptures | 54 | गीता, रामायण, वेद, उपनिषद |
| Rituals | 75 | आरती, यज्ञ, पूजा, मंत्र |
| Mythological | 78 | प्रह्लाद, रावण, भीम, अर्जुन |
| Philosophical | 61 | आत्मा, ब्रह्म, मोक्ष, चेतना |
| Common Hindi | 80 | करना, होना, घर, परिवार |
| **Total Unique** | **580** | **All categories combined** |

## ⚙️ Advanced Options

### Vocabulary Size Control
```bash
--vocab-size 50    # Smaller, faster
--vocab-size 100   # Balanced (default)
--vocab-size 200   # Maximum coverage
```

### Model Selection
```bash
--model large --force-model    # Best accuracy (slow)
--model medium --force-model   # Good balance
--model base                   # Fastest (auto-selected on CPU)
```

### Progress Display
```bash
--progress      # Show progress bar (default)
--no-progress   # Silent processing
--quiet         # Minimal output
```

## 🔍 Quality Comparison

| Approach | Model | Accuracy | Speed | Vocabulary |
|----------|-------|----------|-------|------------|
| Basic | base (auto) | ⭐⭐ | ⭐⭐⭐⭐ | Limited |
| Large Model | large-v3 | ⭐⭐⭐⭐ | ⭐⭐ | Limited |
| **Enhanced** | large-v3 | ⭐⭐⭐⭐⭐ | ⭐⭐ | **580+ terms** |

## 🎬 Content-Specific Examples

### Religious/Devotional Content
```bash
python -m ai_video_editor.cli.features transcribe "bhajan.mp4" \
  --preset hindi-religious --vocab-size 100 --output bhajan_transcript.json
```

### Mythological Stories (Ramayana/Mahabharata)
```bash
python -m ai_video_editor.cli.features transcribe "story.mp4" \
  --preset mythological --vocab-size 150 --output story_transcript.json
```

### Yoga/Meditation Content
```bash
python -m ai_video_editor.cli.features transcribe "yoga.mp4" \
  --preset sanskrit-classical --vocab-size 120 --output yoga_transcript.json
```

### Mixed Content (Best for Unknown Content)
```bash
python -m ai_video_editor.cli.features transcribe "mixed.mp4" \
  --preset comprehensive --vocab-size 200 --output mixed_transcript.json
```

## 🛠️ Technical Details

### Built-in Vocabulary System
- **No external files needed** - everything is built-in
- **Context-aware selection** - automatically chooses relevant terms
- **Randomized sampling** - prevents overfitting to specific word orders
- **Comprehensive coverage** - religious, classical, mythological, common terms

### Performance Optimization
- **Smart model selection** - automatic downgrading on resource constraints
- **Progress tracking** - real-time feedback with ETA
- **Memory efficient** - vocabulary loaded on-demand
- **Error recovery** - graceful fallbacks for API/resource issues

### Output Features
- **Dual text format** - original Devanagari + romanized (Hinglish)
- **Segment timing** - precise word-level timestamps
- **Processing metrics** - model used, processing time, accuracy stats
- **Multiple export formats** - JSON, SRT, VTT subtitles

## 🚨 Troubleshooting

### OpenMP Error
```bash
# Set environment variable
set KMP_DUPLICATE_LIB_OK=TRUE
# Or run the fix script
cd transcription/scripts
fix_openmp_conflict.bat
```

### Memory Issues
```bash
# Use smaller vocabulary
--vocab-size 50

# Use smaller model
--model medium  # instead of large
```

### Slow Processing
```bash
# Use GPU if available
--device cuda

# Reduce vocabulary size
--vocab-size 75

# Use smaller model
--model base
```

## 📈 Results

The enhanced system provides:
- **Better recognition** of Sanskrit/Hindi religious terms
- **Improved accuracy** for mythological names and concepts
- **No dependency** on external vocabulary files
- **Flexible configuration** for different content types
- **Professional output** with timing and romanization

Perfect for transcribing:
- 🎵 Bhajans and devotional songs
- 📖 Religious discourse and lectures  
- 🎭 Mythological stories and epics
- 🧘 Yoga and meditation content
- 📚 Classical Sanskrit texts
- 🎬 General Hindi content with religious references