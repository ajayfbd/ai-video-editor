#!/usr/bin/env python3
"""Compare different transcription results."""

import json
import sys
from pathlib import Path


def load_transcript(file_path):
    """Load transcript JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def print_transcript_comparison(transcripts, labels):
    """Print side-by-side comparison of transcripts."""
    print("📝 TRANSCRIPTION COMPARISON")
    print("=" * 80)
    
    # Print headers
    header = ""
    for i, label in enumerate(labels):
        header += f"{label:35s} | "
    print(header)
    print("-" * 80)
    
    # Get max segments
    max_segments = max(len(t.get('segments', [])) for t in transcripts if t)
    
    # Compare segment by segment
    for i in range(max_segments):
        print(f"\nSegment {i+1}:")
        
        for j, transcript in enumerate(transcripts):
            if not transcript:
                text = "N/A"
                timing = "N/A"
            else:
                segments = transcript.get('segments', [])
                if i < len(segments):
                    seg = segments[i]
                    text = seg.get('text', 'N/A')[:30] + "..."
                    start = seg.get('start', 0)
                    end = seg.get('end', 0)
                    timing = f"{start:.1f}-{end:.1f}s"
                else:
                    text = "N/A"
                    timing = "N/A"
            
            print(f"  {labels[j]:12s}: {timing:10s} | {text}")


def print_quality_metrics(transcripts, labels):
    """Print quality metrics comparison."""
    print("\n📊 QUALITY METRICS")
    print("=" * 60)
    
    for i, (transcript, label) in enumerate(zip(transcripts, labels)):
        if not transcript:
            continue
            
        print(f"\n{label}:")
        
        # Basic stats
        segments = transcript.get('segments', [])
        total_duration = segments[-1].get('end', 0) if segments else 0
        
        print(f"  Duration: {total_duration:.1f}s")
        print(f"  Segments: {len(segments)}")
        print(f"  Language: {transcript.get('language', 'unknown')}")
        print(f"  Model: {transcript.get('model_used', 'unknown')}")
        
        # Confidence
        if segments and 'confidence' in segments[0]:
            avg_conf = sum(seg.get('confidence', 0) for seg in segments) / len(segments)
            print(f"  Avg Confidence: {avg_conf:.3f}")
        
        # Text analysis
        text = transcript.get('text', '')
        words = text.split()
        print(f"  Word Count: {len(words)}")
        print(f"  Unique Words: {len(set(words))}")
        
        # Romanization info
        if transcript.get('romanized'):
            print(f"  Romanized: Yes ({transcript.get('romanization_scheme', 'unknown')})")
            if 'text_original' in transcript:
                print(f"  Original Script: Available")


def main():
    """Main comparison function."""
    if len(sys.argv) < 3:
        print("Usage: python compare_transcriptions.py <transcript1.json> <transcript2.json> [transcript3.json] ...")
        print("\nExample:")
        print("  python compare_transcriptions.py out/transcript_basic.json out/transcript_hinglish.json detailed_analysis/transcript.json")
        sys.exit(1)
    
    # Load all transcripts
    transcript_files = sys.argv[1:]
    transcripts = []
    labels = []
    
    for file_path in transcript_files:
        path = Path(file_path)
        if path.exists():
            transcript = load_transcript(path)
            transcripts.append(transcript)
            labels.append(path.stem)
        else:
            print(f"⚠️  File not found: {file_path}")
            transcripts.append(None)
            labels.append(path.stem + " (missing)")
    
    if not any(transcripts):
        print("No valid transcripts found!")
        return
    
    print("🎬 AI VIDEO EDITOR - TRANSCRIPTION COMPARISON")
    print("=" * 80)
    
    # Print comparison
    print_transcript_comparison(transcripts, labels)
    print_quality_metrics(transcripts, labels)
    
    print("\n" + "=" * 80)
    print("✅ Comparison complete!")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    
    # Find best quality
    best_conf = -999
    best_idx = -1
    for i, transcript in enumerate(transcripts):
        if transcript and transcript.get('segments'):
            segments = transcript['segments']
            if 'confidence' in segments[0]:
                avg_conf = sum(seg.get('confidence', 0) for seg in segments) / len(segments)
                if avg_conf > best_conf:
                    best_conf = avg_conf
                    best_idx = i
    
    if best_idx >= 0:
        print(f"  • Best quality: {labels[best_idx]} (confidence: {best_conf:.3f})")
    
    # Check for romanization
    romanized = [i for i, t in enumerate(transcripts) if t and t.get('romanized')]
    if romanized:
        print(f"  • Romanized versions: {', '.join(labels[i] for i in romanized)}")
    
    # Check for different models
    models = set()
    for t in transcripts:
        if t and t.get('model_used'):
            models.add(t['model_used'])
    
    if len(models) > 1:
        print(f"  • Models compared: {', '.join(models)}")


if __name__ == "__main__":
    main()