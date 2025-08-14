#!/usr/bin/env python3
"""Compare different analysis methods and their performance."""

import json
import time
from pathlib import Path


def load_analysis(analysis_dir):
    """Load analysis results."""
    analysis_dir = Path(analysis_dir)
    complete_file = analysis_dir / 'complete_analysis.json'
    
    if complete_file.exists():
        with open(complete_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def print_comparison_table(analyses, labels):
    """Print comparison table."""
    print("📊 ANALYSIS METHOD COMPARISON")
    print("=" * 80)
    
    # Headers
    header = f"{'Metric':<25}"
    for label in labels:
        header += f"{label:<20}"
    print(header)
    print("-" * 80)
    
    # Transcription quality
    row = f"{'Transcription Model':<25}"
    for analysis in analyses:
        if analysis and 'audio' in analysis:
            model = analysis['audio'].get('statistics', {}).get('model_used', 'N/A')
            row += f"{model:<20}"
        else:
            row += f"{'N/A':<20}"
    print(row)
    
    # Processing time
    row = f"{'Processing Time':<25}"
    for analysis in analyses:
        if analysis and 'metadata' in analysis:
            time_val = analysis['metadata'].get('analysis_time', 0)
            row += f"{time_val:.1f}s{'':<15}"
        else:
            row += f"{'N/A':<20}"
    print(row)
    
    # Audio segments
    row = f"{'Audio Segments':<25}"
    for analysis in analyses:
        if analysis and 'audio' in analysis:
            segments = analysis['audio'].get('statistics', {}).get('segment_count', 0)
            row += f"{segments:<20}"
        else:
            row += f"{'N/A':<20}"
    print(row)
    
    # Video scenes
    row = f"{'Video Scenes':<25}"
    for analysis in analyses:
        if analysis and 'video' in analysis:
            scenes = len(analysis['video'].get('scenes', []))
            row += f"{scenes:<20}"
        else:
            row += f"{'N/A':<20}"
    print(row)
    
    # Face detections
    row = f"{'Face Detections':<25}"
    for analysis in analyses:
        if analysis and 'video' in analysis:
            faces = len(analysis['video'].get('faces', []))
            row += f"{faces:<20}"
        else:
            row += f"{'N/A':<20}"
    print(row)
    
    # Analysis quality
    row = f"{'Analysis Quality':<25}"
    for analysis in analyses:
        if analysis and 'video' in analysis:
            quality = analysis['video'].get('statistics', {}).get('analysis_quality', 'N/A')
            row += f"{quality:<20}"
        else:
            row += f"{'N/A':<20}"
    print(row)


def print_transcript_quality(analyses, labels):
    """Print transcript quality comparison."""
    print("\n📝 TRANSCRIPT QUALITY COMPARISON")
    print("=" * 80)
    
    for i, (analysis, label) in enumerate(zip(analyses, labels)):
        if not analysis or 'audio' not in analysis:
            continue
            
        print(f"\n{label.upper()}:")
        
        audio = analysis['audio']
        transcript = audio.get('transcript', {})
        
        # Show first segment
        segments = transcript.get('segments', [])
        if segments:
            first_seg = segments[0]
            print(f"  First segment: {first_seg.get('text', 'N/A')[:60]}...")
            
            # Show romanization info
            if 'text_original' in first_seg:
                print(f"  Original:      {first_seg.get('text_original', 'N/A')[:60]}...")
                print(f"  Romanized:     Yes")
            else:
                print(f"  Romanized:     No")
        
        # Show statistics
        stats = audio.get('statistics', {})
        print(f"  Duration:      {stats.get('total_duration', 0):.1f}s")
        print(f"  Confidence:    {stats.get('average_confidence', 0):.3f}")
        
        # Show content analysis
        content = audio.get('content_analysis', {})
        theme = content.get('dominant_theme', 'unknown')
        print(f"  Theme:         {theme.title()}")


def print_performance_summary(analyses, labels):
    """Print performance summary."""
    print("\n⚡ PERFORMANCE SUMMARY")
    print("=" * 50)
    
    fastest_time = float('inf')
    fastest_method = ""
    
    for analysis, label in zip(analyses, labels):
        if analysis and 'metadata' in analysis:
            time_val = analysis['metadata'].get('analysis_time', 0)
            if time_val < fastest_time:
                fastest_time = time_val
                fastest_method = label
    
    if fastest_method:
        print(f"🏆 Fastest Method: {fastest_method} ({fastest_time:.1f}s)")
    
    # Quality assessment
    best_quality = ""
    best_score = 0
    
    for analysis, label in zip(analyses, labels):
        if analysis and 'video' in analysis:
            quality = analysis['video'].get('statistics', {}).get('analysis_quality', 'unknown')
            score = {'high': 3, 'medium': 2, 'low': 1, 'minimal': 0}.get(quality, 0)
            if score > best_score:
                best_score = score
                best_quality = label
    
    if best_quality:
        print(f"🎯 Best Quality: {best_quality}")
    
    print("\n💡 RECOMMENDATIONS:")
    print("  • Use faster-whisper for best speed/quality balance")
    print("  • Use medium/large models for better transcription accuracy")
    print("  • Enable --force-model for CPU if you need larger models")
    print("  • Video analysis works well with OpenCV implementation")


def main():
    """Main comparison function."""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python compare_analysis_methods.py <analysis_dir1> <analysis_dir2> [analysis_dir3] ...")
        print("\nExample:")
        print("  python compare_analysis_methods.py detailed_analysis improved_analysis final_analysis")
        sys.exit(1)
    
    # Load all analyses
    analysis_dirs = sys.argv[1:]
    analyses = []
    labels = []
    
    for dir_path in analysis_dirs:
        path = Path(dir_path)
        if path.exists():
            analysis = load_analysis(path)
            analyses.append(analysis)
            labels.append(path.name)
        else:
            print(f"⚠️  Directory not found: {dir_path}")
            analyses.append(None)
            labels.append(path.name + " (missing)")
    
    if not any(analyses):
        print("No valid analyses found!")
        return
    
    print("🎬 AI VIDEO EDITOR - ANALYSIS METHOD COMPARISON")
    print("=" * 80)
    
    # Print comparisons
    print_comparison_table(analyses, labels)
    print_transcript_quality(analyses, labels)
    print_performance_summary(analyses, labels)
    
    print("\n" + "=" * 80)
    print("✅ Comparison complete!")


if __name__ == "__main__":
    main()