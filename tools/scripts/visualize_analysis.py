#!/usr/bin/env python3
"""Visualize analysis results in a readable format."""

import json
import sys
from pathlib import Path


def load_analysis(analysis_dir):
    """Load all analysis files."""
    analysis_dir = Path(analysis_dir)
    
    files = {
        'complete': analysis_dir / 'complete_analysis.json',
        'audio': analysis_dir / 'audio_analysis.json', 
        'report': analysis_dir / 'analysis_report.json',
        'transcript': analysis_dir / 'transcript.json'
    }
    
    data = {}
    for name, file_path in files.items():
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data[name] = json.load(f)
        else:
            print(f"⚠️  {file_path} not found")
    
    return data


def print_audio_analysis(data):
    """Print detailed audio analysis."""
    if 'audio' not in data:
        return
    
    audio = data['audio']
    print("🎵 AUDIO ANALYSIS")
    print("=" * 50)
    
    # Statistics
    stats = audio.get('statistics', {})
    print(f"Duration: {stats.get('total_duration', 0):.1f} seconds")
    print(f"Segments: {stats.get('segment_count', 0)}")
    print(f"Language: {stats.get('language', 'unknown')}")
    print(f"Model: {stats.get('model_used', 'unknown')}")
    print(f"Avg Confidence: {stats.get('average_confidence', 0):.3f}")
    
    # Content Analysis
    content = audio.get('content_analysis', {})
    print(f"\n📝 Content Themes:")
    themes = content.get('themes', {})
    for theme, count in themes.items():
        if count > 0:
            print(f"  {theme.title()}: {count} keywords detected")
    
    print(f"Dominant Theme: {content.get('dominant_theme', 'unknown').title()}")
    print(f"Speaking Rate: {content.get('speaking_rate', 0):.1f} words/second")
    print(f"Vocabulary: {content.get('unique_words', 0)}/{content.get('word_count', 0)} unique words")
    
    # Repeated phrases
    repeated = content.get('repeated_phrases', [])
    if repeated:
        print(f"\n🔄 Repeated Phrases ({len(repeated)} found):")
        for phrase in repeated[:3]:  # Show first 3
            print(f"  Similarity: {phrase['similarity']:.2f}")
            print(f"  Text 1: {phrase['text1'][:50]}...")
            print(f"  Text 2: {phrase['text2'][:50]}...")
            print()


def print_transcript_segments(data):
    """Print transcript segments with timing."""
    if 'transcript' not in data:
        return
    
    segments = data['transcript'].get('segments', [])
    if not segments:
        return
    
    print("\n📝 TRANSCRIPT SEGMENTS")
    print("=" * 50)
    
    for i, seg in enumerate(segments):
        start = seg.get('start', 0)
        end = seg.get('end', 0)
        text = seg.get('text', '')
        confidence = seg.get('confidence', 0)
        
        print(f"[{i+1:2d}] {start:6.2f}s - {end:6.2f}s | Conf: {confidence:6.3f}")
        print(f"     {text}")
        print()


def print_video_analysis(data):
    """Print video analysis results."""
    if 'complete' not in data or 'video' not in data['complete']:
        return
    
    video = data['complete']['video']
    print("\n🎥 VIDEO ANALYSIS")
    print("=" * 50)
    
    # Metadata
    metadata = video.get('metadata', {})
    if metadata:
        print(f"Resolution: {metadata.get('width', 0)}x{metadata.get('height', 0)}")
        print(f"FPS: {metadata.get('fps', 0)}")
        print(f"Duration: {metadata.get('duration', 0):.1f}s")
        print(f"Codec: {metadata.get('codec', 'unknown')}")
    
    # Scene detection
    scenes = video.get('scenes', [])
    print(f"\n🎬 Scenes Detected: {len(scenes)}")
    for i, scene in enumerate(scenes[:5]):  # Show first 5
        print(f"  [{i+1}] {scene.get('timestamp', 0):.2f}s - {scene.get('description', 'Scene change')}")
    
    # Face detection
    faces = video.get('faces', [])
    print(f"\n👤 Faces Detected: {len(faces)}")
    for i, face in enumerate(faces[:5]):  # Show first 5
        print(f"  [{i+1}] {face.get('timestamp', 0):.2f}s - {face.get('expression', 'neutral')} (conf: {face.get('confidence', 0):.2f})")
    
    # Statistics
    stats = video.get('statistics', {})
    print(f"\nAnalysis Quality: {stats.get('analysis_quality', 'unknown').title()}")


def print_recommendations(data):
    """Print editing recommendations."""
    if 'report' not in data:
        return
    
    report = data['report']
    print("\n💡 RECOMMENDATIONS")
    print("=" * 50)
    
    recommendations = report.get('recommendations', [])
    if recommendations:
        print("General Recommendations:")
        for rec in recommendations:
            print(f"  • {rec}")
    
    suggestions = report.get('editing_suggestions', [])
    if suggestions:
        print("\nEditing Suggestions:")
        for sug in suggestions:
            print(f"  • {sug}")
    
    if not recommendations and not suggestions:
        print("No specific recommendations generated.")


def main():
    """Main visualization function."""
    if len(sys.argv) != 2:
        print("Usage: python visualize_analysis.py <analysis_directory>")
        sys.exit(1)
    
    analysis_dir = sys.argv[1]
    if not Path(analysis_dir).exists():
        print(f"Error: Directory {analysis_dir} not found")
        sys.exit(1)
    
    print("🎬 AI VIDEO EDITOR - ANALYSIS VISUALIZATION")
    print("=" * 60)
    
    # Load all analysis data
    data = load_analysis(analysis_dir)
    
    if not data:
        print("No analysis files found!")
        return
    
    # Print each section
    print_audio_analysis(data)
    print_transcript_segments(data)
    print_video_analysis(data)
    print_recommendations(data)
    
    print("\n" + "=" * 60)
    print("✅ Analysis visualization complete!")


if __name__ == "__main__":
    main()