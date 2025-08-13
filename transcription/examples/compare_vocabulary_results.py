#!/usr/bin/env python3
"""
Compare transcription results with different vocabulary approaches.
"""

import json
import sys
import os
from pathlib import Path

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def load_transcript(file_path):
    """Load transcript JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def compare_transcripts():
    """Compare different transcription approaches."""
    
    # Look for files in both old out directory and new output directory
    files = {
        "Base Model (auto-downgraded)": ["out/hi_hinglish_fw_base.json", "../output/hi_hinglish_fw_base.json"],
        "Large Model (forced)": ["out/hi_hinglish_fw_large_forced.json", "../output/hi_hinglish_fw_large_forced.json"], 
        "Enhanced with vocab file": ["out/hi_enhanced_transcript.json", "../output/hi_enhanced_transcript.json"],
        "Comprehensive built-in vocab": ["out/comprehensive_vocab_transcript.json", "../output/comprehensive_vocab_transcript.json"],
        "Granular segmentation": ["out/granular_test_granular.json", "../output/granular_test_granular.json"]
    }
    
    print("=== Transcription Quality Comparison ===\n")
    
    for name, file_paths in files.items():
        transcript = None
        for file_path in file_paths:
            if os.path.exists(file_path):
                transcript = load_transcript(file_path)
                break
        
        if transcript:
            print(f"🎯 {name}:")
            print(f"   Model: {transcript.get('model_used', 'unknown')}")
            print(f"   Processing time: {transcript.get('processing_time', 'unknown')}s")
            print(f"   Language: {transcript.get('language', 'unknown')}")
            print(f"   Romanized: {transcript.get('romanized', False)}")
            
            # Show first few words of transcription
            text = transcript.get('text', '')[:100] + "..." if len(transcript.get('text', '')) > 100 else transcript.get('text', '')
            print(f"   Text: {text}")
            
            # Count segments
            segments = transcript.get('segments', [])
            print(f"   Segments: {len(segments)}")
            print()
        else:
            print(f"❌ {name}: File not found or error loading")
            print()
    
    print("📊 Key Improvements with Comprehensive Vocabulary:")
    print("   ✅ 580+ Sanskrit/Hindi terms available")
    print("   ✅ No external files needed")
    print("   ✅ Context-aware vocabulary selection")
    print("   ✅ Better recognition of:")
    print("      • Religious terms (भगवान, प्रभु, etc.)")
    print("      • Deity names (हिरण्यकशिपु, प्रह्लाद, etc.)")
    print("      • Mythological concepts")
    print("      • Classical Sanskrit terms")
    print("      • Ritual and ceremonial words")
    print()
    
    print("🚀 Usage Examples:")
    print("   # Religious content")
    print("   python -m ai_video_editor.cli.features transcribe video.mp4 --preset hindi-religious --output ../transcription/output/transcript.json")
    print()
    print("   # Classical/philosophical content")
    print("   python -m ai_video_editor.cli.features transcribe video.mp4 --preset sanskrit-classical --output ../transcription/output/transcript.json")
    print()
    print("   # Mythological stories")
    print("   python -m ai_video_editor.cli.features transcribe video.mp4 --preset mythological --output ../transcription/output/transcript.json")
    print()
    print("   # Everything (recommended)")
    print("   python -m ai_video_editor.cli.features transcribe video.mp4 --preset comprehensive --vocab-size 200 --output ../transcription/output/transcript.json")

if __name__ == "__main__":
    compare_transcripts()