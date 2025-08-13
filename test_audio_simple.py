#!/usr/bin/env python3
"""
Simple test of audio transcription feature.
"""

import sys
from pathlib import Path

# Add the feature directory to path
sys.path.insert(0, str(Path("features/01-audio-processing")))

from demo import AudioProcessor

def test_simple_transcription():
    """Test transcription with our test audio file."""
    
    print("🎵 Simple Audio Transcription Test")
    print("=" * 50)
    
    # Check if test file exists
    test_file = "test_audio.wav"
    if not Path(test_file).exists():
        print(f"❌ Test file not found: {test_file}")
        print("💡 Run: python create_test_audio.py")
        return False
    
    try:
        print(f"📁 Testing with: {test_file}")
        
        # Use tiny model for speed
        processor = AudioProcessor(model_size="tiny")
        
        # Process the file
        result = processor.process_file(test_file)
        
        if "error" in result:
            print(f"❌ Processing failed: {result['error']}")
            return False
        
        # Display results
        print("\n📊 Results:")
        print("=" * 30)
        
        transcript = result.get('transcript', {})
        metadata = result.get('audio_metadata', {})
        
        print(f"✅ Transcription successful!")
        print(f"📄 Text: {transcript.get('full_text', 'No text')}")
        print(f"🎵 Duration: {metadata.get('duration', 0)}s")
        print(f"🌍 Language: {transcript.get('language', 'unknown')}")
        print(f"📊 Segments: {len(transcript.get('segments', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_simple_transcription()
    if success:
        print("\n🎉 Audio processing feature is working!")
    else:
        print("\n❌ Audio processing test failed.")
    
    sys.exit(0 if success else 1)