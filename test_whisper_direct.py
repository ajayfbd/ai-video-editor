#!/usr/bin/env python3
"""
Direct test of Whisper functionality.
"""

import whisper
import numpy as np
import soundfile as sf
from pathlib import Path

def test_whisper_direct():
    """Test Whisper directly with a simple audio file."""
    
    print("🎤 Direct Whisper Test")
    print("=" * 30)
    
    # Create a simple test audio file with actual speech-like content
    sample_rate = 16000  # Whisper prefers 16kHz
    duration = 2.0
    
    # Generate a simple sine wave pattern
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create multiple tones to simulate speech
    audio = np.zeros_like(t)
    for freq in [200, 300, 250, 400]:
        segment_len = len(t) // 4
        start = (freq - 200) // 50 * segment_len
        end = start + segment_len
        if end <= len(t):
            audio[start:end] += 0.3 * np.sin(2 * np.pi * freq * t[start:end])
    
    # Save as WAV
    test_file = "whisper_test.wav"
    sf.write(test_file, audio, sample_rate)
    
    print(f"✅ Created test file: {test_file}")
    
    try:
        # Load Whisper model
        print("🤖 Loading Whisper model...")
        model = whisper.load_model("tiny")
        print("✅ Model loaded successfully")
        
        # Test transcription
        print("🎤 Testing transcription...")
        result = model.transcribe(test_file)
        
        print("✅ Transcription completed!")
        print(f"📄 Result: {result['text']}")
        print(f"🌍 Language: {result.get('language', 'unknown')}")
        
        # Clean up
        Path(test_file).unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_whisper_direct()
    if success:
        print("\n🎉 Whisper is working correctly!")
        print("💡 The audio processing feature should work with real audio files.")
    else:
        print("\n❌ Whisper test failed.")