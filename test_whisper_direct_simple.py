#!/usr/bin/env python3
"""
Direct test of Whisper with our audio files to debug transcription issues.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_whisper_direct():
    """Test Whisper directly without caching."""
    
    print("🎤 Direct Whisper Test (No Cache)")
    print("=" * 40)
    
    try:
        import whisper
        
        # Load a small model for testing
        print("Loading Whisper model...")
        model = whisper.load_model("base")
        print(f"✅ Model loaded: {model.is_multilingual}")
        
        # Test files
        test_files = ["test_audio.wav", "whisper_test.wav"]
        
        for audio_file in test_files:
            if not Path(audio_file).exists():
                print(f"❌ File not found: {audio_file}")
                continue
            
            print(f"\n🎵 Testing: {audio_file}")
            file_size = Path(audio_file).stat().st_size
            print(f"   File size: {file_size:,} bytes")
            
            # Transcribe with detailed options
            result = model.transcribe(
                audio_file,
                language=None,  # Auto-detect
                word_timestamps=True,
                verbose=True,
                temperature=0.0,  # More deterministic
                best_of=1,
                beam_size=1
            )
            
            print(f"   Language detected: {result.get('language', 'unknown')}")
            print(f"   Text length: {len(result['text'])}")
            print(f"   Segments: {len(result.get('segments', []))}")
            
            if result['text'].strip():
                print(f"   📝 Transcribed text: '{result['text']}'")
            else:
                print("   ❌ No text transcribed")
            
            # Show segment details
            for i, segment in enumerate(result.get('segments', [])[:3]):
                print(f"   Segment {i+1}: {segment['start']:.2f}s-{segment['end']:.2f}s: '{segment['text']}'")
                print(f"      Confidence: {segment.get('avg_logprob', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Whisper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_audio_properties():
    """Analyze audio file properties to understand why Whisper might not work."""
    
    print("\n🔍 Audio File Analysis")
    print("=" * 40)
    
    try:
        import soundfile as sf
        import numpy as np
        
        test_files = ["test_audio.wav", "whisper_test.wav"]
        
        for audio_file in test_files:
            if not Path(audio_file).exists():
                continue
            
            print(f"\n📊 Analyzing: {audio_file}")
            
            # Load audio
            data, sample_rate = sf.read(audio_file)
            
            print(f"   Sample rate: {sample_rate} Hz")
            print(f"   Duration: {len(data) / sample_rate:.2f} seconds")
            print(f"   Channels: {data.shape[1] if len(data.shape) > 1 else 1}")
            print(f"   Data type: {data.dtype}")
            print(f"   Min value: {np.min(data):.6f}")
            print(f"   Max value: {np.max(data):.6f}")
            print(f"   RMS level: {np.sqrt(np.mean(data**2)):.6f}")
            
            # Check for silence
            silence_threshold = 0.001
            non_silent_samples = np.sum(np.abs(data) > silence_threshold)
            silence_ratio = 1.0 - (non_silent_samples / len(data))
            
            print(f"   Silence ratio: {silence_ratio:.2%}")
            
            if silence_ratio > 0.9:
                print("   ⚠️  File appears to be mostly silent")
            elif silence_ratio > 0.5:
                print("   ⚠️  File has significant silence")
            else:
                print("   ✅ File has reasonable audio content")
            
            # Frequency analysis
            fft = np.fft.fft(data)
            freqs = np.fft.fftfreq(len(data), 1/sample_rate)
            magnitude = np.abs(fft)
            
            # Find dominant frequencies
            dominant_freq_idx = np.argmax(magnitude[:len(magnitude)//2])
            dominant_freq = abs(freqs[dominant_freq_idx])
            
            print(f"   Dominant frequency: {dominant_freq:.1f} Hz")
            
            # Check if frequency content is in speech range
            if 80 <= dominant_freq <= 8000:
                print("   ✅ Dominant frequency in speech range")
            else:
                print("   ⚠️  Dominant frequency outside typical speech range")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio analysis failed: {e}")
        return False

def main():
    """Run direct Whisper tests."""
    
    print("🚀 Direct Whisper Testing & Audio Analysis")
    
    # Analyze audio properties first
    analyze_audio_properties()
    
    # Test Whisper directly
    whisper_success = test_whisper_direct()
    
    if not whisper_success:
        print("\n💡 Suggestions:")
        print("   1. Try recording actual speech with a microphone")
        print("   2. Use a text-to-speech tool to create realistic audio")
        print("   3. Download sample speech audio from the internet")
        print("   4. The synthesized audio might not be realistic enough for Whisper")
    
    return whisper_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)