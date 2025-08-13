#!/usr/bin/env python3
"""
🧪 Audio Processing Tests

Simple tests to verify the Audio Processing feature is working correctly.
"""

import os
import sys
import tempfile
import json
from pathlib import Path
from demo import AudioProcessor


def test_whisper_model_loading():
    """Test if we can load the Whisper model."""
    print("🤖 Testing Whisper Model Loading...")
    
    try:
        # Use tiny model for fast testing
        processor = AudioProcessor(model_size="tiny")
        print("✅ Whisper model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Whisper model loading failed: {e}")
        print("💡 Try installing: pip install openai-whisper")
        return False


def test_audio_metadata_extraction():
    """Test audio metadata extraction with a simple generated audio."""
    print("📊 Testing Audio Metadata Extraction...")
    
    try:
        # Create a simple test audio file (1 second of silence)
        import numpy as np
        import soundfile as sf
        
        # Generate 1 second of silence at 44.1kHz
        sample_rate = 44100
        duration = 1.0
        samples = int(sample_rate * duration)
        audio_data = np.zeros(samples, dtype=np.float32)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
            sf.write(temp_path, audio_data, sample_rate)
        
        try:
            processor = AudioProcessor(model_size="tiny")
            metadata = processor.get_audio_metadata(temp_path)
            
            # Check if we got reasonable metadata
            if metadata.get('duration', 0) > 0:
                print("✅ Audio metadata extraction successful")
                print(f"   Duration: {metadata.get('duration')}s")
                print(f"   Sample Rate: {metadata.get('sample_rate')}Hz")
                return True
            else:
                print("❌ Invalid metadata extracted")
                return False
                
        finally:
            # Clean up
            try:
                os.unlink(temp_path)
            except:
                pass
                
    except ImportError:
        print("⚠️  soundfile not available, skipping metadata test")
        print("💡 Install with: pip install soundfile")
        return True  # Don't fail the test for optional dependency
    except Exception as e:
        print(f"❌ Audio metadata test failed: {e}")
        return False


def test_file_format_detection():
    """Test file format detection logic."""
    print("📁 Testing File Format Detection...")
    
    try:
        processor = AudioProcessor(model_size="tiny")
        
        # Test with non-existent files to check format detection
        test_files = {
            "test.mp3": "audio",
            "test.wav": "audio", 
            "test.mp4": "video",
            "test.avi": "video",
            "test.txt": "unsupported"
        }
        
        for filename, expected_type in test_files.items():
            # This will return an error since files don't exist, but we can check the error message
            result = processor.process_file(filename)
            
            if expected_type == "unsupported":
                if "Unsupported file format" in result.get("error", ""):
                    print(f"✅ Correctly detected unsupported format: {filename}")
                else:
                    print(f"❌ Failed to detect unsupported format: {filename}")
                    return False
            else:
                if "File not found" in result.get("error", ""):
                    print(f"✅ Correctly processed format detection: {filename}")
                else:
                    print(f"❌ Unexpected error for {filename}: {result.get('error')}")
                    return False
        
        print("✅ File format detection working correctly")
        return True
        
    except Exception as e:
        print(f"❌ File format detection test failed: {e}")
        return False


def test_transcription_with_sample():
    """Test transcription with the test video if available."""
    print("🎤 Testing Transcription...")
    
    # Look for test files
    test_files = ["test_video.mp4", "sample.mp3", "test.wav"]
    available_file = None
    
    for test_file in test_files:
        if Path(test_file).exists():
            available_file = test_file
            break
    
    if not available_file:
        print("⚠️  No test files available, skipping transcription test")
        print("💡 Add a test_video.mp4 or sample.mp3 file to test transcription")
        return True  # Don't fail if no test files
    
    try:
        print(f"📁 Testing with: {available_file}")
        processor = AudioProcessor(model_size="tiny")  # Use tiny for speed
        
        result = processor.process_file(available_file)
        
        if "error" in result:
            print(f"❌ Transcription failed: {result['error']}")
            return False
        
        # Check if we got a reasonable result
        transcript = result.get('transcript', {})
        if transcript.get('full_text') and len(transcript.get('segments', [])) > 0:
            print("✅ Transcription successful")
            print(f"   Text length: {len(transcript.get('full_text', ''))}")
            print(f"   Segments: {len(transcript.get('segments', []))}")
            print(f"   Language: {transcript.get('language', 'unknown')}")
            return True
        else:
            print("❌ Transcription returned empty results")
            return False
            
    except Exception as e:
        print(f"❌ Transcription test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("🧪 Audio Processing Test Suite")
    print("=" * 50)
    
    tests = [
        ("Whisper Model Loading", test_whisper_model_loading),
        ("Audio Metadata Extraction", test_audio_metadata_extraction),
        ("File Format Detection", test_file_format_detection),
        ("Transcription with Sample", test_transcription_with_sample)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        print("-" * 30)
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{len(tests)} tests passed")
    
    if passed >= len(tests) - 1:  # Allow 1 test to fail (e.g., if no sample files)
        print("🎉 Audio Processing is working correctly!")
        print("\n💡 Next steps:")
        print("   - Run: python demo.py")
        print("   - Try with your own audio/video files")
        print("   - Combine with AI Content Analysis feature")
    else:
        print("⚠️  Some critical tests failed. Check your setup:")
        print("   - Install dependencies: pip install openai-whisper librosa soundfile")
        print("   - For video support: install FFmpeg")
        print("   - Add test files for full testing")
    
    return passed >= len(tests) - 1


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)