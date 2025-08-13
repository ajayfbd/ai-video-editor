#!/usr/bin/env python3
"""
Create realistic test content for audio/video analysis testing.
"""

import os
import sys
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf

def create_realistic_audio_with_tts():
    """Create realistic audio using text-to-speech if available."""
    
    # Test text with financial content
    financial_text = """
    Welcome to financial education. Today we will discuss investment strategies and portfolio management.
    Understanding compound interest is crucial for long-term wealth building.
    Diversification helps reduce risk in your investment portfolio.
    Let's analyze some data and charts to understand market trends.
    """
    
    print("🎤 Attempting to create realistic audio with TTS...")
    
    # Try different TTS approaches
    tts_methods = [
        ("pyttsx3", create_audio_with_pyttsx3),
        ("gTTS", create_audio_with_gtts),
        ("espeak", create_audio_with_espeak)
    ]
    
    for method_name, method_func in tts_methods:
        try:
            print(f"   Trying {method_name}...")
            success = method_func(financial_text)
            if success:
                print(f"   ✅ Successfully created audio with {method_name}")
                return True
        except Exception as e:
            print(f"   ❌ {method_name} failed: {e}")
            continue
    
    print("   ⚠️  No TTS methods available, using synthesized speech")
    return create_advanced_synthesized_speech()

def create_audio_with_pyttsx3(text):
    """Create audio using pyttsx3 (offline TTS)."""
    try:
        import pyttsx3
        
        engine = pyttsx3.init()
        
        # Configure voice settings
        voices = engine.getProperty('voices')
        if voices:
            engine.setProperty('voice', voices[0].id)
        
        engine.setProperty('rate', 150)  # Speaking rate
        engine.setProperty('volume', 0.8)
        
        # Save to file
        engine.save_to_file(text, 'test_audio_tts.wav')
        engine.runAndWait()
        
        # Convert to our test files
        if Path('test_audio_tts.wav').exists():
            # Copy to our test files
            import shutil
            shutil.copy('test_audio_tts.wav', 'test_audio.wav')
            shutil.copy('test_audio_tts.wav', 'whisper_test.wav')
            return True
        
        return False
        
    except ImportError:
        return False

def create_audio_with_gtts(text):
    """Create audio using Google Text-to-Speech."""
    try:
        from gtts import gTTS
        import io
        from pydub import AudioSegment
        
        # Create TTS
        tts = gTTS(text=text, lang='en', slow=False)
        
        # Save to temporary file
        tts.save("temp_tts.mp3")
        
        # Convert to WAV
        audio = AudioSegment.from_mp3("temp_tts.mp3")
        audio = audio.set_frame_rate(16000)  # Whisper prefers 16kHz
        audio.export("test_audio_gtts.wav", format="wav")
        
        # Copy to our test files
        import shutil
        shutil.copy('test_audio_gtts.wav', 'test_audio.wav')
        shutil.copy('test_audio_gtts.wav', 'whisper_test.wav')
        
        # Cleanup
        os.remove("temp_tts.mp3")
        
        return True
        
    except ImportError:
        return False

def create_audio_with_espeak(text):
    """Create audio using espeak (if available on system)."""
    try:
        # Try to use espeak command line tool
        result = subprocess.run([
            'espeak', 
            '-s', '150',  # Speed
            '-v', 'en',   # Voice
            '-w', 'test_audio_espeak.wav',  # Output file
            text
        ], capture_output=True, text=True)
        
        if result.returncode == 0 and Path('test_audio_espeak.wav').exists():
            # Copy to our test files
            import shutil
            shutil.copy('test_audio_espeak.wav', 'test_audio.wav')
            shutil.copy('test_audio_espeak.wav', 'whisper_test.wav')
            return True
        
        return False
        
    except FileNotFoundError:
        return False

def create_advanced_synthesized_speech():
    """Create more advanced synthesized speech that might work better with Whisper."""
    
    print("   Creating advanced synthesized speech...")
    
    # Parameters for more realistic speech
    sample_rate = 16000
    duration = 10.0
    
    # Create time array
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create more complex speech-like signal
    speech = np.zeros(len(t))
    
    # Define phoneme-like segments with realistic formant frequencies
    phonemes = [
        # "Welcome" - /w/ /e/ /l/ /k/ /ʌ/ /m/
        (0.0, 0.8, [200, 800, 2400], 0.6),    # "wel"
        (0.8, 1.2, [300, 1200, 2800], 0.7),   # "come"
        
        # Pause
        (1.2, 1.5, [0], 0.1),
        
        # "to financial"
        (1.5, 2.0, [400, 900, 2200], 0.6),    # "to"
        (2.0, 3.2, [350, 1800, 2600], 0.8),   # "financial"
        
        # Pause
        (3.2, 3.5, [0], 0.1),
        
        # "education"
        (3.5, 5.0, [400, 1400, 2400], 0.7),   # "education"
        
        # Pause
        (5.0, 5.3, [0], 0.1),
        
        # "investment strategies"
        (5.3, 6.5, [300, 1100, 2300], 0.8),   # "investment"
        (6.5, 8.0, [450, 1300, 2500], 0.7),   # "strategies"
        
        # Final pause
        (8.0, 10.0, [0], 0.1),
    ]
    
    for start_time, end_time, formants, amplitude in phonemes:
        start_idx = int(start_time * sample_rate)
        end_idx = int(end_time * sample_rate)
        
        if end_idx > len(t):
            end_idx = len(t)
        
        segment_t = t[start_idx:end_idx]
        segment_duration = end_time - start_time
        
        if formants[0] == 0:  # Silence
            continue
        
        # Create formant structure
        segment_speech = np.zeros(len(segment_t))
        
        for i, freq in enumerate(formants):
            if freq > 0:
                # Add formant with decreasing amplitude
                formant_amplitude = amplitude * (0.8 ** i)
                
                # Add slight frequency modulation for naturalness
                freq_mod = freq * (1 + 0.02 * np.sin(2 * np.pi * 5 * segment_t))
                
                # Create formant
                formant = formant_amplitude * np.sin(2 * np.pi * freq_mod * segment_t)
                
                # Add harmonic content
                for harmonic in range(2, 4):
                    harmonic_freq = freq * harmonic
                    if harmonic_freq < sample_rate / 2:  # Avoid aliasing
                        harmonic_amp = formant_amplitude / (harmonic * 2)
                        formant += harmonic_amp * np.sin(2 * np.pi * harmonic_freq * segment_t)
                
                segment_speech += formant
        
        # Apply amplitude envelope
        if segment_duration > 0:
            # Create smooth envelope
            envelope_t = np.linspace(0, 1, len(segment_t))
            envelope = np.sin(np.pi * envelope_t) ** 0.5  # Smooth attack and decay
            segment_speech = segment_speech * envelope
        
        # Add to main speech signal
        speech[start_idx:end_idx] += segment_speech
    
    # Add realistic background noise
    noise_level = 0.02
    noise = noise_level * np.random.randn(len(t))
    speech = speech + noise
    
    # Apply realistic filtering (simulate microphone response)
    # Simple high-pass filter to remove very low frequencies
    from scipy import signal
    b, a = signal.butter(2, 80, btype='high', fs=sample_rate)
    speech = signal.filtfilt(b, a, speech)
    
    # Normalize
    speech = speech / np.max(np.abs(speech)) * 0.8
    
    # Save files
    sf.write("test_audio.wav", speech, sample_rate)
    sf.write("whisper_test.wav", speech, sample_rate)
    
    print(f"   ✅ Created advanced synthesized speech ({len(speech)} samples)")
    return True

def create_simple_test_video():
    """Create a simple test video using OpenCV."""
    
    print("🎬 Creating simple test video...")
    
    try:
        import cv2
        
        # Video parameters
        width, height = 1920, 1080
        fps = 30
        duration = 5  # seconds
        total_frames = fps * duration
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('test_video_simple.mp4', fourcc, fps, (width, height))
        
        for frame_num in range(total_frames):
            # Create a simple animated frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Add some color variation over time
            color_shift = int(frame_num * 255 / total_frames)
            frame[:, :, 0] = color_shift  # Blue channel
            frame[:, :, 1] = 128  # Green channel
            frame[:, :, 2] = 255 - color_shift  # Red channel
            
            # Add some geometric shapes
            center_x, center_y = width // 2, height // 2
            radius = int(50 + 30 * np.sin(frame_num * 0.1))
            
            cv2.circle(frame, (center_x, center_y), radius, (255, 255, 255), -1)
            
            # Add text
            text = f"Frame {frame_num + 1}/{total_frames}"
            cv2.putText(frame, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            
            # Add financial-related text occasionally
            if frame_num % 30 == 0:  # Every second
                financial_texts = [
                    "INVESTMENT ANALYSIS",
                    "PORTFOLIO MANAGEMENT", 
                    "FINANCIAL EDUCATION",
                    "MARKET TRENDS",
                    "RISK ASSESSMENT"
                ]
                text_idx = (frame_num // 30) % len(financial_texts)
                financial_text = financial_texts[text_idx]
                cv2.putText(frame, financial_text, (50, height - 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        
        # Replace the corrupted test_video.mp4
        if Path('test_video_simple.mp4').exists():
            import shutil
            shutil.copy('test_video_simple.mp4', 'test_video.mp4')
            print("   ✅ Created and replaced test_video.mp4")
            return True
        
        return False
        
    except Exception as e:
        print(f"   ❌ Video creation failed: {e}")
        return False

def main():
    """Create realistic test content."""
    
    print("🚀 Creating Realistic Test Content for Audio/Video Analysis")
    print("=" * 60)
    
    # Check dependencies
    print("📦 Checking dependencies...")
    
    try:
        import scipy
        print("   ✅ SciPy available")
    except ImportError:
        print("   ❌ SciPy not available - installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "scipy"])
    
    # Create realistic audio
    audio_success = create_realistic_audio_with_tts()
    
    # Create simple test video
    video_success = create_simple_test_video()
    
    # Summary
    print("\n📊 CONTENT CREATION SUMMARY:")
    print(f"   🎵 Audio files: {'✅ SUCCESS' if audio_success else '❌ FAILED'}")
    print(f"   🎬 Video files: {'✅ SUCCESS' if video_success else '❌ FAILED'}")
    
    if audio_success or video_success:
        print("\n🎉 Test content created successfully!")
        print("You can now run the audio/video analysis tests with more realistic content.")
    else:
        print("\n⚠️  Some content creation failed, but existing synthesized content should still work.")
    
    return audio_success or video_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)