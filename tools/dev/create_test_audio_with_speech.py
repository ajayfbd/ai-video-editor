#!/usr/bin/env python3
"""
Create a test audio file with synthesized speech for testing.
"""

import numpy as np
import soundfile as sf
from pathlib import Path

def create_speech_like_audio():
    """Create a simple audio file that resembles speech patterns."""
    
    # Audio parameters
    sample_rate = 16000  # Whisper prefers 16kHz
    duration = 5.0  # 5 seconds
    
    # Create time array
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create speech-like patterns with multiple frequencies
    # Simulate formants (resonant frequencies in speech)
    f1 = 800   # First formant
    f2 = 1200  # Second formant
    f3 = 2400  # Third formant
    
    # Create base speech signal
    speech = (
        0.3 * np.sin(2 * np.pi * f1 * t) +
        0.2 * np.sin(2 * np.pi * f2 * t) +
        0.1 * np.sin(2 * np.pi * f3 * t)
    )
    
    # Add speech-like modulation (amplitude and frequency variations)
    modulation_freq = 5  # 5 Hz modulation (typical speech rate)
    amplitude_modulation = 0.5 + 0.5 * np.sin(2 * np.pi * modulation_freq * t)
    speech = speech * amplitude_modulation
    
    # Add some noise to make it more realistic
    noise = 0.05 * np.random.randn(len(t))
    speech = speech + noise
    
    # Create pauses (silence) to simulate word boundaries
    pause_positions = [0.8, 1.6, 2.4, 3.2, 4.0]  # Pause positions in seconds
    pause_duration = 0.1  # 100ms pauses
    
    for pause_pos in pause_positions:
        start_idx = int(pause_pos * sample_rate)
        end_idx = int((pause_pos + pause_duration) * sample_rate)
        if end_idx < len(speech):
            speech[start_idx:end_idx] *= 0.1  # Reduce amplitude for pauses
    
    # Normalize to prevent clipping
    speech = speech / np.max(np.abs(speech)) * 0.8
    
    return speech, sample_rate

def create_financial_speech_audio():
    """Create audio that might contain financial keywords."""
    
    # Audio parameters
    sample_rate = 16000
    duration = 8.0  # 8 seconds for more content
    
    # Create time array
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create more complex speech patterns
    # Simulate different phonemes with varying frequencies
    segments = [
        # "Welcome to financial education"
        (0.0, 1.5, [600, 1000, 1800]),  # "Welcome"
        (1.5, 2.0, [400, 800, 1600]),   # "to"
        (2.0, 3.5, [700, 1200, 2000]),  # "financial"
        (3.5, 5.0, [650, 1100, 1900]),  # "education"
        
        # "Investment and portfolio management"
        (5.2, 6.5, [750, 1300, 2100]),  # "investment"
        (6.5, 7.0, [500, 900, 1700]),   # "and"
        (7.0, 8.0, [680, 1150, 1950]),  # "portfolio"
    ]
    
    speech = np.zeros(len(t))
    
    for start_time, end_time, formants in segments:
        start_idx = int(start_time * sample_rate)
        end_idx = int(end_time * sample_rate)
        
        if end_idx > len(t):
            end_idx = len(t)
        
        segment_t = t[start_idx:end_idx]
        segment_duration = end_time - start_time
        
        # Create segment with formants
        segment_speech = np.zeros(len(segment_t))
        for i, freq in enumerate(formants):
            amplitude = 0.4 / (i + 1)  # Decreasing amplitude for higher formants
            segment_speech += amplitude * np.sin(2 * np.pi * freq * segment_t)
        
        # Add pitch variation (prosody)
        pitch_variation = 1 + 0.1 * np.sin(2 * np.pi * 3 * segment_t)
        segment_speech = segment_speech * pitch_variation
        
        # Add amplitude envelope
        envelope = np.exp(-2 * np.abs(segment_t - (start_time + segment_duration/2)))
        envelope = envelope / np.max(envelope)
        segment_speech = segment_speech * envelope
        
        speech[start_idx:end_idx] += segment_speech
    
    # Add background noise
    noise = 0.03 * np.random.randn(len(t))
    speech = speech + noise
    
    # Normalize
    speech = speech / np.max(np.abs(speech)) * 0.7
    
    return speech, sample_rate

def main():
    """Create test audio files."""
    
    print("🎤 Creating test audio files with speech-like content...")
    
    # Create basic speech-like audio
    print("Creating basic speech audio...")
    speech1, sr1 = create_speech_like_audio()
    sf.write("test_speech_basic.wav", speech1, sr1)
    print(f"✅ Created: test_speech_basic.wav ({len(speech1)} samples, {sr1} Hz)")
    
    # Create financial content audio
    print("Creating financial speech audio...")
    speech2, sr2 = create_financial_speech_audio()
    sf.write("test_speech_financial.wav", speech2, sr2)
    print(f"✅ Created: test_speech_financial.wav ({len(speech2)} samples, {sr2} Hz)")
    
    # Replace existing test files
    print("Updating existing test files...")
    
    # Update test_audio.wav with financial content
    sf.write("test_audio.wav", speech2, sr2)
    print("✅ Updated: test_audio.wav")
    
    # Update whisper_test.wav with basic speech
    sf.write("whisper_test.wav", speech1, sr1)
    print("✅ Updated: whisper_test.wav")
    
    print("\n🎉 Test audio files created successfully!")
    print("These files contain synthesized speech-like patterns that should be")
    print("detectable by Whisper, even if not perfectly transcribable.")

if __name__ == "__main__":
    main()