#!/usr/bin/env python3
"""
Create a simple test audio file for the audio processing demo.
"""

import numpy as np
import soundfile as sf

def create_test_audio():
    """Create a simple test audio file with speech-like content."""
    
    # Parameters
    sample_rate = 44100
    duration = 5.0  # 5 seconds
    
    # Generate a simple tone pattern that resembles speech
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a pattern with varying frequencies (like speech)
    frequencies = [200, 300, 250, 400, 180, 350]  # Hz
    audio = np.zeros_like(t)
    
    segment_length = len(t) // len(frequencies)
    
    for i, freq in enumerate(frequencies):
        start_idx = i * segment_length
        end_idx = min((i + 1) * segment_length, len(t))
        
        # Create a tone with some modulation
        segment_t = t[start_idx:end_idx]
        tone = np.sin(2 * np.pi * freq * segment_t)
        
        # Add some amplitude modulation to make it more speech-like
        modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 5 * segment_t)
        tone = tone * modulation
        
        # Apply envelope to avoid clicks
        envelope = np.ones_like(segment_t)
        fade_samples = int(0.1 * sample_rate)  # 0.1 second fade
        if len(envelope) > 2 * fade_samples:
            envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
            envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        
        audio[start_idx:end_idx] = tone * envelope * 0.3  # Reduce volume
    
    # Save as WAV file
    output_file = "test_audio.wav"
    sf.write(output_file, audio, sample_rate)
    
    print(f"✅ Created test audio file: {output_file}")
    print(f"   Duration: {duration}s")
    print(f"   Sample Rate: {sample_rate}Hz")
    print(f"   File size: {len(audio) * 4 / 1024:.1f} KB")
    
    return output_file

if __name__ == "__main__":
    create_test_audio()