#!/usr/bin/env python3
"""
🎵 Audio Processing Demo

This demo shows how to extract and transcribe audio from any audio or video file
using OpenAI's Whisper. Works with music files, podcasts, videos, and more!
"""

import os
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
import whisper
import librosa


class AudioProcessor:
    """Processes audio files and extracts transcriptions."""
    
    def __init__(self, model_size: str = "base"):
        """Initialize with Whisper model.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large, large-v3)
                       - tiny: fastest, least accurate
                       - base: good balance (recommended for demo)
                       - large-v3: most accurate, slower
        """
        print(f"🤖 Loading Whisper model: {model_size}")
        print("   (This may take a moment on first run...)")
        
        try:
            self.model = whisper.load_model(model_size)
            print(f"✅ Whisper {model_size} model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load Whisper model: {e}")
            raise
    
    def extract_audio_from_video(self, video_path: str, output_path: str) -> bool:
        """Extract audio from video file using ffmpeg."""
        try:
            print(f"🎬 Extracting audio from video: {Path(video_path).name}")
            
            # Use ffmpeg to extract audio
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn',  # No video
                '-acodec', 'libmp3lame',  # MP3 codec
                '-ab', '192k',  # Bitrate
                '-ar', '44100',  # Sample rate
                '-y',  # Overwrite output
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Audio extracted to: {Path(output_path).name}")
                return True
            else:
                print(f"❌ FFmpeg error: {result.stderr}")
                return False
                
        except FileNotFoundError:
            print("❌ FFmpeg not found. Please install FFmpeg:")
            print("   - Windows: Download from https://ffmpeg.org/download.html")
            print("   - Mac: brew install ffmpeg")
            print("   - Linux: sudo apt install ffmpeg")
            return False
        except Exception as e:
            print(f"❌ Error extracting audio: {e}")
            return False
    
    def get_audio_metadata(self, audio_path: str) -> Dict[str, Any]:
        """Get metadata about the audio file."""
        try:
            # Load audio with librosa
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Get file info
            file_path = Path(audio_path)
            file_size = file_path.stat().st_size
            
            metadata = {
                "duration": round(duration, 2),
                "sample_rate": sr,
                "channels": 1 if len(y.shape) == 1 else y.shape[0],
                "format": file_path.suffix.lower().replace('.', ''),
                "file_size": f"{file_size / (1024*1024):.1f} MB",
                "file_name": file_path.name
            }
            
            return metadata
            
        except Exception as e:
            print(f"⚠️  Could not extract metadata: {e}")
            return {
                "duration": 0,
                "sample_rate": 0,
                "channels": 0,
                "format": "unknown",
                "file_size": "unknown",
                "file_name": Path(audio_path).name
            }
    
    def transcribe_audio(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe audio file using Whisper."""
        try:
            audio_path = str(Path(audio_path).resolve())  # Convert to absolute path
            print(f"🎤 Transcribing audio: {Path(audio_path).name}")
            print("   (This may take a few moments...)")
            
            # Transcribe with Whisper
            options = {}
            if language:
                options['language'] = language
            
            result = self.model.transcribe(audio_path, **options)
            
            # Extract segments with timestamps
            segments = []
            for segment in result.get('segments', []):
                segments.append({
                    "text": segment['text'].strip(),
                    "start": round(segment['start'], 2),
                    "end": round(segment['end'], 2),
                    "confidence": round(segment.get('avg_logprob', 0), 3)
                })
            
            # Get audio metadata
            metadata = self.get_audio_metadata(audio_path)
            
            transcript_data = {
                "transcript": {
                    "full_text": result['text'].strip(),
                    "segments": segments,
                    "language": result.get('language', 'unknown'),
                    "duration": metadata['duration']
                },
                "audio_metadata": metadata
            }
            
            print(f"✅ Transcription completed!")
            print(f"   Language: {result.get('language', 'unknown')}")
            print(f"   Duration: {metadata['duration']}s")
            print(f"   Segments: {len(segments)}")
            
            return transcript_data
            
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            return {"error": str(e)}
    
    def process_file(self, file_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Process any audio or video file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"error": f"File not found: {file_path}"}
        
        print(f"📁 Processing file: {file_path.name}")
        
        # Check if it's a video file that needs audio extraction
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        audio_extensions = {'.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg'}
        
        file_ext = file_path.suffix.lower()
        
        if file_ext in video_extensions:
            # Extract audio from video
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            try:
                if self.extract_audio_from_video(str(file_path), temp_audio_path):
                    result = self.transcribe_audio(temp_audio_path, language)
                    result['source_type'] = 'video'
                    result['original_file'] = str(file_path)
                else:
                    result = {"error": "Failed to extract audio from video"}
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass
            
            return result
            
        elif file_ext in audio_extensions:
            # Direct audio transcription
            result = self.transcribe_audio(str(file_path), language)
            result['source_type'] = 'audio'
            result['original_file'] = str(file_path)
            return result
            
        else:
            return {"error": f"Unsupported file format: {file_ext}. Supported formats: {', '.join(sorted(video_extensions | audio_extensions))}"}


def demo_with_sample_audio():
    """Demo with a sample audio file (if available)."""
    print("🎵 Audio Processing Demo")
    print("=" * 50)
    
    # Look for sample files in the project
    sample_files = []
    
    # Check for common sample files
    possible_samples = [
        "test_audio.wav",  # Our created test file
        "test_video.mp4",
        "sample.mp3",
        "test.wav",
        "demo.m4a"
    ]
    
    for sample in possible_samples:
        if Path(sample).exists():
            sample_files.append(sample)
    
    if not sample_files:
        print("📁 No sample files found in current directory.")
        print("💡 To test with your own file, choose option 2 in the main menu.")
        print("\n🎯 Creating a simple text-to-speech demo instead...")
        demo_text_to_speech()
        return
    
    print(f"📁 Found sample file: {sample_files[0]}")
    
    try:
        # Use base model for faster demo (you can change to 'large-v3' for better accuracy)
        processor = AudioProcessor(model_size="base")
        
        # Process the file
        result = processor.process_file(sample_files[0])
        
        if "error" in result:
            print(f"❌ Processing failed: {result['error']}")
            return
        
        # Display results
        print("\n📊 Processing Results:")
        print("=" * 50)
        
        transcript = result.get('transcript', {})
        metadata = result.get('audio_metadata', {})
        
        print(f"📄 Full Transcript:")
        print(f"   {transcript.get('full_text', 'No text found')[:200]}...")
        
        print(f"\n🎵 Audio Info:")
        print(f"   Duration: {metadata.get('duration', 0)}s")
        print(f"   Language: {transcript.get('language', 'unknown')}")
        print(f"   Segments: {len(transcript.get('segments', []))}")
        
        print(f"\n⏱️  First Few Segments:")
        for i, segment in enumerate(transcript.get('segments', [])[:3]):
            print(f"   [{segment['start']:.1f}s-{segment['end']:.1f}s]: {segment['text']}")
        
        print(f"\n💾 Complete Results (JSON):")
        print(json.dumps(result, indent=2))
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")


def demo_text_to_speech():
    """Simple demo explanation when no audio files are available."""
    print("\n🎯 Audio Processing Feature Overview")
    print("=" * 50)
    print("This feature can transcribe:")
    print("📹 Video files: MP4, AVI, MOV, MKV, etc.")
    print("🎵 Audio files: MP3, WAV, M4A, AAC, FLAC, etc.")
    print("\nExample output structure:")
    
    example_result = {
        "transcript": {
            "full_text": "Welcome to this tutorial on AI video editing...",
            "segments": [
                {"text": "Welcome to this tutorial", "start": 0.0, "end": 2.5, "confidence": 0.95},
                {"text": "on AI video editing", "start": 2.5, "end": 4.8, "confidence": 0.92}
            ],
            "language": "en",
            "duration": 120.5
        },
        "audio_metadata": {
            "duration": 120.5,
            "sample_rate": 44100,
            "channels": 2,
            "format": "mp3",
            "file_size": "2.1 MB"
        }
    }
    
    print(json.dumps(example_result, indent=2))


def demo_with_custom_file():
    """Demo with user-provided file."""
    print("🎵 Audio Processing - Custom File")
    print("=" * 50)
    
    file_path = input("📁 Enter path to your audio/video file: ").strip()
    
    if not file_path:
        print("❌ No file path provided.")
        return
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return
    
    # Ask for language (optional)
    language = input("🌍 Enter language code (optional, e.g., 'en', 'es', 'fr'): ").strip()
    if not language:
        language = None
    
    try:
        print(f"\n🚀 Processing your file...")
        processor = AudioProcessor(model_size="base")  # Use base for speed
        
        result = processor.process_file(file_path, language)
        
        if "error" in result:
            print(f"❌ Processing failed: {result['error']}")
            return
        
        print("\n📊 Your File Results:")
        print("=" * 50)
        print(json.dumps(result, indent=2))
        
        # Save results to file
        output_file = Path(file_path).stem + "_transcript.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")


if __name__ == "__main__":
    print("🎵 Audio Processing Feature Demo")
    print("=" * 50)
    print("Choose an option:")
    print("1. Demo with sample file (if available)")
    print("2. Demo with your own audio/video file")
    print("3. Show example output structure")
    
    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    
    if choice == "2":
        demo_with_custom_file()
    elif choice == "3":
        demo_text_to_speech()
    else:
        demo_with_sample_audio()