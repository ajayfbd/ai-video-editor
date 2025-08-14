#!/usr/bin/env python3
"""
Quick Start Script - Process a single video with AI Director
"""

import sys
import subprocess
from pathlib import Path

def quick_process(video_file: str):
    """Process a single video file quickly."""
    
    video_path = Path(video_file)
    if not video_path.exists():
        print(f"❌ Video file not found: {video_file}")
        return False
    
    print(f"🎬 Quick processing: {video_path.name}")
    
    # Generate output names
    base_name = video_path.stem
    transcript_file = f"transcripts/{base_name}_transcript.json"
    plan_file = f"ai_plans/{base_name}_plan.json"
    output_file = f"outputs/{base_name}_final.mp4"
    
    try:
        # Step 1: Transcribe
        print("🎤 Transcribing audio...")
        subprocess.run([
            'python', '-m', 'ai_video_editor.cli.features', 'transcribe',
            str(video_path), '--output', transcript_file
        ], check=True, cwd=Path(__file__).parent.parent)
        
        # Step 2: Generate AI plan
        print("🤖 Generating AI Director plan...")
        subprocess.run([
            'python', '-m', 'ai_video_editor.cli.features', 'generate-plan',
            str(video_path), '--transcript', transcript_file,
            '--output', plan_file, '--ai-director', '--mock-ai',
            '--style', 'dynamic', '--analyze-scenes'
        ], check=True, cwd=Path(__file__).parent.parent)
        
        # Step 3: Render directly from AI plan
        print("🎬 Rendering final video...")
        subprocess.run([
            'python', '-m', 'ai_video_editor.cli.features', 'render',
            '--ai-plan', plan_file, '--videos', str(video_path),
            '--output', output_file, '--quality', 'high'
        ], check=True, cwd=Path(__file__).parent.parent)
        
        print(f"✅ SUCCESS! Output: {output_file}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Processing failed: {e}")
        return False

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python quick_start.py <video_file>")
        print("Example: python quick_start.py input_clips/my_video.mp4")
        sys.exit(1)
    
    success = quick_process(sys.argv[1])
    sys.exit(0 if success else 1)