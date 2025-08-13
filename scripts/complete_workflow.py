#!/usr/bin/env python3
"""Complete workflow example for AI Video Editor features."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Success: {description}")
        if result.stdout.strip():
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {description}")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Run complete workflow example."""
    if len(sys.argv) != 2:
        print("Usage: python complete_workflow.py <input_video.mp4>")
        sys.exit(1)
    
    input_video = Path(sys.argv[1])
    if not input_video.exists():
        print(f"Error: Input video {input_video} not found")
        sys.exit(1)
    
    print("🎬 AI Video Editor - Complete Workflow")
    print(f"Input: {input_video}")
    
    # Create output directory
    output_dir = Path("workflow_output")
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Transcribe
    transcript_path = output_dir / "transcript.json"
    if not run_command([
        "python", "-m", "ai_video_editor.cli.features", "transcribe",
        str(input_video),
        "--language", "hi",
        "--backend", "faster-whisper",
        "--model", "base",
        "--enhance-audio",
        "--romanize",
        "--output", str(transcript_path)
    ], "Transcribing audio to Hinglish"):
        return
    
    # Step 2: Generate subtitles
    srt_path = output_dir / "subtitles.srt"
    if not run_command([
        "python", "-m", "ai_video_editor.cli.features", "to-srt",
        str(transcript_path),
        "--output", str(srt_path),
        "--format", "srt"
    ], "Generating SRT subtitles"):
        return
    
    # Step 3: Generate editing plan
    plan_path = output_dir / "editing_plan.json"
    if not run_command([
        "python", "-m", "ai_video_editor.cli.features", "generate-plan",
        str(transcript_path),
        "--output", str(plan_path),
        "--style", "devotional",
        "--detect-chorus",
        "--add-transitions"
    ], "Generating editing plan"):
        return
    
    # Step 4: Create execution timeline
    timeline_path = output_dir / "timeline.json"
    if not run_command([
        "python", "-m", "ai_video_editor.cli.features", "plan-execute",
        "--ai-plan", str(plan_path),
        "--output", str(timeline_path)
    ], "Creating execution timeline"):
        return
    
    print("\n🎉 Workflow completed successfully!")
    print(f"📁 Output files in: {output_dir}")
    print(f"   📝 Transcript: {transcript_path}")
    print(f"   📺 Subtitles: {srt_path}")
    print(f"   📋 Editing Plan: {plan_path}")
    print(f"   ⏱️  Timeline: {timeline_path}")
    
    print("\n🚀 Next steps:")
    print("   • Review the editing plan and timeline")
    print("   • Use the render command to create final video")
    print("   • Adjust parameters and re-run as needed")


if __name__ == "__main__":
    main()