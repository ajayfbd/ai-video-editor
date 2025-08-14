#!/usr/bin/env python3
"""
AI Video Editor - Batch Processing Script

Automatically processes all video clips in input_clips/ folder and generates
finished videos using the complete AI Director pipeline.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any
import subprocess
import shutil

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

class BatchVideoProcessor:
    """Automated batch video processing using AI Director pipeline."""
    
    def __init__(self, project_dir: Path = None):
        """Initialize batch processor."""
        self.project_dir = project_dir or Path(__file__).parent
        self.input_dir = self.project_dir / "input_clips"
        self.transcripts_dir = self.project_dir / "transcripts"
        self.plans_dir = self.project_dir / "ai_plans"
        self.timelines_dir = self.project_dir / "timelines"
        self.outputs_dir = self.project_dir / "outputs"
        
        # Supported video formats
        self.video_formats = {'.mp4', '.avi', '.mov', '.mkv', '.m4v', '.webm'}
        self.audio_formats = {'.mp3', '.wav', '.m4a', '.aac', '.flac'}
        
        # Processing settings
        self.settings = {
            'quality': 'high',
            'resolution': '1080p',
            'fps': 30,
            'use_ai_director': True,
            'mock_ai': False,  # Set to True if API issues
            'content_type': 'general',  # auto-detect or specify
            'style': 'dynamic'  # basic, dynamic, devotional
        }
        
        print(f"🎬 AI Video Editor Batch Processor")
        print(f"📁 Project directory: {self.project_dir}")
        print(f"⚙️  Settings: {self.settings}")
    
    def find_video_files(self) -> List[Path]:
        """Find all video files in input directory."""
        video_files = []
        
        if not self.input_dir.exists():
            print(f"❌ Input directory not found: {self.input_dir}")
            return video_files
        
        for file_path in self.input_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.video_formats:
                video_files.append(file_path)
        
        print(f"📹 Found {len(video_files)} video files:")
        for i, video in enumerate(video_files, 1):
            print(f"   {i}. {video.name}")
        
        return video_files
    
    def detect_content_type(self, video_path: Path) -> str:
        """Auto-detect content type based on filename and duration."""
        name = video_path.name.lower()
        
        # Devotional/Religious content detection
        devotional_keywords = ['bhajan', 'devotional', 'spiritual', 'religious', 'mantra', 'aarti', 'kirtan']
        if any(keyword in name for keyword in devotional_keywords):
            return 'devotional'
        
        # Educational content detection
        educational_keywords = ['tutorial', 'lesson', 'education', 'learn', 'course', 'training']
        if any(keyword in name for keyword in educational_keywords):
            return 'educational'
        
        # Music content detection
        music_keywords = ['song', 'music', 'audio', 'track', 'album']
        if any(keyword in name for keyword in music_keywords):
            return 'music'
        
        return 'general'
    
    def run_command(self, command: List[str], description: str) -> bool:
        """Run a command and return success status."""
        try:
            print(f"🔄 {description}...")
            print(f"   Command: {' '.join(command)}")
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=self.project_dir.parent,
                timeout=1800  # 30 minute timeout
            )
            
            if result.returncode == 0:
                print(f"✅ {description} completed successfully")
                return True
            else:
                print(f"❌ {description} failed:")
                print(f"   Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {description} timed out (30 minutes)")
            return False
        except Exception as e:
            print(f"💥 {description} error: {e}")
            return False
    
    def process_single_video(self, video_path: Path) -> bool:
        """Process a single video through the complete AI pipeline."""
        print(f"\n🎬 Processing: {video_path.name}")
        print("=" * 60)
        
        # Generate output filenames
        base_name = video_path.stem
        transcript_file = self.transcripts_dir / f"{base_name}_transcript.json"
        plan_file = self.plans_dir / f"{base_name}_ai_plan.json"
        timeline_file = self.timelines_dir / f"{base_name}_timeline.json"
        output_file = self.outputs_dir / f"{base_name}_final.mp4"
        
        # Auto-detect content type and style
        content_type = self.detect_content_type(video_path)
        style = 'devotional' if content_type == 'devotional' else self.settings['style']
        
        print(f"📊 Detected content type: {content_type}")
        print(f"🎨 Using style: {style}")
        
        start_time = time.time()
        
        # Step 1: Transcribe audio
        if not transcript_file.exists():
            transcribe_cmd = [
                'python', '-m', 'ai_video_editor.cli.features', 'transcribe',
                str(video_path),
                '--output', str(transcript_file),
                '--model', 'large',
                '--enhance-audio'
            ]
            
            if not self.run_command(transcribe_cmd, "Audio transcription"):
                return False
        else:
            print(f"✅ Using existing transcript: {transcript_file.name}")
        
        # Step 2: Generate AI Director plan
        if not plan_file.exists():
            plan_cmd = [
                'python', '-m', 'ai_video_editor.cli.features', 'generate-plan',
                str(video_path),
                '--transcript', str(transcript_file),
                '--output', str(plan_file),
                '--style', style,
                '--content-type', content_type,
                '--analyze-scenes',
                '--detect-faces'
            ]
            
            # Add AI Director flags
            if self.settings['use_ai_director']:
                plan_cmd.append('--ai-director')
                if self.settings['mock_ai']:
                    plan_cmd.append('--mock-ai')
            
            if not self.run_command(plan_cmd, "AI Director plan generation"):
                return False
        else:
            print(f"✅ Using existing AI plan: {plan_file.name}")
        
        # Step 3: Execute plan to create timeline
        if not timeline_file.exists():
            execute_cmd = [
                'python', '-m', 'ai_video_editor.cli.features', 'plan-execute',
                '--ai-plan', str(plan_file),
                '--output', str(timeline_file)
            ]
            
            if not self.run_command(execute_cmd, "Plan execution"):
                return False
        else:
            print(f"✅ Using existing timeline: {timeline_file.name}")
        
        # Step 4: Render final video
        if not output_file.exists():
            render_cmd = [
                'python', '-m', 'ai_video_editor.cli.features', 'render',
                '--timeline', str(timeline_file),
                '--videos', str(video_path),
                '--output', str(output_file),
                '--quality', self.settings['quality'],
                '--resolution', self.settings['resolution'],
                '--fps', str(self.settings['fps'])
            ]
            
            if not self.run_command(render_cmd, "Video rendering"):
                return False
        else:
            print(f"✅ Output already exists: {output_file.name}")
        
        # Success summary
        processing_time = time.time() - start_time
        print(f"\n🎉 SUCCESS: {video_path.name} processed in {processing_time:.1f}s")
        print(f"📁 Output: {output_file}")
        
        # Show file sizes
        if output_file.exists():
            input_size = video_path.stat().st_size / (1024*1024)
            output_size = output_file.stat().st_size / (1024*1024)
            print(f"📊 Size: {input_size:.1f}MB → {output_size:.1f}MB")
        
        return True
    
    def process_all_videos(self) -> Dict[str, Any]:
        """Process all videos in the input directory."""
        print(f"\n🚀 Starting batch processing...")
        
        # Find all video files
        video_files = self.find_video_files()
        
        if not video_files:
            print(f"❌ No video files found in {self.input_dir}")
            print(f"💡 Add video files (.mp4, .avi, .mov, etc.) to the input_clips folder")
            return {'success': False, 'processed': 0, 'failed': 0}
        
        # Process each video
        results = {'success': True, 'processed': 0, 'failed': 0, 'files': []}
        
        for i, video_path in enumerate(video_files, 1):
            print(f"\n📹 Processing {i}/{len(video_files)}: {video_path.name}")
            
            try:
                success = self.process_single_video(video_path)
                
                if success:
                    results['processed'] += 1
                    results['files'].append({'file': video_path.name, 'status': 'success'})
                else:
                    results['failed'] += 1
                    results['files'].append({'file': video_path.name, 'status': 'failed'})
                    
            except KeyboardInterrupt:
                print(f"\n⏹️  Processing interrupted by user")
                results['success'] = False
                break
            except Exception as e:
                print(f"💥 Unexpected error processing {video_path.name}: {e}")
                results['failed'] += 1
                results['files'].append({'file': video_path.name, 'status': 'error', 'error': str(e)})
        
        # Final summary
        print(f"\n" + "="*60)
        print(f"🏁 BATCH PROCESSING COMPLETE")
        print(f"✅ Processed: {results['processed']}")
        print(f"❌ Failed: {results['failed']}")
        print(f"📁 Outputs in: {self.outputs_dir}")
        
        if results['processed'] > 0:
            print(f"\n🎬 Finished videos:")
            for file_info in results['files']:
                if file_info['status'] == 'success':
                    output_name = file_info['file'].replace('.mp4', '_final.mp4')
                    print(f"   ✅ {output_name}")
        
        return results

def main():
    """Main entry point for batch processing."""
    
    # Initialize processor
    processor = BatchVideoProcessor()
    
    # Check if input directory has files
    if not processor.input_dir.exists():
        print(f"❌ Input directory not found: {processor.input_dir}")
        print(f"💡 Create the directory and add your video files")
        return
    
    # Process all videos
    results = processor.process_all_videos()
    
    # Exit with appropriate code
    if results['success'] and results['processed'] > 0:
        print(f"\n🎉 All videos processed successfully!")
        sys.exit(0)
    else:
        print(f"\n💥 Some videos failed to process")
        sys.exit(1)

if __name__ == '__main__':
    main()