#!/usr/bin/env python3
"""
Setup Workspace - Initialize the video processing workspace
"""

import os
from pathlib import Path

def setup_workspace():
    """Set up the complete video processing workspace."""
    
    project_dir = Path(__file__).parent
    
    # Create all necessary directories
    directories = [
        'input_clips',
        'transcripts', 
        'ai_plans',
        'timelines',
        'outputs',
        'temp',
        'logs'
    ]
    
    print("🏗️  Setting up AI Video Editor workspace...")
    
    for dir_name in directories:
        dir_path = project_dir / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"✅ Created: {dir_name}/")
    
    # Create .gitignore
    gitignore_content = """
# Generated files
transcripts/
ai_plans/
timelines/
outputs/
temp/
logs/

# Keep directories but ignore contents
!transcripts/.gitkeep
!ai_plans/.gitkeep
!timelines/.gitkeep
!outputs/.gitkeep
!temp/.gitkeep
!logs/.gitkeep

# Input clips (optional - you may want to track these)
# input_clips/

# Python cache
__pycache__/
*.pyc
*.pyo

# OS files
.DS_Store
Thumbs.db
"""
    
    with open(project_dir / '.gitignore', 'w') as f:
        f.write(gitignore_content.strip())
    
    # Create .gitkeep files
    for dir_name in ['transcripts', 'ai_plans', 'timelines', 'outputs', 'temp', 'logs']:
        gitkeep_file = project_dir / dir_name / '.gitkeep'
        gitkeep_file.touch()
    
    print("\n📋 Workspace setup complete!")
    print(f"📁 Project directory: {project_dir}")
    print("\n🚀 Next steps:")
    print("1. Add your video files to input_clips/")
    print("2. Run: python batch_process.py")
    print("3. Get finished videos from outputs/")
    
    # Check if sample video exists
    sample_video = project_dir / 'input_clips' / 'sample_devotional.mp4'
    if sample_video.exists():
        print(f"\n🎬 Sample video ready: {sample_video.name}")
        print("   Test with: python quick_start.py input_clips/sample_devotional.mp4")

if __name__ == '__main__':
    setup_workspace()