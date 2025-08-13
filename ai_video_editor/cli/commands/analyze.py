"""Analyze command - Comprehensive video and audio analysis with detailed outputs."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import click

from ai_video_editor.utils.logging_config import get_logger
from ai_video_editor.cli.bridge import CLIBridge
from .utils import save_json, load_json

logger = get_logger(__name__)


@click.command("analyze")
@click.argument("video_file", type=click.Path(exists=True, path_type=Path))
@click.option("--output-dir", type=click.Path(path_type=Path), default="analysis_output",
              help="Output directory for analysis files")
@click.option("--audio-analysis", is_flag=True, help="Perform audio analysis and transcription")
@click.option("--video-analysis", is_flag=True, help="Perform video content analysis")
@click.option("--scene-detection", is_flag=True, help="Detect scene changes")
@click.option("--face-detection", is_flag=True, help="Detect faces and expressions")
@click.option("--visual-highlights", is_flag=True, help="Extract visual highlights")
@click.option("--all", "analyze_all", is_flag=True, help="Perform all available analyses")
@click.option("--language", type=str, default="hi", help="Language for audio transcription")
@click.option("--model", type=str, default="medium", help="Whisper model size for transcription")
@click.option("--scene-threshold", type=float, default=0.3, help="Scene detection sensitivity")
@click.option("--save-frames", is_flag=True, help="Save sample frames from detected scenes")
@click.option("--detailed", is_flag=True, help="Generate detailed analysis reports")
@click.option("--force-model", is_flag=True, help="Force use of specified model size even on CPU")
@click.option("--vocab-file", type=click.Path(exists=True, path_type=Path), help="Vocabulary file for better transcription")
@click.option("--preset", type=click.Choice(["hindi-religious", "sanskrit-classical", "mythological", "comprehensive", "general"]), help="Preset vocabulary for common use cases")
@click.option("--initial-prompt", type=str, help="Initial prompt to bias transcription (e.g., lyrics)")
def analyze_cmd(video_file: Path, output_dir: Path, audio_analysis: bool, video_analysis: bool,
                scene_detection: bool, face_detection: bool, visual_highlights: bool,
                analyze_all: bool, language: str, model: str, scene_threshold: float,
                save_frames: bool, detailed: bool, force_model: bool, vocab_file: Optional[Path],
                preset: Optional[str], initial_prompt: Optional[str]):
    """Comprehensive video and audio analysis with detailed output files.
    
    Analyzes video content and saves detailed results including:
    - Audio transcription and analysis
    - Scene detection with timestamps
    - Face detection and expressions
    - Visual highlights and key frames
    - Comprehensive analysis reports
    """
    try:
        start_time = time.time()
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        click.echo(f"🎬 Analyzing video: {video_file}")
        click.echo(f"📁 Output directory: {output_dir}")
        
        # Set analysis flags
        if analyze_all:
            audio_analysis = video_analysis = scene_detection = face_detection = visual_highlights = True
            save_frames = detailed = True
        
        # Initialize CLI bridge
        bridge = CLIBridge(cache_dir=str(output_dir / "cache"))
        
        # Create ContentContext from CLI arguments
        context = bridge.create_context_from_cli_args(
            video_file=video_file,
            language=language,
            model=model,
            vocab_file=vocab_file,
            preset=preset,
            initial_prompt=initial_prompt,
            force_model=force_model
        )
        
        # Perform comprehensive analysis using core modules
        click.echo("\n🔍 Performing comprehensive analysis using core modules...")
        context = bridge.perform_comprehensive_analysis(
            context=context,
            audio_analysis=audio_analysis,
            video_analysis=video_analysis,
            scene_detection=scene_detection,
            face_detection=face_detection,
            visual_highlights=visual_highlights
        )
        
        analysis_results = {}
        
        # Format results for CLI output compatibility
        if audio_analysis:
            click.echo("🎵 Formatting audio analysis results...")
            audio_results = bridge.format_audio_results_for_cli(context)
            analysis_results["audio"] = audio_results
            save_json(audio_results, output_dir / "audio_analysis.json")
            
            # Save transcript separately for compatibility
            if audio_results.get("transcript"):
                save_json(audio_results["transcript"], output_dir / "transcript.json")
            
            click.echo(f"✅ Audio analysis complete - saved to {output_dir}/audio_analysis.json")
        
        if video_analysis or scene_detection or face_detection or visual_highlights:
            click.echo("🎥 Formatting video analysis results...")
            video_results = bridge.format_video_results_for_cli(context)
            analysis_results["video"] = video_results
            save_json(video_results, output_dir / "video_analysis.json")
            
            # Save individual components for compatibility
            if video_results.get("scenes"):
                save_json(video_results["scenes"], output_dir / "scenes.json")
            if video_results.get("faces"):
                save_json(video_results["faces"], output_dir / "faces.json")
            if video_results.get("visual_highlights"):
                save_json(video_results["visual_highlights"], output_dir / "visual_highlights.json")
            
            click.echo(f"✅ Video analysis complete - saved to {output_dir}/video_analysis.json")
        
        # Generate comprehensive report
        if detailed:
            click.echo("\n📊 Generating detailed analysis report...")
            report = bridge.generate_analysis_report_for_cli(context, video_file, output_dir)
            save_json(report, output_dir / "analysis_report.json")
            click.echo(f"✅ Analysis report saved to {output_dir}/analysis_report.json")
        
        # Save combined results with metadata
        analysis_results["metadata"] = {
            "video_file": str(video_file),
            "analysis_time": time.time() - start_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "analyses_performed": {
                "audio": audio_analysis,
                "video": video_analysis,
                "scenes": scene_detection,
                "faces": face_detection,
                "highlights": visual_highlights
            },
            "core_modules_used": True,
            "content_context_integration": True
        }
        
        save_json(analysis_results, output_dir / "complete_analysis.json")
        
        total_time = time.time() - start_time
        click.echo(f"\n🎉 Analysis complete in {total_time:.1f}s")
        click.echo(f"📁 All results saved to: {output_dir}")
        
        # Show summary using core module results
        _show_analysis_summary_from_context(context, analysis_results)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        click.echo(f"[ERROR] Analysis failed: {e}")
        sys.exit(1)


def _show_analysis_summary_from_context(context, analysis_results: dict):
    """Show analysis summary based on ContentContext results."""
    click.echo("\n📊 Analysis Summary:")
    
    # Audio summary
    if "audio" in analysis_results:
        audio = analysis_results["audio"]
        stats = audio.get("statistics", {})
        content = audio.get("content_analysis", {})
        
        click.echo(f"🎵 Audio Analysis:")
        click.echo(f"   Duration: {stats.get('total_duration', 0):.1f}s")
        click.echo(f"   Segments: {stats.get('segment_count', 0)}")
        click.echo(f"   Language: {stats.get('language', 'unknown')}")
        click.echo(f"   Confidence: {stats.get('average_confidence', 0):.2f}")
        click.echo(f"   Theme: {content.get('dominant_theme', 'general')}")
        click.echo(f"   Speaking rate: {content.get('speaking_rate', 0):.1f} words/sec")
    
    # Video summary
    if "video" in analysis_results:
        video = analysis_results["video"]
        stats = video.get("statistics", {})
        
        click.echo(f"🎥 Video Analysis:")
        click.echo(f"   Resolution: {stats.get('resolution', 'unknown')}")
        click.echo(f"   FPS: {stats.get('fps', 0)}")
        click.echo(f"   Scenes: {stats.get('total_scenes', 0)}")
        click.echo(f"   Faces: {stats.get('total_faces', 0)}")
        click.echo(f"   Highlights: {stats.get('total_highlights', 0)}")
        click.echo(f"   Quality: {stats.get('analysis_quality', 'unknown')}")
    
    # Content insights from ContentContext
    if context.key_concepts:
        click.echo(f"🔍 Key Concepts: {', '.join(context.key_concepts[:5])}")
    
    if context.emotional_markers:
        high_peaks = [p for p in context.emotional_markers if p.intensity > 0.7]
        click.echo(f"😊 Emotional Peaks: {len(high_peaks)} high-intensity moments")
    
    if hasattr(context, 'content_type'):
        click.echo(f"📝 Content Type: {context.content_type.value}")
    
    click.echo("\n✨ Analysis completed using AI Video Editor core modules")
    click.echo("   Benefits: ContentContext integration, Memory learning, AI Director compatibility")