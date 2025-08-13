"""Render command - Modern video rendering with AI Director integration."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, List

import click

from ai_video_editor.utils.logging_config import get_logger
from .utils import load_json, format_time_duration

logger = get_logger(__name__)


@click.command("render")
@click.option("--timeline", "timeline_path", type=click.Path(exists=True, path_type=Path), 
              help="Execution timeline JSON path")
@click.option("--ai-plan", "ai_plan_path", type=click.Path(exists=True, path_type=Path), 
              help="AI plan JSON (alternative to --timeline)")
@click.option("--videos", multiple=True, type=click.Path(exists=True, path_type=Path), 
              help="Input video files (can repeat)")
@click.option("--assets", "assets_dir", type=click.Path(exists=True, path_type=Path), 
              help="B-roll assets directory")
@click.option("--output", "output_mp4", type=click.Path(path_type=Path), required=False, 
              help="Output MP4 path (default: out/<basename>_final.mp4)")
@click.option("--quality", type=click.Choice(["low", "medium", "high", "ultra"]), default="high", 
              help="Render quality preset (low=fast/small, high=slow/quality)")
@click.option("--resolution", type=click.Choice(["720p", "1080p", "1440p", "4k"]), default="1080p",
              help="Output resolution")
@click.option("--fps", type=click.IntRange(15, 60), default=30,
              help="Output frame rate (15-60 fps)")
@click.option("--project-id", default="cli_render", help="Project ID for context")
@click.option("--temp-dir", type=click.Path(path_type=Path), 
              help="Temporary directory for processing")
@click.option("--cleanup/--no-cleanup", default=True,
              help="Clean up temporary files after rendering")
def render_cmd(timeline_path: Optional[Path], ai_plan_path: Optional[Path], 
               videos: List[Path], assets_dir: Optional[Path], output_mp4: Path, 
               quality: str, resolution: str, fps: int, project_id: str,
               temp_dir: Optional[Path], cleanup: bool):
    """🎬 Render final MP4 from execution timeline or AI plan.
    
    Requires either --timeline (execution timeline JSON) or --ai-plan (AI Director plan JSON).
    
    Examples:
        # Render from AI plan with high quality
        python -m ai_video_editor.cli.features render --ai-plan plan.json --videos video.mp4 --output final.mp4 --quality high
        
        # Render from timeline with custom settings
        python -m ai_video_editor.cli.features render --timeline timeline.json --videos video.mp4 --output final.mp4 --resolution 4k --fps 60
    """
    start_time = time.time()
    
    try:
        from ai_video_editor.core.content_context import ContentContext, ContentType, UserPreferences
        from ai_video_editor.modules.video_processing.composer import VideoComposer

        if not timeline_path and not ai_plan_path:
            raise click.UsageError("❌ Provide either --timeline or --ai-plan.")

        if not videos:
            raise click.UsageError("❌ At least one input video file is required (--videos).")

        # Parse resolution
        resolution_map = {
            "720p": (1280, 720),
            "1080p": (1920, 1080), 
            "1440p": (2560, 1440),
            "4k": (3840, 2160)
        }
        width, height = resolution_map[resolution]

        click.echo(f"🎬 Starting video render...")
        click.echo(f"📁 Output: {output_mp4}")
        click.echo(f"🎯 Quality: {quality} | Resolution: {resolution} ({width}x{height}) | FPS: {fps}")

        # Create context with user preferences
        user_prefs = UserPreferences()
        user_prefs.quality_mode = quality
        user_prefs.output_resolution = (width, height)
        user_prefs.output_fps = fps

        context = ContentContext(
            project_id=project_id,
            video_files=[str(v) for v in videos],
            content_type=ContentType.GENERAL,
            user_preferences=user_prefs
        )

        # Initialize composer with temp directory
        temp_directory = temp_dir or output_mp4.parent / "temp"
        # Derive default output path if omitted
        if output_mp4 is None:
            out_dir = Path("out"); out_dir.mkdir(exist_ok=True)
            base = (videos[0].stem if videos else Path("project").name)
            output_mp4 = out_dir / f"{base}_final.mp4"

        composer = VideoComposer(
            output_dir=str(output_mp4.parent),
            temp_dir=str(temp_directory)
        )

        if timeline_path:
            # Render from timeline
            click.echo("📋 Loading execution timeline...")
            timeline_data = load_json(timeline_path)
            timeline = _timeline_from_dict(timeline_data)
            
            click.echo(f"⚙️  Processing {len(timeline.operations)} operations...")
            
            # Create composition plan from timeline
            composition_plan = composer.create_composition_plan_from_timeline(timeline, context)
            
            # Apply resolution settings
            composition_plan.output_settings.width = width
            composition_plan.output_settings.height = height
            composition_plan.output_settings.fps = fps
            
            # Render
            click.echo("🎥 Rendering video composition...")
            result = composer.write_video(composition_plan, str(output_mp4), quality_preset=quality)
            
        else:
            # Render from AI plan
            click.echo("🤖 Loading AI Director plan...")
            ai_plan = load_json(ai_plan_path)
            context.processed_video = ai_plan
            
            # Count plan elements
            editing_decisions = len(ai_plan.get('editing_decisions', []))
            broll_plans = len(ai_plan.get('broll_plans', []))
            click.echo(f"📝 Plan contains {editing_decisions} editing decisions and {broll_plans} B-roll elements")
            
            # Use full AI plan rendering with custom settings
            click.echo("🎬 Executing AI Director plan...")
            result = composer.compose_video_with_ai_plan(context)
            
            # Now render the composed result
            if 'composition_result' in context.processed_video:
                comp_result = context.processed_video['composition_result']
                # Copy the rendered file to desired output location
                import shutil
                if Path(comp_result['output_path']).exists():
                    shutil.copy2(comp_result['output_path'], output_mp4)
                    result = comp_result
                else:
                    raise RuntimeError("AI Director composition failed to produce output file")

        # Clean up if requested
        if cleanup:
            click.echo("🧹 Cleaning up temporary files...")
            composer.cleanup_temp_files()

        # Display results
        total_time = time.time() - start_time
        
        click.echo("\n✅ Video rendering complete!")
        click.echo(f"📁 Output file: {output_mp4}")
        
        if isinstance(result, dict):
            if 'duration' in result:
                click.echo(f"⏱️  Duration: {format_time_duration(result['duration'])}")
            if 'file_size_mb' in result:
                click.echo(f"💾 File size: {result['file_size_mb']:.1f} MB")
            if 'layers_rendered' in result:
                click.echo(f"🎞️  Layers: {result['layers_rendered']} | Transitions: {result.get('transitions_applied', 0)} | Effects: {result.get('effects_applied', 0)}")
            if 'render_time' in result:
                click.echo(f"⚡ Render time: {format_time_duration(result['render_time'])}")
        
        click.echo(f"🎉 Total processing time: {format_time_duration(total_time)}")

    except ImportError as e:
        if "movis" in str(e):
            click.echo("❌ Missing dependency: movis library is required for video rendering")
            click.echo("💡 Install with: pip install movis")
        else:
            click.echo(f"❌ Missing dependency: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Render failed: {e}")
        click.echo(f"❌ Render failed: {e}")
        if "No such file or directory" in str(e):
            click.echo("💡 Check that all input files exist and paths are correct")
        elif "Permission denied" in str(e):
            click.echo("💡 Check file permissions and ensure output directory is writable")
        sys.exit(1)


def _timeline_from_dict(data):
    """Reconstruct ExecutionTimeline from dict."""
    from ai_video_editor.modules.video_processing.plan_execution import (
        ExecutionTimeline, TrackOperation, SynchronizationPoint
    )

    ops = []
    for op in data.get("operations", []):
        ops.append(
            TrackOperation(
                operation_id=op.get("operation_id"),
                track_id=op.get("track_id"),
                track_type=op.get("track_type"),
                start_time=float(op.get("start_time", 0.0)),
                end_time=float(op.get("end_time", 0.0)),
                operation_type=op.get("operation_type"),
                parameters=op.get("parameters", {}),
                priority=int(op.get("priority", 0)),
                source_decision=op.get("source_decision"),
            )
        )

    sync_points = []
    for sp in data.get("sync_points", []):
        sync_points.append(
            SynchronizationPoint(
                timestamp=float(sp.get("timestamp", 0.0)),
                sync_type=sp.get("sync_type", ""),
                affected_tracks=sp.get("affected_tracks", []),
                tolerance=float(sp.get("tolerance", 0.1)),
            )
        )

    return ExecutionTimeline(
        total_duration=float(data.get("total_duration", 0.0)),
        operations=ops,
        sync_points=sync_points,
        track_mapping=data.get("track_mapping", {}),
        conflicts_resolved=int(data.get("conflicts_resolved", 0)),
        optimization_applied=bool(data.get("optimization_applied", False)),
    )