"""Feature-focused CLI (ai-ve) for modular tools.

Commands:
- transcribe: Whisper transcription -> JSON with smart defaults and Hinglish support
- to-srt: Convert transcript JSON to SRT/VTT subtitles
- plan-execute: AI Director decisions + b-roll plans -> execution_timeline.json
- render: Render from timeline.json (or AI plan JSON) using movis
"""
from __future__ import annotations

import click

from ai_video_editor.utils.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.version_option(version="0.1.0", prog_name="ai-ve")
def cli(debug: bool):
    """AI Video Editor feature tools (modular CLI).
    
    Smart defaults for Windows/low VRAM systems:
    - Uses faster-whisper with CPU int8 quantization by default
    - Auto-romanizes Hindi to Hinglish for better readability
    - Includes audio enhancement with FFmpeg fallback
    - CUDA auto-fallback for out-of-memory situations
    """
    setup_logging("DEBUG" if debug else "INFO")
    if debug:
        logger.debug("Debug mode enabled")


# Import and register commands from separate modules
from .commands.transcribe import transcribe_cmd
from .commands.subtitles import to_srt_cmd  # includes default out path
from .commands.analyze import analyze_cmd
from .commands.generate_plan import generate_plan_cmd
from .commands.plan_execute import plan_execute_cmd
from .commands.render import render_cmd

cli.add_command(transcribe_cmd)
cli.add_command(to_srt_cmd)
cli.add_command(analyze_cmd)
cli.add_command(generate_plan_cmd)
cli.add_command(plan_execute_cmd)
cli.add_command(render_cmd)


def main():
    """Main entry point for ai-ve CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\n[INFO] Operation cancelled by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        click.echo(f"[ERROR] Unexpected error: {e}")


if __name__ == "__main__":
    main()