"""Subtitles command - Convert transcript JSON to SRT/VTT."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict, Any

import click

from ai_video_editor.utils.logging_config import get_logger
from .utils import load_json

logger = get_logger(__name__)


@click.command("to-srt")
@click.argument("transcript_json", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "output_path", type=click.Path(path_type=Path), required=True, 
              help="Output subtitle file path")
@click.option("--format", "fmt", type=click.Choice(["srt", "vtt"]), default="srt", 
              help="Subtitle format")
@click.option("--use-original", is_flag=True, 
              help="Use original script text instead of romanized (if available)")
@click.option("--max-line-len", type=int, default=42, 
              help="Maximum characters per line for word wrapping")
@click.option("--strip", is_flag=True, help="Strip extra whitespace from text")
def to_srt_cmd(transcript_json: Path, output_path: Path, fmt: str, use_original: bool, 
               max_line_len: int, strip: bool):
    """Convert transcript JSON to SRT or VTT subtitle format.
    
    Uses romanized text by default, or original script with --use-original.
    """
    try:
        data = load_json(transcript_json)
        segments = data.get("segments", [])
        
        if not segments:
            raise ValueError("No segments found in transcript JSON")
        
        # Generate subtitle content
        if fmt == "srt":
            content = _generate_srt(segments, use_original, max_line_len, strip)
        else:  # vtt
            content = _generate_vtt(segments, use_original, max_line_len, strip)
        
        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        
        click.echo(f"[OK] Subtitles written to {output_path} ({fmt.upper()})")
        
    except Exception as e:
        logger.error(f"Subtitle conversion failed: {e}")
        click.echo(f"[ERROR] Subtitle conversion failed: {e}")
        sys.exit(1)


def _generate_srt(segments: List[Dict[str, Any]], use_original: bool, max_line_len: int, strip: bool) -> str:
    """Generate SRT format subtitles."""
    lines = []
    
    for i, seg in enumerate(segments, 1):
        # Get text (original or romanized)
        text = _get_segment_text(seg, use_original)
        if not text:
            continue
        
        # Process text
        text = _process_text(text, max_line_len, strip)
        
        # Format timestamps
        start_time = _format_srt_timestamp(seg.get("start", 0.0))
        end_time = _format_srt_timestamp(seg.get("end", 0.0))
        
        # Add SRT entry
        lines.extend([
            str(i),
            f"{start_time} --> {end_time}",
            text,
            ""  # Empty line separator
        ])
    
    return "\n".join(lines)


def _generate_vtt(segments: List[Dict[str, Any]], use_original: bool, max_line_len: int, strip: bool) -> str:
    """Generate VTT format subtitles."""
    lines = ["WEBVTT", ""]  # VTT header
    
    for seg in segments:
        # Get text (original or romanized)
        text = _get_segment_text(seg, use_original)
        if not text:
            continue
        
        # Process text
        text = _process_text(text, max_line_len, strip)
        
        # Format timestamps
        start_time = _format_vtt_timestamp(seg.get("start", 0.0))
        end_time = _format_vtt_timestamp(seg.get("end", 0.0))
        
        # Add VTT entry
        lines.extend([
            f"{start_time} --> {end_time}",
            text,
            ""  # Empty line separator
        ])
    
    return "\n".join(lines)


def _get_segment_text(seg: Dict[str, Any], use_original: bool) -> str:
    """Get text from segment, preferring original or romanized based on flag."""
    if use_original and "text_original" in seg:
        return seg["text_original"]
    return seg.get("text", "")


def _process_text(text: str, max_line_len: int, strip: bool) -> str:
    """Process text with wrapping and stripping."""
    if strip:
        text = " ".join(text.split())  # Collapse whitespace
    
    # Simple word wrapping
    if len(text) <= max_line_len:
        return text
    
    words = text.split()
    lines = []
    current_line = []
    current_len = 0
    
    for word in words:
        if current_len + len(word) + 1 <= max_line_len:
            current_line.append(word)
            current_len += len(word) + 1
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_len = len(word)
    
    if current_line:
        lines.append(" ".join(current_line))
    
    return "\n".join(lines)


def _format_srt_timestamp(seconds: float) -> str:
    """Format timestamp for SRT format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_vtt_timestamp(seconds: float) -> str:
    """Format timestamp for VTT format (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"