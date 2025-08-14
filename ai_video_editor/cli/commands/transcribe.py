"""Transcribe command - Whisper transcription with smart defaults."""
from __future__ import annotations

import time
import sys
from pathlib import Path
from typing import Optional

import click

from ai_video_editor.utils.logging_config import get_logger
from .utils import (
    save_json, enhance_audio_fallback, romanize_hindi_text,
    get_smart_device_settings, get_smart_model_name, format_time_duration
)

logger = get_logger(__name__)


@click.command("transcribe")
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option("--model", "model_size", type=click.Choice(["tiny", "base", "small", "medium", "large", "turbo"]), 
              default="large", help="Model size (default: large for best quality)")
@click.option("--language", type=str, default=None, help="Language code (auto-detect if omitted)")
@click.option("--output", "output_path", type=click.Path(path_type=Path), default=None, 
              help="Output transcript JSON path (default: workspace/outputs/<filename>_transcript.json)")
@click.option("--backend", type=click.Choice(["whisper", "faster-whisper"]), default="faster-whisper", 
              help="ASR backend (default: faster-whisper)")
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]), default="auto", 
              help="Device for backend execution (smart default: auto)")
@click.option("--compute-type", "compute_type", type=click.Choice(["int8", "int8_float32", "float16", "float32"]), 
              default=None, help="faster-whisper compute type")
@click.option("--vad/--no-vad", default=True, help="Use voice activity detection for better segmentation")
@click.option("--vad-threshold", type=float, default=0.3, 
              help="VAD threshold (0.0-1.0, lower = more sensitive, more segments)")
@click.option("--min-silence-duration", type=int, default=500, 
              help="Minimum silence duration in ms to split segments (lower = more segments)")
@click.option("--word-timestamps", default=True, is_flag=True, help="Enable word-level timestamps (default: enabled)")
@click.option("--segment-length", type=int, default=3, 
              help="Force maximum segment length in seconds (default: 3 seconds)")
@click.option("--initial-prompt", type=str, default=None, 
              help="Initial prompt to bias recognition (e.g., script hints)")
@click.option("--enhance-audio/--no-enhance-audio", default=True, 
              help="Preprocess audio: normalize + light denoise before ASR (default: enabled)")
@click.option("--task", type=click.Choice(["transcribe", "translate"]), default="transcribe", 
              help="Recognition task")
@click.option("--romanize/--no-romanize", default=True, 
              help="If Hindi is detected, output Hinglish (romanized) text by default (smart default: on)")
@click.option("--romanize-scheme", type=click.Choice(["hk", "itrans", "iast"]), default="hk", 
              help="Romanization scheme for Hindi")
@click.option("--force-model", default=True, is_flag=True, 
              help="Force use of specified model size even on CPU (default: enabled for large model)")
@click.option("--vocab-file", type=click.Path(exists=True, path_type=Path), 
              help="Sanskrit/Hindi vocabulary file for better word prediction")
@click.option("--preset", type=click.Choice(["hindi-religious", "sanskrit-classical", "mythological", "comprehensive", "general"]), 
              help="Preset configuration for common use cases")
@click.option("--vocab-size", type=int, default=100, 
              help="Number of vocabulary words to use in prompt (default: 100)")
@click.option("--progress/--no-progress", default=True, help="Show progress bar during transcription")
@click.option("--quiet", is_flag=True, help="Suppress progress and info output")
def transcribe_cmd(input_file: Path, model_size: str, language: Optional[str], output_path: Optional[Path], 
                   backend: str, device: str, compute_type: Optional[str], vad: bool, vad_threshold: float, 
                   min_silence_duration: int, word_timestamps: bool, segment_length: int, 
                   initial_prompt: Optional[str], enhance_audio: bool, task: str, romanize: bool, 
                   romanize_scheme: str, force_model: bool, vocab_file: Optional[Path], 
                   preset: Optional[str], progress: bool, quiet: bool, vocab_size: int):
    """Transcribe audio/video to JSON transcript using Whisper.
    
    If language is Hindi (hi) and --romanize is enabled (default), the output JSON's 
    text/segments.text are Hinglish (romanized), with originals preserved in text_original.
    """
    try:
        start_time = time.time()
        
        # Generate default output path if not provided
        if output_path is None:
            # Create workspace/outputs directory if it doesn't exist
            output_dir = Path("workspace/outputs")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename: <input_name>_transcript.json
            input_stem = input_file.stem
            output_path = output_dir / f"{input_stem}_transcript.json"
            
            if not quiet:
                click.echo(f"[INFO] Using default output path: {output_path}")
        
        # Apply preset configurations with comprehensive vocabulary
        if preset:
            initial_prompt = _apply_preset(preset, initial_prompt, vocab_size, quiet, language)
        
        # Load vocabulary file if provided
        if vocab_file:
            initial_prompt = _load_vocab_file(vocab_file, initial_prompt, quiet)
        
        # Built-in comprehensive Sanskrit/Hindi vocabulary fallback
        if not initial_prompt and (language == "hi" or not language):
            initial_prompt = _apply_default_vocab(vocab_size, quiet)
        
        # Optional audio enhancement pre-step
        input_media_path = str(input_file)
        if enhance_audio:
            input_media_path = _enhance_audio(input_file)
        
        # Transcribe using selected backend
        if backend == "whisper":
            data = _transcribe_with_whisper(input_media_path, model_size, language)
        else:
            data = _transcribe_with_faster_whisper(
                input_media_path, model_size, language, device, compute_type, vad, 
                vad_threshold, min_silence_duration, word_timestamps, segment_length,
                initial_prompt, task, force_model, progress, quiet, start_time
            )
        
        # Auto-romanize for Hindi
        if romanize and data.get('language') == 'hi':
            data = romanize_hindi_text(data, romanize_scheme)
        
        # Add final processing time
        total_time = time.time() - start_time
        if data:
            data["processing_time"] = round(total_time, 2)
        
        # Save output
        save_json(data, output_path, ensure_ascii=False)
        
        if not quiet:
            click.echo(f"[OK] Transcript written to {output_path}")
            click.echo(f"[INFO] Total processing time: {format_time_duration(total_time)}")
        else:
            click.echo(f"[OK] Transcript written to {output_path}")
            
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        click.echo(f"[ERROR] Transcription failed: {e}")
        sys.exit(1)


def _apply_preset(preset: str, initial_prompt: Optional[str], vocab_size: int, quiet: bool, language: Optional[str]) -> Optional[str]:
    """Apply preset configurations with comprehensive vocabulary."""
    try:
        from ai_video_editor.utils.sanskrit_hindi_vocab import sanskrit_hindi_vocab
        
        if not language:
            language = "hi"  # All presets use Hindi detection
        
        preset_map = {
            "hindi-religious": "religious",
            "sanskrit-classical": "classical", 
            "mythological": "mythological",
            "comprehensive": "comprehensive",
            "general": "common"
        }
        
        vocab_type = preset_map.get(preset)
        if vocab_type:
            vocab_prompt = sanskrit_hindi_vocab.get_vocabulary_prompt(vocab_type, vocab_size)
            if not initial_prompt:
                initial_prompt = vocab_prompt
            if not quiet:
                click.echo(f"[INFO] Applied {preset} preset: {len(vocab_prompt.split(', '))} terms")
        
        return initial_prompt
    except ImportError:
        logger.warning("Sanskrit/Hindi vocabulary module not available")
        return initial_prompt


def _load_vocab_file(vocab_file: Path, initial_prompt: Optional[str], quiet: bool) -> Optional[str]:
    """Load vocabulary file and combine with initial prompt."""
    try:
        with vocab_file.open("r", encoding="utf-8") as f:
            vocab_words = [line.strip() for line in f if line.strip()]
        if not quiet:
            click.echo(f"[INFO] Loaded {len(vocab_words)} vocabulary words from {vocab_file}")
        
        if vocab_words:
            vocab_prompt = ", ".join(vocab_words[:50])  # Limit to first 50 words
            if initial_prompt:
                initial_prompt = f"{initial_prompt}, {vocab_prompt}"
            else:
                initial_prompt = vocab_prompt
        
        return initial_prompt
    except Exception as e:
        logger.warning(f"Failed to load vocabulary file: {e}")
        return initial_prompt


def _apply_default_vocab(vocab_size: int, quiet: bool) -> Optional[str]:
    """Apply default comprehensive vocabulary for Hindi."""
    try:
        from ai_video_editor.utils.sanskrit_hindi_vocab import sanskrit_hindi_vocab
        
        initial_prompt = sanskrit_hindi_vocab.get_vocabulary_prompt("comprehensive", min(vocab_size, 80))
        if not quiet:
            vocab_stats = sanskrit_hindi_vocab.get_vocabulary_stats()
            click.echo(f"[INFO] Using comprehensive Sanskrit/Hindi vocabulary ({vocab_stats['total_unique']} total terms available)")
            click.echo(f"[INFO] Selected {len(initial_prompt.split(', '))} terms for ASR prompt")
        
        return initial_prompt
    except ImportError:
        return None


def _enhance_audio(input_file: Path) -> str:
    """Enhance audio with fallback to FFmpeg."""
    try:
        from ai_video_editor.modules.enhancement.audio_enhancement import AudioEnhancementEngine, AudioEnhancementSettings
        from ai_video_editor.core.content_context import ContentContext, ContentType, UserPreferences
        
        ctx = ContentContext(
            project_id="transcribe_cli", 
            video_files=[str(input_file)], 
            content_type=ContentType.GENERAL, 
            user_preferences=UserPreferences()
        )
        engine = AudioEnhancementEngine(
            output_dir="temp/audio_enhancement", 
            settings=AudioEnhancementSettings(noise_reduction_strength=0.3)
        )
        res = engine.enhance_audio(ctx, audio_file_path=str(input_file))
        enhanced_audio_path = res.enhanced_audio_path or str(input_file)
        logger.info(f"Using enhanced audio for transcription (internal): {enhanced_audio_path}")
        return enhanced_audio_path
    except Exception as e:
        logger.warning(f"Internal audio enhancement failed: {e}; falling back to FFmpeg loudnorm")
        return enhance_audio_fallback(input_file)


def _transcribe_with_whisper(input_media_path: str, model_size: str, language: Optional[str]) -> dict:
    """Transcribe using standard Whisper backend."""
    from ai_video_editor.modules.content_analysis.audio_analyzer import FinancialContentAnalyzer
    
    analyzer = FinancialContentAnalyzer()
    transcript = analyzer.transcribe_audio(input_media_path, model_size=model_size, language=language)
    return transcript.to_dict() if hasattr(transcript, 'to_dict') else transcript


def _transcribe_with_faster_whisper(input_media_path: str, model_size: str, language: Optional[str], 
                                   device: str, compute_type: Optional[str], vad: bool, vad_threshold: float,
                                   min_silence_duration: int, word_timestamps: bool, segment_length: int,
                                   initial_prompt: Optional[str], task: str, force_model: bool, 
                                   progress: bool, quiet: bool, start_time: float) -> dict:
    """Transcribe using faster-whisper backend."""
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        raise RuntimeError("faster-whisper backend requested but package not installed. pip install faster-whisper") from e
    
    # Get smart device settings
    dev, ct = get_smart_device_settings(device, compute_type)
    
    # Get smart model name
    model_name = get_smart_model_name(model_size, dev, force_model)
    
    # Build and transcribe with fallback
    def _build_and_transcribe(_dev: str, _ct: str, _model_name: str):
        _model = WhisperModel(_model_name, device=_dev, compute_type=_ct)
        
        # Configure VAD parameters
        vad_parameters = None
        if vad:
            vad_parameters = {
                "threshold": vad_threshold,
                "min_silence_duration_ms": min_silence_duration,
            }
            if not quiet:
                click.echo(f"[INFO] Using VAD for granular segmentation: threshold={vad_threshold}, min_silence={min_silence_duration}ms")
        else:
            if not quiet:
                click.echo("[INFO] VAD disabled - may produce longer segments")
        
        return _model.transcribe(
            input_media_path,
            language=language or None,
            task=task,
            vad_filter=vad,
            vad_parameters=vad_parameters,
            initial_prompt=initial_prompt,
            word_timestamps=word_timestamps,
        )
    
    # Try transcription with CUDA fallback
    try:
        seg_iter, info = _build_and_transcribe(dev, ct, model_name)
    except Exception as ex:
        emsg = str(ex).lower()
        if "cuda" in emsg or "out of memory" in emsg or "cudnn" in emsg:
            logger.warning("CUDA path failed (OOM). Falling back to CPU int8 'base'.")
            dev = "cpu"
            ct = "int8"
            model_name = "base"
            seg_iter, info = _build_and_transcribe(dev, ct, model_name)
        else:
            raise
    
    # Process segments with progress tracking
    segments, full_text = _process_segments(
        seg_iter, info, input_media_path, segment_length, progress, quiet, start_time
    )
    
    return {
        "text": " ".join(full_text).strip(),
        "segments": segments,
        "language": (language or getattr(info, "language", "unknown")),
        "model_used": f"faster-whisper:{model_name}:{dev}:{ct}",
        "processing_time": None,
    }


def _process_segments(seg_iter, info, input_media_path: str, segment_length: int, 
                     progress: bool, quiet: bool, start_time: float):
    """Process segments with optional progress tracking and length splitting."""
    # Get audio duration for progress tracking
    audio_duration = getattr(info, 'duration', None)
    if not audio_duration:
        try:
            import subprocess
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'csv=p=0', input_media_path
            ], capture_output=True, text=True)
            audio_duration = float(result.stdout.strip()) if result.stdout.strip() else None
        except Exception:
            audio_duration = None
    
    segments = []
    full_text = []
    
    # Progress tracking setup
    if progress and not quiet and audio_duration:
        progress_bar_width = 50
        last_progress = 0
        
        def update_progress(current_time):
            nonlocal last_progress
            if audio_duration and current_time:
                percent = min(100, (current_time / audio_duration) * 100)
                if percent - last_progress >= 2:  # Update every 2%
                    filled = int(progress_bar_width * percent / 100)
                    bar = '█' * filled + '░' * (progress_bar_width - filled)
                    elapsed = time.time() - start_time
                    if percent > 0:
                        eta = (elapsed / percent) * (100 - percent)
                        eta_str = format_time_duration(eta)
                    else:
                        eta_str = "--:--"
                    sys.stdout.write(f'\r[{bar}] {percent:5.1f}% | Elapsed: {format_time_duration(elapsed)} | ETA: {eta_str}')
                    sys.stdout.flush()
                    last_progress = percent
        
        if not quiet:
            click.echo(f"[INFO] Transcribing {audio_duration:.1f}s audio...")
    
    # Collect all segments first
    raw_segments = []
    for s in seg_iter:
        seg = {
            "text": s.text.strip(),
            "start": float(s.start) if s.start is not None else 0.0,
            "end": float(s.end) if s.end is not None else 0.0,
        }
        raw_segments.append(seg)
        
        # Update progress
        if progress and not quiet and audio_duration:
            update_progress(seg["end"])
    
    # Post-process segments for granular splitting if requested
    if segment_length > 0:
        if not quiet:
            click.echo(f"[INFO] Splitting segments to maximum {segment_length} seconds each...")
        
        segments, full_text = _split_long_segments(raw_segments, segment_length)
    else:
        segments = raw_segments
        full_text = [seg["text"] for seg in segments]
    
    # Complete progress bar
    if progress and not quiet and audio_duration:
        elapsed = time.time() - start_time
        sys.stdout.write(f'\r[{"█" * 50}] 100.0% | Completed in {format_time_duration(elapsed)}\n')
        sys.stdout.flush()
        
    if not quiet:
        click.echo(f"[INFO] Generated {len(segments)} segments")
    
    return segments, full_text


def _split_long_segments(raw_segments, segment_length: int):
    """Split long segments into smaller chunks."""
    segments = []
    full_text = []
    
    for seg in raw_segments:
        duration = seg["end"] - seg["start"]
        if duration <= segment_length:
            segments.append(seg)
            full_text.append(seg["text"])
        else:
            # Split long segment into smaller chunks
            words = seg["text"].split()
            if len(words) <= 1:
                segments.append(seg)
                full_text.append(seg["text"])
                continue
            
            words_per_chunk = max(1, len(words) * segment_length // int(duration))
            
            for i in range(0, len(words), words_per_chunk):
                chunk_words = words[i:i + words_per_chunk]
                chunk_text = " ".join(chunk_words)
                
                # Calculate proportional timing
                chunk_start = seg["start"] + (duration * i / len(words))
                chunk_end = seg["start"] + (duration * min(i + words_per_chunk, len(words)) / len(words))
                
                chunk_seg = {
                    "text": chunk_text,
                    "start": round(chunk_start, 2),
                    "end": round(chunk_end, 2),
                }
                segments.append(chunk_seg)
                full_text.append(chunk_text)
    
    return segments, full_text