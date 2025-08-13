"""Shared utilities for CLI commands."""
from __future__ import annotations

import json
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file with proper encoding."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Path, ensure_ascii: bool = False) -> None:
    """Save JSON file with proper encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=ensure_ascii)


def enhance_audio_fallback(input_file: Path) -> str:
    """
    Try to enhance audio using FFmpeg loudnorm as fallback.
    Returns path to enhanced audio file, or original path if enhancement fails.
    """
    try:
        Path("temp").mkdir(exist_ok=True)
        tmp_wav = Path(tempfile.gettempdir()) / f"enh_{os.getpid()}.wav"
        cmd = [
            "ffmpeg", "-y", "-i", str(input_file),
            "-af", "loudnorm", "-ar", "16000", "-ac", "1",
            str(tmp_wav)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"Using enhanced audio for transcription (ffmpeg): {tmp_wav}")
        return str(tmp_wav)
    except Exception as e:
        logger.warning(f"FFmpeg enhancement failed: {e}; proceeding with original audio")
        return str(input_file)


def guess_script(text: str) -> str:
    """Guess if text is in Urdu, Devanagari, or unknown script."""
    import re
    ar = len(re.findall(r"[\u0600-\u06FF]", text or ""))
    dev = len(re.findall(r"[\u0900-\u097F]", text or ""))
    if ar > dev and ar > 5:
        return 'urdu'
    if dev > ar and dev > 5:
        return 'devanagari'
    return 'unknown'


def urdu_to_devanagari(text: str) -> str:
    """Convert Urdu script to Devanagari."""
    try:
        from aksharamukha import transliterate as _ak
        return _ak.process('Urdu', 'Devanagari', text or '')
    except Exception:
        return text or ''


def devanagari_to_roman(text: str, scheme: str = 'hk') -> str:
    """Convert Devanagari to Roman script."""
    try:
        from indic_transliteration import sanscript
        schemes = {
            'hk': sanscript.HK,
            'itrans': sanscript.ITRANS,
            'iast': sanscript.IAST,
        }
        target = schemes.get(scheme.lower(), sanscript.HK)
        return sanscript.transliterate(text or '', sanscript.DEVANAGARI, target)
    except Exception:
        # If romanization libs not available, keep original
        return text or ''


def romanize_hindi_text(data: Dict[str, Any], scheme: str = 'hk') -> Dict[str, Any]:
    """
    Romanize Hindi text in transcript data.
    Converts Urdu/Devanagari to Roman script and preserves originals.
    """
    if not data.get('language') == 'hi':
        return data
    
    logger.info("Hindi detected: generating Hinglish (romanized) transcript")
    
    # Process main text
    text_orig = data.get('text', '')
    script = guess_script(text_orig)
    
    # Normalize to Devanagari first
    if script == 'urdu':
        dev_text = urdu_to_devanagari(text_orig)
    elif script == 'devanagari':
        dev_text = text_orig
    else:
        # Best effort: try Urdu->Dev; if unchanged, assume already Latin or mixed
        tmp = urdu_to_devanagari(text_orig)
        dev_text = tmp if tmp != text_orig else text_orig

    roman_text = devanagari_to_roman(dev_text, scheme)
    data['text_original'] = text_orig
    data['text'] = roman_text
    data['romanized'] = True
    data['romanization_scheme'] = scheme.lower()

    # Process segments
    for seg in data.get('segments', []) or []:
        seg_text = seg.get('text', '')
        s_script = guess_script(seg_text)
        if s_script == 'urdu':
            dev_seg = urdu_to_devanagari(seg_text)
        elif s_script == 'devanagari':
            dev_seg = seg_text
        else:
            tmp = urdu_to_devanagari(seg_text)
            dev_seg = tmp if tmp != seg_text else seg_text
        seg['text_original'] = seg_text
        seg['text'] = devanagari_to_roman(dev_seg, scheme)
    
    return data


def get_smart_device_settings(device: str, compute_type: Optional[str] = None):
    """
    Get smart device and compute type settings based on available hardware.
    Returns (device, compute_type) tuple.
    """
    if device == "auto":
        use_cuda = False
        try:
            import torch as _torch
            use_cuda = _torch.cuda.is_available()
            if use_cuda:
                try:
                    vram = _torch.cuda.get_device_properties(0).total_memory
                    if vram < 3_000_000_000:  # <3GB
                        logger.info("Low GPU memory detected (<3GB). Using CPU int8 for stability.")
                        use_cuda = False
                except Exception:
                    pass
        except Exception:
            use_cuda = False
    else:
        use_cuda = (device == "cuda")

    dev = "cuda" if use_cuda else "cpu"
    ct = compute_type or ("float16" if use_cuda else "int8")
    
    return dev, ct


def get_smart_model_name(model_size: str, device: str, force_model: bool = False) -> str:
    """
    Get smart model name with automatic downgrade for CPU performance.
    """
    model_mapping = {
        "tiny": "tiny",
        "base": "base", 
        "small": "small",
        "medium": "medium",
        "large": "large-v3",
        "turbo": "medium"  # map turbo to medium for now
    }
    
    model_name = model_mapping[model_size]
    
    # Warn about performance implications for large models on CPU
    if device == "cpu" and model_name in ("large-v3", "medium"):
        if force_model:
            logger.warning(f"Using {model_name} model on CPU as requested. This may be significantly slower than GPU or smaller models.")
        else:
            logger.info(f"CPU backend detected; downgrading model '{model_name}' to 'base' for performance. Use --force-model to override.")
            model_name = "base"
    
    return model_name


def format_time_duration(seconds: float) -> str:
    """Format seconds as MM:SS."""
    return f"{int(seconds//60):02d}:{int(seconds%60):02d}"