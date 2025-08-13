"""Configuration management for AI Video Editor."""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

from pydantic import field_validator
from pydantic_settings import BaseSettings


class ContentType(str, Enum):
    """Supported content types."""
    EDUCATIONAL = "educational"
    MUSIC = "music"
    GENERAL = "general"


class VideoQuality(str, Enum):
    """Video quality settings."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


class OutputFormat(str, Enum):
    """Supported output formats."""
    MP4 = "mp4"
    AVI = "avi"
    MOV = "mov"
    MKV = "mkv"


@dataclass
class ProjectSettings:
    """Settings for a video project."""
    content_type: ContentType = ContentType.GENERAL
    target_duration: Optional[int] = None  # in seconds
    auto_enhance: bool = True
    output_format: OutputFormat = OutputFormat.MP4
    quality: VideoQuality = VideoQuality.HIGH
    enable_analytics: bool = True
    enable_character_animation: bool = False
    enable_b_roll_generation: bool = True
    enable_thumbnail_generation: bool = True
    
    def __post_init__(self):
        """Set default target duration based on content type."""
        if self.target_duration is None:
            duration_map = {
                ContentType.EDUCATIONAL: 900,  # 15 minutes
                ContentType.MUSIC: 360,        # 6 minutes
                ContentType.GENERAL: 180,      # 3 minutes
            }
            self.target_duration = duration_map[self.content_type]


class Settings(BaseSettings):
    """Application settings loaded from environment variables and config files."""
    
    # Application settings
    app_name: str = "AI Video Editor"
    version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # File paths
    project_dir: Path = Path.cwd()
    output_dir: Path = Path.cwd() / "out"
    temp_dir: Path = Path.cwd() / "temp"
    logs_dir: Path = Path.cwd() / "out" / "logs"
    
    # API Configuration
    gemini_api_key: Optional[str] = None
    imagen_api_key: Optional[str] = None
    google_cloud_project: Optional[str] = None
    
    # Processing settings
    max_concurrent_processes: int = 2
    max_memory_usage_gb: float = 8.0
    enable_gpu_acceleration: bool = True
    
    # Default project settings
    default_content_type: ContentType = ContentType.GENERAL
    default_quality: VideoQuality = VideoQuality.HIGH
    default_output_format: OutputFormat = OutputFormat.MP4
    
    # Supported formats
    supported_video_formats: List[str] = field(default_factory=lambda: [
        ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"
    ])
    supported_audio_formats: List[str] = field(default_factory=lambda: [
        ".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a"
    ])
    
    # Performance settings
    whisper_model_size: str = "large-v3"
    video_processing_timeout: int = 3600  # 1 hour
    api_request_timeout: int = 30
    max_retries: int = 3
    
    class Config:
        env_prefix = "AI_VIDEO_EDITOR_"
        env_file = ".env"
        case_sensitive = False
    
    @field_validator("project_dir", "output_dir", "temp_dir", "logs_dir", mode="before")
    @classmethod
    def ensure_path_exists(cls, v):
        """Ensure directories exist."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @field_validator("gemini_api_key", "imagen_api_key")
    @classmethod
    def validate_api_keys(cls, v, info):
        """Validate API keys are provided when needed."""
        if v is None:
            import warnings
            warnings.warn(f"{info.field_name} not provided. Some features may not work.")
        return v
    
    def get_project_settings(self, **overrides) -> ProjectSettings:
        """Get project settings with optional overrides."""
        # Start with overrides to ensure content_type is set correctly before __post_init__
        content_type = overrides.get('content_type', self.default_content_type)
        
        settings = ProjectSettings(
            content_type=content_type,
            quality=overrides.get('quality', self.default_quality),
            output_format=overrides.get('output_format', self.default_output_format),
        )
        
        # Apply any remaining overrides
        for key, value in overrides.items():
            if hasattr(settings, key) and key not in ['content_type', 'quality', 'output_format']:
                setattr(settings, key, value)
        
        return settings


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def load_settings(config_file: Optional[Path] = None) -> Settings:
    """Load settings from file and environment variables."""
    global _settings
    
    if config_file and config_file.exists():
        # Load from specific config file
        _settings = Settings(_env_file=str(config_file))
    else:
        # Load from default locations
        _settings = Settings()
    
    return _settings


def create_default_config(config_path: Path) -> None:
    """Create a default configuration file."""
    default_config = """# AI Video Editor Configuration

# API Keys (required for full functionality)
AI_VIDEO_EDITOR_GEMINI_API_KEY=your_gemini_api_key_here
AI_VIDEO_EDITOR_IMAGEN_API_KEY=your_imagen_api_key_here
AI_VIDEO_EDITOR_GOOGLE_CLOUD_PROJECT=your_project_id_here

# Application Settings
AI_VIDEO_EDITOR_DEBUG=false
AI_VIDEO_EDITOR_LOG_LEVEL=INFO

# Processing Settings
AI_VIDEO_EDITOR_MAX_CONCURRENT_PROCESSES=2
AI_VIDEO_EDITOR_MAX_MEMORY_USAGE_GB=8.0
AI_VIDEO_EDITOR_ENABLE_GPU_ACCELERATION=true

# Default Project Settings
AI_VIDEO_EDITOR_DEFAULT_CONTENT_TYPE=general
AI_VIDEO_EDITOR_DEFAULT_QUALITY=high
AI_VIDEO_EDITOR_DEFAULT_OUTPUT_FORMAT=mp4

# Performance Settings
AI_VIDEO_EDITOR_WHISPER_MODEL_SIZE=large-v3
AI_VIDEO_EDITOR_VIDEO_PROCESSING_TIMEOUT=3600
AI_VIDEO_EDITOR_API_REQUEST_TIMEOUT=30
AI_VIDEO_EDITOR_MAX_RETRIES=3
"""
    
    config_path.write_text(default_config)


# Configuration validation
def validate_environment() -> Dict[str, Any]:
    """Validate the current environment and return status."""
    settings = get_settings()
    status = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "api_keys": {},
        "directories": {},
        "system": {}
    }
    
    # Check API keys
    if not settings.gemini_api_key:
        status["warnings"].append("Gemini API key not configured")
        status["api_keys"]["gemini"] = False
    else:
        status["api_keys"]["gemini"] = True
    
    if not settings.imagen_api_key:
        status["warnings"].append("Imagen API key not configured")
        status["api_keys"]["imagen"] = False
    else:
        status["api_keys"]["imagen"] = True
    
    # Check directories
    for dir_name, dir_path in [
        ("project", settings.project_dir),
        ("output", settings.output_dir),
        ("temp", settings.temp_dir),
        ("logs", settings.logs_dir),
    ]:
        if dir_path.exists() and dir_path.is_dir():
            status["directories"][dir_name] = True
        else:
            status["errors"].append(f"{dir_name} directory not accessible: {dir_path}")
            status["directories"][dir_name] = False
            status["valid"] = False
    
    # Check system resources
    import psutil
    memory_gb = psutil.virtual_memory().total / (1024**3)
    status["system"]["memory_gb"] = round(memory_gb, 1)
    
    if memory_gb < settings.max_memory_usage_gb:
        status["warnings"].append(
            f"Available memory ({memory_gb:.1f}GB) is less than configured limit "
            f"({settings.max_memory_usage_gb}GB)"
        )
    
    return status