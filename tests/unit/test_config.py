"""Unit tests for configuration management."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from ai_video_editor.core.config import (
    Settings, 
    ProjectSettings, 
    ContentType, 
    VideoQuality, 
    OutputFormat,
    get_settings,
    validate_environment
)


class TestProjectSettings:
    """Test ProjectSettings class."""
    
    def test_default_initialization(self):
        """Test default project settings initialization."""
        settings = ProjectSettings()
        
        assert settings.content_type == ContentType.GENERAL
        assert settings.target_duration == 180  # 3 minutes for general content
        assert settings.auto_enhance is True
        assert settings.output_format == OutputFormat.MP4
        assert settings.quality == VideoQuality.HIGH
    
    def test_educational_content_duration(self):
        """Test educational content gets correct default duration."""
        settings = ProjectSettings(content_type=ContentType.EDUCATIONAL)
        
        assert settings.target_duration == 900  # 15 minutes
    
    def test_music_content_duration(self):
        """Test music content gets correct default duration."""
        settings = ProjectSettings(content_type=ContentType.MUSIC)
        
        assert settings.target_duration == 360  # 6 minutes
    
    def test_custom_duration_override(self):
        """Test custom duration overrides default."""
        settings = ProjectSettings(
            content_type=ContentType.EDUCATIONAL,
            target_duration=600  # 10 minutes
        )
        
        assert settings.target_duration == 600


class TestSettings:
    """Test Settings class."""
    
    def test_default_settings(self):
        """Test default settings initialization."""
        settings = Settings()
        
        assert settings.app_name == "AI Video Editor"
        assert settings.version == "0.1.0"
        assert settings.debug is False
        assert settings.log_level == "INFO"
        assert settings.max_concurrent_processes == 2
        assert settings.default_content_type == ContentType.GENERAL
    
    @patch.dict('os.environ', {
        'AI_VIDEO_EDITOR_DEBUG': 'true',
        'AI_VIDEO_EDITOR_LOG_LEVEL': 'DEBUG',
        'AI_VIDEO_EDITOR_GEMINI_API_KEY': 'test_key'
    })
    def test_environment_variable_override(self):
        """Test settings can be overridden by environment variables."""
        settings = Settings()
        
        assert settings.debug is True
        assert settings.log_level == "DEBUG"
        assert settings.gemini_api_key == "test_key"
    
    def test_get_project_settings(self):
        """Test getting project settings with overrides."""
        settings = Settings()
        project_settings = settings.get_project_settings(
            content_type=ContentType.EDUCATIONAL,
            quality=VideoQuality.ULTRA
        )
        
        assert project_settings.content_type == ContentType.EDUCATIONAL
        assert project_settings.quality == VideoQuality.ULTRA
        assert project_settings.target_duration == 900  # Educational default


class TestValidateEnvironment:
    """Test environment validation."""
    
    @patch('psutil.virtual_memory')
    def test_validate_environment_success(self, mock_memory):
        """Test successful environment validation."""
        # Mock system memory
        mock_memory.return_value = MagicMock()
        mock_memory.return_value.total = 16 * 1024**3  # 16GB
        
        with patch('ai_video_editor.core.config.get_settings') as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.gemini_api_key = "test_key"
            mock_settings.imagen_api_key = "test_key"
            mock_settings.project_dir = Path("/tmp/test")
            mock_settings.output_dir = Path("/tmp/test/output")
            mock_settings.temp_dir = Path("/tmp/test/temp")
            mock_settings.logs_dir = Path("/tmp/test/logs")
            mock_settings.max_memory_usage_gb = 8.0
            mock_get_settings.return_value = mock_settings
            
            # Mock Path.exists() and Path.is_dir()
            with patch.object(Path, 'exists', return_value=True), \
                 patch.object(Path, 'is_dir', return_value=True):
                
                status = validate_environment()
                
                assert status["valid"] is True
                assert status["api_keys"]["gemini"] is True
                assert status["api_keys"]["imagen"] is True
                assert all(status["directories"].values())
    
    @patch('psutil.virtual_memory')
    def test_validate_environment_missing_api_keys(self, mock_memory):
        """Test environment validation with missing API keys."""
        # Mock system memory
        mock_memory.return_value = MagicMock()
        mock_memory.return_value.total = 16 * 1024**3  # 16GB
        
        with patch('ai_video_editor.core.config.get_settings') as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.gemini_api_key = None
            mock_settings.imagen_api_key = None
            mock_settings.project_dir = Path("/tmp/test")
            mock_settings.output_dir = Path("/tmp/test/output")
            mock_settings.temp_dir = Path("/tmp/test/temp")
            mock_settings.logs_dir = Path("/tmp/test/logs")
            mock_settings.max_memory_usage_gb = 8.0
            mock_get_settings.return_value = mock_settings
            
            # Mock Path.exists() and Path.is_dir()
            with patch.object(Path, 'exists', return_value=True), \
                 patch.object(Path, 'is_dir', return_value=True):
                
                status = validate_environment()
                
                assert status["valid"] is True  # Still valid, just warnings
                assert status["api_keys"]["gemini"] is False
                assert status["api_keys"]["imagen"] is False
                assert len(status["warnings"]) >= 2