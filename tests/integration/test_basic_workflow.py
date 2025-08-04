"""Integration tests for basic workflow."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from ai_video_editor.core.config import get_settings, validate_environment
from ai_video_editor.utils.logging_config import setup_logging, get_logger


class TestBasicWorkflow:
    """Test basic application workflow."""
    
    def test_application_initialization(self):
        """Test that the application can be initialized properly."""
        # Setup logging
        setup_logging()
        logger = get_logger(__name__)
        
        # Load settings
        settings = get_settings()
        
        # Verify basic configuration
        assert settings.app_name == "AI Video Editor"
        assert settings.version == "0.1.0"
        assert isinstance(settings.project_dir, Path)
        assert isinstance(settings.output_dir, Path)
        
        logger.info("Application initialization test passed")
    
    def test_environment_validation_workflow(self):
        """Test environment validation workflow."""
        # Validate environment
        status = validate_environment()
        
        # Should have all required keys
        assert "valid" in status
        assert "warnings" in status
        assert "errors" in status
        assert "api_keys" in status
        assert "directories" in status
        assert "system" in status
        
        # Should detect missing API keys as warnings, not errors
        assert "gemini" in status["api_keys"]
        assert "imagen" in status["api_keys"]
        
        # Directories should be accessible
        for dir_name, accessible in status["directories"].items():
            assert accessible, f"{dir_name} directory should be accessible"
    
    def test_project_settings_workflow(self):
        """Test project settings creation workflow."""
        settings = get_settings()
        
        # Test different content types
        educational_settings = settings.get_project_settings(
            content_type="educational"
        )
        assert educational_settings.target_duration == 900  # 15 minutes
        
        music_settings = settings.get_project_settings(
            content_type="music"
        )
        assert music_settings.target_duration == 360  # 6 minutes
        
        general_settings = settings.get_project_settings(
            content_type="general"
        )
        assert general_settings.target_duration == 180  # 3 minutes
    
    @patch('ai_video_editor.core.config.Path.mkdir')
    def test_directory_creation_workflow(self, mock_mkdir):
        """Test that required directories are created."""
        from ai_video_editor.core.config import Settings
        
        # This should trigger directory creation
        settings = Settings()
        
        # Verify mkdir was called for each directory
        assert mock_mkdir.call_count >= 4  # project, output, temp, logs
    
    def test_logging_workflow(self):
        """Test logging workflow."""
        setup_logging("INFO")
        
        # Get different loggers
        main_logger = get_logger("ai_video_editor")
        module_logger = get_logger("ai_video_editor.modules.test")
        
        # Test logging at different levels
        main_logger.info("Test info message")
        main_logger.debug("Test debug message")
        main_logger.warning("Test warning message")
        
        module_logger.info("Module test message")
        
        # Should not raise any exceptions
        assert True


class TestErrorHandlingWorkflow:
    """Test error handling in workflows."""
    
    def test_configuration_error_handling(self):
        """Test configuration error handling."""
        from ai_video_editor.core.exceptions import ConfigurationError
        
        # This should not raise an exception even with missing API keys
        settings = get_settings()
        assert settings.gemini_api_key is None
        assert settings.imagen_api_key is None
    
    def test_logging_error_handling(self):
        """Test logging error handling."""
        # Should handle invalid log levels gracefully
        setup_logging("INVALID_LEVEL")
        
        logger = get_logger("test")
        logger.info("This should still work")
        
        # Should not raise exceptions
        assert True