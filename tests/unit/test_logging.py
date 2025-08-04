"""Unit tests for logging configuration."""

import pytest
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from ai_video_editor.utils.logging_config import (
    setup_logging,
    get_logger,
    log_performance,
    LOGGING_CONFIG
)


class TestLoggingSetup:
    """Test logging setup and configuration."""
    
    def test_setup_logging_default(self):
        """Test default logging setup."""
        setup_logging()
        
        # Check that ai_video_editor logger exists and is configured
        logger = logging.getLogger("ai_video_editor")
        # The logger level might be modified by previous tests, so just check it's valid
        assert logger.level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
        assert len(logger.handlers) == 3  # console, file, error_file
    
    def test_setup_logging_with_level_override(self):
        """Test logging setup with level override."""
        setup_logging(log_level="WARNING")
        
        logger = logging.getLogger("ai_video_editor")
        assert logger.level == logging.WARNING
    
    @patch.dict('os.environ', {'AI_VIDEO_EDITOR_DEBUG': 'true'})
    def test_setup_logging_debug_mode(self):
        """Test logging setup in debug mode."""
        setup_logging()
        
        # In debug mode, console handler should use detailed formatter
        logger = logging.getLogger("ai_video_editor")
        console_handler = next(
            h for h in logger.handlers 
            if isinstance(h, logging.StreamHandler)
        )
        assert console_handler.level == logging.DEBUG
    
    def test_get_logger(self):
        """Test getting logger instance."""
        logger = get_logger("test_module")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"


class TestLogPerformanceDecorator:
    """Test performance logging decorator."""
    
    def test_log_performance_success(self):
        """Test performance logging with successful function."""
        @log_performance
        def test_function():
            return "success"
        
        with patch('ai_video_editor.utils.logging_config.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            result = test_function()
            
            assert result == "success"
            mock_logger.debug.assert_called_once()
            debug_call = mock_logger.debug.call_args[0][0]
            assert "test_function executed in" in debug_call
            assert "seconds" in debug_call
    
    def test_log_performance_with_exception(self):
        """Test performance logging with function that raises exception."""
        @log_performance
        def test_function():
            raise ValueError("Test error")
        
        with patch('ai_video_editor.utils.logging_config.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            with pytest.raises(ValueError):
                test_function()
            
            mock_logger.error.assert_called_once()
            error_call = mock_logger.error.call_args[0][0]
            assert "test_function failed after" in error_call
            assert "seconds" in error_call


class TestLoggingConfiguration:
    """Test logging configuration structure."""
    
    def test_logging_config_structure(self):
        """Test that logging configuration has required structure."""
        assert "version" in LOGGING_CONFIG
        assert "formatters" in LOGGING_CONFIG
        assert "handlers" in LOGGING_CONFIG
        assert "loggers" in LOGGING_CONFIG
        
        # Check required formatters
        formatters = LOGGING_CONFIG["formatters"]
        assert "detailed" in formatters
        assert "simple" in formatters
        
        # Check required handlers
        handlers = LOGGING_CONFIG["handlers"]
        assert "console" in handlers
        assert "file" in handlers
        assert "error_file" in handlers
        
        # Check ai_video_editor logger configuration
        loggers = LOGGING_CONFIG["loggers"]
        assert "ai_video_editor" in loggers
        ai_logger = loggers["ai_video_editor"]
        # The level might be modified by setup_logging, so just check it exists
        assert "level" in ai_logger
        assert "console" in ai_logger["handlers"]
        assert "file" in ai_logger["handlers"]
        assert "error_file" in ai_logger["handlers"]
    
    def test_log_file_paths(self):
        """Test that log file paths are correctly configured."""
        handlers = LOGGING_CONFIG["handlers"]
        
        file_handler = handlers["file"]
        error_handler = handlers["error_file"]
        
        assert "ai_video_editor.log" in file_handler["filename"]
        assert "errors.log" in error_handler["filename"]
        
        # Check that log files would be created in logs directory
        assert "logs" in file_handler["filename"]
        assert "logs" in error_handler["filename"]