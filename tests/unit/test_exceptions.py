"""Unit tests for custom exceptions."""

import pytest
from unittest.mock import MagicMock, patch

from ai_video_editor.core.exceptions import (
    VideoEditorError,
    InputValidationError,
    FileNotFoundError,
    UnsupportedFormatError,
    ProcessingError,
    ResourceError,
    NetworkError,
    APIError,
    ConfigurationError,
    AuthenticationError,
    TimeoutError,
    handle_errors,
    retry_on_error
)


class TestVideoEditorError:
    """Test base VideoEditorError class."""
    
    def test_basic_error(self):
        """Test basic error creation."""
        error = VideoEditorError("Test error")
        
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.error_code is None
        assert error.details == {}
    
    def test_error_with_code_and_details(self):
        """Test error with error code and details."""
        details = {"key": "value"}
        error = VideoEditorError("Test error", error_code="TEST_ERROR", details=details)
        
        assert str(error) == "[TEST_ERROR] Test error"
        assert error.error_code == "TEST_ERROR"
        assert error.details == details


class TestSpecificErrors:
    """Test specific error types."""
    
    def test_input_validation_error(self):
        """Test InputValidationError."""
        error = InputValidationError("Invalid input", field="test_field")
        
        assert error.error_code == "INPUT_VALIDATION"
        assert error.field == "test_field"
    
    def test_file_not_found_error(self):
        """Test FileNotFoundError."""
        error = FileNotFoundError("/path/to/file.mp4")
        
        assert error.error_code == "FILE_NOT_FOUND"
        assert error.file_path == "/path/to/file.mp4"
        assert "File not found: /path/to/file.mp4" in str(error)
    
    def test_unsupported_format_error(self):
        """Test UnsupportedFormatError."""
        supported = [".mp4", ".avi"]
        error = UnsupportedFormatError(".xyz", supported_formats=supported)
        
        assert error.error_code == "UNSUPPORTED_FORMAT"
        assert error.file_format == ".xyz"
        assert error.supported_formats == supported
        assert ".mp4, .avi" in str(error)
    
    def test_processing_error(self):
        """Test ProcessingError."""
        error = ProcessingError("video_merge", reason="Insufficient memory")
        
        assert error.error_code == "PROCESSING_ERROR"
        assert error.operation == "video_merge"
        assert error.reason == "Insufficient memory"
    
    def test_resource_error(self):
        """Test ResourceError."""
        error = ResourceError("memory", required="8GB", available="4GB")
        
        assert error.error_code == "RESOURCE_ERROR"
        assert error.resource_type == "memory"
        assert "required: 8GB, available: 4GB" in str(error)
    
    def test_network_error(self):
        """Test NetworkError."""
        error = NetworkError("Gemini API", reason="Connection timeout")
        
        assert error.error_code == "NETWORK_ERROR"
        assert error.service == "Gemini API"
        assert error.reason == "Connection timeout"
    
    def test_api_error(self):
        """Test APIError."""
        error = APIError("Gemini API", status_code=429, response_text="Rate limited")
        
        assert error.error_code == "API_ERROR"
        assert error.api_name == "Gemini API"
        assert error.status_code == 429
        assert error.response_text == "Rate limited"
        assert "HTTP 429" in str(error)
    
    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError("api_key", reason="Missing required key")
        
        assert error.error_code == "CONFIGURATION_ERROR"
        assert error.config_key == "api_key"
        assert error.reason == "Missing required key"
    
    def test_authentication_error(self):
        """Test AuthenticationError."""
        error = AuthenticationError("Gemini API")
        
        assert error.error_code == "AUTHENTICATION_ERROR"
        assert error.service == "Gemini API"
    
    def test_timeout_error(self):
        """Test TimeoutError."""
        error = TimeoutError("video_processing", timeout_seconds=30.0)
        
        assert error.error_code == "TIMEOUT_ERROR"
        assert error.operation == "video_processing"
        assert error.timeout_seconds == 30.0


class TestErrorHandling:
    """Test error handling decorators."""
    
    def test_handle_errors_decorator_success(self):
        """Test handle_errors decorator with successful function."""
        @handle_errors()
        def test_function():
            return "success"
        
        result = test_function()
        assert result == "success"
    
    def test_handle_errors_decorator_video_editor_error(self):
        """Test handle_errors decorator with VideoEditorError."""
        @handle_errors()
        def test_function():
            raise InputValidationError("Test error")
        
        with pytest.raises(InputValidationError):
            test_function()
    
    def test_handle_errors_decorator_unexpected_error(self):
        """Test handle_errors decorator with unexpected error."""
        @handle_errors()
        def test_function():
            raise ValueError("Unexpected error")
        
        with pytest.raises(VideoEditorError) as exc_info:
            test_function()
        
        assert "Unexpected error in test_function" in str(exc_info.value)
        assert exc_info.value.error_code == "UNEXPECTED_ERROR"
    
    def test_retry_on_error_decorator_success(self):
        """Test retry_on_error decorator with successful function."""
        @retry_on_error(max_retries=2)
        def test_function():
            return "success"
        
        result = test_function()
        assert result == "success"
    
    def test_retry_on_error_decorator_retryable_error(self):
        """Test retry_on_error decorator with retryable error."""
        call_count = 0
        
        @retry_on_error(max_retries=2, delay=0.1)
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkError("Test service", "Connection failed")
            return "success"
        
        result = test_function()
        assert result == "success"
        assert call_count == 3
    
    def test_retry_on_error_decorator_non_retryable_error(self):
        """Test retry_on_error decorator with non-retryable error."""
        call_count = 0
        
        @retry_on_error(max_retries=2)
        def test_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Non-retryable error")
        
        with pytest.raises(ValueError):
            test_function()
        
        assert call_count == 1  # Should not retry