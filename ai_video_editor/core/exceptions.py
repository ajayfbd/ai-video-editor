"""Custom exceptions for AI Video Editor."""

from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .content_context import ContentContext


class VideoEditorError(Exception):
    """Base exception for all AI Video Editor errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class InputValidationError(VideoEditorError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="INPUT_VALIDATION", **kwargs)
        self.field = field


class FileNotFoundError(VideoEditorError):
    """Raised when required files are not found."""
    
    def __init__(self, file_path: str, **kwargs):
        message = f"File not found: {file_path}"
        super().__init__(message, error_code="FILE_NOT_FOUND", **kwargs)
        self.file_path = file_path


class UnsupportedFormatError(VideoEditorError):
    """Raised when file format is not supported."""
    
    def __init__(self, file_format: str, supported_formats: list = None, **kwargs):
        message = f"Unsupported format: {file_format}"
        if supported_formats:
            message += f". Supported formats: {', '.join(supported_formats)}"
        super().__init__(message, error_code="UNSUPPORTED_FORMAT", **kwargs)
        self.file_format = file_format
        self.supported_formats = supported_formats or []


class ProcessingError(VideoEditorError):
    """Raised when video processing operations fail."""
    
    def __init__(self, operation: str, reason: Optional[str] = None, **kwargs):
        message = f"Processing failed: {operation}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, error_code="PROCESSING_ERROR", **kwargs)
        self.operation = operation
        self.reason = reason


class ResourceError(VideoEditorError):
    """Raised when system resources are insufficient."""
    
    def __init__(self, resource_type: str, required: str = None, available: str = None, **kwargs):
        message = f"Insufficient {resource_type}"
        if required and available:
            message += f" (required: {required}, available: {available})"
        super().__init__(message, error_code="RESOURCE_ERROR", **kwargs)
        self.resource_type = resource_type
        self.required = required
        self.available = available


class NetworkError(VideoEditorError):
    """Raised when network operations fail."""
    
    def __init__(self, service: str, reason: Optional[str] = None, **kwargs):
        message = f"Network error accessing {service}"
        if reason:
            message += f": {reason}"
        super().__init__(message, error_code="NETWORK_ERROR", **kwargs)
        self.service = service
        self.reason = reason


class APIError(VideoEditorError):
    """Raised when API calls fail."""
    
    def __init__(
        self, 
        api_name: str, 
        status_code: Optional[int] = None, 
        response_text: Optional[str] = None,
        **kwargs
    ):
        message = f"API error: {api_name}"
        if status_code:
            message += f" (HTTP {status_code})"
        if response_text:
            message += f" - {response_text}"
        super().__init__(message, error_code="API_ERROR", **kwargs)
        self.api_name = api_name
        self.status_code = status_code
        self.response_text = response_text


class ConfigurationError(VideoEditorError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, config_key: str, reason: Optional[str] = None, **kwargs):
        message = f"Configuration error: {config_key}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, error_code="CONFIGURATION_ERROR", **kwargs)
        self.config_key = config_key
        self.reason = reason


class AuthenticationError(VideoEditorError):
    """Raised when authentication fails."""
    
    def __init__(self, service: str, **kwargs):
        message = f"Authentication failed for {service}"
        super().__init__(message, error_code="AUTHENTICATION_ERROR", **kwargs)
        self.service = service


class TimeoutError(VideoEditorError):
    """Raised when operations timeout."""
    
    def __init__(self, operation: str, timeout_seconds: float, **kwargs):
        message = f"Operation timed out: {operation} (after {timeout_seconds}s)"
        super().__init__(message, error_code="TIMEOUT_ERROR", **kwargs)
        self.operation = operation
        self.timeout_seconds = timeout_seconds


# ContentContext-specific exceptions
class ContentContextError(VideoEditorError):
    """Base exception for ContentContext-related errors."""
    
    def __init__(self, message: str, context_state: Optional['ContentContext'] = None, **kwargs):
        super().__init__(message, error_code="CONTENT_CONTEXT_ERROR", **kwargs)
        self.context_state = context_state
        self.recovery_checkpoint = None


class ContextIntegrityError(ContentContextError):
    """Raised when ContentContext data is corrupted or invalid."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, **kwargs)


class ModuleIntegrationError(ContentContextError):
    """Raised when modules fail to integrate properly."""
    
    def __init__(self, module_name: str, reason: Optional[str] = None, **kwargs):
        message = f"Module integration failed: {module_name}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, **kwargs)
        self.module_name = module_name
        self.reason = reason


class SynchronizationError(ContentContextError):
    """Raised when thumbnail-metadata synchronization fails."""
    
    def __init__(self, sync_type: str, reason: Optional[str] = None, **kwargs):
        message = f"Synchronization failed: {sync_type}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, **kwargs)
        self.sync_type = sync_type
        self.reason = reason


class APIIntegrationError(ContentContextError):
    """Raised when external API calls fail."""
    
    def __init__(self, api_service: str, operation: str, reason: Optional[str] = None, **kwargs):
        message = f"API integration failed: {api_service}.{operation}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, **kwargs)
        self.api_service = api_service
        self.operation = operation
        self.reason = reason


class GeminiAPIError(APIIntegrationError):
    """Specific error for Gemini API failures."""
    
    def __init__(self, operation: str, reason: Optional[str] = None, **kwargs):
        super().__init__("gemini", operation, reason, **kwargs)


class ImagenAPIError(APIIntegrationError):
    """Specific error for Imagen API failures."""
    
    def __init__(self, operation: str, reason: Optional[str] = None, **kwargs):
        super().__init__("imagen", operation, reason, **kwargs)


class ResourceConstraintError(ContentContextError):
    """Raised when system resources are insufficient."""
    
    def __init__(self, resource_type: str, constraint: str, **kwargs):
        message = f"Resource constraint: {resource_type} - {constraint}"
        super().__init__(message, error_code="RESOURCE_CONSTRAINT_ERROR", **kwargs)
        self.resource_type = resource_type
        self.constraint = constraint


class MemoryConstraintError(ResourceConstraintError):
    """Raised when memory usage exceeds limits."""
    
    def __init__(self, current_usage: Optional[int] = None, limit: Optional[int] = None, **kwargs):
        constraint = "memory limit exceeded"
        if current_usage and limit:
            constraint = f"memory usage {current_usage}MB exceeds limit {limit}MB"
        super().__init__("memory", constraint, error_code="MEMORY_CONSTRAINT_ERROR", **kwargs)
        self.current_usage = current_usage
        self.limit = limit


class ProcessingTimeoutError(ResourceConstraintError):
    """Raised when processing takes too long."""
    
    def __init__(self, operation: str, timeout_seconds: float, **kwargs):
        constraint = f"processing timeout after {timeout_seconds}s"
        super().__init__("processing_time", constraint, error_code="PROCESSING_TIMEOUT_ERROR", **kwargs)
        self.operation = operation
        self.timeout_seconds = timeout_seconds


class BatchProcessingError(ContentContextError):
    """Raised when batch processing operations fail."""
    
    def __init__(self, batch_id: str, operation: str, reason: Optional[str] = None, **kwargs):
        message = f"Batch processing failed: {batch_id}.{operation}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, error_code="BATCH_PROCESSING_ERROR", **kwargs)
        self.batch_id = batch_id
        self.operation = operation
        self.reason = reason


# Error handling utilities
def handle_errors(logger=None):
    """
    Decorator for comprehensive error handling and logging.
    
    Args:
        logger: Logger instance to use for error logging
    """
    def decorator(func):
        import functools
        from ai_video_editor.utils.logging_config import get_logger
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_logger = logger or get_logger(func.__module__)
            
            try:
                return func(*args, **kwargs)
            except VideoEditorError as e:
                func_logger.error(f"Video editor error in {func.__name__}: {e}")
                if e.details:
                    func_logger.debug(f"Error details: {e.details}")
                raise
            except Exception as e:
                func_logger.error(f"Unexpected error in {func.__name__}: {str(e)}", exc_info=True)
                raise VideoEditorError(
                    f"Unexpected error in {func.__name__}: {str(e)}",
                    error_code="UNEXPECTED_ERROR",
                    details={"original_exception": str(e)}
                )
        
        return wrapper
    return decorator


def retry_on_error(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator to retry operations on specific errors.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
    """
    def decorator(func):
        import functools
        import time
        from ai_video_editor.utils.logging_config import get_logger
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (NetworkError, APIError, TimeoutError) as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
                        raise
                except Exception as e:
                    # Don't retry on non-retryable errors
                    raise
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator