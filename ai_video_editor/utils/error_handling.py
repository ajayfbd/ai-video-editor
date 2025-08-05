"""
Error Handling Utilities - Custom exceptions and error handling patterns.

This module provides custom exceptions and error handling utilities
for the AI Video Editor system.
"""

from typing import Optional, Any, Dict
import logging

logger = logging.getLogger(__name__)


class ContentContextError(Exception):
    """Base exception for ContentContext-related errors."""
    
    def __init__(self, message: str, context_state: Optional[Any] = None):
        super().__init__(message)
        self.context_state = context_state
        self.recovery_checkpoint = None


class ContextIntegrityError(ContentContextError):
    """Raised when ContentContext data is corrupted or invalid."""
    pass


class APIIntegrationError(ContentContextError):
    """Raised when external API calls fail."""
    pass


class GeminiAPIError(APIIntegrationError):
    """Specific error for Gemini API failures."""
    pass


class ImagenAPIError(APIIntegrationError):
    """Specific error for Imagen API failures."""
    pass


class ResourceConstraintError(ContentContextError):
    """Raised when system resources are insufficient."""
    pass


class MemoryConstraintError(ResourceConstraintError):
    """Raised when memory usage exceeds limits."""
    pass


class ProcessingTimeoutError(ResourceConstraintError):
    """Raised when processing takes too long."""
    pass


class VideoAnalysisError(ContentContextError):
    """Raised when video analysis fails."""
    pass


class AudioAnalysisError(ContentContextError):
    """Raised when audio analysis fails."""
    pass


def handle_error_with_context(error: Exception, context: Optional[Any] = None) -> ContentContextError:
    """
    Convert generic exceptions to ContentContextError with context preservation.
    
    Args:
        error: Original exception
        context: ContentContext or other state to preserve
        
    Returns:
        ContentContextError with preserved context
    """
    if isinstance(error, ContentContextError):
        return error
    
    # Map common exceptions to specific error types
    if isinstance(error, MemoryError):
        return MemoryConstraintError(f"Memory error: {str(error)}", context)
    elif isinstance(error, TimeoutError):
        return ProcessingTimeoutError(f"Timeout error: {str(error)}", context)
    elif "api" in str(error).lower():
        return APIIntegrationError(f"API error: {str(error)}", context)
    else:
        return ContentContextError(f"Unexpected error: {str(error)}", context)


def log_error_with_context(error: Exception, context: Optional[Any] = None, 
                          module_name: str = "unknown") -> None:
    """
    Log error with context information.
    
    Args:
        error: Exception to log
        context: ContentContext or other relevant context
        module_name: Name of the module where error occurred
    """
    error_info = {
        'module': module_name,
        'error_type': type(error).__name__,
        'error_message': str(error),
        'has_context': context is not None
    }
    
    if context and hasattr(context, 'project_id'):
        error_info['project_id'] = context.project_id
    
    logger.error(f"Error in {module_name}: {error_info}")


class ErrorRecoveryManager:
    """Manages error recovery strategies."""
    
    def __init__(self):
        self.recovery_strategies = {}
    
    def register_recovery_strategy(self, error_type: type, strategy_func):
        """Register a recovery strategy for a specific error type."""
        self.recovery_strategies[error_type] = strategy_func
    
    def attempt_recovery(self, error: Exception, context: Optional[Any] = None) -> Optional[Any]:
        """
        Attempt to recover from an error using registered strategies.
        
        Args:
            error: Exception to recover from
            context: Context to use for recovery
            
        Returns:
            Recovered context or None if recovery failed
        """
        error_type = type(error)
        
        if error_type in self.recovery_strategies:
            try:
                return self.recovery_strategies[error_type](error, context)
            except Exception as recovery_error:
                logger.error(f"Recovery strategy failed: {recovery_error}")
        
        return None


# Global error recovery manager
_global_recovery_manager = ErrorRecoveryManager()


def get_recovery_manager() -> ErrorRecoveryManager:
    """Get global error recovery manager."""
    return _global_recovery_manager