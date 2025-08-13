"""Logging configuration for AI Video Editor."""

import logging
import logging.config
import os
from pathlib import Path
from typing import Dict, Any

# Create logs directory if it doesn't exist
LOGS_DIR = Path("out") / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Logging configuration
LOGGING_CONFIG: Dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple": {
            "format": "%(levelname)s - %(message)s",
        },
        "json": {
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": str(LOGS_DIR / "ai_video_editor.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8",
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "detailed",
            "filename": str(LOGS_DIR / "errors.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8",
        },
    },
    "loggers": {
        "ai_video_editor": {
            "level": "DEBUG",
            "handlers": ["console", "file", "error_file"],
            "propagate": False,
        },
        "ai_video_editor.modules": {
            "level": "DEBUG",
            "handlers": ["file", "error_file"],
            "propagate": False,
        },
    },
    "root": {
        "level": "WARNING",
        "handlers": ["console"],
    },
}


def setup_logging(log_level: str = None) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: Override the default log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Validate log level
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    
    # Override log level if specified and valid
    if log_level:
        log_level = log_level.upper()
        if log_level in valid_levels:
            LOGGING_CONFIG["loggers"]["ai_video_editor"]["level"] = log_level
            LOGGING_CONFIG["handlers"]["console"]["level"] = log_level
        else:
            # Use INFO as fallback for invalid levels
            LOGGING_CONFIG["loggers"]["ai_video_editor"]["level"] = "INFO"
            LOGGING_CONFIG["handlers"]["console"]["level"] = "INFO"
    
    # Apply environment-specific overrides
    if os.getenv("AI_VIDEO_EDITOR_DEBUG", "").lower() == "true":
        LOGGING_CONFIG["handlers"]["console"]["level"] = "DEBUG"
        LOGGING_CONFIG["handlers"]["console"]["formatter"] = "detailed"
    
    # Configure logging
    logging.config.dictConfig(LOGGING_CONFIG)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the specified module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Performance logging decorator
def log_performance(func):
    """Decorator to log function execution time."""
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {str(e)}")
            raise
    
    return wrapper