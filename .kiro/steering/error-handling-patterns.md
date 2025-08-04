# Error Handling Patterns for AI Video Editor

## Core Principle: ContentContext Preservation

All error handling must preserve ContentContext state to enable recovery and maintain processing continuity.

## Error Categories and Handling

### 1. ContentContext Integrity Errors
```python
class ContentContextError(Exception):
    """Base exception for ContentContext-related errors"""
    def __init__(self, message: str, context_state: Optional[ContentContext] = None):
        super().__init__(message)
        self.context_state = context_state
        self.recovery_checkpoint = None

class ContextIntegrityError(ContentContextError):
    """Raised when ContentContext data is corrupted or invalid"""
    pass

# Error handling pattern
@contextmanager
def preserve_context_on_error(context: ContentContext, checkpoint_name: str):
    try:
        # Save checkpoint before risky operation
        context_manager.save_checkpoint(context, checkpoint_name)
        yield context
    except Exception as e:
        # Restore from checkpoint on error
        restored_context = context_manager.load_checkpoint(context.project_id, checkpoint_name)
        logger.error(f"Error occurred, restored from checkpoint {checkpoint_name}: {str(e)}")
        raise ContentContextError(f"Processing failed: {str(e)}", restored_context)
```

### 2. API Integration Errors
```python
class APIIntegrationError(ContentContextError):
    """Raised when external API calls fail"""
    pass

class GeminiAPIError(APIIntegrationError):
    """Specific error for Gemini API failures"""
    pass

class ImagenAPIError(APIIntegrationError):
    """Specific error for Imagen API failures"""
    pass

# Retry pattern with exponential backoff
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(APIIntegrationError)
)
def call_api_with_retry(api_function, context: ContentContext, *args, **kwargs):
    try:
        return api_function(*args, **kwargs)
    except requests.RequestException as e:
        logger.warning(f"API call failed, retrying: {str(e)}")
        raise APIIntegrationError(f"API call failed: {str(e)}", context)
```

### 3. Resource Constraint Errors
```python
class ResourceConstraintError(ContentContextError):
    """Raised when system resources are insufficient"""
    pass

class MemoryConstraintError(ResourceConstraintError):
    """Raised when memory usage exceeds limits"""
    pass

class ProcessingTimeoutError(ResourceConstraintError):
    """Raised when processing takes too long"""
    pass

# Resource monitoring pattern
def monitor_resources(func):
    @wraps(func)
    def wrapper(context: ContentContext, *args, **kwargs):
        initial_memory = psutil.Process().memory_info().rss
        start_time = time.time()
        
        try:
            result = func(context, *args, **kwargs)
            
            # Check resource usage
            final_memory = psutil.Process().memory_info().rss
            processing_time = time.time() - start_time
            
            if final_memory - initial_memory > 8_000_000_000:  # 8GB
                logger.warning(f"High memory usage in {func.__name__}: {(final_memory - initial_memory) / 1_000_000_000:.2f}GB")
            
            if processing_time > 300:  # 5 minutes
                logger.warning(f"Long processing time in {func.__name__}: {processing_time:.2f}s")
            
            return result
            
        except MemoryError:
            raise MemoryConstraintError(f"Out of memory in {func.__name__}", context)
        except TimeoutError:
            raise ProcessingTimeoutError(f"Processing timeout in {func.__name__}", context)
    
    return wrapper
```

## Graceful Degradation Strategies

### 1. API Failure Fallbacks
```python
class GracefulDegradationManager:
    def __init__(self, context: ContentContext):
        self.context = context
        self.fallback_strategies = {
            'gemini_api': self._handle_gemini_failure,
            'imagen_api': self._handle_imagen_failure,
            'whisper_api': self._handle_whisper_failure
        }
    
    def _handle_gemini_failure(self, context: ContentContext) -> ContentContext:
        """Fallback for Gemini API failures"""
        logger.warning("Gemini API failed, using cached keyword research")
        
        # Use cached keywords or basic analysis
        if hasattr(context, 'cached_keywords'):
            context.trending_keywords = context.cached_keywords
        else:
            # Generate basic keywords from content concepts
            context.trending_keywords = self._generate_basic_keywords(context.key_concepts)
        
        context.processing_metrics.add_fallback_used('gemini_api')
        return context
    
    def _handle_imagen_failure(self, context: ContentContext) -> ContentContext:
        """Fallback for Imagen API failures"""
        logger.warning("Imagen API failed, using procedural generation")
        
        # Switch to procedural thumbnail generation
        context.thumbnail_generation_strategy = 'procedural_only'
        context.processing_metrics.add_fallback_used('imagen_api')
        return context
```

### 2. Quality vs Performance Trade-offs
```python
class QualityManager:
    def adjust_for_constraints(self, context: ContentContext, available_memory: int) -> ContentContext:
        """Adjust processing quality based on available resources"""
        
        if available_memory < 4_000_000_000:  # Less than 4GB
            logger.info("Low memory detected, reducing processing quality")
            context.processing_preferences.thumbnail_resolution = (1280, 720)  # Reduce from 1920x1080
            context.processing_preferences.batch_size = 1
            context.processing_preferences.enable_aggressive_caching = True
            
        elif available_memory < 8_000_000_000:  # Less than 8GB
            logger.info("Medium memory detected, using balanced processing")
            context.processing_preferences.batch_size = 2
            context.processing_preferences.parallel_processing = False
            
        return context
```

## Error Recovery Patterns

### 1. Checkpoint-Based Recovery
```python
class CheckpointManager:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
    
    def save_checkpoint(self, context: ContentContext, checkpoint_name: str) -> bool:
        """Save ContentContext state for recovery"""
        try:
            checkpoint_path = os.path.join(self.storage_path, f"{context.project_id}_{checkpoint_name}.json")
            with open(checkpoint_path, 'w') as f:
                json.dump(asdict(context), f, default=str)
            return True
        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint_name}: {str(e)}")
            return False
    
    def load_checkpoint(self, project_id: str, checkpoint_name: str) -> Optional[ContentContext]:
        """Load ContentContext from checkpoint"""
        try:
            checkpoint_path = os.path.join(self.storage_path, f"{project_id}_{checkpoint_name}.json")
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
            return ContentContext(**data)
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_name}: {str(e)}")
            return None
```

### 2. Partial Processing Recovery
```python
def recover_partial_processing(context: ContentContext, failed_module: str) -> ContentContext:
    """Recover from partial processing failures"""
    
    recovery_strategies = {
        'audio_analysis': lambda ctx: ctx,  # Can continue without audio analysis
        'video_analysis': lambda ctx: _use_basic_video_analysis(ctx),
        'keyword_research': lambda ctx: _use_cached_keywords(ctx),
        'thumbnail_generation': lambda ctx: _use_procedural_thumbnails(ctx),
        'metadata_generation': lambda ctx: _use_basic_metadata(ctx)
    }
    
    if failed_module in recovery_strategies:
        logger.info(f"Recovering from {failed_module} failure")
        context = recovery_strategies[failed_module](context)
        context.processing_metrics.add_recovery_action(failed_module)
    
    return context
```

## Error Reporting and User Communication

### 1. User-Friendly Error Messages
```python
class ErrorMessageFormatter:
    def format_for_user(self, error: ContentContextError) -> str:
        """Convert technical errors to user-friendly messages"""
        
        error_messages = {
            MemoryConstraintError: "Your video is quite large and requires more memory. Try closing other applications or processing a shorter video segment.",
            GeminiAPIError: "We're having trouble connecting to our content analysis service. Your video will be processed with basic analysis instead.",
            ImagenAPIError: "Thumbnail generation is temporarily using alternative methods. Your thumbnails will still be created but may look slightly different.",
            ProcessingTimeoutError: "Processing is taking longer than expected. This might be due to a very long video or high system load."
        }
        
        error_type = type(error)
        if error_type in error_messages:
            return error_messages[error_type]
        else:
            return f"An unexpected error occurred: {str(error)}. Please try again or contact support."
```

### 2. Recovery Suggestions
```python
class RecoverySuggestionEngine:
    def suggest_recovery_actions(self, error: ContentContextError) -> List[str]:
        """Suggest specific actions user can take to recover"""
        
        suggestions = {
            MemoryConstraintError: [
                "Close other applications to free up memory",
                "Process video in smaller segments",
                "Reduce video resolution before processing",
                "Restart the application to clear memory"
            ],
            APIIntegrationError: [
                "Check your internet connection",
                "Verify your API keys are correct",
                "Try again in a few minutes",
                "Use offline processing mode if available"
            ],
            ProcessingTimeoutError: [
                "Try processing a shorter video segment",
                "Close other applications to free up resources",
                "Check if your video file is corrupted",
                "Restart the application and try again"
            ]
        }
        
        return suggestions.get(type(error), ["Try again or contact support"])
```

This error handling framework ensures that the ContentContext system remains robust and recoverable, providing users with clear feedback and maintaining processing continuity even when individual components fail.