"""
GeminiClient - Wrapper for Google GenAI API with enhanced error handling and caching.

This module provides a robust client for the Gemini API with structured response
handling, retry logic, JSON schema validation, and intelligent caching integration.
"""

import json
import logging
import time
import asyncio
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from functools import wraps
import os

from google import genai
from google.genai import types, errors
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ...core.cache_manager import CacheManager
from ...core.content_context import ContentContext
from ...core.exceptions import (
    GeminiAPIError, 
    APIIntegrationError, 
    ContentContextError,
    AuthenticationError,
    ConfigurationError,
    NetworkError,
    ProcessingTimeoutError
)


logger = logging.getLogger(__name__)


@dataclass
class GeminiResponse:
    """Structured response from Gemini API with metadata."""
    content: str
    model_used: str
    timestamp: datetime
    processing_time: float
    token_count: Optional[int] = None
    finish_reason: Optional[str] = None
    safety_ratings: Optional[List[Dict[str, Any]]] = None
    raw_response: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for caching."""
        return {
            'content': self.content,
            'model_used': self.model_used,
            'timestamp': self.timestamp.isoformat(),
            'processing_time': self.processing_time,
            'token_count': self.token_count,
            'finish_reason': self.finish_reason,
            'safety_ratings': self.safety_ratings
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GeminiResponse':
        """Create response from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class GeminiConfig:
    """Configuration for Gemini API requests."""
    model: str = "gemini-2.0-flash-exp"
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_output_tokens: Optional[int] = None
    candidate_count: int = 1
    stop_sequences: Optional[List[str]] = None
    safety_settings: Optional[List[Dict[str, Any]]] = None
    
    def to_generate_config(self) -> types.GenerateContentConfig:
        """Convert to GenAI GenerateContentConfig."""
        config_dict = {
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'candidate_count': self.candidate_count
        }
        
        if self.max_output_tokens:
            config_dict['max_output_tokens'] = self.max_output_tokens
        
        if self.stop_sequences:
            config_dict['stop_sequences'] = self.stop_sequences
        
        if self.safety_settings:
            config_dict['safety_settings'] = self.safety_settings
        
        return types.GenerateContentConfig(**config_dict)


class GeminiClient:
    """
    Enhanced Gemini API client with caching, retry logic, and error handling.
    
    This client provides a robust interface to the Gemini API with:
    - Intelligent caching for API responses
    - Exponential backoff retry logic
    - JSON schema validation for structured responses
    - ContentContext integration for tracking API usage
    - Comprehensive error handling and recovery
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_manager: Optional[CacheManager] = None,
        default_config: Optional[GeminiConfig] = None,
        enable_caching: bool = True,
        cache_ttl: int = 3600,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        timeout: float = 120.0
    ):
        """
        Initialize GeminiClient.
        
        Args:
            api_key: Gemini API key (if None, uses GEMINI_API_KEY env var)
            cache_manager: CacheManager instance for response caching
            default_config: Default configuration for API requests
            enable_caching: Whether to enable response caching
            cache_ttl: Cache time-to-live in seconds
            max_retries: Maximum retry attempts for failed requests
            base_delay: Base delay for exponential backoff
            max_delay: Maximum delay between retries
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise AuthenticationError(
                "gemini",
                details={"message": "API key not provided and GEMINI_API_KEY environment variable not set"}
            )
        
        self.cache_manager = cache_manager
        self.enable_caching = enable_caching and cache_manager is not None
        self.cache_ttl = cache_ttl
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
        
        # Default configuration
        self.default_config = default_config or GeminiConfig()
        
        # Initialize the GenAI client
        try:
            # Configure HTTP options with timeout
            http_options = types.HttpOptions(
                timeout=self.timeout,
                retry_options=types.HttpRetryOptions(
                    attempts=self.max_retries,
                    initial_delay=self.base_delay,
                    max_delay=self.max_delay,
                    exp_base=2.0,
                    jitter=0.1,
                    http_status_codes=[429, 500, 502, 503, 504]
                )
            )
            
            self.client = genai.Client(
                api_key=self.api_key,
                http_options=http_options
            )
            
            logger.info("GeminiClient initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize GeminiClient: {str(e)}")
            raise ConfigurationError(
                "gemini_client_init",
                reason=str(e),
                details={"api_key_provided": bool(self.api_key)}
            )
        
        # API usage tracking
        self.usage_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'retry_attempts': 0
        }
    
    def _generate_cache_key(self, prompt: str, config: GeminiConfig, **kwargs) -> str:
        """Generate cache key for request."""
        if not self.cache_manager:
            return ""
        
        cache_data = {
            'prompt': prompt,
            'model': config.model,
            'temperature': config.temperature,
            'top_p': config.top_p,
            'top_k': config.top_k,
            'max_output_tokens': config.max_output_tokens,
            **kwargs
        }
        
        return self.cache_manager._generate_key("gemini_api", **cache_data)
    
    def _estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Estimate API cost based on token usage."""
        # Rough cost estimates (these should be updated with actual pricing)
        cost_per_1k_tokens = {
            'gemini-2.0-flash-exp': 0.0001,  # Example pricing
            'gemini-2.5-pro-latest': 0.0005,
            'gemini-2.0-flash': 0.0001
        }
        
        rate = cost_per_1k_tokens.get(model, 0.0001)
        total_tokens = input_tokens + output_tokens
        return (total_tokens / 1000) * rate
    
    def _validate_json_response(self, response_text: str, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate JSON response against schema.
        
        Args:
            response_text: Response text to validate
            schema: Optional JSON schema for validation
            
        Returns:
            Parsed JSON data
            
        Raises:
            GeminiAPIError: If JSON is invalid or doesn't match schema
        """
        try:
            # Try to parse JSON
            data = json.loads(response_text)
            
            # Basic schema validation if provided
            if schema:
                self._validate_schema(data, schema)
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {str(e)}")
            raise GeminiAPIError(
                "json_validation",
                reason=f"Invalid JSON response: {str(e)}",
                details={"response_text": response_text[:500]}
            )
        except Exception as e:
            logger.error(f"Schema validation failed: {str(e)}")
            raise GeminiAPIError(
                "schema_validation",
                reason=f"Schema validation failed: {str(e)}",
                details={"response_text": response_text[:500], "schema": schema}
            )
    
    def _validate_schema(self, data: Dict[str, Any], schema: Dict[str, Any]):
        """Basic schema validation."""
        required_fields = schema.get('required', [])
        
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Required field '{field}' missing from response")
        
        # Type validation for specified fields
        field_types = schema.get('properties', {})
        for field, type_spec in field_types.items():
            if field in data:
                expected_type = type_spec.get('type')
                if expected_type == 'string' and not isinstance(data[field], str):
                    raise ValueError(f"Field '{field}' should be string, got {type(data[field])}")
                elif expected_type == 'array' and not isinstance(data[field], list):
                    raise ValueError(f"Field '{field}' should be array, got {type(data[field])}")
                elif expected_type == 'object' and not isinstance(data[field], dict):
                    raise ValueError(f"Field '{field}' should be object, got {type(data[field])}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type((NetworkError, ProcessingTimeoutError, errors.APIError))
    )
    def _make_api_request(
        self,
        prompt: str,
        config: GeminiConfig,
        system_instruction: Optional[str] = None,
        tools: Optional[List[Any]] = None
    ) -> Any:
        """
        Make API request with retry logic.
        
        Args:
            prompt: Input prompt
            config: Request configuration
            system_instruction: Optional system instruction
            tools: Optional tools for function calling
            
        Returns:
            Raw API response
            
        Raises:
            GeminiAPIError: If API request fails after retries
        """
        try:
            # Prepare content
            contents = [types.Content(
                role='user',
                parts=[types.Part.from_text(text=prompt)]
            )]
            
            # Prepare generation config
            gen_config = config.to_generate_config()
            if system_instruction:
                gen_config.system_instruction = system_instruction
            if tools:
                gen_config.tools = tools
            
            # Make API request
            start_time = time.time()
            response = self.client.models.generate_content(
                model=config.model,
                contents=contents,
                config=gen_config
            )
            processing_time = time.time() - start_time
            
            # Update usage stats
            self.usage_stats['total_requests'] += 1
            self.usage_stats['successful_requests'] += 1
            
            logger.debug(f"Gemini API request successful in {processing_time:.2f}s")
            return response, processing_time
            
        except errors.APIError as e:
            self.usage_stats['failed_requests'] += 1
            self.usage_stats['retry_attempts'] += 1
            
            logger.error(f"Gemini API error: {e.code} - {e.message}")
            
            # Map specific error codes to appropriate exceptions
            if e.code == 401:
                raise AuthenticationError("gemini", details={"message": e.message})
            elif e.code == 429:
                raise NetworkError("gemini", reason="Rate limit exceeded")
            elif e.code >= 500:
                raise NetworkError("gemini", reason=f"Server error: {e.message}")
            else:
                raise GeminiAPIError(
                    "api_request",
                    reason=f"API error {e.code}: {e.message}",
                    details={"status_code": e.code, "message": e.message}
                )
        
        except Exception as e:
            # Handle mock errors and other exceptions
            if hasattr(e, 'code') and hasattr(e, 'message'):
                self.usage_stats['failed_requests'] += 1
                self.usage_stats['retry_attempts'] += 1
                
                logger.error(f"API error: {e.code} - {e.message}")
                
                # Map specific error codes to appropriate exceptions
                if e.code == 401:
                    raise AuthenticationError("gemini", details={"message": e.message})
                elif e.code == 429:
                    raise NetworkError("gemini", reason="Rate limit exceeded")
                elif e.code >= 500:
                    raise NetworkError("gemini", reason=f"Server error: {e.message}")
                else:
                    raise GeminiAPIError(
                        "api_request",
                        reason=f"API error {e.code}: {e.message}",
                        details={"status_code": e.code, "message": e.message}
                    )
            else:
                self.usage_stats['failed_requests'] += 1
                logger.error(f"Unexpected error in Gemini API request: {str(e)}")
                raise GeminiAPIError(
                    "api_request",
                    reason=f"Unexpected error: {str(e)}",
                    details={"error_type": type(e).__name__}
                )
    
    def generate_content(
        self,
        prompt: str,
        config: Optional[GeminiConfig] = None,
        system_instruction: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        context: Optional[ContentContext] = None,
        enable_caching: Optional[bool] = None,
        json_schema: Optional[Dict[str, Any]] = None
    ) -> GeminiResponse:
        """
        Generate content using Gemini API.
        
        Args:
            prompt: Input prompt for content generation
            config: Request configuration (uses default if None)
            system_instruction: Optional system instruction
            tools: Optional tools for function calling
            context: ContentContext for tracking usage
            enable_caching: Override default caching behavior
            json_schema: Optional JSON schema for response validation
            
        Returns:
            GeminiResponse with generated content and metadata
            
        Raises:
            GeminiAPIError: If content generation fails
        """
        config = config or self.default_config
        use_caching = enable_caching if enable_caching is not None else self.enable_caching
        
        # Generate cache key
        cache_key = ""
        if use_caching:
            cache_key = self._generate_cache_key(
                prompt, config, 
                system_instruction=system_instruction,
                tools=bool(tools),
                json_schema=bool(json_schema)
            )
            
            # Try to get from cache
            cached_response = self.cache_manager.get(cache_key)
            if cached_response:
                self.usage_stats['cache_hits'] += 1
                logger.debug("Retrieved response from cache")
                
                # Update context if provided
                if context:
                    context.processing_metrics.add_api_call('gemini_cached', 1)
                
                return GeminiResponse.from_dict(cached_response)
        
        self.usage_stats['cache_misses'] += 1
        
        try:
            # Make API request
            raw_response, processing_time = self._make_api_request(
                prompt, config, system_instruction, tools
            )
            
            # Extract response content
            if not raw_response.candidates:
                raise GeminiAPIError(
                    "empty_response",
                    reason="No candidates in response",
                    details={"raw_response": str(raw_response)}
                )
            
            candidate = raw_response.candidates[0]
            content = candidate.content.parts[0].text if candidate.content.parts else ""
            
            # Validate JSON if schema provided
            if json_schema and content.strip():
                try:
                    self._validate_json_response(content, json_schema)
                except GeminiAPIError:
                    # If JSON validation fails, try to extract JSON from response
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        content = json_match.group()
                        self._validate_json_response(content, json_schema)
                    else:
                        raise
            
            # Create structured response
            response = GeminiResponse(
                content=content,
                model_used=config.model,
                timestamp=datetime.now(),
                processing_time=processing_time,
                finish_reason=getattr(candidate, 'finish_reason', None),
                safety_ratings=getattr(candidate, 'safety_ratings', None),
                raw_response=raw_response
            )
            
            # Estimate token usage and cost
            input_tokens = len(prompt.split()) * 1.3  # Rough estimate
            output_tokens = len(content.split()) * 1.3
            estimated_cost = self._estimate_cost(int(input_tokens), int(output_tokens), config.model)
            
            response.token_count = int(input_tokens + output_tokens)
            self.usage_stats['total_tokens'] += response.token_count
            self.usage_stats['total_cost'] += estimated_cost
            
            # Cache response if enabled
            if use_caching and cache_key:
                self.cache_manager.cache_api_response(
                    service="gemini",
                    endpoint="generate_content",
                    params={
                        "prompt": prompt[:100],  # Truncated for cache key
                        "model": config.model,
                        "temperature": config.temperature
                    },
                    response=response.to_dict(),
                    cost=estimated_cost
                )
            
            # Update context if provided
            if context:
                context.processing_metrics.add_api_call('gemini', 1)
                context.cost_tracking.add_cost('gemini', estimated_cost)
            
            logger.info(f"Generated content successfully: {len(content)} chars, ~{response.token_count} tokens")
            return response
            
        except Exception as e:
            if isinstance(e, (GeminiAPIError, AuthenticationError, NetworkError)):
                raise
            
            logger.error(f"Unexpected error in content generation: {str(e)}")
            raise GeminiAPIError(
                "content_generation",
                reason=f"Unexpected error: {str(e)}",
                details={"error_type": type(e).__name__, "prompt_length": len(prompt)}
            )
    
    def generate_structured_response(
        self,
        prompt: str,
        response_schema: Dict[str, Any],
        config: Optional[GeminiConfig] = None,
        system_instruction: Optional[str] = None,
        context: Optional[ContentContext] = None,
        max_attempts: int = 3
    ) -> Dict[str, Any]:
        """
        Generate structured JSON response with schema validation.
        
        Args:
            prompt: Input prompt
            response_schema: JSON schema for response validation
            config: Request configuration
            system_instruction: Optional system instruction
            context: ContentContext for tracking
            max_attempts: Maximum attempts to get valid JSON
            
        Returns:
            Parsed and validated JSON response
            
        Raises:
            GeminiAPIError: If unable to generate valid structured response
        """
        # Enhance prompt for JSON output
        json_prompt = f"""
{prompt}

Please respond with a valid JSON object that matches this schema:
{json.dumps(response_schema, indent=2)}

Ensure your response is valid JSON and includes all required fields.
"""
        
        if system_instruction:
            system_instruction += "\n\nAlways respond with valid JSON format."
        else:
            system_instruction = "Always respond with valid JSON format."
        
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                response = self.generate_content(
                    prompt=json_prompt,
                    config=config,
                    system_instruction=system_instruction,
                    context=context,
                    json_schema=response_schema
                )
                
                # Parse and validate JSON
                return self._validate_json_response(response.content, response_schema)
                
            except GeminiAPIError as e:
                last_error = e
                if attempt < max_attempts - 1:
                    logger.warning(f"Attempt {attempt + 1} failed for structured response: {e}")
                    time.sleep(1.0 * (attempt + 1))  # Progressive delay
                else:
                    logger.error(f"All {max_attempts} attempts failed for structured response")
        
        # If all attempts failed, raise the last error
        raise last_error or GeminiAPIError(
            "structured_response",
            reason="Failed to generate valid structured response after all attempts"
        )
    
    async def generate_content_async(
        self,
        prompt: str,
        config: Optional[GeminiConfig] = None,
        system_instruction: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        context: Optional[ContentContext] = None,
        enable_caching: Optional[bool] = None,
        json_schema: Optional[Dict[str, Any]] = None
    ) -> GeminiResponse:
        """
        Asynchronous version of generate_content.
        
        Args:
            Same as generate_content
            
        Returns:
            GeminiResponse with generated content and metadata
        """
        # Run synchronous method in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.generate_content,
            prompt,
            config,
            system_instruction,
            tools,
            context,
            enable_caching,
            json_schema
        )
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get API usage statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        stats = self.usage_stats.copy()
        
        # Calculate derived metrics
        if stats['total_requests'] > 0:
            stats['success_rate'] = stats['successful_requests'] / stats['total_requests']
            stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
        else:
            stats['success_rate'] = 0.0
            stats['cache_hit_rate'] = 0.0
        
        stats['average_tokens_per_request'] = (
            stats['total_tokens'] / stats['successful_requests'] 
            if stats['successful_requests'] > 0 else 0
        )
        
        return stats
    
    def reset_usage_stats(self):
        """Reset usage statistics."""
        self.usage_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'retry_attempts': 0
        }
        logger.info("Usage statistics reset")
    
    def update_api_usage_patterns(self, context: ContentContext):
        """
        Update memory with API usage patterns for optimization.
        
        Args:
            context: ContentContext with usage data
        """
        if not context:
            return
        
        usage_pattern = {
            'timestamp': datetime.now().isoformat(),
            'project_id': context.project_id,
            'content_type': context.content_type.value,
            'api_calls': context.processing_metrics.api_calls_made.get('gemini', 0),
            'processing_time': context.processing_metrics.module_processing_times.get('gemini_client', 0.0),
            'cost': context.cost_tracking.gemini_api_cost,
            'cache_hit_rate': self.get_usage_stats()['cache_hit_rate']
        }
        
        # Store pattern for future optimization
        logger.debug(f"API usage pattern: {usage_pattern}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type:
            logger.error(f"GeminiClient context exited with error: {exc_val}")
        else:
            logger.debug("GeminiClient context exited successfully")


# Utility functions for common use cases
def create_financial_analysis_prompt(transcript: str, concepts: List[str]) -> str:
    """Create prompt for financial content analysis."""
    return f"""
Analyze this financial education transcript and provide structured insights:

Transcript: {transcript}

Key concepts mentioned: {', '.join(concepts)}

Please provide analysis in the following areas:
1. Educational value and complexity level
2. Key financial concepts explained
3. Opportunities for visual aids (charts, graphs, animations)
4. Emotional engagement points
5. SEO-relevant keywords and themes

Focus on actionable insights for video editing and content optimization.
"""


def create_keyword_research_prompt(content_summary: str, target_audience: str) -> str:
    """Create prompt for keyword research and SEO optimization."""
    return f"""
Perform comprehensive keyword research for this content:

Content Summary: {content_summary}
Target Audience: {target_audience}

Please provide:
1. Primary keywords (3-5 high-volume, relevant terms)
2. Long-tail keywords (5-10 specific phrases)
3. Trending hashtags related to the topic
4. Seasonal keywords if applicable
5. Competitor analysis keywords
6. YouTube title suggestions (3-5 options)
7. Video description template with keyword integration
8. Recommended tags (10-15 tags)

Focus on current trends and search volume potential.
"""


def create_thumbnail_concept_prompt(visual_highlights: List[str], emotional_peaks: List[str]) -> str:
    """Create prompt for thumbnail concept generation."""
    return f"""
Generate thumbnail concepts based on this video analysis:

Visual Highlights: {', '.join(visual_highlights)}
Emotional Peaks: {', '.join(emotional_peaks)}

Please provide:
1. 3-5 thumbnail concepts with descriptions
2. Text overlay suggestions for each concept
3. Color scheme recommendations
4. Visual hierarchy guidelines
5. A/B testing variations
6. Hook text that aligns with video titles

Focus on high click-through rate potential and brand consistency.
"""