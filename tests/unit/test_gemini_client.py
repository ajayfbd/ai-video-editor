"""
Unit tests for GeminiClient with comprehensive mocking.

This test suite covers all aspects of the GeminiClient including
API interactions, caching, error handling, and ContentContext integration.
"""

import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any

from ai_video_editor.modules.intelligence.gemini_client import (
    GeminiClient,
    GeminiConfig,
    GeminiResponse,
    create_financial_analysis_prompt,
    create_keyword_research_prompt,
    create_thumbnail_concept_prompt
)
from ai_video_editor.core.cache_manager import CacheManager
from ai_video_editor.core.content_context import ContentContext, ContentType, UserPreferences
from ai_video_editor.core.exceptions import (
    GeminiAPIError,
    AuthenticationError,
    ConfigurationError,
    NetworkError
)


class TestGeminiConfig:
    """Test GeminiConfig dataclass and conversion methods."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = GeminiConfig()
        
        assert config.model == "gemini-2.0-flash-exp"
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 40
        assert config.max_output_tokens is None
        assert config.candidate_count == 1
        assert config.stop_sequences is None
        assert config.safety_settings is None
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = GeminiConfig(
            model="gemini-2.5-pro-latest",
            temperature=0.3,
            max_output_tokens=1000,
            stop_sequences=["END"]
        )
        
        assert config.model == "gemini-2.5-pro-latest"
        assert config.temperature == 0.3
        assert config.max_output_tokens == 1000
        assert config.stop_sequences == ["END"]
    
    @patch('ai_video_editor.modules.intelligence.gemini_client.types')
    def test_to_generate_config(self, mock_types):
        """Test conversion to GenerateContentConfig."""
        config = GeminiConfig(
            temperature=0.5,
            max_output_tokens=500,
            stop_sequences=["STOP"]
        )
        
        mock_generate_config = Mock()
        mock_types.GenerateContentConfig.return_value = mock_generate_config
        
        result = config.to_generate_config()
        
        mock_types.GenerateContentConfig.assert_called_once_with(
            temperature=0.5,
            top_p=0.9,
            top_k=40,
            candidate_count=1,
            max_output_tokens=500,
            stop_sequences=["STOP"]
        )
        assert result == mock_generate_config


class TestGeminiResponse:
    """Test GeminiResponse dataclass and serialization."""
    
    def test_response_creation(self):
        """Test creating GeminiResponse."""
        timestamp = datetime.now()
        response = GeminiResponse(
            content="Test response",
            model_used="gemini-2.0-flash-exp",
            timestamp=timestamp,
            processing_time=1.5,
            token_count=100,
            finish_reason="STOP"
        )
        
        assert response.content == "Test response"
        assert response.model_used == "gemini-2.0-flash-exp"
        assert response.timestamp == timestamp
        assert response.processing_time == 1.5
        assert response.token_count == 100
        assert response.finish_reason == "STOP"
    
    def test_to_dict(self):
        """Test converting response to dictionary."""
        timestamp = datetime.now()
        response = GeminiResponse(
            content="Test response",
            model_used="gemini-2.0-flash-exp",
            timestamp=timestamp,
            processing_time=1.5
        )
        
        result = response.to_dict()
        
        assert result['content'] == "Test response"
        assert result['model_used'] == "gemini-2.0-flash-exp"
        assert result['timestamp'] == timestamp.isoformat()
        assert result['processing_time'] == 1.5
    
    def test_from_dict(self):
        """Test creating response from dictionary."""
        timestamp = datetime.now()
        data = {
            'content': "Test response",
            'model_used': "gemini-2.0-flash-exp",
            'timestamp': timestamp.isoformat(),
            'processing_time': 1.5,
            'token_count': 100
        }
        
        response = GeminiResponse.from_dict(data)
        
        assert response.content == "Test response"
        assert response.model_used == "gemini-2.0-flash-exp"
        assert response.timestamp == timestamp
        assert response.processing_time == 1.5
        assert response.token_count == 100


class TestGeminiClient:
    """Test GeminiClient functionality with comprehensive mocking."""
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Mock CacheManager for testing."""
        cache_manager = Mock(spec=CacheManager)
        cache_manager._generate_key.return_value = "test_cache_key"
        cache_manager.get.return_value = None
        cache_manager.cache_api_response.return_value = None
        return cache_manager
    
    @pytest.fixture
    def mock_content_context(self):
        """Mock ContentContext for testing."""
        context = Mock(spec=ContentContext)
        context.project_id = "test_project"
        context.content_type = ContentType.EDUCATIONAL
        context.processing_metrics = Mock()
        context.cost_tracking = Mock()
        return context
    
    @pytest.fixture
    def mock_genai_client(self):
        """Mock Google GenAI client."""
        with patch('ai_video_editor.modules.intelligence.gemini_client.genai') as mock_genai:
            mock_client = Mock()
            mock_genai.Client.return_value = mock_client
            
            # Mock response structure
            mock_response = Mock()
            mock_candidate = Mock()
            mock_content = Mock()
            mock_part = Mock()
            mock_part.text = "Generated response text"
            mock_content.parts = [mock_part]
            mock_candidate.content = mock_content
            mock_candidate.finish_reason = "STOP"
            mock_response.candidates = [mock_candidate]
            
            mock_client.models.generate_content.return_value = mock_response
            
            yield mock_genai, mock_client, mock_response
    
    def test_init_with_api_key(self, mock_genai_client):
        """Test initialization with API key."""
        mock_genai, mock_client, _ = mock_genai_client
        
        client = GeminiClient(api_key="test_api_key")
        
        assert client.api_key == "test_api_key"
        assert client.enable_caching is False  # No cache manager provided
        mock_genai.Client.assert_called_once()
    
    def test_init_with_env_var(self, mock_genai_client):
        """Test initialization with environment variable."""
        mock_genai, mock_client, _ = mock_genai_client
        
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'env_api_key'}):
            client = GeminiClient()
            assert client.api_key == "env_api_key"
    
    def test_init_without_api_key(self, mock_genai_client):
        """Test initialization failure without API key."""
        mock_genai, mock_client, _ = mock_genai_client
        
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(AuthenticationError):
                GeminiClient()
    
    def test_init_with_cache_manager(self, mock_genai_client, mock_cache_manager):
        """Test initialization with cache manager."""
        mock_genai, mock_client, _ = mock_genai_client
        
        client = GeminiClient(
            api_key="test_key",
            cache_manager=mock_cache_manager
        )
        
        assert client.cache_manager == mock_cache_manager
        assert client.enable_caching is True
    
    @patch('ai_video_editor.modules.intelligence.gemini_client.time.time')
    def test_generate_content_success(self, mock_time, mock_genai_client, mock_cache_manager):
        """Test successful content generation."""
        mock_genai, mock_client, mock_response = mock_genai_client
        mock_time.side_effect = [1000.0, 1001.5]  # Start and end times
        
        client = GeminiClient(
            api_key="test_key",
            cache_manager=mock_cache_manager
        )
        
        result = client.generate_content("Test prompt")
        
        assert isinstance(result, GeminiResponse)
        assert result.content == "Generated response text"
        assert result.model_used == "gemini-2.0-flash-exp"
        assert result.processing_time == 1.5
        assert result.finish_reason == "STOP"
        
        # Verify API call was made
        mock_client.models.generate_content.assert_called_once()
        
        # Verify caching was attempted
        mock_cache_manager.cache_api_response.assert_called_once()
    
    def test_generate_content_with_cache_hit(self, mock_genai_client, mock_cache_manager):
        """Test content generation with cache hit."""
        mock_genai, mock_client, _ = mock_genai_client
        
        # Mock cache hit
        cached_response = {
            'content': "Cached response",
            'model_used': "gemini-2.0-flash-exp",
            'timestamp': datetime.now().isoformat(),
            'processing_time': 0.5
        }
        mock_cache_manager.get.return_value = cached_response
        
        client = GeminiClient(
            api_key="test_key",
            cache_manager=mock_cache_manager
        )
        
        result = client.generate_content("Test prompt")
        
        assert result.content == "Cached response"
        assert client.usage_stats['cache_hits'] == 1
        
        # Verify API was not called
        mock_client.models.generate_content.assert_not_called()
    
    def test_generate_content_with_context(self, mock_genai_client, mock_content_context):
        """Test content generation with ContentContext tracking."""
        mock_genai, mock_client, _ = mock_genai_client
        
        client = GeminiClient(api_key="test_key")
        
        result = client.generate_content("Test prompt", context=mock_content_context)
        
        assert isinstance(result, GeminiResponse)
        
        # Verify context was updated
        mock_content_context.processing_metrics.add_api_call.assert_called_with('gemini', 1)
        mock_content_context.cost_tracking.add_cost.assert_called()
    
    def test_generate_content_api_error(self, mock_genai_client):
        """Test handling of API errors."""
        mock_genai, mock_client, _ = mock_genai_client
        
        # Create a mock exception that looks like an API error
        class MockAPIError(Exception):
            def __init__(self, message):
                super().__init__(message)
                self.code = 429
                self.message = "Rate limit exceeded"
        
        mock_client.models.generate_content.side_effect = MockAPIError("Test error")
        
        client = GeminiClient(api_key="test_key")
        
        # After retries, it should raise a GeminiAPIError
        with pytest.raises(GeminiAPIError) as exc_info:
            client.generate_content("Test prompt")
        
        # Should have 3 failed requests due to retries
        assert client.usage_stats['failed_requests'] == 3
    
    def test_generate_content_authentication_error(self, mock_genai_client):
        """Test handling of authentication errors."""
        mock_genai, mock_client, _ = mock_genai_client
        
        # Create a mock exception that looks like an authentication error
        class MockAPIError(Exception):
            def __init__(self, message):
                super().__init__(message)
                self.code = 401
                self.message = "Invalid API key"
        
        mock_client.models.generate_content.side_effect = MockAPIError("Unauthorized")
        
        client = GeminiClient(api_key="test_key")
        
        with pytest.raises(AuthenticationError):
            client.generate_content("Test prompt")
    
    def test_generate_structured_response_success(self, mock_genai_client):
        """Test successful structured response generation."""
        mock_genai, mock_client, mock_response = mock_genai_client
        
        # Mock JSON response
        json_response = {"key": "value", "number": 42}
        mock_response.candidates[0].content.parts[0].text = json.dumps(json_response)
        
        client = GeminiClient(api_key="test_key")
        
        schema = {
            "type": "object",
            "required": ["key", "number"],
            "properties": {
                "key": {"type": "string"},
                "number": {"type": "number"}
            }
        }
        
        result = client.generate_structured_response(
            "Generate JSON",
            response_schema=schema
        )
        
        assert result == json_response
    
    def test_generate_structured_response_invalid_json(self, mock_genai_client):
        """Test handling of invalid JSON in structured response."""
        mock_genai, mock_client, mock_response = mock_genai_client
        
        # Mock invalid JSON response
        mock_response.candidates[0].content.parts[0].text = "Invalid JSON response"
        
        client = GeminiClient(api_key="test_key")
        
        schema = {"type": "object", "required": ["key"]}
        
        with pytest.raises(GeminiAPIError) as exc_info:
            client.generate_structured_response(
                "Generate JSON",
                response_schema=schema,
                max_attempts=1
            )
        
        assert "json_validation" in str(exc_info.value)
    
    def test_validate_json_response_success(self, mock_genai_client):
        """Test successful JSON validation."""
        mock_genai, mock_client, _ = mock_genai_client
        
        client = GeminiClient(api_key="test_key")
        
        json_data = {"name": "test", "value": 123}
        json_string = json.dumps(json_data)
        
        schema = {
            "required": ["name", "value"],
            "properties": {
                "name": {"type": "string"},
                "value": {"type": "number"}
            }
        }
        
        result = client._validate_json_response(json_string, schema)
        assert result == json_data
    
    def test_validate_json_response_invalid_json(self, mock_genai_client):
        """Test JSON validation with invalid JSON."""
        mock_genai, mock_client, _ = mock_genai_client
        
        client = GeminiClient(api_key="test_key")
        
        with pytest.raises(GeminiAPIError) as exc_info:
            client._validate_json_response("Invalid JSON", {})
        
        assert exc_info.value.operation == "json_validation"
    
    def test_validate_json_response_schema_mismatch(self, mock_genai_client):
        """Test JSON validation with schema mismatch."""
        mock_genai, mock_client, _ = mock_genai_client
        
        client = GeminiClient(api_key="test_key")
        
        json_data = {"name": "test"}
        json_string = json.dumps(json_data)
        
        schema = {
            "required": ["name", "required_field"],
            "properties": {
                "name": {"type": "string"},
                "required_field": {"type": "string"}
            }
        }
        
        with pytest.raises(GeminiAPIError) as exc_info:
            client._validate_json_response(json_string, schema)
        
        assert exc_info.value.operation == "schema_validation"
    
    def test_estimate_cost(self, mock_genai_client):
        """Test cost estimation."""
        mock_genai, mock_client, _ = mock_genai_client
        
        client = GeminiClient(api_key="test_key")
        
        cost = client._estimate_cost(1000, 500, "gemini-2.0-flash-exp")
        
        # Should be (1500 tokens / 1000) * 0.0001 = 0.00015
        assert cost == 0.00015
    
    def test_get_usage_stats(self, mock_genai_client):
        """Test usage statistics retrieval."""
        mock_genai, mock_client, _ = mock_genai_client
        
        client = GeminiClient(api_key="test_key")
        
        # Simulate some usage
        client.usage_stats['total_requests'] = 10
        client.usage_stats['successful_requests'] = 8
        client.usage_stats['cache_hits'] = 3
        client.usage_stats['cache_misses'] = 7
        client.usage_stats['total_tokens'] = 1000
        
        stats = client.get_usage_stats()
        
        assert stats['total_requests'] == 10
        assert stats['successful_requests'] == 8
        assert stats['success_rate'] == 0.8
        assert stats['cache_hit_rate'] == 0.3
        assert stats['average_tokens_per_request'] == 125.0
    
    def test_reset_usage_stats(self, mock_genai_client):
        """Test usage statistics reset."""
        mock_genai, mock_client, _ = mock_genai_client
        
        client = GeminiClient(api_key="test_key")
        
        # Set some usage stats
        client.usage_stats['total_requests'] = 10
        client.usage_stats['total_cost'] = 5.0
        
        client.reset_usage_stats()
        
        assert client.usage_stats['total_requests'] == 0
        assert client.usage_stats['total_cost'] == 0.0
    
    @pytest.mark.asyncio
    async def test_generate_content_async(self, mock_genai_client):
        """Test asynchronous content generation."""
        mock_genai, mock_client, _ = mock_genai_client
        
        client = GeminiClient(api_key="test_key")
        
        result = await client.generate_content_async("Test prompt")
        
        assert isinstance(result, GeminiResponse)
        assert result.content == "Generated response text"
    
    def test_context_manager(self, mock_genai_client):
        """Test context manager functionality."""
        mock_genai, mock_client, _ = mock_genai_client
        
        with GeminiClient(api_key="test_key") as client:
            assert isinstance(client, GeminiClient)
            result = client.generate_content("Test prompt")
            assert isinstance(result, GeminiResponse)
    
    def test_update_api_usage_patterns(self, mock_genai_client, mock_content_context):
        """Test API usage pattern tracking."""
        mock_genai, mock_client, _ = mock_genai_client
        
        client = GeminiClient(api_key="test_key")
        
        # Mock context data
        mock_content_context.processing_metrics.api_calls_made = {'gemini': 5}
        mock_content_context.processing_metrics.module_processing_times = {'gemini_client': 2.5}
        mock_content_context.cost_tracking.gemini_api_cost = 0.05
        
        # Should not raise any exceptions
        client.update_api_usage_patterns(mock_content_context)


class TestUtilityFunctions:
    """Test utility functions for prompt creation."""
    
    def test_create_financial_analysis_prompt(self):
        """Test financial analysis prompt creation."""
        transcript = "Today we'll discuss compound interest and investment strategies."
        concepts = ["compound interest", "investment", "portfolio"]
        
        prompt = create_financial_analysis_prompt(transcript, concepts)
        
        assert "compound interest" in prompt
        assert "investment strategies" in prompt
        assert "Educational value" in prompt
        assert "SEO-relevant keywords" in prompt
    
    def test_create_keyword_research_prompt(self):
        """Test keyword research prompt creation."""
        content_summary = "Video about personal finance basics"
        target_audience = "young adults"
        
        prompt = create_keyword_research_prompt(content_summary, target_audience)
        
        assert "personal finance basics" in prompt
        assert "young adults" in prompt
        assert "Primary keywords" in prompt
        assert "YouTube title suggestions" in prompt
    
    def test_create_thumbnail_concept_prompt(self):
        """Test thumbnail concept prompt creation."""
        visual_highlights = ["excited expression", "chart visualization"]
        emotional_peaks = ["excitement", "curiosity"]
        
        prompt = create_thumbnail_concept_prompt(visual_highlights, emotional_peaks)
        
        assert "excited expression" in prompt
        assert "excitement" in prompt
        assert "thumbnail concepts" in prompt
        assert "click-through rate" in prompt


class TestErrorHandling:
    """Test comprehensive error handling scenarios."""
    
    def test_network_error_handling(self, mock_genai_client):
        """Test network error handling and retry logic."""
        mock_genai, mock_client, _ = mock_genai_client
        
        # Mock network error
        mock_client.models.generate_content.side_effect = Exception("Connection failed")
        
        client = GeminiClient(api_key="test_key")
        
        with pytest.raises(GeminiAPIError) as exc_info:
            client.generate_content("Test prompt")
        
        assert "Connection failed" in str(exc_info.value)
    
    def test_empty_response_handling(self, mock_genai_client):
        """Test handling of empty API responses."""
        mock_genai, mock_client, mock_response = mock_genai_client
        
        # Mock empty response
        mock_response.candidates = []
        
        client = GeminiClient(api_key="test_key")
        
        with pytest.raises(GeminiAPIError) as exc_info:
            client.generate_content("Test prompt")
        
        assert exc_info.value.operation == "empty_response"
    
    def test_configuration_error_handling(self):
        """Test configuration error handling."""
        with patch('ai_video_editor.modules.intelligence.gemini_client.genai.Client') as mock_client:
            mock_client.side_effect = Exception("Configuration error")
            
            with pytest.raises(ConfigurationError):
                GeminiClient(api_key="test_key")


class TestCacheIntegration:
    """Test cache integration scenarios."""
    
    def test_cache_key_generation(self, mock_genai_client, mock_cache_manager):
        """Test cache key generation."""
        mock_genai, mock_client, _ = mock_genai_client
        
        client = GeminiClient(
            api_key="test_key",
            cache_manager=mock_cache_manager
        )
        
        config = GeminiConfig(temperature=0.5)
        key = client._generate_cache_key("test prompt", config)
        
        mock_cache_manager._generate_key.assert_called_once()
        assert key == "test_cache_key"
    
    def test_cache_disabled(self, mock_genai_client):
        """Test behavior when caching is disabled."""
        mock_genai, mock_client, _ = mock_genai_client
        
        client = GeminiClient(api_key="test_key", enable_caching=False)
        
        result = client.generate_content("Test prompt")
        
        assert isinstance(result, GeminiResponse)
        assert client.usage_stats['cache_hits'] == 0
        assert client.usage_stats['cache_misses'] == 1


if __name__ == "__main__":
    pytest.main([__file__])