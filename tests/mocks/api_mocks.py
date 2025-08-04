# tests/mocks/api_mocks.py
"""
Comprehensive API mocking utilities for AI Video Editor testing.
Provides realistic mock responses for all external APIs.
"""

import json
import time
import random
from typing import Dict, List, Any, Optional, Callable
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta

from tests.data.sample_data import get_sample_data


class GeminiAPIMock:
    """Mock for Gemini API with realistic response patterns."""
    
    def __init__(self):
        self.call_count = 0
        self.call_history = []
        self.response_delay = 0.1  # Simulate network latency
        self.failure_rate = 0.0  # Probability of failure (0.0 to 1.0)
        self.rate_limit_calls = 100  # Calls per minute before rate limiting
        self.rate_limit_window = []
        
    def analyze_content(self, content: str, content_type: str = "educational", **kwargs) -> Dict[str, Any]:
        """Mock content analysis with Gemini API."""
        self._record_call("analyze_content", {"content": content, "content_type": content_type})
        
        # Simulate rate limiting
        if self._check_rate_limit():
            raise Exception("Rate limit exceeded")
        
        # Simulate random failures
        if random.random() < self.failure_rate:
            raise Exception("Gemini API temporarily unavailable")
        
        # Simulate processing delay
        time.sleep(self.response_delay)
        
        # Return realistic response based on content type
        base_response = get_sample_data("gemini_response", content_type)
        
        # Customize response based on input content
        if "financial" in content.lower():
            base_response["content_analysis"]["key_concepts"].extend(["financial planning", "budgeting"])
        if "investment" in content.lower():
            base_response["content_analysis"]["key_concepts"].extend(["portfolio", "risk management"])
        
        return base_response
    
    def research_keywords(self, concepts: List[str], niche: str = "", **kwargs) -> Dict[str, Any]:
        """Mock keyword research functionality."""
        self._record_call("research_keywords", {"concepts": concepts, "niche": niche})
        
        if self._check_rate_limit():
            raise Exception("Rate limit exceeded")
        
        if random.random() < self.failure_rate:
            raise Exception("Gemini API temporarily unavailable")
        
        time.sleep(self.response_delay)
        
        # Generate keywords based on input concepts
        keywords = []
        for concept in concepts:
            keywords.extend([
                f"{concept} explained",
                f"how to {concept}",
                f"{concept} for beginners",
                f"best {concept} tips"
            ])
        
        return {
            "primary_keywords": keywords[:5],
            "long_tail_keywords": keywords[5:10],
            "trending_hashtags": [f"#{concept.replace(' ', '')}" for concept in concepts[:5]],
            "search_volume_data": {keyword: random.randint(1000, 50000) for keyword in keywords[:5]},
            "competition_level": {keyword: random.choice(["low", "medium", "high"]) for keyword in keywords[:5]}
        }
    
    def analyze_competitors(self, niche: str, **kwargs) -> Dict[str, Any]:
        """Mock competitor analysis functionality."""
        self._record_call("analyze_competitors", {"niche": niche})
        
        if self._check_rate_limit():
            raise Exception("Rate limit exceeded")
        
        time.sleep(self.response_delay * 2)  # Competitor analysis takes longer
        
        return {
            "successful_titles": [
                f"How I Mastered {niche} in 30 Days",
                f"The {niche} Mistake That Cost Me Everything",
                f"Why Everyone Gets {niche} Wrong (And How to Fix It)"
            ],
            "thumbnail_patterns": [
                {"pattern": "shocked_face", "success_rate": 0.85},
                {"pattern": "before_after", "success_rate": 0.78},
                {"pattern": "red_arrow", "success_rate": 0.72}
            ],
            "common_hooks": [
                "This changed my life",
                "I wish I knew this earlier",
                "The secret nobody tells you"
            ]
        }
    
    def predict_engagement(self, title: str, description: str, **kwargs) -> Dict[str, Any]:
        """Mock engagement prediction functionality."""
        self._record_call("predict_engagement", {"title": title, "description": description})
        
        time.sleep(self.response_delay)
        
        # Simple heuristics for engagement prediction
        title_score = len(title.split()) / 10  # Prefer 6-10 word titles
        description_score = min(len(description) / 1000, 1.0)  # Prefer longer descriptions
        
        base_ctr = 0.08 + (title_score * 0.02) + (description_score * 0.02)
        base_ctr = min(max(base_ctr, 0.02), 0.20)  # Clamp between 2% and 20%
        
        return {
            "estimated_ctr": base_ctr,
            "estimated_watch_time": random.uniform(0.4, 0.8),
            "estimated_retention": random.uniform(0.5, 0.9),
            "virality_score": random.uniform(0.1, 0.7),
            "confidence": random.uniform(0.7, 0.95)
        }
    
    def set_failure_rate(self, rate: float):
        """Set the probability of API failures for testing error handling."""
        self.failure_rate = max(0.0, min(1.0, rate))
    
    def set_response_delay(self, delay: float):
        """Set the simulated response delay."""
        self.response_delay = max(0.0, delay)
    
    def reset_call_history(self):
        """Reset call history and counters."""
        self.call_count = 0
        self.call_history.clear()
        self.rate_limit_window.clear()
    
    def get_call_count(self, method: Optional[str] = None) -> int:
        """Get the number of calls made to the API."""
        if method is None:
            return self.call_count
        return sum(1 for call in self.call_history if call["method"] == method)
    
    def _record_call(self, method: str, params: Dict[str, Any]):
        """Record an API call for tracking."""
        self.call_count += 1
        self.call_history.append({
            "method": method,
            "params": params,
            "timestamp": datetime.now(),
            "call_number": self.call_count
        })
        
        # Track for rate limiting
        self.rate_limit_window.append(datetime.now())
        # Remove calls older than 1 minute
        cutoff = datetime.now() - timedelta(minutes=1)
        self.rate_limit_window = [t for t in self.rate_limit_window if t > cutoff]
    
    def _check_rate_limit(self) -> bool:
        """Check if rate limit is exceeded."""
        return len(self.rate_limit_window) >= self.rate_limit_calls


class ImagenAPIMock:
    """Mock for Imagen API with realistic image generation simulation."""
    
    def __init__(self):
        self.call_count = 0
        self.call_history = []
        self.response_delay = 2.0  # Image generation takes longer
        self.failure_rate = 0.0
        self.cost_per_generation = 0.05
        self.total_cost = 0.0
        
    def generate_background(self, prompt: str, style: str = "modern_professional", **kwargs) -> Dict[str, Any]:
        """Mock background image generation."""
        self._record_call("generate_background", {"prompt": prompt, "style": style})
        
        if random.random() < self.failure_rate:
            raise Exception("Imagen API service unavailable")
        
        # Simulate generation time
        generation_time = self.response_delay + random.uniform(-0.5, 1.0)
        time.sleep(min(generation_time, 0.1))  # Cap sleep for tests
        
        # Calculate cost
        cost = self.cost_per_generation * (1 + random.uniform(-0.2, 0.3))
        self.total_cost += cost
        
        # Generate mock response
        image_id = f"mock_image_{self.call_count}_{int(time.time())}"
        
        return {
            "image_url": f"https://mock-imagen-api.com/{image_id}.jpg",
            "image_data": self._generate_mock_image_data(prompt, style),
            "generation_metadata": {
                "prompt_used": prompt,
                "style": style,
                "resolution": kwargs.get("resolution", "1920x1080"),
                "generation_time": generation_time,
                "cost": cost,
                "model_version": "imagen-3.0",
                "safety_rating": "safe"
            },
            "quality_metrics": {
                "overall_quality": random.uniform(0.8, 0.95),
                "concept_alignment": random.uniform(0.75, 0.92),
                "visual_appeal": random.uniform(0.8, 0.95),
                "text_readability": random.uniform(0.7, 0.9),
                "brand_consistency": random.uniform(0.8, 0.9)
            }
        }
    
    def generate_variations(self, base_prompt: str, count: int = 3, **kwargs) -> List[Dict[str, Any]]:
        """Mock generation of multiple variations."""
        variations = []
        for i in range(count):
            variation_prompt = f"{base_prompt} (variation {i+1})"
            variation = self.generate_background(variation_prompt, **kwargs)
            variations.append(variation)
        
        return variations
    
    def get_total_cost(self) -> float:
        """Get total cost of all generations."""
        return self.total_cost
    
    def reset_cost_tracking(self):
        """Reset cost tracking."""
        self.total_cost = 0.0
    
    def set_failure_rate(self, rate: float):
        """Set the probability of API failures."""
        self.failure_rate = max(0.0, min(1.0, rate))
    
    def set_response_delay(self, delay: float):
        """Set the simulated response delay."""
        self.response_delay = max(0.0, delay)
    
    def _generate_mock_image_data(self, prompt: str, style: str) -> bytes:
        """Generate mock image data."""
        # Create a simple mock image data based on prompt and style
        header = f"MOCK_IMAGE_{style}_{len(prompt)}_chars"
        data_size = 1024 * 100  # 100KB mock image
        
        mock_data = header.encode('utf-8')
        mock_data += b'\x00' * (data_size - len(mock_data))
        
        return mock_data
    
    def _record_call(self, method: str, params: Dict[str, Any]):
        """Record an API call."""
        self.call_count += 1
        self.call_history.append({
            "method": method,
            "params": params,
            "timestamp": datetime.now(),
            "call_number": self.call_count
        })


class WhisperAPIMock:
    """Mock for Whisper API with realistic transcription simulation."""
    
    def __init__(self):
        self.call_count = 0
        self.call_history = []
        self.response_delay = 1.0  # Per minute of audio
        self.failure_rate = 0.0
        self.accuracy_rate = 0.95  # Transcription accuracy
        
    def transcribe(self, audio_file: str, language: str = "auto", **kwargs) -> Dict[str, Any]:
        """Mock audio transcription."""
        self._record_call("transcribe", {"audio_file": audio_file, "language": language})
        
        if random.random() < self.failure_rate:
            raise Exception("Whisper API service unavailable")
        
        # Simulate processing time based on audio length
        # For testing, we'll use a fixed short delay
        time.sleep(min(self.response_delay, 0.1))
        
        # Get sample transcript data
        transcript_data = get_sample_data("transcript", "educational")
        
        # Simulate accuracy by occasionally introducing errors
        if random.random() > self.accuracy_rate:
            # Introduce some transcription errors
            transcript_data = self._introduce_transcription_errors(transcript_data)
        
        return {
            "text": transcript_data["full_text"],
            "segments": transcript_data["segments"],
            "language": language if language != "auto" else transcript_data["language"],
            "confidence": transcript_data["overall_confidence"],
            "processing_time": self.response_delay,
            "model_version": "whisper-large-v3"
        }
    
    def transcribe_with_timestamps(self, audio_file: str, **kwargs) -> Dict[str, Any]:
        """Mock transcription with detailed timestamps."""
        result = self.transcribe(audio_file, **kwargs)
        
        # Add more detailed timestamp information
        for segment in result["segments"]:
            if "words" not in segment:
                segment["words"] = self._generate_word_timestamps(segment["text"], 
                                                                segment["start_time"], 
                                                                segment["end_time"])
        
        return result
    
    def identify_speakers(self, audio_file: str, **kwargs) -> Dict[str, Any]:
        """Mock speaker identification."""
        self._record_call("identify_speakers", {"audio_file": audio_file})
        
        result = self.transcribe(audio_file, **kwargs)
        
        # Add speaker information
        speakers = ["main", "guest"] if random.random() > 0.7 else ["main"]
        
        for segment in result["segments"]:
            segment["speaker"] = random.choice(speakers)
            segment["speaker_confidence"] = random.uniform(0.8, 0.98)
        
        result["speakers_detected"] = len(speakers)
        result["speaker_labels"] = speakers
        
        return result
    
    def set_accuracy_rate(self, rate: float):
        """Set transcription accuracy rate."""
        self.accuracy_rate = max(0.0, min(1.0, rate))
    
    def set_failure_rate(self, rate: float):
        """Set API failure rate."""
        self.failure_rate = max(0.0, min(1.0, rate))
    
    def _introduce_transcription_errors(self, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """Introduce realistic transcription errors."""
        # Common transcription errors
        error_patterns = [
            ("financial", "financial"),  # No error - most words are correct
            ("interest", "interest"),
            ("compound", "compound"),
            ("investment", "investment"),
            ("you", "you"),
            ("the", "the"),
            ("and", "and"),
            ("to", "to"),
            ("your", "your"),
            ("that", "that")
        ]
        
        # Occasionally replace words with similar sounding ones
        if random.random() < 0.1:  # 10% chance of word substitution
            transcript_data["full_text"] = transcript_data["full_text"].replace("interest", "interests")
        
        # Reduce confidence slightly
        transcript_data["overall_confidence"] *= 0.95
        
        return transcript_data
    
    def _generate_word_timestamps(self, text: str, start_time: float, end_time: float) -> List[Dict[str, Any]]:
        """Generate word-level timestamps."""
        words = text.split()
        duration = end_time - start_time
        word_duration = duration / len(words) if words else 0
        
        word_timestamps = []
        current_time = start_time
        
        for word in words:
            word_end = current_time + word_duration
            word_timestamps.append({
                "word": word,
                "start": current_time,
                "end": word_end,
                "confidence": random.uniform(0.85, 0.98)
            })
            current_time = word_end
        
        return word_timestamps
    
    def _record_call(self, method: str, params: Dict[str, Any]):
        """Record an API call."""
        self.call_count += 1
        self.call_history.append({
            "method": method,
            "params": params,
            "timestamp": datetime.now(),
            "call_number": self.call_count
        })


class ComprehensiveAPIMocker:
    """Comprehensive API mocker that manages all external API mocks."""
    
    def __init__(self):
        self.gemini_mock = GeminiAPIMock()
        self.imagen_mock = ImagenAPIMock()
        self.whisper_mock = WhisperAPIMock()
        self.active_patches = []
        
    def mock_all_apis(self):
        """Mock all external APIs."""
        self.mock_gemini_api()
        self.mock_imagen_api()
        self.mock_whisper_api()
    
    def mock_gemini_api(self):
        """Mock Gemini API calls."""
        # For now, just store the mock without patching non-existent modules
        # When the actual modules are implemented, these patches can be activated
        self.gemini_patches_ready = True
        print("Gemini API mock ready (modules not yet implemented)")
    
    def mock_imagen_api(self):
        """Mock Imagen API calls."""
        # For now, just store the mock without patching non-existent modules
        self.imagen_patches_ready = True
        print("Imagen API mock ready (modules not yet implemented)")
    
    def mock_whisper_api(self):
        """Mock Whisper API calls."""
        # For now, just store the mock without patching non-existent modules
        self.whisper_patches_ready = True
        print("Whisper API mock ready (modules not yet implemented)")
    
    def set_failure_rates(self, gemini: float = 0.0, imagen: float = 0.0, whisper: float = 0.0):
        """Set failure rates for all APIs."""
        self.gemini_mock.set_failure_rate(gemini)
        self.imagen_mock.set_failure_rate(imagen)
        self.whisper_mock.set_failure_rate(whisper)
    
    def set_response_delays(self, gemini: float = 0.1, imagen: float = 2.0, whisper: float = 1.0):
        """Set response delays for all APIs."""
        self.gemini_mock.set_response_delay(gemini)
        self.imagen_mock.set_response_delay(imagen)
        self.whisper_mock.response_delay = whisper
    
    def get_total_api_calls(self) -> Dict[str, int]:
        """Get total API calls for all services."""
        return {
            "gemini": self.gemini_mock.call_count,
            "imagen": self.imagen_mock.call_count,
            "whisper": self.whisper_mock.call_count,
            "total": self.gemini_mock.call_count + self.imagen_mock.call_count + self.whisper_mock.call_count
        }
    
    def get_total_cost(self) -> float:
        """Get total cost of all API calls."""
        # Estimate costs (these would be real costs in production)
        gemini_cost = self.gemini_mock.call_count * 0.02  # $0.02 per call
        imagen_cost = self.imagen_mock.get_total_cost()
        whisper_cost = self.whisper_mock.call_count * 0.01  # $0.01 per minute
        
        return gemini_cost + imagen_cost + whisper_cost
    
    def reset_all_mocks(self):
        """Reset all mock states."""
        self.gemini_mock.reset_call_history()
        self.imagen_mock.call_count = 0
        self.imagen_mock.call_history.clear()
        self.imagen_mock.reset_cost_tracking()
        self.whisper_mock.call_count = 0
        self.whisper_mock.call_history.clear()
    
    def cleanup(self):
        """Stop all patches."""
        for patch_obj in self.active_patches:
            patch_obj.stop()
        self.active_patches.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()