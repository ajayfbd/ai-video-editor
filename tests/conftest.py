# tests/conftest.py
"""
Comprehensive pytest configuration and fixtures for AI Video Editor testing.
Provides mock fixtures for all external dependencies and test utilities.
"""

import pytest
import json
import tempfile
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, MagicMock, patch
import psutil
import time

from ai_video_editor.core.content_context import (
    ContentContext, EmotionalPeak, VisualHighlight, TrendingKeywords,
    ProcessingMetrics, CostMetrics, ContentType, UserPreferences, FaceDetection
)


# ============================================================================
# MOCK FIXTURES FOR EXTERNAL APIS
# ============================================================================

@pytest.fixture
def mock_gemini_response():
    """Mock Gemini API response with comprehensive content analysis."""
    return {
        "content_analysis": {
            "key_concepts": ["financial literacy", "investment basics", "compound interest"],
            "content_themes": ["education", "finance", "personal development"],
            "educational_opportunities": ["mathematical visualization", "chart generation"],
            "content_type_suggestion": "educational"
        },
        "trending_keywords": {
            "primary_keywords": ["financial education", "investment tips", "money management"],
            "long_tail_keywords": ["how to invest for beginners", "compound interest explained"],
            "trending_hashtags": ["#FinancialLiteracy", "#InvestmentTips", "#MoneyManagement"],
            "seasonal_keywords": ["2025 investment trends", "new year financial goals"],
            "search_volume_data": {
                "financial education": 12000,
                "investment tips": 8500,
                "money management": 6200
            }
        },
        "emotional_analysis": {
            "peaks": [
                {"timestamp": 30.5, "emotion": "excitement", "intensity": 0.8, "context": "explaining compound interest"},
                {"timestamp": 95.2, "emotion": "curiosity", "intensity": 0.7, "context": "investment examples"},
                {"timestamp": 180.1, "emotion": "confidence", "intensity": 0.9, "context": "call to action"}
            ]
        },
        "competitor_insights": {
            "successful_titles": [
                "How I Made $10,000 From Compound Interest",
                "Investment Mistakes That Cost Me $50,000",
                "The Simple Investment Strategy That Changed My Life"
            ],
            "thumbnail_patterns": [
                {"pattern": "shocked_face_with_money", "success_rate": 0.85},
                {"pattern": "before_after_charts", "success_rate": 0.78},
                {"pattern": "red_arrow_pointing_up", "success_rate": 0.72}
            ]
        },
        "engagement_predictions": {
            "estimated_ctr": 0.12,
            "estimated_watch_time": 0.68,
            "virality_score": 0.45,
            "educational_value": 0.89
        }
    }


@pytest.fixture
def mock_imagen_response():
    """Mock Imagen API response for thumbnail background generation."""
    return {
        "image_url": "https://mock-imagen-api.com/generated-background-123.jpg",
        "image_data": b"mock_image_data_bytes_here",
        "generation_metadata": {
            "prompt_used": "Professional financial education background with charts and graphs",
            "style": "modern_professional",
            "resolution": "1920x1080",
            "generation_time": 2.3,
            "cost": 0.05
        },
        "quality_score": 0.92,
        "concept_alignment": 0.88
    }


@pytest.fixture
def mock_whisper_response():
    """Mock Whisper API response with detailed transcription."""
    return {
        "text": "Welcome to financial education. Today we're going to learn about compound interest and how it can transform your financial future. Let me show you some examples.",
        "segments": [
            {
                "text": "Welcome to financial education.",
                "start": 0.0,
                "end": 2.5,
                "confidence": 0.95,
                "words": [
                    {"word": "Welcome", "start": 0.0, "end": 0.8, "confidence": 0.98},
                    {"word": "to", "start": 0.8, "end": 1.0, "confidence": 0.99},
                    {"word": "financial", "start": 1.0, "end": 1.8, "confidence": 0.94},
                    {"word": "education", "start": 1.8, "end": 2.5, "confidence": 0.96}
                ]
            },
            {
                "text": "Today we're going to learn about compound interest",
                "start": 2.5,
                "end": 6.2,
                "confidence": 0.92,
                "words": [
                    {"word": "Today", "start": 2.5, "end": 3.0, "confidence": 0.97},
                    {"word": "we're", "start": 3.0, "end": 3.3, "confidence": 0.89},
                    {"word": "going", "start": 3.3, "end": 3.7, "confidence": 0.94},
                    {"word": "to", "start": 3.7, "end": 3.9, "confidence": 0.99},
                    {"word": "learn", "start": 3.9, "end": 4.3, "confidence": 0.96},
                    {"word": "about", "start": 4.3, "end": 4.7, "confidence": 0.98},
                    {"word": "compound", "start": 4.7, "end": 5.4, "confidence": 0.91},
                    {"word": "interest", "start": 5.4, "end": 6.2, "confidence": 0.93}
                ]
            }
        ],
        "language": "en",
        "confidence": 0.94,
        "processing_time": 15.2
    }


# ============================================================================
# MOCK VIDEO AND AUDIO FILES
# ============================================================================

@pytest.fixture
def mock_video_properties():
    """Mock video file properties for consistent testing."""
    return {
        "file_path": "mock://test-video.mp4",
        "duration": 300.0,  # 5 minutes
        "resolution": (1920, 1080),
        "fps": 30,
        "bitrate": 5000000,
        "codec": "h264",
        "file_size": 125000000,  # ~125MB
        "format": "mp4",
        "has_audio": True,
        "audio_codec": "aac",
        "audio_bitrate": 128000,
        "creation_date": "2025-01-01T10:00:00Z"
    }


@pytest.fixture
def mock_audio_transcript():
    """Mock audio transcript with emotional markers and key concepts."""
    return {
        "full_text": "Welcome to financial education. Today we're going to learn about compound interest and how it can transform your financial future. The key is to start early and be consistent. Let me show you some real examples of how compound interest works in practice.",
        "segments": [
            {
                "text": "Welcome to financial education.",
                "start_time": 0.0,
                "end_time": 2.5,
                "confidence": 0.95,
                "speaker": "main",
                "emotional_markers": [
                    {"emotion": "welcoming", "intensity": 0.7, "timestamp": 1.2}
                ]
            },
            {
                "text": "Today we're going to learn about compound interest",
                "start_time": 2.5,
                "end_time": 6.2,
                "confidence": 0.92,
                "speaker": "main",
                "emotional_markers": [
                    {"emotion": "excitement", "intensity": 0.8, "timestamp": 4.5}
                ]
            },
            {
                "text": "and how it can transform your financial future.",
                "start_time": 6.2,
                "end_time": 9.8,
                "confidence": 0.89,
                "speaker": "main",
                "emotional_markers": [
                    {"emotion": "confidence", "intensity": 0.9, "timestamp": 8.0}
                ]
            }
        ],
        "key_concepts_identified": [
            "compound interest", "financial education", "investment strategy",
            "long-term planning", "financial transformation"
        ],
        "filler_words": ["um", "uh", "like"],
        "filler_word_count": 3,
        "speech_pace": "moderate",
        "overall_confidence": 0.92,
        "language": "en-US"
    }


@pytest.fixture
def mock_video_analysis():
    """Mock video analysis results with face detection and visual highlights."""
    return {
        "visual_highlights": [
            {
                "timestamp": 45.2,
                "description": "Speaker pointing to chart on screen",
                "visual_elements": ["chart", "pointing_gesture", "professional_background"],
                "thumbnail_potential": 0.85,
                "faces_detected": 1,
                "dominant_colors": ["#2E86AB", "#A23B72", "#F18F01"]
            },
            {
                "timestamp": 120.7,
                "description": "Animated graph showing compound growth",
                "visual_elements": ["animated_graph", "upward_trend", "numbers"],
                "thumbnail_potential": 0.92,
                "faces_detected": 0,
                "dominant_colors": ["#00A86B", "#FFD700", "#1E3A8A"]
            },
            {
                "timestamp": 200.3,
                "description": "Speaker with excited expression",
                "visual_elements": ["excited_face", "hand_gestures", "eye_contact"],
                "thumbnail_potential": 0.78,
                "faces_detected": 1,
                "dominant_colors": ["#FF6B6B", "#4ECDC4", "#45B7D1"]
            }
        ],
        "face_detections": [
            {
                "timestamp": 45.2,
                "face_box": {"x": 640, "y": 200, "width": 400, "height": 500},
                "expression": "neutral",
                "confidence": 0.94,
                "eye_contact": True,
                "face_angle": 5.2
            },
            {
                "timestamp": 200.3,
                "face_box": {"x": 620, "y": 180, "width": 420, "height": 520},
                "expression": "excited",
                "confidence": 0.89,
                "eye_contact": True,
                "face_angle": -2.1
            }
        ],
        "quality_assessment": {
            "overall_score": 0.87,
            "lighting_quality": 0.92,
            "composition_score": 0.85,
            "stability_score": 0.88,
            "audio_sync": 0.96,
            "resolution_adequacy": 0.95
        },
        "scene_changes": [
            {"timestamp": 0.0, "scene_type": "intro"},
            {"timestamp": 30.5, "scene_type": "explanation"},
            {"timestamp": 120.0, "scene_type": "demonstration"},
            {"timestamp": 250.0, "scene_type": "conclusion"}
        ]
    }


# ============================================================================
# CONTENT CONTEXT FIXTURES
# ============================================================================

@pytest.fixture
def sample_content_context():
    """Create a sample ContentContext for testing."""
    return ContentContext(
        project_id="test_project_123",
        video_files=["test_video.mp4"],
        content_type=ContentType.EDUCATIONAL,
        user_preferences=UserPreferences(
            quality_mode="balanced",
            max_api_cost=2.0
        ),
        key_concepts=["financial literacy", "investment basics"],
        emotional_markers=[
            EmotionalPeak(
                timestamp=30.5,
                emotion="excitement",
                intensity=0.8,
                confidence=0.9,
                context="explaining compound interest"
            )
        ],
        visual_highlights=[
            VisualHighlight(
                timestamp=45.2,
                description="Speaker pointing to chart",
                faces=[],
                visual_elements=["chart", "pointing_gesture"],
                thumbnail_potential=0.85
            )
        ]
    )


@pytest.fixture
def populated_content_context(sample_content_context, mock_gemini_response):
    """Create a fully populated ContentContext for integration testing."""
    context = sample_content_context
    
    # Add trending keywords
    context.trending_keywords = TrendingKeywords(
        primary_keywords=mock_gemini_response["trending_keywords"]["primary_keywords"],
        long_tail_keywords=mock_gemini_response["trending_keywords"]["long_tail_keywords"],
        trending_hashtags=mock_gemini_response["trending_keywords"]["trending_hashtags"],
        seasonal_keywords=mock_gemini_response["trending_keywords"]["seasonal_keywords"],
        competitor_keywords=["finance tips", "money advice"],
        search_volume_data=mock_gemini_response["trending_keywords"]["search_volume_data"],
        research_timestamp=datetime.now()
    )
    
    # Add thumbnail concepts
    context.thumbnail_concepts = [
        {
            "concept_id": "concept_1",
            "emotional_theme": "excitement",
            "visual_elements": ["chart", "upward_arrow"],
            "text_overlay": "Transform Your Financial Future!",
            "background_type": "ai_generated",
            "target_keywords": ["financial education", "investment tips"],
            "ctr_prediction": 0.12
        }
    ]
    
    # Add metadata variations
    context.metadata_variations = [
        {
            "platform": "youtube",
            "title": "How Compound Interest Can Transform Your Financial Future",
            "description": "Learn the power of compound interest with real examples...",
            "tags": ["financial education", "investment tips", "money management"],
            "hashtags": [],
            "thumbnail_alignment_score": 0.89
        }
    ]
    
    return context


# ============================================================================
# PERFORMANCE TESTING FIXTURES
# ============================================================================

@pytest.fixture
def performance_monitor():
    """Performance monitoring fixture for testing."""
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            self.metrics = {}
        
        def start_monitoring(self):
            self.start_time = time.time()
            self.start_memory = psutil.Process().memory_info().rss
        
        def stop_monitoring(self):
            if self.start_time is None:
                raise ValueError("Monitoring not started")
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            self.metrics = {
                "processing_time": end_time - self.start_time,
                "memory_used": end_memory - self.start_memory,
                "peak_memory": end_memory,
                "memory_efficiency": (end_memory - self.start_memory) / max(1, end_time - self.start_time)
            }
            
            return self.metrics
        
        def assert_performance_targets(self, max_time=300, max_memory=16_000_000_000):
            """Assert that performance targets are met."""
            if "processing_time" not in self.metrics:
                raise ValueError("No metrics available - call stop_monitoring() first")
            
            assert self.metrics["processing_time"] < max_time, f"Processing took {self.metrics['processing_time']:.2f}s, expected < {max_time}s"
            assert self.metrics["memory_used"] < max_memory, f"Memory usage {self.metrics['memory_used']} bytes, expected < {max_memory} bytes"
    
    return PerformanceMonitor()


@pytest.fixture
def memory_profiler():
    """Memory profiling fixture for detailed memory analysis."""
    class MemoryProfiler:
        def __init__(self):
            self.snapshots = []
        
        def take_snapshot(self, label: str):
            """Take a memory snapshot with a label."""
            memory_info = psutil.Process().memory_info()
            self.snapshots.append({
                "label": label,
                "timestamp": time.time(),
                "rss": memory_info.rss,
                "vms": memory_info.vms
            })
        
        def get_memory_diff(self, start_label: str, end_label: str):
            """Get memory difference between two snapshots."""
            start_snapshot = next((s for s in self.snapshots if s["label"] == start_label), None)
            end_snapshot = next((s for s in self.snapshots if s["label"] == end_label), None)
            
            if not start_snapshot or not end_snapshot:
                raise ValueError(f"Snapshots not found: {start_label}, {end_label}")
            
            return {
                "rss_diff": end_snapshot["rss"] - start_snapshot["rss"],
                "vms_diff": end_snapshot["vms"] - start_snapshot["vms"],
                "time_diff": end_snapshot["timestamp"] - start_snapshot["timestamp"]
            }
        
        def assert_memory_growth(self, start_label: str, end_label: str, max_growth: int):
            """Assert that memory growth is within limits."""
            diff = self.get_memory_diff(start_label, end_label)
            assert diff["rss_diff"] < max_growth, f"Memory grew by {diff['rss_diff']} bytes, expected < {max_growth}"
    
    return MemoryProfiler()


# ============================================================================
# TEST DATA MANAGEMENT
# ============================================================================

@pytest.fixture
def test_data_manager():
    """Test data management fixture for sample files and expected outputs."""
    class TestDataManager:
        def __init__(self):
            self.temp_dir = tempfile.mkdtemp(prefix="ai_video_editor_test_")
            self.sample_files = {}
            self.expected_outputs = {}
        
        def create_sample_video_file(self, filename: str, properties: Dict[str, Any]):
            """Create a mock video file with specified properties."""
            file_path = os.path.join(self.temp_dir, filename)
            
            # Create a minimal mock file
            with open(file_path, 'wb') as f:
                f.write(b"MOCK_VIDEO_DATA")
            
            self.sample_files[filename] = {
                "path": file_path,
                "properties": properties
            }
            
            return file_path
        
        def create_sample_audio_file(self, filename: str, transcript: Dict[str, Any]):
            """Create a mock audio file with transcript data."""
            file_path = os.path.join(self.temp_dir, filename)
            
            with open(file_path, 'wb') as f:
                f.write(b"MOCK_AUDIO_DATA")
            
            # Store transcript separately
            transcript_path = file_path + ".transcript.json"
            with open(transcript_path, 'w') as f:
                json.dump(transcript, f)
            
            self.sample_files[filename] = {
                "path": file_path,
                "transcript_path": transcript_path,
                "transcript": transcript
            }
            
            return file_path
        
        def load_expected_output(self, test_name: str, output_type: str):
            """Load expected output for a specific test."""
            key = f"{test_name}_{output_type}"
            return self.expected_outputs.get(key)
        
        def save_expected_output(self, test_name: str, output_type: str, data: Any):
            """Save expected output for future test runs."""
            key = f"{test_name}_{output_type}"
            self.expected_outputs[key] = data
        
        def cleanup(self):
            """Clean up temporary files."""
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
    
    manager = TestDataManager()
    yield manager
    manager.cleanup()


# ============================================================================
# API MOCKING UTILITIES
# ============================================================================

@pytest.fixture
def api_mocker():
    """Comprehensive API mocking utility."""
    class APIMocker:
        def __init__(self):
            self.patches = []
            self.call_counts = {}
        
        def mock_gemini_api(self, response_data: Optional[Dict] = None):
            """Mock Gemini API calls."""
            if response_data is None:
                response_data = {
                    "content_analysis": {"key_concepts": ["test_concept"]},
                    "trending_keywords": {"primary_keywords": ["test_keyword"]},
                    "emotional_analysis": {"peaks": []},
                    "competitor_insights": {"successful_titles": []},
                    "engagement_predictions": {"estimated_ctr": 0.1}
                }
            
            mock_client_instance = MagicMock()
            mock_client_instance.analyze_content.return_value = response_data

            def mock_call(*args, **kwargs):
                self.call_counts["gemini"] = self.call_counts.get("gemini", 0) + 1
                return mock_client_instance
            
            patcher = patch('ai_video_editor.modules.intelligence.ai_director.GeminiClient', side_effect=mock_call)
            self.patches.append(patcher)
            return patcher.start()
        
        def mock_imagen_api(self, response_data: Optional[Dict] = None):
            """Mock Imagen API calls."""
            if response_data is None:
                response_data = {
                    "image_url": "mock://generated-image.jpg",
                    "image_data": b"mock_image_data",
                    "generation_metadata": {"cost": 0.05}
                }
            
            def mock_call(*args, **kwargs):
                self.call_counts["imagen"] = self.call_counts.get("imagen", 0) + 1
                return response_data
            
            patcher = patch('ai_video_editor.modules.thumbnail_generation.imagen_api.generate_background', side_effect=mock_call)
            self.patches.append(patcher)
            return patcher.start()
        
        def mock_whisper_api(self, response_data: Optional[Dict] = None):
            """Mock Whisper API calls."""
            if response_data is None:
                response_data = {
                    "text": "Mock transcription text",
                    "segments": [],
                    "language": "en",
                    "confidence": 0.95
                }
            
            def mock_call(*args, **kwargs):
                self.call_counts["whisper"] = self.call_counts.get("whisper", 0) + 1
                return response_data
            
            patcher = patch('ai_video_editor.modules.audio_analysis.whisper_api.transcribe', side_effect=mock_call)
            self.patches.append(patcher)
            return patcher.start()
        
        def get_call_count(self, api_name: str) -> int:
            """Get the number of calls made to a specific API."""
            return self.call_counts.get(api_name, 0)
        
        def reset_call_counts(self):
            """Reset all call counts."""
            self.call_counts.clear()
        
        def cleanup(self):
            """Stop all patches."""
            for patcher in self.patches:
                patcher.stop()
            self.patches.clear()
    
    mocker = APIMocker()
    yield mocker
    mocker.cleanup()


# ============================================================================
# INTEGRATION TEST UTILITIES
# ============================================================================

@pytest.fixture
def integration_test_helper():
    """Helper utilities for integration testing."""
    class IntegrationTestHelper:
        def __init__(self):
            self.context_snapshots = {}
        
        def save_context_snapshot(self, context: ContentContext, label: str):
            """Save a snapshot of ContentContext for comparison."""
            # Create a deep copy of the context for comparison
            snapshot = {
                "project_id": context.project_id,
                "video_files": context.video_files.copy(),
                "content_type": context.content_type,
                "key_concepts": context.key_concepts.copy(),
                "emotional_markers_count": len(context.emotional_markers),
                "visual_highlights_count": len(context.visual_highlights),
                "has_trending_keywords": context.trending_keywords is not None,
                "thumbnail_concepts_count": len(context.thumbnail_concepts),
                "metadata_variations_count": len(context.metadata_variations)
            }
            self.context_snapshots[label] = snapshot
        
        def compare_context_snapshots(self, label1: str, label2: str):
            """Compare two ContentContext snapshots."""
            snapshot1 = self.context_snapshots.get(label1)
            snapshot2 = self.context_snapshots.get(label2)
            
            if not snapshot1 or not snapshot2:
                raise ValueError(f"Snapshots not found: {label1}, {label2}")
            
            differences = {}
            for key in snapshot1:
                if snapshot1[key] != snapshot2[key]:
                    differences[key] = {
                        "before": snapshot1[key],
                        "after": snapshot2[key]
                    }
            
            return differences
        
        def validate_context_integrity(self, context: ContentContext):
            """Validate that ContentContext maintains integrity."""
            assert context.project_id is not None, "Project ID should not be None"
            assert len(context.video_files) > 0, "Should have at least one video file"
            assert context.content_type is not None, "Content type should be specified"
            
            # Validate emotional markers
            for marker in context.emotional_markers:
                assert 0 <= marker.intensity <= 1, f"Emotional intensity should be 0-1, got {marker.intensity}"
                assert 0 <= marker.confidence <= 1, f"Confidence should be 0-1, got {marker.confidence}"
            
            # Validate visual highlights
            for highlight in context.visual_highlights:
                assert 0 <= highlight.thumbnail_potential <= 1, f"Thumbnail potential should be 0-1, got {highlight.thumbnail_potential}"
            
            # Validate thumbnail concepts
            for concept in context.thumbnail_concepts:
                assert 0 <= concept.ctr_prediction <= 1, f"CTR prediction should be 0-1, got {concept.ctr_prediction}"
    
    return IntegrationTestHelper()