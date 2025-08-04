"""
Unit tests for ContentContext system.

Tests the core ContentContext dataclass and related data models
with comprehensive mocking strategies.
"""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from ai_video_editor.core.content_context import (
    ContentContext, ContentType, UserPreferences, EmotionalPeak,
    VisualHighlight, FaceDetection, TrendingKeywords, ProcessingMetrics,
    CostMetrics
)
from ai_video_editor.core.exceptions import ContentContextError


class TestEmotionalPeak:
    """Test EmotionalPeak data model."""
    
    def test_emotional_peak_creation(self):
        """Test creating an EmotionalPeak instance."""
        peak = EmotionalPeak(
            timestamp=30.5,
            emotion="excitement",
            intensity=0.8,
            confidence=0.9,
            context="explaining compound interest"
        )
        
        assert peak.timestamp == 30.5
        assert peak.emotion == "excitement"
        assert peak.intensity == 0.8
        assert peak.confidence == 0.9
        assert peak.context == "explaining compound interest"
    
    def test_emotional_peak_serialization(self):
        """Test EmotionalPeak to_dict and from_dict methods."""
        peak = EmotionalPeak(
            timestamp=45.2,
            emotion="curiosity",
            intensity=0.7,
            confidence=0.85,
            context="introducing new concept"
        )
        
        # Test to_dict
        peak_dict = peak.to_dict()
        expected_dict = {
            'timestamp': 45.2,
            'emotion': 'curiosity',
            'intensity': 0.7,
            'confidence': 0.85,
            'context': 'introducing new concept'
        }
        assert peak_dict == expected_dict
        
        # Test from_dict
        restored_peak = EmotionalPeak.from_dict(peak_dict)
        assert restored_peak.timestamp == peak.timestamp
        assert restored_peak.emotion == peak.emotion
        assert restored_peak.intensity == peak.intensity
        assert restored_peak.confidence == peak.confidence
        assert restored_peak.context == peak.context


class TestFaceDetection:
    """Test FaceDetection data model."""
    
    def test_face_detection_creation(self):
        """Test creating a FaceDetection instance."""
        face = FaceDetection(
            bbox=[100, 150, 200, 250],
            confidence=0.95,
            expression="happy",
            landmarks=[[120, 170], [180, 170], [150, 200]]
        )
        
        assert face.bbox == [100, 150, 200, 250]
        assert face.confidence == 0.95
        assert face.expression == "happy"
        assert face.landmarks == [[120, 170], [180, 170], [150, 200]]
    
    def test_face_detection_serialization(self):
        """Test FaceDetection serialization."""
        face = FaceDetection(
            bbox=[50, 75, 150, 175],
            confidence=0.88,
            expression="neutral"
        )
        
        face_dict = face.to_dict()
        restored_face = FaceDetection.from_dict(face_dict)
        
        assert restored_face.bbox == face.bbox
        assert restored_face.confidence == face.confidence
        assert restored_face.expression == face.expression
        assert restored_face.landmarks == face.landmarks


class TestVisualHighlight:
    """Test VisualHighlight data model."""
    
    def test_visual_highlight_creation(self):
        """Test creating a VisualHighlight instance."""
        faces = [
            FaceDetection([100, 100, 200, 200], 0.9, "excited"),
            FaceDetection([300, 100, 400, 200], 0.85, "focused")
        ]
        
        highlight = VisualHighlight(
            timestamp=60.0,
            description="Key explanation moment",
            faces=faces,
            visual_elements=["whiteboard", "gestures", "eye_contact"],
            thumbnail_potential=0.9
        )
        
        assert highlight.timestamp == 60.0
        assert highlight.description == "Key explanation moment"
        assert len(highlight.faces) == 2
        assert highlight.visual_elements == ["whiteboard", "gestures", "eye_contact"]
        assert highlight.thumbnail_potential == 0.9
    
    def test_visual_highlight_serialization(self):
        """Test VisualHighlight serialization with nested FaceDetection."""
        faces = [FaceDetection([50, 50, 100, 100], 0.8)]
        highlight = VisualHighlight(
            timestamp=120.5,
            description="Demo moment",
            faces=faces,
            visual_elements=["screen", "pointer"],
            thumbnail_potential=0.75
        )
        
        highlight_dict = highlight.to_dict()
        restored_highlight = VisualHighlight.from_dict(highlight_dict)
        
        assert restored_highlight.timestamp == highlight.timestamp
        assert restored_highlight.description == highlight.description
        assert len(restored_highlight.faces) == 1
        assert restored_highlight.faces[0].bbox == faces[0].bbox
        assert restored_highlight.visual_elements == highlight.visual_elements
        assert restored_highlight.thumbnail_potential == highlight.thumbnail_potential


class TestTrendingKeywords:
    """Test TrendingKeywords data model."""
    
    def test_trending_keywords_creation(self):
        """Test creating a TrendingKeywords instance."""
        research_time = datetime.now()
        keywords = TrendingKeywords(
            primary_keywords=["financial literacy", "investment"],
            long_tail_keywords=["beginner investment guide", "how to start investing"],
            trending_hashtags=["#investing", "#finance", "#money"],
            seasonal_keywords=["tax season", "year end planning"],
            competitor_keywords=["wealth building", "passive income"],
            search_volume_data={"financial literacy": 10000, "investment": 50000},
            research_timestamp=research_time
        )
        
        assert keywords.primary_keywords == ["financial literacy", "investment"]
        assert len(keywords.long_tail_keywords) == 2
        assert len(keywords.trending_hashtags) == 3
        assert keywords.search_volume_data["investment"] == 50000
        assert keywords.research_timestamp == research_time
    
    def test_trending_keywords_serialization(self):
        """Test TrendingKeywords serialization."""
        research_time = datetime(2025, 1, 15, 10, 30, 0)
        keywords = TrendingKeywords(
            primary_keywords=["test", "keywords"],
            long_tail_keywords=["test long tail"],
            trending_hashtags=["#test"],
            seasonal_keywords=["seasonal"],
            competitor_keywords=["competitor"],
            search_volume_data={"test": 1000},
            research_timestamp=research_time
        )
        
        keywords_dict = keywords.to_dict()
        restored_keywords = TrendingKeywords.from_dict(keywords_dict)
        
        assert restored_keywords.primary_keywords == keywords.primary_keywords
        assert restored_keywords.research_timestamp == keywords.research_timestamp
        assert restored_keywords.search_volume_data == keywords.search_volume_data


class TestProcessingMetrics:
    """Test ProcessingMetrics data model."""
    
    def test_processing_metrics_creation(self):
        """Test creating ProcessingMetrics instance."""
        metrics = ProcessingMetrics()
        
        assert metrics.total_processing_time == 0.0
        assert metrics.module_processing_times == {}
        assert metrics.memory_peak_usage == 0
        assert metrics.api_calls_made == {}
        assert metrics.cache_hit_rate == 0.0
        assert metrics.fallbacks_used == []
        assert metrics.recovery_actions == []
    
    def test_add_module_metrics(self):
        """Test adding module metrics."""
        metrics = ProcessingMetrics()
        
        metrics.add_module_metrics("audio_analysis", 15.5, 2048000000)
        metrics.add_module_metrics("video_analysis", 25.2, 1024000000)
        
        assert metrics.total_processing_time == 40.7
        assert metrics.module_processing_times["audio_analysis"] == 15.5
        assert metrics.module_processing_times["video_analysis"] == 25.2
        assert metrics.memory_peak_usage == 2048000000  # Should be the higher value
    
    def test_add_api_call(self):
        """Test adding API call tracking."""
        metrics = ProcessingMetrics()
        
        metrics.add_api_call("gemini", 3)
        metrics.add_api_call("imagen", 1)
        metrics.add_api_call("gemini", 2)  # Should add to existing
        
        assert metrics.api_calls_made["gemini"] == 5
        assert metrics.api_calls_made["imagen"] == 1
    
    def test_add_fallback_and_recovery(self):
        """Test adding fallback and recovery tracking."""
        metrics = ProcessingMetrics()
        
        metrics.add_fallback_used("gemini_api")
        metrics.add_recovery_action("restored_from_checkpoint")
        
        assert "gemini_api" in metrics.fallbacks_used
        assert "restored_from_checkpoint" in metrics.recovery_actions


class TestCostMetrics:
    """Test CostMetrics data model."""
    
    def test_cost_metrics_creation(self):
        """Test creating CostMetrics instance."""
        costs = CostMetrics()
        
        assert costs.gemini_api_cost == 0.0
        assert costs.imagen_api_cost == 0.0
        assert costs.total_cost == 0.0
        assert costs.cost_per_asset == {}
        assert costs.optimization_savings == 0.0
    
    def test_add_cost(self):
        """Test adding costs."""
        costs = CostMetrics()
        
        costs.add_cost("gemini", 0.50)
        costs.add_cost("imagen", 1.25)
        costs.add_cost("gemini", 0.25)  # Should add to existing
        
        assert costs.gemini_api_cost == 0.75
        assert costs.imagen_api_cost == 1.25
        assert costs.total_cost == 2.0
    
    def test_add_asset_cost(self):
        """Test adding asset-specific costs."""
        costs = CostMetrics()
        
        costs.add_asset_cost("thumbnail", 0.30)
        costs.add_asset_cost("metadata", 0.15)
        costs.add_asset_cost("thumbnail", 0.20)  # Should add to existing
        
        assert costs.cost_per_asset["thumbnail"] == 0.50
        assert costs.cost_per_asset["metadata"] == 0.15
    
    def test_add_optimization_savings(self):
        """Test adding optimization savings."""
        costs = CostMetrics()
        
        costs.add_optimization_savings(0.35)
        costs.add_optimization_savings(0.15)
        
        assert costs.optimization_savings == 0.50


class TestUserPreferences:
    """Test UserPreferences data model."""
    
    def test_user_preferences_defaults(self):
        """Test UserPreferences default values."""
        prefs = UserPreferences()
        
        assert prefs.quality_mode == "balanced"
        assert prefs.thumbnail_resolution == (1920, 1080)
        assert prefs.batch_size == 3
        assert prefs.enable_aggressive_caching is False
        assert prefs.parallel_processing is True
        assert prefs.max_api_cost == 2.0
    
    def test_user_preferences_serialization(self):
        """Test UserPreferences serialization."""
        prefs = UserPreferences(
            quality_mode="high",
            thumbnail_resolution=(2560, 1440),
            batch_size=5,
            enable_aggressive_caching=True,
            parallel_processing=False,
            max_api_cost=5.0
        )
        
        prefs_dict = prefs.to_dict()
        restored_prefs = UserPreferences.from_dict(prefs_dict)
        
        assert restored_prefs.quality_mode == prefs.quality_mode
        assert restored_prefs.thumbnail_resolution == prefs.thumbnail_resolution
        assert restored_prefs.batch_size == prefs.batch_size
        assert restored_prefs.enable_aggressive_caching == prefs.enable_aggressive_caching
        assert restored_prefs.parallel_processing == prefs.parallel_processing
        assert restored_prefs.max_api_cost == prefs.max_api_cost


class TestContentContext:
    """Test ContentContext main class."""
    
    @pytest.fixture
    def sample_context(self):
        """Create a sample ContentContext for testing."""
        return ContentContext(
            project_id="test_project_123",
            video_files=["test_video1.mp4", "test_video2.mp4"],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
    
    def test_content_context_creation(self, sample_context):
        """Test creating a ContentContext instance."""
        assert sample_context.project_id == "test_project_123"
        assert len(sample_context.video_files) == 2
        assert sample_context.content_type == ContentType.EDUCATIONAL
        assert isinstance(sample_context.user_preferences, UserPreferences)
        assert isinstance(sample_context.processing_metrics, ProcessingMetrics)
        assert isinstance(sample_context.cost_tracking, CostMetrics)
    
    def test_content_context_auto_id_generation(self):
        """Test automatic project ID generation."""
        context = ContentContext(
            project_id="",  # Empty ID should trigger auto-generation
            video_files=["test.mp4"],
            content_type=ContentType.GENERAL,
            user_preferences=UserPreferences()
        )
        
        assert context.project_id != ""
        assert len(context.project_id) > 0
    
    def test_add_emotional_marker(self, sample_context):
        """Test adding emotional markers."""
        initial_count = len(sample_context.emotional_markers)
        
        sample_context.add_emotional_marker(
            timestamp=30.5,
            emotion="excitement",
            intensity=0.8,
            confidence=0.9,
            context="explaining key concept"
        )
        
        assert len(sample_context.emotional_markers) == initial_count + 1
        marker = sample_context.emotional_markers[-1]
        assert marker.timestamp == 30.5
        assert marker.emotion == "excitement"
        assert marker.intensity == 0.8
        assert marker.confidence == 0.9
        assert marker.context == "explaining key concept"
    
    def test_add_visual_highlight(self, sample_context):
        """Test adding visual highlights."""
        faces = [FaceDetection([100, 100, 200, 200], 0.9)]
        initial_count = len(sample_context.visual_highlights)
        
        sample_context.add_visual_highlight(
            timestamp=45.0,
            description="Key demonstration",
            faces=faces,
            visual_elements=["whiteboard", "gestures"],
            thumbnail_potential=0.85
        )
        
        assert len(sample_context.visual_highlights) == initial_count + 1
        highlight = sample_context.visual_highlights[-1]
        assert highlight.timestamp == 45.0
        assert highlight.description == "Key demonstration"
        assert len(highlight.faces) == 1
        assert highlight.visual_elements == ["whiteboard", "gestures"]
        assert highlight.thumbnail_potential == 0.85
    
    def test_get_synchronized_concepts(self, sample_context):
        """Test getting synchronized concepts."""
        # Add some test data
        sample_context.key_concepts = ["finance", "investment", "education"]
        sample_context.content_themes = ["learning", "finance", "beginner"]
        
        # Add high-intensity emotional marker
        sample_context.add_emotional_marker(30.0, "excitement", 0.8, 0.9, "key moment")
        
        # Add high-potential visual highlight
        faces = [FaceDetection([100, 100, 200, 200], 0.9)]
        sample_context.add_visual_highlight(
            45.0, "demo", faces, ["chart", "pointer"], 0.9
        )
        
        concepts = sample_context.get_synchronized_concepts()
        
        # Should include concepts from key_concepts, content_themes, emotions, and visuals
        assert "finance" in concepts  # From both key_concepts and content_themes
        assert "education" in concepts
        assert "learning" in concepts
        assert "excitement_content" in concepts  # From high-intensity emotion
        assert "chart" in concepts  # From high-potential visual
        assert "pointer" in concepts
    
    def test_get_top_emotional_peaks(self, sample_context):
        """Test getting top emotional peaks."""
        # Add emotional markers with different intensities
        sample_context.add_emotional_marker(10.0, "curiosity", 0.6, 0.8, "intro")
        sample_context.add_emotional_marker(30.0, "excitement", 0.9, 0.95, "key point")
        sample_context.add_emotional_marker(50.0, "surprise", 0.7, 0.85, "revelation")
        sample_context.add_emotional_marker(70.0, "satisfaction", 0.8, 0.9, "conclusion")
        
        top_peaks = sample_context.get_top_emotional_peaks(count=2)
        
        assert len(top_peaks) == 2
        assert top_peaks[0].emotion == "excitement"  # Highest intensity (0.9)
        assert top_peaks[1].emotion == "satisfaction"  # Second highest (0.8)
    
    def test_get_best_visual_highlights(self, sample_context):
        """Test getting best visual highlights."""
        faces = [FaceDetection([100, 100, 200, 200], 0.9)]
        
        # Add visual highlights with different thumbnail potentials
        sample_context.add_visual_highlight(10.0, "intro", faces, ["text"], 0.6)
        sample_context.add_visual_highlight(30.0, "key moment", faces, ["chart"], 0.9)
        sample_context.add_visual_highlight(50.0, "demo", faces, ["screen"], 0.7)
        sample_context.add_visual_highlight(70.0, "conclusion", faces, ["summary"], 0.8)
        
        best_highlights = sample_context.get_best_visual_highlights(count=2)
        
        assert len(best_highlights) == 2
        assert best_highlights[0].description == "key moment"  # Highest potential (0.9)
        assert best_highlights[1].description == "conclusion"  # Second highest (0.8)
    
    def test_update_processing_stage(self, sample_context):
        """Test updating processing stage."""
        initial_stage = sample_context._processing_stage
        initial_modified = sample_context._last_modified
        
        # Add small delay to ensure timestamp difference
        import time
        time.sleep(0.001)
        
        sample_context.update_processing_stage("audio_analysis")
        
        assert sample_context._processing_stage == "audio_analysis"
        assert sample_context._last_modified >= initial_modified
    
    def test_add_checkpoint(self, sample_context):
        """Test adding checkpoints."""
        initial_count = len(sample_context._checkpoints)
        initial_modified = sample_context._last_modified
        
        # Add small delay to ensure timestamp difference
        import time
        time.sleep(0.001)
        
        sample_context.add_checkpoint("after_audio_analysis")
        
        assert len(sample_context._checkpoints) == initial_count + 1
        assert "after_audio_analysis" in sample_context._checkpoints
        assert sample_context._last_modified >= initial_modified
    
    def test_content_context_serialization(self, sample_context):
        """Test ContentContext serialization and deserialization."""
        # Add some test data
        sample_context.key_concepts = ["test", "concepts"]
        sample_context.add_emotional_marker(30.0, "excitement", 0.8, 0.9, "test")
        
        faces = [FaceDetection([100, 100, 200, 200], 0.9)]
        sample_context.add_visual_highlight(45.0, "test highlight", faces, ["element"], 0.8)
        
        # Test to_dict
        context_dict = sample_context.to_dict()
        assert context_dict["project_id"] == sample_context.project_id
        assert context_dict["content_type"] == sample_context.content_type.value
        assert len(context_dict["emotional_markers"]) == 1
        assert len(context_dict["visual_highlights"]) == 1
        
        # Test from_dict
        restored_context = ContentContext.from_dict(context_dict)
        assert restored_context.project_id == sample_context.project_id
        assert restored_context.content_type == sample_context.content_type
        assert len(restored_context.emotional_markers) == 1
        assert len(restored_context.visual_highlights) == 1
        assert restored_context.key_concepts == sample_context.key_concepts
    
    def test_content_context_json_serialization(self, sample_context):
        """Test ContentContext JSON serialization."""
        # Add some test data
        sample_context.audio_transcript = "Test transcript"
        sample_context.key_concepts = ["json", "test"]
        
        # Test to_json
        json_str = sample_context.to_json()
        assert isinstance(json_str, str)
        
        # Verify it's valid JSON
        parsed_json = json.loads(json_str)
        assert parsed_json["project_id"] == sample_context.project_id
        assert parsed_json["audio_transcript"] == "Test transcript"
        
        # Test from_json
        restored_context = ContentContext.from_json(json_str)
        assert restored_context.project_id == sample_context.project_id
        assert restored_context.audio_transcript == sample_context.audio_transcript
        assert restored_context.key_concepts == sample_context.key_concepts
    
    def test_content_context_with_trending_keywords(self, sample_context):
        """Test ContentContext with TrendingKeywords."""
        keywords = TrendingKeywords(
            primary_keywords=["test", "keywords"],
            long_tail_keywords=["test long tail"],
            trending_hashtags=["#test"],
            seasonal_keywords=["seasonal"],
            competitor_keywords=["competitor"],
            search_volume_data={"test": 1000},
            research_timestamp=datetime.now()
        )
        
        sample_context.trending_keywords = keywords
        
        # Test serialization with trending keywords
        context_dict = sample_context.to_dict()
        assert context_dict["trending_keywords"] is not None
        assert context_dict["trending_keywords"]["primary_keywords"] == ["test", "keywords"]
        
        # Test deserialization
        restored_context = ContentContext.from_dict(context_dict)
        assert restored_context.trending_keywords is not None
        assert restored_context.trending_keywords.primary_keywords == ["test", "keywords"]
        assert isinstance(restored_context.trending_keywords.research_timestamp, datetime)


class TestContentContextIntegration:
    """Test ContentContext integration scenarios."""
    
    def test_complete_processing_workflow_simulation(self):
        """Test a complete processing workflow simulation with mocked data."""
        # Create context
        context = ContentContext(
            project_id="integration_test",
            video_files=["test_video.mp4"],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences(quality_mode="high")
        )
        
        # Simulate audio analysis
        context.audio_transcript = "Welcome to this educational video about financial literacy..."
        context.add_emotional_marker(15.0, "enthusiasm", 0.7, 0.85, "introduction")
        context.add_emotional_marker(45.0, "excitement", 0.9, 0.92, "key concept explanation")
        
        # Simulate video analysis
        faces = [FaceDetection([200, 150, 400, 350], 0.88, "engaged")]
        context.add_visual_highlight(
            45.0, "presenter explaining with gestures", faces, 
            ["whiteboard", "gestures", "eye_contact"], 0.85
        )
        
        # Simulate content analysis
        context.key_concepts = ["financial literacy", "budgeting", "saving", "investing"]
        context.content_themes = ["education", "personal finance", "beginner guide"]
        
        # Simulate keyword research
        context.trending_keywords = TrendingKeywords(
            primary_keywords=["financial literacy", "personal finance", "budgeting"],
            long_tail_keywords=["beginner financial literacy guide", "how to budget money"],
            trending_hashtags=["#personalfinance", "#budgeting", "#financialliteracy"],
            seasonal_keywords=["tax season", "new year budgeting"],
            competitor_keywords=["money management", "financial planning"],
            search_volume_data={"financial literacy": 15000, "budgeting": 25000},
            research_timestamp=datetime.now()
        )
        
        # Simulate processing metrics
        context.processing_metrics.add_module_metrics("audio_analysis", 12.5, 1500000000)
        context.processing_metrics.add_module_metrics("video_analysis", 18.3, 2100000000)
        context.processing_metrics.add_module_metrics("content_analysis", 8.7, 800000000)
        context.processing_metrics.add_api_call("gemini", 3)
        context.processing_metrics.add_api_call("imagen", 1)
        
        # Simulate cost tracking
        context.cost_tracking.add_cost("gemini", 0.45)
        context.cost_tracking.add_cost("imagen", 0.75)
        context.cost_tracking.add_asset_cost("thumbnail", 0.30)
        context.cost_tracking.add_asset_cost("metadata", 0.15)
        
        # Test synchronized concepts
        synchronized_concepts = context.get_synchronized_concepts()
        assert "financial literacy" in synchronized_concepts
        assert "budgeting" in synchronized_concepts
        assert "excitement_content" in synchronized_concepts  # From high-intensity emotion
        assert "whiteboard" in synchronized_concepts  # From high-potential visual
        
        # Test top emotional peaks
        top_peaks = context.get_top_emotional_peaks(count=2)
        assert len(top_peaks) == 2
        assert top_peaks[0].emotion == "excitement"  # Higher intensity
        
        # Test serialization of complete context
        context_dict = context.to_dict()
        restored_context = ContentContext.from_dict(context_dict)
        
        # Verify all data is preserved
        assert restored_context.project_id == context.project_id
        assert restored_context.audio_transcript == context.audio_transcript
        assert len(restored_context.emotional_markers) == 2
        assert len(restored_context.visual_highlights) == 1
        assert restored_context.key_concepts == context.key_concepts
        assert restored_context.trending_keywords.primary_keywords == context.trending_keywords.primary_keywords
        assert restored_context.processing_metrics.total_processing_time == context.processing_metrics.total_processing_time
        assert restored_context.cost_tracking.total_cost == context.cost_tracking.total_cost
    
    def test_error_recovery_scenario(self):
        """Test error recovery scenario with partial data."""
        context = ContentContext(
            project_id="error_recovery_test",
            video_files=["test_video.mp4"],
            content_type=ContentType.MUSIC,
            user_preferences=UserPreferences()
        )
        
        # Simulate successful audio analysis
        context.audio_transcript = "Music video transcript..."
        context.add_emotional_marker(30.0, "energy", 0.8, 0.9, "beat drop")
        
        # Simulate failed video analysis (no visual highlights added)
        context.processing_metrics.add_fallback_used("video_analysis")
        context.processing_metrics.add_recovery_action("used_audio_only_analysis")
        
        # Simulate partial keyword research (some data missing)
        context.trending_keywords = TrendingKeywords(
            primary_keywords=["music", "beat"],
            long_tail_keywords=[],  # Empty due to API failure
            trending_hashtags=["#music"],
            seasonal_keywords=[],
            competitor_keywords=[],
            search_volume_data={"music": 50000},
            research_timestamp=datetime.now()
        )
        
        # Test that context is still functional with partial data
        synchronized_concepts = context.get_synchronized_concepts()
        # Note: key_concepts and content_themes are empty, so only emotional content should be present
        assert "energy_content" in synchronized_concepts
        
        # Test that fallbacks are recorded
        assert "video_analysis" in context.processing_metrics.fallbacks_used
        assert "used_audio_only_analysis" in context.processing_metrics.recovery_actions
        
        # Test serialization works with partial data
        context_dict = context.to_dict()
        restored_context = ContentContext.from_dict(context_dict)
        
        assert restored_context.project_id == context.project_id
        assert len(restored_context.visual_highlights) == 0  # No visual data due to failure
        assert len(restored_context.emotional_markers) == 1  # Audio analysis succeeded
        assert restored_context.trending_keywords.primary_keywords == ["music", "beat"]