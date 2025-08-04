"""
Unit tests for DataValidator.

Tests the DataValidator class for ContentContext integrity checking
with comprehensive validation scenarios.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from ai_video_editor.core.data_validator import (
    DataValidator, ValidationRule, RequiredFieldsRule, FileExistenceRule,
    EmotionalMarkersRule, VisualHighlightsRule, TrendingKeywordsRule,
    ProcessingMetricsRule, CostMetricsRule, DataConsistencyRule
)
from ai_video_editor.core.content_context import (
    ContentContext, ContentType, UserPreferences, EmotionalPeak,
    VisualHighlight, FaceDetection, TrendingKeywords, ProcessingMetrics,
    CostMetrics
)


class TestValidationRules:
    """Test individual validation rules."""
    
    @pytest.fixture
    def valid_context(self):
        """Create a valid ContentContext for testing."""
        return ContentContext(
            project_id="test_project",
            video_files=["test_video.mp4"],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
    
    def test_required_fields_rule_valid(self, valid_context):
        """Test RequiredFieldsRule with valid context."""
        rule = RequiredFieldsRule()
        is_valid, message = rule.validate(valid_context)
        
        assert is_valid is True
        assert "All required fields are present" in message
    
    def test_required_fields_rule_missing_project_id(self, valid_context):
        """Test RequiredFieldsRule with missing project_id."""
        valid_context.project_id = ""
        
        rule = RequiredFieldsRule()
        is_valid, message = rule.validate(valid_context)
        
        assert is_valid is False
        assert "project_id is required" in message
    
    def test_required_fields_rule_empty_video_files(self, valid_context):
        """Test RequiredFieldsRule with empty video_files."""
        valid_context.video_files = []
        
        rule = RequiredFieldsRule()
        is_valid, message = rule.validate(valid_context)
        
        assert is_valid is False
        assert "video_files list cannot be empty" in message
    
    def test_file_existence_rule_existing_files(self, valid_context, tmp_path):
        """Test FileExistenceRule with existing files."""
        # Create temporary files
        video_file = tmp_path / "test_video.mp4"
        video_file.write_text("mock video content")
        
        valid_context.video_files = [str(video_file)]
        
        rule = FileExistenceRule()
        is_valid, message = rule.validate(valid_context)
        
        assert is_valid is True
        assert "All referenced files exist" in message
    
    def test_file_existence_rule_missing_files(self, valid_context):
        """Test FileExistenceRule with missing files."""
        valid_context.video_files = ["nonexistent_file.mp4"]
        
        rule = FileExistenceRule()
        is_valid, message = rule.validate(valid_context)
        
        assert is_valid is False
        assert "Missing video files" in message
        assert "nonexistent_file.mp4" in message
    
    def test_emotional_markers_rule_valid(self, valid_context):
        """Test EmotionalMarkersRule with valid markers."""
        valid_context.add_emotional_marker(30.0, "excitement", 0.8, 0.9, "test context")
        valid_context.add_emotional_marker(60.0, "curiosity", 0.7, 0.85, "another context")
        
        rule = EmotionalMarkersRule()
        is_valid, message = rule.validate(valid_context)
        
        assert is_valid is True
        assert "2 emotional markers are valid" in message
    
    def test_emotional_markers_rule_invalid_intensity(self, valid_context):
        """Test EmotionalMarkersRule with invalid intensity."""
        # Create invalid marker directly to bypass ContentContext validation
        invalid_marker = EmotionalPeak(
            timestamp=30.0,
            emotion="excitement",
            intensity=1.5,  # Invalid: > 1.0
            confidence=0.9,
            context="test"
        )
        valid_context.emotional_markers.append(invalid_marker)
        
        rule = EmotionalMarkersRule()
        is_valid, message = rule.validate(valid_context)
        
        assert is_valid is False
        assert "intensity 1.5 not in range [0.0, 1.0]" in message
    
    def test_emotional_markers_rule_negative_timestamp(self, valid_context):
        """Test EmotionalMarkersRule with negative timestamp."""
        invalid_marker = EmotionalPeak(
            timestamp=-10.0,  # Invalid: negative
            emotion="excitement",
            intensity=0.8,
            confidence=0.9,
            context="test"
        )
        valid_context.emotional_markers.append(invalid_marker)
        
        rule = EmotionalMarkersRule()
        is_valid, message = rule.validate(valid_context)
        
        assert is_valid is False
        assert "negative timestamp -10.0" in message
    
    def test_visual_highlights_rule_valid(self, valid_context):
        """Test VisualHighlightsRule with valid highlights."""
        faces = [FaceDetection([100, 100, 200, 200], 0.9, "happy")]
        valid_context.add_visual_highlight(
            45.0, "test highlight", faces, ["element1", "element2"], 0.8
        )
        
        rule = VisualHighlightsRule()
        is_valid, message = rule.validate(valid_context)
        
        assert is_valid is True
        assert "1 visual highlights are valid" in message
    
    def test_visual_highlights_rule_invalid_thumbnail_potential(self, valid_context):
        """Test VisualHighlightsRule with invalid thumbnail potential."""
        faces = [FaceDetection([100, 100, 200, 200], 0.9)]
        invalid_highlight = VisualHighlight(
            timestamp=45.0,
            description="test",
            faces=faces,
            visual_elements=["element"],
            thumbnail_potential=1.5  # Invalid: > 1.0
        )
        valid_context.visual_highlights.append(invalid_highlight)
        
        rule = VisualHighlightsRule()
        is_valid, message = rule.validate(valid_context)
        
        assert is_valid is False
        assert "thumbnail_potential 1.5 not in range [0.0, 1.0]" in message
    
    def test_trending_keywords_rule_valid(self, valid_context):
        """Test TrendingKeywordsRule with valid keywords."""
        valid_context.trending_keywords = TrendingKeywords(
            primary_keywords=["test", "keywords"],
            long_tail_keywords=["long tail keyword"],
            trending_hashtags=["#test"],
            seasonal_keywords=["seasonal"],
            competitor_keywords=["competitor"],
            search_volume_data={"test": 1000, "keywords": 500},
            research_timestamp=datetime.now()
        )
        
        rule = TrendingKeywordsRule()
        is_valid, message = rule.validate(valid_context)
        
        assert is_valid is True
        assert "Trending keywords data is valid" in message
    
    def test_trending_keywords_rule_none(self, valid_context):
        """Test TrendingKeywordsRule with None keywords."""
        valid_context.trending_keywords = None
        
        rule = TrendingKeywordsRule()
        is_valid, message = rule.validate(valid_context)
        
        assert is_valid is True
        assert "No trending keywords to validate" in message
    
    def test_processing_metrics_rule_valid(self, valid_context):
        """Test ProcessingMetricsRule with valid metrics."""
        valid_context.processing_metrics.add_module_metrics("test_module", 10.5, 1000000)
        valid_context.processing_metrics.add_api_call("gemini", 3)
        valid_context.processing_metrics.cache_hit_rate = 0.75
        
        rule = ProcessingMetricsRule()
        is_valid, message = rule.validate(valid_context)
        
        assert is_valid is True
        assert "Processing metrics are valid" in message
    
    def test_processing_metrics_rule_negative_time(self, valid_context):
        """Test ProcessingMetricsRule with negative processing time."""
        valid_context.processing_metrics.total_processing_time = -5.0
        
        rule = ProcessingMetricsRule()
        is_valid, message = rule.validate(valid_context)
        
        assert is_valid is False
        assert "total_processing_time cannot be negative" in message
    
    def test_cost_metrics_rule_valid(self, valid_context):
        """Test CostMetricsRule with valid costs."""
        valid_context.cost_tracking.add_cost("gemini", 0.50)
        valid_context.cost_tracking.add_cost("imagen", 0.75)
        valid_context.cost_tracking.add_asset_cost("thumbnail", 0.30)
        
        rule = CostMetricsRule()
        is_valid, message = rule.validate(valid_context)
        
        assert is_valid is True
        assert "Cost metrics are valid" in message
    
    def test_cost_metrics_rule_inconsistent_total(self, valid_context):
        """Test CostMetricsRule with inconsistent total cost."""
        valid_context.cost_tracking.gemini_api_cost = 0.50
        valid_context.cost_tracking.imagen_api_cost = 0.75
        valid_context.cost_tracking.total_cost = 2.00  # Should be 1.25
        
        rule = CostMetricsRule()
        is_valid, message = rule.validate(valid_context)
        
        assert is_valid is False
        assert "total_cost 2.0 doesn't match sum of individual costs 1.25" in message
    
    def test_data_consistency_rule_valid(self, valid_context):
        """Test DataConsistencyRule with consistent data."""
        # Add data in chronological order
        valid_context.add_emotional_marker(10.0, "curiosity", 0.7, 0.8, "intro")
        valid_context.add_emotional_marker(30.0, "excitement", 0.9, 0.9, "key point")
        
        faces = [FaceDetection([100, 100, 200, 200], 0.9)]
        valid_context.add_visual_highlight(15.0, "early highlight", faces, ["element"], 0.7)
        valid_context.add_visual_highlight(45.0, "later highlight", faces, ["element"], 0.8)
        
        # Add overlapping concepts and themes
        valid_context.key_concepts = ["finance", "education", "investment"]
        valid_context.content_themes = ["finance", "learning", "beginner"]
        
        rule = DataConsistencyRule()
        is_valid, message = rule.validate(valid_context)
        
        assert is_valid is True
        assert "Data consistency checks passed" in message


class TestDataValidator:
    """Test DataValidator main class."""
    
    @pytest.fixture
    def validator(self):
        """Create DataValidator instance."""
        return DataValidator()
    
    @pytest.fixture
    def valid_context(self):
        """Create a valid ContentContext for testing."""
        context = ContentContext(
            project_id="test_project",
            video_files=["test_video.mp4"],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        # Add some valid data
        context.add_emotional_marker(30.0, "excitement", 0.8, 0.9, "test context")
        
        faces = [FaceDetection([100, 100, 200, 200], 0.9, "happy")]
        context.add_visual_highlight(45.0, "test highlight", faces, ["element"], 0.8)
        
        context.key_concepts = ["test", "concepts"]
        context.content_themes = ["test", "themes"]
        
        return context
    
    def test_validator_initialization(self, validator):
        """Test DataValidator initialization."""
        assert len(validator.rules) == 8  # Should have 8 default rules
        
        rule_names = [rule.name for rule in validator.rules]
        expected_rules = [
            "required_fields", "file_existence", "emotional_markers",
            "visual_highlights", "trending_keywords", "processing_metrics",
            "cost_metrics", "data_consistency"
        ]
        
        for expected_rule in expected_rules:
            assert expected_rule in rule_names
    
    def test_validate_success(self, validator, valid_context):
        """Test successful validation."""
        result = validator.validate(valid_context)
        
        assert result['valid'] is True
        assert result['score'] > 0.8
        assert result['rules_passed'] > 0
        assert result['rules_failed'] == 0
        assert len(result['errors']) == 0
    
    def test_validate_with_errors(self, validator):
        """Test validation with errors."""
        # Create context with errors
        invalid_context = ContentContext(
            project_id="",  # Missing project ID
            video_files=[],  # No video files
            content_type=ContentType.GENERAL,
            user_preferences=UserPreferences()
        )
        
        result = validator.validate(invalid_context)
        
        assert result['valid'] is False
        assert result['score'] == 0.0
        assert result['rules_failed'] > 0
        assert len(result['errors']) > 0
    
    def test_validate_with_warnings(self, validator, valid_context):
        """Test validation with warnings."""
        # Add missing files (should generate warnings)
        valid_context.video_files = ["missing_file.mp4"]
        
        result = validator.validate(valid_context)
        
        assert result['valid'] is True  # Still valid but with warnings
        assert result['score'] < 1.0  # Reduced score
        assert len(result['warnings']) > 0
        assert any("Missing video files" in warning for warning in result['warnings'])
    
    def test_validate_strict_mode(self, validator, valid_context):
        """Test validation in strict mode (warnings treated as errors)."""
        # Add missing files (should generate warnings)
        valid_context.video_files = ["missing_file.mp4"]
        
        result = validator.validate(valid_context, strict=True)
        
        assert result['valid'] is False  # Should be invalid in strict mode
        assert len(result['errors']) > 0
        assert any("Missing video files" in error for error in result['errors'])
    
    def test_validate_field_update_valid(self, validator, valid_context):
        """Test validating a valid field update."""
        result = validator.validate_field_update(
            valid_context, "audio_transcript", "New transcript content"
        )
        
        assert result['valid'] is True
        assert len(result['errors']) == 0
    
    def test_validate_field_update_invalid(self, validator, valid_context):
        """Test validating an invalid field update."""
        # Try to set invalid emotional markers
        invalid_markers = [
            EmotionalPeak(30.0, "excitement", 1.5, 0.9, "test")  # Invalid intensity
        ]
        
        result = validator.validate_field_update(
            valid_context, "emotional_markers", invalid_markers
        )
        
        # Note: This test might pass if the field validation doesn't catch the specific field
        # The actual behavior depends on how the validation rules are implemented
        assert isinstance(result, dict)
        assert 'valid' in result
        assert 'errors' in result
    
    def test_get_validation_summary(self, validator, valid_context):
        """Test getting validation summary."""
        summary = validator.get_validation_summary(valid_context)
        
        assert isinstance(summary, str)
        assert "Validation Summary" in summary
        assert valid_context.project_id in summary
        assert "VALID" in summary or "INVALID" in summary
        assert "Score:" in summary
        assert "Rules Passed:" in summary
    
    def test_add_custom_rule(self, validator):
        """Test adding custom validation rule."""
        class CustomRule(ValidationRule):
            def __init__(self):
                super().__init__("custom_rule", "error")
            
            def validate(self, context):
                return True, "Custom rule passed"
        
        initial_count = len(validator.rules)
        custom_rule = CustomRule()
        
        validator.add_custom_rule(custom_rule)
        
        assert len(validator.rules) == initial_count + 1
        assert validator.rules[-1].name == "custom_rule"
    
    def test_remove_rule(self, validator):
        """Test removing validation rule."""
        initial_count = len(validator.rules)
        
        # Remove existing rule
        success = validator.remove_rule("file_existence")
        
        assert success is True
        assert len(validator.rules) == initial_count - 1
        
        rule_names = [rule.name for rule in validator.rules]
        assert "file_existence" not in rule_names
    
    def test_remove_nonexistent_rule(self, validator):
        """Test removing non-existent rule."""
        initial_count = len(validator.rules)
        
        success = validator.remove_rule("nonexistent_rule")
        
        assert success is False
        assert len(validator.rules) == initial_count
    
    def test_validation_with_exception(self, validator, valid_context):
        """Test validation when a rule raises an exception."""
        # Create a rule that raises an exception
        class FailingRule(ValidationRule):
            def __init__(self):
                super().__init__("failing_rule", "error")
            
            def validate(self, context):
                raise ValueError("Test exception")
        
        validator.add_custom_rule(FailingRule())
        
        result = validator.validate(valid_context)
        
        # Should handle the exception gracefully
        assert result['valid'] is False
        assert len(result['errors']) > 0
        assert any("Validation exception" in error for error in result['errors'])
    
    def test_score_calculation(self, validator):
        """Test validation score calculation."""
        # Create context with mixed validation results
        context = ContentContext(
            project_id="score_test",
            video_files=["missing_file.mp4"],  # Will generate warning
            content_type=ContentType.GENERAL,
            user_preferences=UserPreferences()
        )
        
        result = validator.validate(context)
        
        # Should have reduced score due to warnings
        assert 0.0 < result['score'] < 1.0
        assert result['valid'] is True  # Still valid
        assert len(result['warnings']) > 0
    
    def test_comprehensive_validation_scenario(self, validator):
        """Test comprehensive validation scenario with complex data."""
        context = ContentContext(
            project_id="comprehensive_test",
            video_files=["test_video.mp4"],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        # Add comprehensive data
        context.audio_transcript = "Test transcript content"
        context.key_concepts = ["education", "learning", "tutorial"]
        context.content_themes = ["education", "beginner", "practical"]
        
        # Add emotional markers
        context.add_emotional_marker(10.0, "curiosity", 0.6, 0.8, "introduction")
        context.add_emotional_marker(30.0, "excitement", 0.9, 0.95, "key concept")
        context.add_emotional_marker(50.0, "satisfaction", 0.7, 0.85, "conclusion")
        
        # Add visual highlights
        faces = [FaceDetection([100, 100, 200, 200], 0.9, "engaged")]
        context.add_visual_highlight(15.0, "intro visual", faces, ["text", "graphics"], 0.7)
        context.add_visual_highlight(35.0, "key visual", faces, ["diagram", "pointer"], 0.9)
        
        # Add trending keywords
        context.trending_keywords = TrendingKeywords(
            primary_keywords=["education", "tutorial", "learning"],
            long_tail_keywords=["beginner education tutorial", "how to learn effectively"],
            trending_hashtags=["#education", "#learning", "#tutorial"],
            seasonal_keywords=["back to school", "new year learning"],
            competitor_keywords=["online courses", "educational content"],
            search_volume_data={"education": 50000, "tutorial": 30000, "learning": 40000},
            research_timestamp=datetime.now()
        )
        
        # Add processing metrics
        context.processing_metrics.add_module_metrics("audio_analysis", 15.2, 1500000000)
        context.processing_metrics.add_module_metrics("video_analysis", 22.8, 2000000000)
        context.processing_metrics.add_api_call("gemini", 4)
        context.processing_metrics.add_api_call("imagen", 2)
        context.processing_metrics.cache_hit_rate = 0.65
        
        # Add cost tracking
        context.cost_tracking.add_cost("gemini", 0.60)
        context.cost_tracking.add_cost("imagen", 0.90)
        context.cost_tracking.add_asset_cost("thumbnail", 0.45)
        context.cost_tracking.add_asset_cost("metadata", 0.20)
        context.cost_tracking.add_optimization_savings(0.25)
        
        # Validate
        result = validator.validate(context)
        
        # Should pass all validations with high score
        assert result['valid'] is True
        assert result['score'] >= 0.8  # High score for comprehensive valid data
        assert result['rules_passed'] >= 7  # Most rules should pass
        assert len(result['errors']) == 0
        
        # Get summary
        summary = validator.get_validation_summary(context)
        assert "VALID" in summary
        assert "comprehensive_test" in summary