"""
Integration test for ContentContext system.

Tests the complete ContentContext system working together with all components.
"""

import pytest
import tempfile
import shutil
from datetime import datetime

from ai_video_editor.core.content_context import ContentContext, ContentType, UserPreferences
from ai_video_editor.core.context_manager import ContextManager
from ai_video_editor.core.data_validator import DataValidator
from ai_video_editor.core.cache_manager import CacheManager
from ai_video_editor.core.state_tracker import StateTracker


class TestContentContextSystemIntegration:
    """Test complete ContentContext system integration."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def system_components(self, temp_dir):
        """Create all system components."""
        context_manager = ContextManager(storage_path=f"{temp_dir}/contexts")
        data_validator = DataValidator()
        cache_manager = CacheManager(cache_dir=f"{temp_dir}/cache")
        state_tracker = StateTracker()
        
        return {
            'context_manager': context_manager,
            'data_validator': data_validator,
            'cache_manager': cache_manager,
            'state_tracker': state_tracker
        }
    
    def test_complete_workflow_integration(self, system_components, temp_dir):
        """Test complete workflow with all system components."""
        context_manager = system_components['context_manager']
        data_validator = system_components['data_validator']
        cache_manager = system_components['cache_manager']
        state_tracker = system_components['state_tracker']
        
        # Create mock video file
        video_file = f"{temp_dir}/test_video.mp4"
        with open(video_file, 'w') as f:
            f.write("mock video content")
        
        # 1. Create ContentContext
        context = context_manager.create_context(
            video_files=[video_file],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences(quality_mode="high")
        )
        
        assert context is not None
        assert context.project_id != ""
        assert context.content_type == ContentType.EDUCATIONAL
        
        # 2. Validate initial context
        validation_result = data_validator.validate(context)
        assert validation_result['valid'] is True
        assert validation_result['score'] > 0.8
        
        # 3. Simulate processing with data population
        context.audio_transcript = "Welcome to this educational video about financial planning..."
        context.add_emotional_marker(15.0, "enthusiasm", 0.8, 0.9, "introduction")
        context.add_emotional_marker(45.0, "excitement", 0.9, 0.95, "key concept")
        
        # Add visual highlights
        from ai_video_editor.core.content_context import FaceDetection
        faces = [FaceDetection([200, 150, 400, 350], 0.88, "engaged")]
        context.add_visual_highlight(
            30.0, "presenter explaining concept", faces,
            ["whiteboard", "gestures", "eye_contact"], 0.85
        )
        
        # Add concepts and themes
        context.key_concepts = ["financial planning", "budgeting", "investment"]
        context.content_themes = ["education", "personal finance", "beginner"]
        
        # 4. Cache some processing results
        cache_manager.cache_processing_result(
            context.project_id, "audio_analysis", "transcription",
            {"transcript": context.audio_transcript, "confidence": 0.92}
        )
        
        cache_manager.cache_keyword_research(
            context.key_concepts, "educational",
            {"primary_keywords": ["financial planning", "budgeting tips"]}
        )
        
        # 5. Create and manage workflow
        workflow = state_tracker.create_workflow(context)
        assert workflow is not None
        assert workflow.context_id == context.project_id
        assert len(workflow.tasks) > 0
        
        # Start workflow
        success = state_tracker.start_workflow(workflow.workflow_id)
        assert success is True
        
        # Simulate task execution
        ready_tasks = state_tracker.get_next_tasks(workflow.workflow_id, max_tasks=2)
        assert len(ready_tasks) > 0
        
        for task in ready_tasks[:1]:  # Process first task
            state_tracker.start_task(workflow.workflow_id, task.task_id)
            state_tracker.update_task_progress(workflow.workflow_id, task.task_id, 0.5)
            state_tracker.complete_task(workflow.workflow_id, task.task_id)
        
        # 6. Update processing metrics
        context.processing_metrics.add_module_metrics("audio_analysis", 12.5, 1500000000)
        context.processing_metrics.add_api_call("gemini", 2)
        context.cost_tracking.add_cost("gemini", 0.35)
        
        # 7. Save context with checkpoint
        context_manager.save_checkpoint(context, "after_audio_analysis")
        save_success = context_manager.save_context(context)
        assert save_success is True
        
        # 8. Validate final context
        final_validation = data_validator.validate(context)
        assert final_validation['valid'] is True
        assert final_validation['score'] > 0.8
        
        # 9. Test context retrieval and cache hits
        cached_result = cache_manager.get_processing_result(
            context.project_id, "audio_analysis", "transcription"
        )
        assert cached_result is not None
        assert cached_result["transcript"] == context.audio_transcript
        
        # 10. Test synchronized concepts
        synchronized_concepts = context.get_synchronized_concepts()
        assert "financial planning" in synchronized_concepts
        assert "budgeting" in synchronized_concepts
        assert "excitement_content" in synchronized_concepts  # From high-intensity emotion
        assert "whiteboard" in synchronized_concepts  # From high-potential visual
        
        # 11. Get workflow status
        workflow_status = state_tracker.get_workflow_status(workflow.workflow_id)
        assert workflow_status['context_id'] == context.project_id
        assert workflow_status['completed_tasks'] >= 1
        
        # 12. Get system statistics
        context_stats = context_manager.get_context_stats(context)
        cache_stats = cache_manager.get_stats()
        tracker_stats = state_tracker.get_statistics()
        
        assert context_stats['project_id'] == context.project_id
        assert context_stats['emotional_markers_count'] == 2
        assert context_stats['visual_highlights_count'] == 1
        assert context_stats['total_cost'] == 0.35
        
        assert cache_stats['hits'] >= 0
        assert cache_stats['puts'] >= 2
        
        assert tracker_stats['total_workflows'] >= 1
        assert tracker_stats['completed_tasks'] >= 1
    
    def test_error_recovery_integration(self, system_components, temp_dir):
        """Test error recovery across system components."""
        context_manager = system_components['context_manager']
        data_validator = system_components['data_validator']
        state_tracker = system_components['state_tracker']
        
        # Create context
        video_file = f"{temp_dir}/test_video.mp4"
        with open(video_file, 'w') as f:
            f.write("mock video content")
        
        context = context_manager.create_context(
            video_files=[video_file],
            content_type=ContentType.MUSIC,
            user_preferences=UserPreferences()
        )
        
        # Simulate partial processing with failures
        context.audio_transcript = "Music video content..."
        context.add_emotional_marker(30.0, "energy", 0.8, 0.9, "beat drop")
        
        # Record fallback usage
        context.processing_metrics.add_fallback_used("video_analysis")
        context.processing_metrics.add_recovery_action("used_audio_only_analysis")
        
        # Save checkpoint before potential failure
        context_manager.save_checkpoint(context, "before_video_analysis")
        
        # Create workflow and simulate task failure
        workflow = state_tracker.create_workflow(context)
        state_tracker.start_workflow(workflow.workflow_id)
        
        # Start and fail a task
        state_tracker.start_task(workflow.workflow_id, "video_analysis")
        state_tracker.fail_task(workflow.workflow_id, "video_analysis", "Mock processing error")
        
        # Validate context is still functional
        validation_result = data_validator.validate(context)
        assert validation_result['valid'] is True  # Should still be valid
        
        # Check that fallbacks are recorded
        assert "video_analysis" in context.processing_metrics.fallbacks_used
        assert "used_audio_only_analysis" in context.processing_metrics.recovery_actions
        
        # Test checkpoint recovery
        recovered_context = context_manager.load_checkpoint(
            context.project_id, "before_video_analysis"
        )
        assert recovered_context is not None
        assert recovered_context.audio_transcript == context.audio_transcript
        
        # Verify workflow tracks the failure
        workflow_status = state_tracker.get_workflow_status(workflow.workflow_id)
        assert workflow_status['failed_tasks'] >= 1
        assert workflow_status['has_failures'] is True
    
    def test_cache_integration_with_context(self, system_components, temp_dir):
        """Test cache integration with ContentContext processing."""
        context_manager = system_components['context_manager']
        cache_manager = system_components['cache_manager']
        
        # Create context
        video_file = f"{temp_dir}/test_video.mp4"
        with open(video_file, 'w') as f:
            f.write("mock video content")
        
        context = context_manager.create_context(
            video_files=[video_file],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        # Cache API responses
        cache_manager.cache_api_response(
            "gemini", "analyze_content",
            {"text": "test content", "context_id": context.project_id},
            {"analysis": "educational content detected", "confidence": 0.92},
            cost=0.25
        )
        
        # Cache keyword research
        cache_manager.cache_keyword_research(
            ["education", "learning"], "educational",
            {"primary_keywords": ["education", "learning", "tutorial"]}
        )
        
        # Test cache retrieval
        cached_api_response = cache_manager.get_api_response(
            "gemini", "analyze_content",
            {"text": "test content", "context_id": context.project_id}
        )
        assert cached_api_response is not None
        assert cached_api_response["analysis"] == "educational content detected"
        
        cached_keywords = cache_manager.get_keyword_research(
            ["education", "learning"], "educational"
        )
        assert cached_keywords is not None
        assert "education" in cached_keywords["primary_keywords"]
        
        # Test cache invalidation by context
        cache_manager.cache_processing_result(
            context.project_id, "test_module", "test_stage", "test_result"
        )
        
        # Verify cached
        cached_result = cache_manager.get_processing_result(
            context.project_id, "test_module", "test_stage"
        )
        assert cached_result == "test_result"
        
        # Invalidate context cache
        invalidated_count = cache_manager.invalidate_context(context.project_id)
        assert invalidated_count >= 1
        
        # Verify invalidated
        cached_result_after = cache_manager.get_processing_result(
            context.project_id, "test_module", "test_stage"
        )
        assert cached_result_after is None
        
        # Check cache statistics
        cache_stats = cache_manager.get_stats()
        assert cache_stats['api_cost_saved'] == 0.25
        assert cache_stats['hits'] >= 2  # At least 2 successful retrievals
        assert cache_stats['misses'] >= 1  # At least 1 miss after invalidation