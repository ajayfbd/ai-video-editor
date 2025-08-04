"""
Unit tests for ContextManager.

Tests the ContextManager class for lifecycle management and state tracking
with comprehensive mocking strategies.
"""

import pytest
import os
import json
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path

from ai_video_editor.core.context_manager import ContextManager
from ai_video_editor.core.content_context import ContentContext, ContentType, UserPreferences
from ai_video_editor.core.exceptions import ContentContextError


class TestContextManager:
    """Test ContextManager functionality."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def context_manager(self, temp_storage):
        """Create ContextManager with temporary storage."""
        return ContextManager(storage_path=temp_storage)
    
    @pytest.fixture
    def sample_video_files(self, temp_storage):
        """Create sample video files for testing."""
        video_files = []
        for i in range(2):
            video_path = os.path.join(temp_storage, f"test_video_{i}.mp4")
            with open(video_path, 'w') as f:
                f.write(f"mock video content {i}")
            video_files.append(video_path)
        return video_files
    
    def test_context_manager_initialization(self, temp_storage):
        """Test ContextManager initialization."""
        manager = ContextManager(storage_path=temp_storage)
        
        assert manager.storage_path == Path(temp_storage)
        assert manager.storage_path.exists()
        assert len(manager._active_contexts) == 0
    
    def test_create_context_success(self, context_manager, sample_video_files):
        """Test successful context creation."""
        user_prefs = UserPreferences(quality_mode="high")
        
        context = context_manager.create_context(
            video_files=sample_video_files,
            content_type=ContentType.EDUCATIONAL,
            user_preferences=user_prefs,
            project_id="test_project"
        )
        
        assert context.project_id == "test_project"
        assert context.video_files == sample_video_files
        assert context.content_type == ContentType.EDUCATIONAL
        assert context.user_preferences.quality_mode == "high"
        
        # Check that context is registered as active
        assert context.project_id in context_manager._active_contexts
        
        # Check that initial checkpoint was created
        assert "initial" in context._checkpoints
    
    def test_create_context_auto_id(self, context_manager, sample_video_files):
        """Test context creation with auto-generated ID."""
        context = context_manager.create_context(
            video_files=sample_video_files,
            content_type=ContentType.GENERAL,
            user_preferences=None  # Should use defaults
        )
        
        assert context.project_id != ""
        assert len(context.project_id) > 0
        assert isinstance(context.user_preferences, UserPreferences)
        assert context.user_preferences.quality_mode == "balanced"  # Default
    
    def test_create_context_missing_files_warning(self, context_manager, temp_storage):
        """Test context creation with missing video files (should warn but not fail)."""
        missing_files = [
            os.path.join(temp_storage, "missing1.mp4"),
            os.path.join(temp_storage, "missing2.mp4")
        ]
        
        with patch('ai_video_editor.core.context_manager.logger') as mock_logger:
            context = context_manager.create_context(
                video_files=missing_files,
                content_type=ContentType.MUSIC,
                user_preferences=UserPreferences()
            )
            
            # Should create context but log warnings
            assert context is not None
            assert mock_logger.warning.call_count == 2
            mock_logger.warning.assert_any_call("Video file not found: " + missing_files[0])
            mock_logger.warning.assert_any_call("Video file not found: " + missing_files[1])
    
    def test_get_context(self, context_manager, sample_video_files):
        """Test getting active context."""
        context = context_manager.create_context(
            video_files=sample_video_files,
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        retrieved_context = context_manager.get_context(context.project_id)
        assert retrieved_context is not None
        assert retrieved_context.project_id == context.project_id
        
        # Test getting non-existent context
        non_existent = context_manager.get_context("non_existent_id")
        assert non_existent is None
    
    def test_validate_context_success(self, context_manager, sample_video_files):
        """Test successful context validation."""
        context = context_manager.create_context(
            video_files=sample_video_files,
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        validation_result = context_manager.validate_context(context)
        
        assert validation_result['valid'] is True
        assert validation_result['score'] > 0.8  # Should have high score
        assert len(validation_result['issues']) == 0
    
    def test_validate_context_with_issues(self, context_manager):
        """Test context validation with issues."""
        # Create context with issues
        context = ContentContext(
            project_id="test_project",  # Valid project ID
            video_files=[],  # No video files - this should cause validation failure
            content_type=ContentType.GENERAL,
            user_preferences=UserPreferences()
        )
        
        validation_result = context_manager.validate_context(context)
        
        assert validation_result['valid'] is False
        assert validation_result['score'] == 0.0
        assert len(validation_result['issues']) > 0
        assert any("video files" in issue for issue in validation_result['issues'])
    
    def test_validate_context_with_warnings(self, context_manager, temp_storage):
        """Test context validation with warnings."""
        # Create context with missing files (warnings)
        missing_files = [os.path.join(temp_storage, "missing.mp4")]
        context = ContentContext(
            project_id="test_warnings",
            video_files=missing_files,
            content_type=ContentType.GENERAL,
            user_preferences=UserPreferences()
        )
        
        validation_result = context_manager.validate_context(context)
        
        assert validation_result['valid'] is True  # Valid but with warnings
        assert validation_result['score'] < 1.0  # Reduced score due to warnings
        assert len(validation_result['warnings']) > 0
        assert any("Missing video files" in warning for warning in validation_result['warnings'])
    
    def test_save_and_load_context(self, context_manager, sample_video_files):
        """Test saving and loading context."""
        # Create and save context
        original_context = context_manager.create_context(
            video_files=sample_video_files,
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences(quality_mode="high")
        )
        
        # Add some data to make it more interesting
        original_context.audio_transcript = "Test transcript"
        original_context.key_concepts = ["test", "concepts"]
        
        save_success = context_manager.save_context(original_context)
        assert save_success is True
        
        # Clear active contexts to simulate fresh load
        context_manager._active_contexts.clear()
        
        # Load context
        loaded_context = context_manager.load_context(original_context.project_id)
        
        assert loaded_context is not None
        assert loaded_context.project_id == original_context.project_id
        assert loaded_context.video_files == original_context.video_files
        assert loaded_context.content_type == original_context.content_type
        assert loaded_context.audio_transcript == original_context.audio_transcript
        assert loaded_context.key_concepts == original_context.key_concepts
        assert loaded_context.user_preferences.quality_mode == "high"
        
        # Should be registered as active again
        assert loaded_context.project_id in context_manager._active_contexts
    
    def test_load_nonexistent_context(self, context_manager):
        """Test loading non-existent context."""
        loaded_context = context_manager.load_context("nonexistent_id")
        assert loaded_context is None
    
    def test_save_checkpoint(self, context_manager, sample_video_files):
        """Test saving checkpoints."""
        context = context_manager.create_context(
            video_files=sample_video_files,
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        # Save checkpoint
        checkpoint_success = context_manager.save_checkpoint(context, "after_audio_analysis")
        assert checkpoint_success is True
        
        # Check that checkpoint was added to context
        assert "after_audio_analysis" in context._checkpoints
        
        # Check that checkpoint file exists
        checkpoint_dir = context_manager.storage_path / "checkpoints" / context.project_id
        assert checkpoint_dir.exists()
        
        checkpoint_files = list(checkpoint_dir.glob("after_audio_analysis_*.json"))
        assert len(checkpoint_files) == 1
    
    def test_load_checkpoint(self, context_manager, sample_video_files):
        """Test loading from checkpoint."""
        # Create context and add some data
        original_context = context_manager.create_context(
            video_files=sample_video_files,
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        original_context.audio_transcript = "Original transcript"
        original_context.key_concepts = ["original", "concepts"]
        
        # Save checkpoint
        context_manager.save_checkpoint(original_context, "test_checkpoint")
        
        # Modify context after checkpoint
        original_context.audio_transcript = "Modified transcript"
        original_context.key_concepts = ["modified", "concepts"]
        
        # Load from checkpoint
        restored_context = context_manager.load_checkpoint(
            original_context.project_id, "test_checkpoint"
        )
        
        assert restored_context is not None
        assert restored_context.project_id == original_context.project_id
        assert restored_context.audio_transcript == "Original transcript"  # From checkpoint
        assert restored_context.key_concepts == ["original", "concepts"]  # From checkpoint
    
    def test_load_nonexistent_checkpoint(self, context_manager, sample_video_files):
        """Test loading non-existent checkpoint."""
        context = context_manager.create_context(
            video_files=sample_video_files,
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        restored_context = context_manager.load_checkpoint(
            context.project_id, "nonexistent_checkpoint"
        )
        assert restored_context is None
    
    def test_list_checkpoints(self, context_manager, sample_video_files):
        """Test listing checkpoints."""
        context = context_manager.create_context(
            video_files=sample_video_files,
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        # Save multiple checkpoints
        context_manager.save_checkpoint(context, "checkpoint1")
        context_manager.save_checkpoint(context, "checkpoint2")
        context_manager.save_checkpoint(context, "checkpoint1")  # Same name, should be newer
        
        checkpoints = context_manager.list_checkpoints(context.project_id)
        
        assert len(checkpoints) == 3
        
        # Should be sorted by timestamp, most recent first
        assert checkpoints[0]['timestamp'] >= checkpoints[1]['timestamp']
        assert checkpoints[1]['timestamp'] >= checkpoints[2]['timestamp']
        
        # Check checkpoint data
        for checkpoint in checkpoints:
            assert 'name' in checkpoint
            assert 'timestamp' in checkpoint
            assert 'file_path' in checkpoint
            assert 'size' in checkpoint
            assert checkpoint['size'] > 0
    
    def test_cleanup_old_checkpoints(self, context_manager, sample_video_files):
        """Test cleaning up old checkpoints."""
        context = context_manager.create_context(
            video_files=sample_video_files,
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        # Save multiple checkpoints
        for i in range(5):
            context_manager.save_checkpoint(context, f"checkpoint_{i}")
        
        # Cleanup, keeping only 3
        removed_count = context_manager.cleanup_old_checkpoints(context.project_id, keep_count=3)
        
        # Should remove at least 2 oldest (might be more due to initial checkpoint)
        assert removed_count >= 2
        
        # Check remaining checkpoints
        remaining_checkpoints = context_manager.list_checkpoints(context.project_id)
        assert len(remaining_checkpoints) == 3
    
    def test_get_context_stats(self, context_manager, sample_video_files):
        """Test getting context statistics."""
        context = context_manager.create_context(
            video_files=sample_video_files,
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        # Add some data
        context.audio_transcript = "Test transcript"
        context.key_concepts = ["concept1", "concept2"]
        context.add_emotional_marker(30.0, "excitement", 0.8, 0.9, "test")
        context.processing_metrics.add_module_metrics("test_module", 10.5, 1000000)
        context.cost_tracking.add_cost("gemini", 0.50)
        
        stats = context_manager.get_context_stats(context)
        
        assert stats['project_id'] == context.project_id
        assert stats['content_type'] == context.content_type.value
        assert stats['video_files_count'] == len(sample_video_files)
        assert stats['emotional_markers_count'] == 1
        assert stats['key_concepts_count'] == 2
        assert stats['total_processing_time'] == 10.5
        assert stats['total_cost'] == 0.50
        assert stats['memory_peak_usage'] == 1000000
        assert stats['has_trending_keywords'] is False
        assert stats['has_competitor_insights'] is False
    
    def test_close_context(self, context_manager, sample_video_files):
        """Test closing context."""
        context = context_manager.create_context(
            video_files=sample_video_files,
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        project_id = context.project_id
        
        # Context should be active
        assert project_id in context_manager._active_contexts
        
        # Close context with save
        close_success = context_manager.close_context(project_id, save=True)
        assert close_success is True
        
        # Context should no longer be active
        assert project_id not in context_manager._active_contexts
        
        # But should be saved to disk
        context_file = context_manager.storage_path / f"{project_id}.json"
        assert context_file.exists()
    
    def test_close_context_without_save(self, context_manager, sample_video_files):
        """Test closing context without saving."""
        context = context_manager.create_context(
            video_files=sample_video_files,
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        project_id = context.project_id
        
        # Close context without save
        close_success = context_manager.close_context(project_id, save=False)
        assert close_success is True
        
        # Context should no longer be active
        assert project_id not in context_manager._active_contexts
        
        # Should not be saved to disk (only initial checkpoint exists)
        context_file = context_manager.storage_path / f"{project_id}.json"
        assert not context_file.exists()
    
    def test_close_nonexistent_context(self, context_manager):
        """Test closing non-existent context."""
        close_success = context_manager.close_context("nonexistent_id")
        assert close_success is False
    
    def test_list_active_contexts(self, context_manager, sample_video_files):
        """Test listing active contexts."""
        # Initially no active contexts
        active_contexts = context_manager.list_active_contexts()
        assert len(active_contexts) == 0
        
        # Create contexts
        context1 = context_manager.create_context(
            video_files=sample_video_files,
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        context2 = context_manager.create_context(
            video_files=sample_video_files,
            content_type=ContentType.MUSIC,
            user_preferences=UserPreferences()
        )
        
        # Should have 2 active contexts
        active_contexts = context_manager.list_active_contexts()
        assert len(active_contexts) == 2
        assert context1.project_id in active_contexts
        assert context2.project_id in active_contexts
        
        # Close one context
        context_manager.close_context(context1.project_id)
        
        # Should have 1 active context
        active_contexts = context_manager.list_active_contexts()
        assert len(active_contexts) == 1
        assert context2.project_id in active_contexts
    
    def test_get_storage_usage(self, context_manager, sample_video_files):
        """Test getting storage usage statistics."""
        # Create and save some contexts
        for i in range(3):
            context = context_manager.create_context(
                video_files=sample_video_files,
                content_type=ContentType.EDUCATIONAL,
                user_preferences=UserPreferences()
            )
            context.audio_transcript = f"Test transcript {i}"
            context_manager.save_context(context)
            context_manager.save_checkpoint(context, f"checkpoint_{i}")
        
        usage = context_manager.get_storage_usage()
        
        assert 'total_size_bytes' in usage
        assert 'total_size_mb' in usage
        assert 'context_files' in usage
        assert 'checkpoint_files' in usage
        assert 'storage_path' in usage
        
        assert usage['total_size_bytes'] > 0
        assert usage['total_size_mb'] > 0
        assert usage['context_files'] == 3
        assert usage['checkpoint_files'] >= 3  # At least 3 checkpoints
        assert usage['storage_path'] == str(context_manager.storage_path)
    
    @patch('ai_video_editor.core.context_manager.logger')
    def test_error_handling_in_save_context(self, mock_logger, context_manager, sample_video_files):
        """Test error handling in save_context."""
        context = context_manager.create_context(
            video_files=sample_video_files,
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        # Mock file write to raise exception
        with patch('builtins.open', side_effect=IOError("Disk full")):
            save_success = context_manager.save_context(context)
            
            assert save_success is False
            mock_logger.error.assert_called_once()
            assert "Failed to save ContentContext" in mock_logger.error.call_args[0][0]
    
    @patch('ai_video_editor.core.context_manager.logger')
    def test_error_handling_in_load_context(self, mock_logger, context_manager):
        """Test error handling in load_context."""
        # Try to load from corrupted file
        corrupted_file = context_manager.storage_path / "corrupted.json"
        with open(corrupted_file, 'w') as f:
            f.write("invalid json content")
        
        loaded_context = context_manager.load_context("corrupted")
        
        assert loaded_context is None
        mock_logger.error.assert_called_once()
        assert "Failed to load ContentContext" in mock_logger.error.call_args[0][0]
    
    def test_validation_with_emotional_markers(self, context_manager, sample_video_files):
        """Test validation with emotional markers."""
        context = context_manager.create_context(
            video_files=sample_video_files,
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        # Add valid emotional marker
        context.add_emotional_marker(30.0, "excitement", 0.8, 0.9, "test context")
        
        validation_result = context_manager.validate_context(context)
        assert validation_result['valid'] is True
        
        # Add invalid emotional marker (intensity out of range)
        context.emotional_markers.append(
            type(context.emotional_markers[0])(
                timestamp=60.0,
                emotion="invalid",
                intensity=1.5,  # Invalid: > 1.0
                confidence=0.9,
                context="test"
            )
        )
        
        validation_result = context_manager.validate_context(context)
        assert validation_result['valid'] is False
        assert any("intensity" in issue for issue in validation_result['issues'])
    
    def test_concurrent_access_safety(self, context_manager, sample_video_files):
        """Test thread safety of ContextManager operations."""
        import threading
        import time
        
        contexts_created = []
        errors = []
        
        def create_context_worker(worker_id):
            try:
                context = context_manager.create_context(
                    video_files=sample_video_files,
                    content_type=ContentType.EDUCATIONAL,
                    user_preferences=UserPreferences(),
                    project_id=f"worker_{worker_id}"
                )
                contexts_created.append(context.project_id)
                
                # Save context
                context_manager.save_context(context)
                
                # Save checkpoint
                context_manager.save_checkpoint(context, f"checkpoint_{worker_id}")
                
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_context_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(contexts_created) == 5
        assert len(set(contexts_created)) == 5  # All unique IDs
        
        # Check that all contexts are active
        active_contexts = context_manager.list_active_contexts()
        assert len(active_contexts) == 5