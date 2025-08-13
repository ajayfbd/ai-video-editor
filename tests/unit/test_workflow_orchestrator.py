"""
Unit tests for WorkflowOrchestrator.

Tests the end-to-end workflow orchestration functionality including
progress tracking, error recovery, resource monitoring, and CLI integration.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta

from ai_video_editor.core.workflow_orchestrator import (
    WorkflowOrchestrator,
    WorkflowConfiguration,
    WorkflowStage,
    ProcessingMode,
    StageProgress
)
from ai_video_editor.core.content_context import ContentContext, ContentType, UserPreferences
from ai_video_editor.core.config import ProjectSettings
from ai_video_editor.core.exceptions import (
    ContentContextError,
    ResourceConstraintError,
    MemoryConstraintError,
    ProcessingTimeoutError,
    APIIntegrationError
)


class TestWorkflowConfiguration:
    """Test WorkflowConfiguration class."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        config = WorkflowConfiguration()
        
        assert config.processing_mode == ProcessingMode.BALANCED
        assert config.enable_parallel_processing is True
        assert config.max_concurrent_stages == 2
        assert config.enable_caching is True
        assert config.enable_recovery is True
        assert config.checkpoint_frequency == 3
        assert config.timeout_per_stage == 1800
        assert config.max_memory_usage_gb == 8.0
        assert config.enable_progress_display is True
        assert config.skip_stages == []
    
    def test_custom_configuration(self):
        """Test custom configuration values."""
        config = WorkflowConfiguration(
            processing_mode=ProcessingMode.FAST,
            enable_parallel_processing=False,
            max_memory_usage_gb=4.0,
            timeout_per_stage=900,
            skip_stages=[WorkflowStage.BROLL_GENERATION]
        )
        
        assert config.processing_mode == ProcessingMode.FAST
        assert config.enable_parallel_processing is False
        assert config.max_memory_usage_gb == 4.0
        assert config.timeout_per_stage == 900
        assert WorkflowStage.BROLL_GENERATION in config.skip_stages


class TestStageProgress:
    """Test StageProgress class."""
    
    def test_stage_progress_initialization(self):
        """Test StageProgress initialization."""
        progress = StageProgress(stage=WorkflowStage.AUDIO_PROCESSING)
        
        assert progress.stage == WorkflowStage.AUDIO_PROCESSING
        assert progress.status == "pending"
        assert progress.start_time is None
        assert progress.end_time is None
        assert progress.progress_percentage == 0.0
        assert progress.current_operation == ""
        assert progress.error_message is None
        assert progress.metrics == {}
    
    def test_stage_progress_properties(self):
        """Test StageProgress properties."""
        progress = StageProgress(stage=WorkflowStage.AUDIO_PROCESSING)
        
        # Test initial state
        assert not progress.is_running
        assert not progress.is_completed
        assert not progress.has_failed
        assert progress.duration is None
        
        # Test running state
        progress.status = "running"
        progress.start_time = datetime.now()
        assert progress.is_running
        assert not progress.is_completed
        assert not progress.has_failed
        
        # Test completed state
        progress.status = "completed"
        progress.end_time = datetime.now()
        assert not progress.is_running
        assert progress.is_completed
        assert not progress.has_failed
        assert progress.duration is not None
        
        # Test failed state
        progress.status = "failed"
        assert not progress.is_running
        assert not progress.is_completed
        assert progress.has_failed


class TestWorkflowOrchestrator:
    """Test WorkflowOrchestrator class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def test_video_files(self, temp_dir):
        """Create test video files."""
        video_files = []
        for i in range(2):
            video_file = temp_dir / f"test_video_{i}.mp4"
            video_file.write_text(f"Mock video content {i}")
            video_files.append(str(video_file))
        return video_files
    
    @pytest.fixture
    def workflow_config(self, temp_dir):
        """Create test workflow configuration."""
        return WorkflowConfiguration(
            processing_mode=ProcessingMode.FAST,
            enable_parallel_processing=False,
            max_memory_usage_gb=4.0,
            timeout_per_stage=60,
            enable_progress_display=False,
            output_directory=temp_dir / "out",
            temp_directory=temp_dir / "temp"
        )
    
    @pytest.fixture
    def project_settings(self):
        """Create test project settings."""
        return ProjectSettings(
            content_type=ContentType.EDUCATIONAL,
            auto_enhance=True,
            enable_b_roll_generation=True,
            enable_thumbnail_generation=True
        )
    
    @pytest.fixture
    def user_preferences(self):
        """Create test user preferences."""
        return UserPreferences(
            quality_mode="balanced",
            batch_size=2,
            parallel_processing=True
        )
    
    @pytest.fixture
    def mock_console(self):
        """Create mock console for testing."""
        return Mock()
    
    def test_orchestrator_initialization(self, workflow_config, mock_console):
        """Test WorkflowOrchestrator initialization."""
        orchestrator = WorkflowOrchestrator(
            config=workflow_config,
            console=mock_console
        )
        
        assert orchestrator.config == workflow_config
        assert orchestrator.console == mock_console
        assert orchestrator.context is None
        assert orchestrator.current_stage is None
        assert orchestrator.workflow_start_time is None
        assert orchestrator.workflow_end_time is None
        assert len(orchestrator.stages) > 0
        assert all(isinstance(stage, WorkflowStage) for stage in orchestrator.stages.keys())
        assert all(isinstance(progress, StageProgress) for progress in orchestrator.stages.values())
    
    def test_orchestrator_default_initialization(self):
        """Test WorkflowOrchestrator with default configuration."""
        orchestrator = WorkflowOrchestrator()
        
        assert isinstance(orchestrator.config, WorkflowConfiguration)
        assert orchestrator.config.processing_mode == ProcessingMode.BALANCED
        assert orchestrator.console is not None
        assert len(orchestrator.stages) > 0
    
    @pytest.mark.asyncio
    async def test_create_content_context(
        self, 
        workflow_config, 
        mock_console, 
        test_video_files, 
        project_settings, 
        user_preferences
    ):
        """Test ContentContext creation."""
        orchestrator = WorkflowOrchestrator(
            config=workflow_config,
            console=mock_console
        )
        
        context = await orchestrator._create_content_context(
            input_files=test_video_files,
            project_settings=project_settings,
            user_preferences=user_preferences
        )
        
        assert isinstance(context, ContentContext)
        assert context.video_files == test_video_files
        assert context.content_type == ContentType.EDUCATIONAL
        assert context.user_preferences == user_preferences
        assert context.project_id.startswith("workflow_")
    
    @pytest.mark.asyncio
    async def test_create_content_context_missing_files(
        self, 
        workflow_config, 
        mock_console
    ):
        """Test ContentContext creation with missing files."""
        orchestrator = WorkflowOrchestrator(
            config=workflow_config,
            console=mock_console
        )
        
        with pytest.raises(ContentContextError, match="Input file not found"):
            await orchestrator._create_content_context(
                input_files=["nonexistent_file.mp4"],
                project_settings=None,
                user_preferences=None
            )
    
    def test_apply_processing_mode(self, workflow_config, mock_console):
        """Test processing mode application to user preferences."""
        orchestrator = WorkflowOrchestrator(
            config=workflow_config,
            console=mock_console
        )
        
        preferences = UserPreferences()
        orchestrator._apply_processing_mode(preferences)
        
        # Fast mode should be applied based on workflow_config
        assert preferences.quality_mode == "fast"
        assert preferences.batch_size == 1
        assert preferences.parallel_processing is False
        assert preferences.enable_aggressive_caching is True
    
    @pytest.mark.asyncio
    async def test_stage_execution_success(self, workflow_config, mock_console):
        """Test successful stage execution."""
        orchestrator = WorkflowOrchestrator(
            config=workflow_config,
            console=mock_console
        )
        
        # Mock a stage function
        mock_stage_func = AsyncMock()
        orchestrator._module_registry[WorkflowStage.INITIALIZATION] = mock_stage_func
        
        # Execute stage
        await orchestrator._execute_stage(WorkflowStage.INITIALIZATION)
        
        # Verify stage was called
        mock_stage_func.assert_called_once()
        
        # Verify stage progress
        stage_progress = orchestrator.stages[WorkflowStage.INITIALIZATION]
        assert stage_progress.status == "completed"
        assert stage_progress.progress_percentage == 100.0
        assert stage_progress.start_time is not None
        assert stage_progress.end_time is not None
        assert stage_progress.duration is not None
    
    @pytest.mark.asyncio
    async def test_stage_execution_failure(self, workflow_config, mock_console):
        """Test stage execution failure."""
        orchestrator = WorkflowOrchestrator(
            config=workflow_config,
            console=mock_console
        )
        
        # Mock a stage function that raises an exception
        mock_stage_func = AsyncMock(side_effect=Exception("Test error"))
        orchestrator._module_registry[WorkflowStage.INITIALIZATION] = mock_stage_func
        
        # Execute stage and expect failure
        with pytest.raises(ContentContextError, match="Stage initialization failed"):
            await orchestrator._execute_stage(WorkflowStage.INITIALIZATION)
        
        # Verify stage progress
        stage_progress = orchestrator.stages[WorkflowStage.INITIALIZATION]
        assert stage_progress.status == "failed"
        assert stage_progress.error_message is not None
        assert "Test error" in stage_progress.error_message
    
    @pytest.mark.asyncio
    async def test_stage_execution_timeout(self, workflow_config, mock_console):
        """Test stage execution timeout."""
        # Set very short timeout
        workflow_config.timeout_per_stage = 1
        
        orchestrator = WorkflowOrchestrator(
            config=workflow_config,
            console=mock_console
        )
        
        # Mock a stage function that takes too long
        async def slow_stage():
            await asyncio.sleep(2)
        
        orchestrator._module_registry[WorkflowStage.INITIALIZATION] = slow_stage
        
        # Execute stage and expect timeout
        with pytest.raises(ProcessingTimeoutError):
            await orchestrator._execute_stage(WorkflowStage.INITIALIZATION)
        
        # Verify stage progress
        stage_progress = orchestrator.stages[WorkflowStage.INITIALIZATION]
        assert stage_progress.status == "failed"
        assert "timed out" in stage_progress.error_message
    
    @pytest.mark.asyncio
    async def test_stage_recovery_success(self, workflow_config, mock_console):
        """Test successful stage recovery."""
        workflow_config.enable_recovery = True
        
        orchestrator = WorkflowOrchestrator(
            config=workflow_config,
            console=mock_console
        )
        
        # Create a context for recovery
        orchestrator.context = ContentContext(
            project_id="test",
            video_files=["test.mp4"],
            content_type=ContentType.GENERAL,
            user_preferences=UserPreferences()
        )
        
        # Mock stage function that fails first time, succeeds second time
        call_count = 0
        async def flaky_stage():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise APIIntegrationError("gemini", "test", "Temporary error")
            # Success on second call
        
        orchestrator._module_registry[WorkflowStage.INITIALIZATION] = flaky_stage
        
        # Mock the execute_stage method to avoid infinite recursion
        original_execute_stage = orchestrator._execute_stage
        
        async def mock_execute_stage(stage):
            if stage == WorkflowStage.INITIALIZATION and call_count == 2:
                # Simulate successful execution on retry
                stage_progress = orchestrator.stages[stage]
                stage_progress.status = "completed"
                stage_progress.progress_percentage = 100.0
                return
            return await original_execute_stage(stage)
        
        orchestrator._execute_stage = mock_execute_stage
        
        # Test recovery
        error = APIIntegrationError("gemini", "test", "Temporary error")
        recovery_success = await orchestrator._attempt_stage_recovery(
            WorkflowStage.INITIALIZATION, error
        )
        
        assert recovery_success is True
        assert orchestrator.recovery_attempts[WorkflowStage.INITIALIZATION] == 1
    
    @pytest.mark.asyncio
    async def test_stage_recovery_max_attempts(self, workflow_config, mock_console):
        """Test stage recovery with maximum attempts reached."""
        workflow_config.enable_recovery = True
        
        orchestrator = WorkflowOrchestrator(
            config=workflow_config,
            console=mock_console
        )
        orchestrator.max_recovery_attempts = 2
        
        # Set recovery attempts to maximum
        orchestrator.recovery_attempts[WorkflowStage.INITIALIZATION] = 2
        
        error = Exception("Persistent error")
        recovery_success = await orchestrator._attempt_stage_recovery(
            WorkflowStage.INITIALIZATION, error
        )
        
        assert recovery_success is False
    
    @pytest.mark.asyncio
    async def test_checkpoint_save_and_load(self, workflow_config, mock_console, temp_dir):
        """Test checkpoint saving and loading."""
        workflow_config.temp_directory = temp_dir
        
        orchestrator = WorkflowOrchestrator(
            config=workflow_config,
            console=mock_console
        )
        
        # Create test context
        context = ContentContext(
            project_id="test",
            video_files=["test.mp4"],
            content_type=ContentType.GENERAL,
            user_preferences=UserPreferences()
        )
        orchestrator.context = context
        
        # Save checkpoint
        await orchestrator._save_checkpoint("test_checkpoint")
        
        # Verify checkpoint was saved
        assert "test_checkpoint" in orchestrator.checkpoints
        assert orchestrator.checkpoints["test_checkpoint"].project_id == "test"
        
        # Verify checkpoint file was created
        checkpoint_file = temp_dir / "test_checkpoint.json"
        assert checkpoint_file.exists()
    
    @pytest.mark.asyncio
    @patch('psutil.Process')
    async def test_resource_monitoring(self, mock_process, workflow_config, mock_console):
        """Test resource monitoring functionality."""
        # Mock memory info
        mock_memory_info = Mock()
        mock_memory_info.rss = 4 * 1024**3  # 4GB
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        workflow_config.max_memory_usage_gb = 2.0  # Set low limit to trigger warning
        
        orchestrator = WorkflowOrchestrator(
            config=workflow_config,
            console=mock_console
        )
        
        # Create test context
        orchestrator.context = ContentContext(
            project_id="test",
            video_files=["test.mp4"],
            content_type=ContentType.GENERAL,
            user_preferences=UserPreferences()
        )
        
        # Start monitoring task
        monitor_task = asyncio.create_task(orchestrator._monitor_resources())
        
        # Let it run briefly
        await asyncio.sleep(0.1)
        
        # Cancel monitoring
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        
        # Verify memory pressure relief was applied
        assert orchestrator.context.user_preferences.batch_size == 1  # Should be reduced
        assert orchestrator.context.user_preferences.enable_aggressive_caching is True
        
        # Verify performance metrics were updated
        assert "current_memory_gb" in orchestrator.performance_metrics
        assert "peak_memory_gb" in orchestrator.performance_metrics
    
    def test_workflow_status(self, workflow_config, mock_console):
        """Test workflow status reporting."""
        orchestrator = WorkflowOrchestrator(
            config=workflow_config,
            console=mock_console
        )
        
        # Set some stage progress
        orchestrator.stages[WorkflowStage.INITIALIZATION].status = "completed"
        orchestrator.stages[WorkflowStage.INITIALIZATION].progress_percentage = 100.0
        orchestrator.stages[WorkflowStage.AUDIO_PROCESSING].status = "running"
        orchestrator.stages[WorkflowStage.AUDIO_PROCESSING].progress_percentage = 50.0
        orchestrator.current_stage = WorkflowStage.AUDIO_PROCESSING
        
        status = orchestrator.get_workflow_status()
        
        assert status["current_stage"] == "audio_processing"
        assert "stages" in status
        assert status["stages"]["initialization"]["status"] == "completed"
        assert status["stages"]["initialization"]["progress"] == 100.0
        assert status["stages"]["audio_processing"]["status"] == "running"
        assert status["stages"]["audio_processing"]["progress"] == 50.0
        assert "performance_metrics" in status
        assert "recovery_attempts" in status
    
    def test_processing_summary(self, workflow_config, mock_console):
        """Test processing summary generation."""
        orchestrator = WorkflowOrchestrator(
            config=workflow_config,
            console=mock_console
        )
        
        # Test without context
        summary = orchestrator.get_processing_summary()
        assert summary == {}
        
        # Create test context
        context = ContentContext(
            project_id="test_project",
            video_files=["test1.mp4", "test2.mp4"],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        orchestrator.context = context
        orchestrator.workflow_start_time = datetime.now()
        orchestrator.workflow_end_time = datetime.now() + timedelta(seconds=120)
        
        summary = orchestrator.get_processing_summary()
        
        assert summary["project_id"] == "test_project"
        assert summary["input_files"] == ["test1.mp4", "test2.mp4"]
        assert summary["content_type"] == "educational"
        assert "processing_metrics" in summary
        assert "cost_tracking" in summary
        assert summary["workflow_duration"] == 120.0
    
    @pytest.mark.asyncio
    async def test_full_workflow_execution(
        self, 
        workflow_config, 
        mock_console, 
        test_video_files, 
        project_settings, 
        user_preferences
    ):
        """Test complete workflow execution."""
        # Skip most stages for faster testing
        workflow_config.skip_stages = [
            WorkflowStage.VIDEO_PROCESSING,
            WorkflowStage.INTELLIGENCE_LAYER,
            WorkflowStage.CONTENT_ANALYSIS,
            WorkflowStage.BROLL_GENERATION,
            WorkflowStage.VIDEO_COMPOSITION,
            WorkflowStage.THUMBNAIL_GENERATION,
            WorkflowStage.METADATA_GENERATION
        ]
        
        orchestrator = WorkflowOrchestrator(
            config=workflow_config,
            console=mock_console
        )
        
        # Mock all stage functions to complete quickly
        for stage in orchestrator._module_registry:
            orchestrator._module_registry[stage] = AsyncMock()
        
        # Execute workflow
        result_context = await orchestrator.process_video(
            input_files=test_video_files,
            project_settings=project_settings,
            user_preferences=user_preferences
        )
        
        # Verify results
        assert isinstance(result_context, ContentContext)
        assert result_context.video_files == test_video_files
        assert result_context.content_type == ContentType.EDUCATIONAL
        assert orchestrator.workflow_start_time is not None
        assert orchestrator.workflow_end_time is not None
        
        # Verify stages that weren't skipped were executed
        executed_stages = [
            WorkflowStage.INITIALIZATION,
            WorkflowStage.INPUT_ANALYSIS,
            WorkflowStage.AUDIO_PROCESSING,
            WorkflowStage.FINALIZATION
        ]
        
        for stage in executed_stages:
            if stage not in workflow_config.skip_stages:
                assert orchestrator.stages[stage].status == "completed"
    
    @pytest.mark.asyncio
    async def test_workflow_failure_handling(
        self, 
        workflow_config, 
        mock_console, 
        test_video_files
    ):
        """Test workflow failure handling."""
        orchestrator = WorkflowOrchestrator(
            config=workflow_config,
            console=mock_console
        )
        
        # Mock initialization stage to fail
        orchestrator._module_registry[WorkflowStage.INITIALIZATION] = AsyncMock(
            side_effect=Exception("Critical failure")
        )
        
        # Execute workflow and expect failure
        with pytest.raises(ContentContextError):
            await orchestrator.process_video(
                input_files=test_video_files,
                project_settings=None,
                user_preferences=None
            )
        
        # Verify failure was handled
        assert orchestrator.workflow_end_time is not None
        assert orchestrator.stages[WorkflowStage.INITIALIZATION].status == "failed"


class TestWorkflowIntegration:
    """Integration tests for workflow orchestrator."""
    
    @pytest.mark.asyncio
    async def test_memory_constraint_handling(self):
        """Test handling of memory constraints during workflow."""
        config = WorkflowConfiguration(
            max_memory_usage_gb=0.1,  # Very low limit
            enable_recovery=True
        )
        
        orchestrator = WorkflowOrchestrator(config=config)
        
        # Create context
        context = ContentContext(
            project_id="test",
            video_files=["test.mp4"],
            content_type=ContentType.GENERAL,
            user_preferences=UserPreferences(batch_size=4)
        )
        orchestrator.context = context
        
        # Simulate memory constraint error
        error = MemoryConstraintError(current_usage=8000, limit=1000)
        
        recovery_success = await orchestrator._attempt_stage_recovery(
            WorkflowStage.AUDIO_PROCESSING, error
        )
        
        # Verify memory pressure relief was applied
        assert context.user_preferences.batch_size < 4  # Should be reduced
        assert context.user_preferences.enable_aggressive_caching is True
    
    @pytest.mark.asyncio
    async def test_api_error_recovery(self):
        """Test recovery from API errors."""
        config = WorkflowConfiguration(enable_recovery=True)
        orchestrator = WorkflowOrchestrator(config=config)
        
        # Create context
        context = ContentContext(
            project_id="test",
            video_files=["test.mp4"],
            content_type=ContentType.GENERAL,
            user_preferences=UserPreferences()
        )
        orchestrator.context = context
        
        # Simulate API error
        error = APIIntegrationError("gemini", "analyze", "Rate limit exceeded")
        
        # Mock successful retry
        orchestrator._execute_stage = AsyncMock()
        
        recovery_success = await orchestrator._attempt_stage_recovery(
            WorkflowStage.INTELLIGENCE_LAYER, error
        )
        
        # Verify recovery was attempted
        assert orchestrator.recovery_attempts[WorkflowStage.INTELLIGENCE_LAYER] == 1
    
    def test_workflow_configuration_validation(self):
        """Test workflow configuration validation."""
        # Test valid configuration
        config = WorkflowConfiguration(
            processing_mode=ProcessingMode.HIGH_QUALITY,
            max_concurrent_stages=4,
            max_memory_usage_gb=16.0
        )
        
        assert config.processing_mode == ProcessingMode.HIGH_QUALITY
        assert config.max_concurrent_stages == 4
        assert config.max_memory_usage_gb == 16.0
        
        # Test configuration with skip stages
        config_with_skips = WorkflowConfiguration(
            skip_stages=[WorkflowStage.BROLL_GENERATION, WorkflowStage.THUMBNAIL_GENERATION]
        )
        
        assert len(config_with_skips.skip_stages) == 2
        assert WorkflowStage.BROLL_GENERATION in config_with_skips.skip_stages
        assert WorkflowStage.THUMBNAIL_GENERATION in config_with_skips.skip_stages


if __name__ == "__main__":
    pytest.main([__file__])