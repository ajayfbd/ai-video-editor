"""
Unit tests for StateTracker.

Tests the StateTracker class for processing progress and dependency management
with comprehensive workflow scenarios.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from ai_video_editor.core.state_tracker import (
    StateTracker, ProcessingTask, ProcessingWorkflow, ProcessingState,
    TaskPriority
)
from ai_video_editor.core.content_context import ContentContext, ContentType, UserPreferences


class TestProcessingTask:
    """Test ProcessingTask functionality."""
    
    def test_processing_task_creation(self):
        """Test creating a ProcessingTask."""
        task = ProcessingTask(
            task_id="test_task",
            name="Test Task",
            module_name="test_module",
            stage="test_stage",
            dependencies=["dep1", "dep2"],
            priority=TaskPriority.HIGH
        )
        
        assert task.task_id == "test_task"
        assert task.name == "Test Task"
        assert task.module_name == "test_module"
        assert task.stage == "test_stage"
        assert task.dependencies == ["dep1", "dep2"]
        assert task.state == ProcessingState.PENDING
        assert task.priority == TaskPriority.HIGH
        assert task.progress == 0.0
        assert task.started_at is None
        assert task.completed_at is None
    
    def test_processing_task_properties(self):
        """Test ProcessingTask properties."""
        task = ProcessingTask("test", "Test", "module", "stage")
        
        # Initial state
        assert task.is_ready is True
        assert task.is_running is False
        assert task.is_completed is False
        assert task.is_failed is False
        assert task.duration is None
        
        # Start task
        task.start()
        assert task.is_ready is False
        assert task.is_running is True
        assert task.is_completed is False
        assert task.started_at is not None
        assert task.progress == 0.0
        
        # Complete task
        task.complete()
        assert task.is_running is False
        assert task.is_completed is True
        assert task.completed_at is not None
        assert task.progress == 1.0
        assert task.duration is not None
    
    def test_processing_task_failure(self):
        """Test ProcessingTask failure handling."""
        task = ProcessingTask("test", "Test", "module", "stage")
        
        task.start()
        task.fail("Test error message")
        
        assert task.is_failed is True
        assert task.is_running is False
        assert task.error_message == "Test error message"
        assert task.completed_at is not None
    
    def test_processing_task_skip(self):
        """Test ProcessingTask skip functionality."""
        task = ProcessingTask("test", "Test", "module", "stage")
        
        task.skip("Dependency failed")
        
        assert task.state == ProcessingState.SKIPPED
        assert task.completed_at is not None
        assert task.metadata["skip_reason"] == "Dependency failed"
    
    def test_processing_task_cancel(self):
        """Test ProcessingTask cancellation."""
        task = ProcessingTask("test", "Test", "module", "stage")
        
        task.start()
        task.cancel()
        
        assert task.state == ProcessingState.CANCELLED
        assert task.completed_at is not None
    
    def test_processing_task_update_progress(self):
        """Test ProcessingTask progress updates."""
        task = ProcessingTask("test", "Test", "module", "stage")
        
        task.update_progress(0.5)
        assert task.progress == 0.5
        
        # Test bounds
        task.update_progress(-0.1)
        assert task.progress == 0.0
        
        task.update_progress(1.5)
        assert task.progress == 1.0
    
    def test_processing_task_serialization(self):
        """Test ProcessingTask serialization."""
        task = ProcessingTask(
            task_id="test_task",
            name="Test Task",
            module_name="test_module",
            stage="test_stage",
            dependencies=["dep1"],
            priority=TaskPriority.HIGH
        )
        
        task.start()
        task.update_progress(0.7)
        task.metadata["custom_data"] = "test_value"
        
        # Test to_dict
        task_dict = task.to_dict()
        
        assert task_dict["task_id"] == "test_task"
        assert task_dict["name"] == "Test Task"
        assert task_dict["state"] == "running"
        assert task_dict["priority"] == 3  # HIGH priority value
        assert task_dict["progress"] == 0.7
        assert task_dict["dependencies"] == ["dep1"]
        assert task_dict["metadata"]["custom_data"] == "test_value"
        
        # Test from_dict
        restored_task = ProcessingTask.from_dict(task_dict)
        
        assert restored_task.task_id == task.task_id
        assert restored_task.name == task.name
        assert restored_task.state == task.state
        assert restored_task.priority == task.priority
        assert restored_task.progress == task.progress
        assert restored_task.dependencies == task.dependencies
        assert restored_task.metadata == task.metadata
        assert restored_task.started_at == task.started_at


class TestProcessingWorkflow:
    """Test ProcessingWorkflow functionality."""
    
    def test_processing_workflow_creation(self):
        """Test creating a ProcessingWorkflow."""
        workflow = ProcessingWorkflow(
            workflow_id="test_workflow",
            context_id="test_context"
        )
        
        assert workflow.workflow_id == "test_workflow"
        assert workflow.context_id == "test_context"
        assert len(workflow.tasks) == 0
        assert workflow.state == ProcessingState.PENDING
        assert workflow.started_at is None
        assert workflow.completed_at is None
    
    def test_processing_workflow_add_task(self):
        """Test adding tasks to workflow."""
        workflow = ProcessingWorkflow("test_workflow", "test_context")
        
        task1 = ProcessingTask("task1", "Task 1", "module1", "stage1")
        task2 = ProcessingTask("task2", "Task 2", "module2", "stage2", dependencies=["task1"])
        
        workflow.add_task(task1)
        workflow.add_task(task2)
        
        assert len(workflow.tasks) == 2
        assert "task1" in workflow.tasks
        assert "task2" in workflow.tasks
        assert workflow.tasks["task2"].dependencies == ["task1"]
    
    def test_processing_workflow_get_ready_tasks(self):
        """Test getting ready tasks from workflow."""
        workflow = ProcessingWorkflow("test_workflow", "test_context")
        
        # Create tasks with dependencies
        task1 = ProcessingTask("task1", "Task 1", "module1", "stage1")  # No dependencies
        task2 = ProcessingTask("task2", "Task 2", "module2", "stage2", dependencies=["task1"])
        task3 = ProcessingTask("task3", "Task 3", "module3", "stage3")  # No dependencies
        task4 = ProcessingTask("task4", "Task 4", "module4", "stage4", dependencies=["task2", "task3"])
        
        workflow.add_task(task1)
        workflow.add_task(task2)
        workflow.add_task(task3)
        workflow.add_task(task4)
        
        # Initially, only tasks without dependencies should be ready
        ready_tasks = workflow.get_ready_tasks()
        ready_ids = [task.task_id for task in ready_tasks]
        
        assert len(ready_tasks) == 2
        assert "task1" in ready_ids
        assert "task3" in ready_ids
        
        # Complete task1
        task1.complete()
        
        # Now task2 should be ready
        ready_tasks = workflow.get_ready_tasks()
        ready_ids = [task.task_id for task in ready_tasks]
        
        assert "task2" in ready_ids
        assert "task4" not in ready_ids  # Still waiting for task3
        
        # Complete task2 and task3
        task2.complete()
        task3.complete()
        
        # Now task4 should be ready
        ready_tasks = workflow.get_ready_tasks()
        ready_ids = [task.task_id for task in ready_tasks]
        
        assert "task4" in ready_ids
    
    def test_processing_workflow_priority_ordering(self):
        """Test that ready tasks are ordered by priority."""
        workflow = ProcessingWorkflow("test_workflow", "test_context")
        
        # Create tasks with different priorities
        task_low = ProcessingTask("low", "Low Priority", "module", "stage", priority=TaskPriority.LOW)
        task_high = ProcessingTask("high", "High Priority", "module", "stage", priority=TaskPriority.HIGH)
        task_normal = ProcessingTask("normal", "Normal Priority", "module", "stage", priority=TaskPriority.NORMAL)
        
        workflow.add_task(task_low)
        workflow.add_task(task_high)
        workflow.add_task(task_normal)
        
        ready_tasks = workflow.get_ready_tasks()
        
        # Should be ordered by priority (high to low)
        assert ready_tasks[0].task_id == "high"
        assert ready_tasks[1].task_id == "normal"
        assert ready_tasks[2].task_id == "low"
    
    def test_processing_workflow_progress(self):
        """Test workflow progress calculation."""
        workflow = ProcessingWorkflow("test_workflow", "test_context")
        
        task1 = ProcessingTask("task1", "Task 1", "module1", "stage1")
        task2 = ProcessingTask("task2", "Task 2", "module2", "stage2")
        task3 = ProcessingTask("task3", "Task 3", "module3", "stage3")
        
        workflow.add_task(task1)
        workflow.add_task(task2)
        workflow.add_task(task3)
        
        # Initial progress should be 0
        assert workflow.total_progress == 0.0
        
        # Update task progress
        task1.update_progress(1.0)  # Complete
        task2.update_progress(0.5)  # Half done
        task3.update_progress(0.0)  # Not started
        
        # Total progress should be average: (1.0 + 0.5 + 0.0) / 3 = 0.5
        assert workflow.total_progress == 0.5
    
    def test_processing_workflow_completion_status(self):
        """Test workflow completion status."""
        workflow = ProcessingWorkflow("test_workflow", "test_context")
        
        task1 = ProcessingTask("task1", "Task 1", "module1", "stage1")
        task2 = ProcessingTask("task2", "Task 2", "module2", "stage2")
        
        workflow.add_task(task1)
        workflow.add_task(task2)
        
        # Initially not completed
        assert workflow.is_completed is False
        assert workflow.has_failures is False
        
        # Complete one task
        task1.complete()
        assert workflow.is_completed is False
        
        # Complete all tasks
        task2.complete()
        assert workflow.is_completed is True
        
        # Test with failure
        task2.fail("Test error")
        assert workflow.has_failures is True
        
        # Test with skipped task
        task2.skip("Test skip")
        assert workflow.is_completed is True  # Skipped tasks count as completed
    
    def test_processing_workflow_serialization(self):
        """Test ProcessingWorkflow serialization."""
        workflow = ProcessingWorkflow("test_workflow", "test_context")
        
        task = ProcessingTask("task1", "Task 1", "module1", "stage1")
        workflow.add_task(task)
        
        workflow.state = ProcessingState.RUNNING
        workflow.started_at = datetime.now()
        
        # Test to_dict
        workflow_dict = workflow.to_dict()
        
        assert workflow_dict["workflow_id"] == "test_workflow"
        assert workflow_dict["context_id"] == "test_context"
        assert workflow_dict["state"] == "running"
        assert "task1" in workflow_dict["tasks"]
        
        # Test from_dict
        restored_workflow = ProcessingWorkflow.from_dict(workflow_dict)
        
        assert restored_workflow.workflow_id == workflow.workflow_id
        assert restored_workflow.context_id == workflow.context_id
        assert restored_workflow.state == workflow.state
        assert len(restored_workflow.tasks) == 1
        assert "task1" in restored_workflow.tasks


class TestStateTracker:
    """Test StateTracker functionality."""
    
    @pytest.fixture
    def state_tracker(self):
        """Create StateTracker instance."""
        return StateTracker()
    
    @pytest.fixture
    def sample_context(self):
        """Create sample ContentContext."""
        return ContentContext(
            project_id="test_context",
            video_files=["test_video.mp4"],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
    
    def test_state_tracker_initialization(self, state_tracker):
        """Test StateTracker initialization."""
        assert len(state_tracker.workflows) == 0
        assert len(state_tracker.active_workflows) == 0
        assert len(state_tracker.progress_callbacks) == 0
        assert len(state_tracker.completion_callbacks) == 0
        assert len(state_tracker.error_callbacks) == 0
    
    def test_create_workflow(self, state_tracker, sample_context):
        """Test creating a workflow."""
        workflow = state_tracker.create_workflow(sample_context)
        
        assert workflow.context_id == sample_context.project_id
        assert workflow.workflow_id in state_tracker.workflows
        assert len(workflow.tasks) > 0  # Should have standard tasks
        
        # Check that standard tasks are created
        task_ids = list(workflow.tasks.keys())
        expected_tasks = [
            "audio_analysis", "video_analysis", "content_analysis",
            "keyword_research", "competitor_analysis", "thumbnail_generation",
            "metadata_generation", "video_processing", "output_synchronization"
        ]
        
        for expected_task in expected_tasks:
            assert expected_task in task_ids
    
    def test_workflow_task_dependencies(self, state_tracker, sample_context):
        """Test that workflow tasks have correct dependencies."""
        workflow = state_tracker.create_workflow(sample_context)
        
        # Check specific dependencies
        content_analysis = workflow.tasks["content_analysis"]
        assert "audio_analysis" in content_analysis.dependencies
        
        keyword_research = workflow.tasks["keyword_research"]
        assert "content_analysis" in keyword_research.dependencies
        
        thumbnail_generation = workflow.tasks["thumbnail_generation"]
        assert "keyword_research" in thumbnail_generation.dependencies
        assert "video_analysis" in thumbnail_generation.dependencies
        
        output_sync = workflow.tasks["output_synchronization"]
        assert "thumbnail_generation" in output_sync.dependencies
        assert "metadata_generation" in output_sync.dependencies
    
    def test_get_workflow(self, state_tracker, sample_context):
        """Test getting workflow by ID."""
        workflow = state_tracker.create_workflow(sample_context)
        
        retrieved_workflow = state_tracker.get_workflow(workflow.workflow_id)
        assert retrieved_workflow is workflow
        
        # Test getting non-existent workflow
        non_existent = state_tracker.get_workflow("non_existent_id")
        assert non_existent is None
    
    def test_get_workflow_by_context(self, state_tracker, sample_context):
        """Test getting workflow by context ID."""
        workflow = state_tracker.create_workflow(sample_context)
        state_tracker.start_workflow(workflow.workflow_id)
        
        retrieved_workflow = state_tracker.get_workflow_by_context(sample_context.project_id)
        assert retrieved_workflow is workflow
        
        # Test getting non-existent context workflow
        non_existent = state_tracker.get_workflow_by_context("non_existent_context")
        assert non_existent is None
    
    def test_start_workflow(self, state_tracker, sample_context):
        """Test starting a workflow."""
        workflow = state_tracker.create_workflow(sample_context)
        
        success = state_tracker.start_workflow(workflow.workflow_id)
        
        assert success is True
        assert workflow.state == ProcessingState.RUNNING
        assert workflow.started_at is not None
        assert workflow.workflow_id in state_tracker.active_workflows
    
    def test_start_nonexistent_workflow(self, state_tracker):
        """Test starting non-existent workflow."""
        success = state_tracker.start_workflow("non_existent_id")
        assert success is False
    
    def test_start_task(self, state_tracker, sample_context):
        """Test starting a task."""
        workflow = state_tracker.create_workflow(sample_context)
        
        # Start a task without dependencies
        success = state_tracker.start_task(workflow.workflow_id, "audio_analysis")
        
        assert success is True
        
        task = workflow.tasks["audio_analysis"]
        assert task.state == ProcessingState.RUNNING
        assert task.started_at is not None
    
    def test_start_task_with_unmet_dependencies(self, state_tracker, sample_context):
        """Test starting a task with unmet dependencies."""
        workflow = state_tracker.create_workflow(sample_context)
        
        # Try to start a task that depends on another task
        success = state_tracker.start_task(workflow.workflow_id, "content_analysis")
        
        assert success is False  # Should fail because audio_analysis is not completed
    
    def test_complete_task(self, state_tracker, sample_context):
        """Test completing a task."""
        workflow = state_tracker.create_workflow(sample_context)
        
        # Start and complete a task
        state_tracker.start_task(workflow.workflow_id, "audio_analysis")
        success = state_tracker.complete_task(workflow.workflow_id, "audio_analysis")
        
        assert success is True
        
        task = workflow.tasks["audio_analysis"]
        assert task.state == ProcessingState.COMPLETED
        assert task.completed_at is not None
        assert task.progress == 1.0
    
    def test_fail_task(self, state_tracker, sample_context):
        """Test failing a task."""
        workflow = state_tracker.create_workflow(sample_context)
        
        # Start and fail a task
        state_tracker.start_task(workflow.workflow_id, "audio_analysis")
        success = state_tracker.fail_task(workflow.workflow_id, "audio_analysis", "Test error")
        
        assert success is True
        
        task = workflow.tasks["audio_analysis"]
        assert task.state == ProcessingState.FAILED
        assert task.error_message == "Test error"
        assert task.completed_at is not None
    
    def test_update_task_progress(self, state_tracker, sample_context):
        """Test updating task progress."""
        workflow = state_tracker.create_workflow(sample_context)
        
        # Start task and update progress
        state_tracker.start_task(workflow.workflow_id, "audio_analysis")
        success = state_tracker.update_task_progress(workflow.workflow_id, "audio_analysis", 0.7)
        
        assert success is True
        
        task = workflow.tasks["audio_analysis"]
        assert task.progress == 0.7
    
    def test_get_next_tasks(self, state_tracker, sample_context):
        """Test getting next ready tasks."""
        workflow = state_tracker.create_workflow(sample_context)
        
        # Initially, tasks without dependencies should be ready
        next_tasks = state_tracker.get_next_tasks(workflow.workflow_id, max_tasks=3)
        
        assert len(next_tasks) > 0
        
        # All returned tasks should be ready
        for task in next_tasks:
            assert task.is_ready
    
    def test_get_workflow_status(self, state_tracker, sample_context):
        """Test getting workflow status."""
        workflow = state_tracker.create_workflow(sample_context)
        state_tracker.start_workflow(workflow.workflow_id)
        
        # Start and complete some tasks
        state_tracker.start_task(workflow.workflow_id, "audio_analysis")
        state_tracker.complete_task(workflow.workflow_id, "audio_analysis")
        
        status = state_tracker.get_workflow_status(workflow.workflow_id)
        
        assert status['workflow_id'] == workflow.workflow_id
        assert status['context_id'] == sample_context.project_id
        assert status['state'] == ProcessingState.RUNNING.value
        assert status['total_tasks'] > 0
        assert status['completed_tasks'] == 1
        assert status['running_tasks'] == 0
        assert status['failed_tasks'] == 0
        assert 'tasks' in status
        assert 'audio_analysis' in status['tasks']
    
    def test_workflow_completion(self, state_tracker, sample_context):
        """Test workflow completion when all tasks are done."""
        workflow = state_tracker.create_workflow(sample_context)
        state_tracker.start_workflow(workflow.workflow_id)
        
        # Complete all tasks
        for task_id in workflow.tasks:
            task = workflow.tasks[task_id]
            if task.is_ready:
                state_tracker.start_task(workflow.workflow_id, task_id)
                state_tracker.complete_task(workflow.workflow_id, task_id)
        
        # Continue until all tasks are completed
        max_iterations = 20  # Prevent infinite loop
        iteration = 0
        
        while not workflow.is_completed and iteration < max_iterations:
            ready_tasks = state_tracker.get_next_tasks(workflow.workflow_id, max_tasks=5)
            for task in ready_tasks:
                state_tracker.start_task(workflow.workflow_id, task.task_id)
                state_tracker.complete_task(workflow.workflow_id, task.task_id)
            iteration += 1
        
        assert workflow.is_completed
        assert workflow.state == ProcessingState.COMPLETED
        assert workflow.completed_at is not None
        assert workflow.workflow_id not in state_tracker.active_workflows
    
    def test_cancel_workflow(self, state_tracker, sample_context):
        """Test cancelling a workflow."""
        workflow = state_tracker.create_workflow(sample_context)
        state_tracker.start_workflow(workflow.workflow_id)
        
        # Start some tasks
        state_tracker.start_task(workflow.workflow_id, "audio_analysis")
        
        success = state_tracker.cancel_workflow(workflow.workflow_id)
        
        assert success is True
        assert workflow.state == ProcessingState.CANCELLED
        assert workflow.completed_at is not None
        assert workflow.workflow_id not in state_tracker.active_workflows
        
        # All tasks should be cancelled
        for task in workflow.tasks.values():
            if task.state in [ProcessingState.PENDING, ProcessingState.RUNNING]:
                assert task.state == ProcessingState.CANCELLED
    
    def test_progress_callbacks(self, state_tracker, sample_context):
        """Test progress callbacks."""
        workflow = state_tracker.create_workflow(sample_context)
        
        callback_calls = []
        
        def progress_callback(workflow_id, task_id, progress):
            callback_calls.append((workflow_id, task_id, progress))
        
        state_tracker.add_progress_callback(workflow.workflow_id, progress_callback)
        
        # Start task and update progress
        state_tracker.start_task(workflow.workflow_id, "audio_analysis")
        state_tracker.update_task_progress(workflow.workflow_id, "audio_analysis", 0.5)
        state_tracker.complete_task(workflow.workflow_id, "audio_analysis")
        
        # Should have received callbacks
        assert len(callback_calls) >= 2  # At least start (0.0) and complete (1.0)
        
        # Check callback data
        for workflow_id, task_id, progress in callback_calls:
            assert workflow_id == workflow.workflow_id
            assert task_id == "audio_analysis"
            assert 0.0 <= progress <= 1.0
    
    def test_completion_callbacks(self, state_tracker, sample_context):
        """Test completion callbacks."""
        workflow = state_tracker.create_workflow(sample_context)
        
        completion_called = []
        
        def completion_callback(workflow_id):
            completion_called.append(workflow_id)
        
        state_tracker.add_completion_callback(workflow.workflow_id, completion_callback)
        
        # Complete all tasks to trigger workflow completion
        # (This is a simplified version - in practice would need to complete all dependencies)
        for task in workflow.tasks.values():
            task.complete()
        
        # Manually trigger completion
        state_tracker._complete_workflow(workflow.workflow_id)
        
        assert len(completion_called) == 1
        assert completion_called[0] == workflow.workflow_id
    
    def test_error_callbacks(self, state_tracker, sample_context):
        """Test error callbacks."""
        workflow = state_tracker.create_workflow(sample_context)
        
        error_calls = []
        
        def error_callback(workflow_id, task_id, error_message):
            error_calls.append((workflow_id, task_id, error_message))
        
        state_tracker.add_error_callback(workflow.workflow_id, error_callback)
        
        # Start and fail a task
        state_tracker.start_task(workflow.workflow_id, "audio_analysis")
        state_tracker.fail_task(workflow.workflow_id, "audio_analysis", "Test error")
        
        assert len(error_calls) == 1
        workflow_id, task_id, error_message = error_calls[0]
        assert workflow_id == workflow.workflow_id
        assert task_id == "audio_analysis"
        assert error_message == "Test error"
    
    def test_cleanup_completed_workflows(self, state_tracker, sample_context):
        """Test cleaning up old completed workflows."""
        # Create and complete a workflow
        workflow = state_tracker.create_workflow(sample_context)
        
        # Manually set completion time to past
        workflow.state = ProcessingState.COMPLETED
        workflow.completed_at = datetime.now() - timedelta(hours=25)  # 25 hours ago
        
        # Create another recent workflow
        recent_context = ContentContext(
            project_id="recent_context",
            video_files=["recent_video.mp4"],
            content_type=ContentType.GENERAL,
            user_preferences=UserPreferences()
        )
        recent_workflow = state_tracker.create_workflow(recent_context)
        recent_workflow.state = ProcessingState.COMPLETED
        recent_workflow.completed_at = datetime.now() - timedelta(hours=1)  # 1 hour ago
        
        initial_count = len(state_tracker.workflows)
        
        # Cleanup workflows older than 24 hours
        cleaned_count = state_tracker.cleanup_completed_workflows(max_age_hours=24)
        
        assert cleaned_count == 1
        assert len(state_tracker.workflows) == initial_count - 1
        assert workflow.workflow_id not in state_tracker.workflows
        assert recent_workflow.workflow_id in state_tracker.workflows
    
    def test_get_statistics(self, state_tracker, sample_context):
        """Test getting state tracker statistics."""
        # Create some workflows with different states
        workflow1 = state_tracker.create_workflow(sample_context)
        workflow1.state = ProcessingState.COMPLETED
        
        context2 = ContentContext(
            project_id="context2",
            video_files=["video2.mp4"],
            content_type=ContentType.MUSIC,
            user_preferences=UserPreferences()
        )
        workflow2 = state_tracker.create_workflow(context2)
        state_tracker.start_workflow(workflow2.workflow_id)
        
        # Complete some tasks
        state_tracker.start_task(workflow2.workflow_id, "audio_analysis")
        state_tracker.complete_task(workflow2.workflow_id, "audio_analysis")
        
        # Fail a task
        state_tracker.start_task(workflow2.workflow_id, "video_analysis")
        state_tracker.fail_task(workflow2.workflow_id, "video_analysis", "Test error")
        
        stats = state_tracker.get_statistics()
        
        assert stats['total_workflows'] == 2
        assert stats['active_workflows'] == 1
        assert stats['completed_workflows'] == 1
        assert stats['failed_workflows'] == 1  # workflow2 has failures
        assert stats['total_tasks'] > 0
        assert stats['completed_tasks'] >= 1
        assert stats['failed_tasks'] >= 1
        assert 'completion_rate' in stats
        assert 'failure_rate' in stats
    
    def test_concurrent_workflow_operations(self, state_tracker):
        """Test concurrent workflow operations."""
        results = []
        errors = []
        
        def workflow_worker(worker_id):
            try:
                context = ContentContext(
                    project_id=f"context_{worker_id}",
                    video_files=[f"video_{worker_id}.mp4"],
                    content_type=ContentType.GENERAL,
                    user_preferences=UserPreferences()
                )
                
                workflow = state_tracker.create_workflow(context)
                state_tracker.start_workflow(workflow.workflow_id)
                
                # Start and complete a task
                state_tracker.start_task(workflow.workflow_id, "audio_analysis")
                state_tracker.complete_task(workflow.workflow_id, "audio_analysis")
                
                results.append(worker_id)
                
            except Exception as e:
                errors.append(f"Worker {worker_id}: {str(e)}")
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=workflow_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
        assert len(state_tracker.workflows) == 5
        assert len(state_tracker.active_workflows) == 5