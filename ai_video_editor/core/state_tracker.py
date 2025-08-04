"""
StateTracker - Tracks processing progress and dependency management.

This module provides comprehensive state tracking for ContentContext processing,
including progress monitoring, dependency management, and workflow coordination.
"""

import logging
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import threading
import json

from .content_context import ContentContext
from .exceptions import ContentContextError


logger = logging.getLogger(__name__)


class ProcessingState(Enum):
    """Processing state enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority enumeration."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ProcessingTask:
    """Represents a processing task with dependencies and state."""
    
    task_id: str
    name: str
    module_name: str
    stage: str
    dependencies: List[str] = field(default_factory=list)
    state: ProcessingState = ProcessingState.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    progress: float = 0.0  # 0.0 to 1.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Get task duration if completed."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
    
    @property
    def is_ready(self) -> bool:
        """Check if task is ready to run (all dependencies completed)."""
        return self.state == ProcessingState.PENDING
    
    @property
    def is_running(self) -> bool:
        """Check if task is currently running."""
        return self.state == ProcessingState.RUNNING
    
    @property
    def is_completed(self) -> bool:
        """Check if task is completed successfully."""
        return self.state == ProcessingState.COMPLETED
    
    @property
    def is_failed(self) -> bool:
        """Check if task has failed."""
        return self.state == ProcessingState.FAILED
    
    def start(self):
        """Mark task as started."""
        self.state = ProcessingState.RUNNING
        self.started_at = datetime.now()
        self.progress = 0.0
    
    def complete(self):
        """Mark task as completed."""
        self.state = ProcessingState.COMPLETED
        self.completed_at = datetime.now()
        self.progress = 1.0
    
    def fail(self, error_message: str):
        """Mark task as failed."""
        self.state = ProcessingState.FAILED
        self.completed_at = datetime.now()
        self.error_message = error_message
    
    def skip(self, reason: str):
        """Mark task as skipped."""
        self.state = ProcessingState.SKIPPED
        self.completed_at = datetime.now()
        self.metadata['skip_reason'] = reason
    
    def cancel(self):
        """Mark task as cancelled."""
        self.state = ProcessingState.CANCELLED
        self.completed_at = datetime.now()
    
    def update_progress(self, progress: float):
        """Update task progress."""
        self.progress = max(0.0, min(1.0, progress))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            'task_id': self.task_id,
            'name': self.name,
            'module_name': self.module_name,
            'stage': self.stage,
            'dependencies': self.dependencies,
            'state': self.state.value,
            'priority': self.priority.value,
            'progress': self.progress,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingTask':
        """Create task from dictionary."""
        task = cls(
            task_id=data['task_id'],
            name=data['name'],
            module_name=data['module_name'],
            stage=data['stage'],
            dependencies=data.get('dependencies', []),
            state=ProcessingState(data.get('state', 'pending')),
            priority=TaskPriority(data.get('priority', 2)),
            progress=data.get('progress', 0.0),
            error_message=data.get('error_message'),
            metadata=data.get('metadata', {})
        )
        
        if data.get('started_at'):
            task.started_at = datetime.fromisoformat(data['started_at'])
        
        if data.get('completed_at'):
            task.completed_at = datetime.fromisoformat(data['completed_at'])
        
        return task


@dataclass
class ProcessingWorkflow:
    """Represents a complete processing workflow with tasks and dependencies."""
    
    workflow_id: str
    context_id: str
    tasks: Dict[str, ProcessingTask] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    state: ProcessingState = ProcessingState.PENDING
    
    def add_task(self, task: ProcessingTask):
        """Add task to workflow."""
        self.tasks[task.task_id] = task
    
    def get_ready_tasks(self) -> List[ProcessingTask]:
        """Get tasks that are ready to run."""
        ready_tasks = []
        
        for task in self.tasks.values():
            if task.state != ProcessingState.PENDING:
                continue
            
            # Check if all dependencies are completed
            dependencies_met = True
            for dep_id in task.dependencies:
                dep_task = self.tasks.get(dep_id)
                if not dep_task or not dep_task.is_completed:
                    dependencies_met = False
                    break
            
            if dependencies_met:
                ready_tasks.append(task)
        
        # Sort by priority
        ready_tasks.sort(key=lambda t: t.priority.value, reverse=True)
        return ready_tasks
    
    def get_running_tasks(self) -> List[ProcessingTask]:
        """Get currently running tasks."""
        return [task for task in self.tasks.values() if task.is_running]
    
    def get_completed_tasks(self) -> List[ProcessingTask]:
        """Get completed tasks."""
        return [task for task in self.tasks.values() if task.is_completed]
    
    def get_failed_tasks(self) -> List[ProcessingTask]:
        """Get failed tasks."""
        return [task for task in self.tasks.values() if task.is_failed]
    
    @property
    def total_progress(self) -> float:
        """Get overall workflow progress."""
        if not self.tasks:
            return 0.0
        
        total_progress = sum(task.progress for task in self.tasks.values())
        return total_progress / len(self.tasks)
    
    @property
    def is_completed(self) -> bool:
        """Check if workflow is completed."""
        return all(task.state in [ProcessingState.COMPLETED, ProcessingState.SKIPPED] 
                  for task in self.tasks.values())
    
    @property
    def has_failures(self) -> bool:
        """Check if workflow has any failures."""
        return any(task.is_failed for task in self.tasks.values())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow to dictionary."""
        return {
            'workflow_id': self.workflow_id,
            'context_id': self.context_id,
            'tasks': {task_id: task.to_dict() for task_id, task in self.tasks.items()},
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'state': self.state.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingWorkflow':
        """Create workflow from dictionary."""
        workflow = cls(
            workflow_id=data['workflow_id'],
            context_id=data['context_id'],
            created_at=datetime.fromisoformat(data['created_at']),
            state=ProcessingState(data.get('state', 'pending'))
        )
        
        if data.get('started_at'):
            workflow.started_at = datetime.fromisoformat(data['started_at'])
        
        if data.get('completed_at'):
            workflow.completed_at = datetime.fromisoformat(data['completed_at'])
        
        # Load tasks
        for task_id, task_data in data.get('tasks', {}).items():
            workflow.tasks[task_id] = ProcessingTask.from_dict(task_data)
        
        return workflow


class StateTracker:
    """
    Tracks processing progress and manages workflow dependencies.
    
    Provides comprehensive state tracking for ContentContext processing including:
    - Task dependency management
    - Progress monitoring
    - Workflow coordination
    - Error handling and recovery
    """
    
    def __init__(self):
        """Initialize StateTracker."""
        self.workflows: Dict[str, ProcessingWorkflow] = {}
        self.active_workflows: Set[str] = set()
        self.progress_callbacks: Dict[str, List[Callable]] = {}
        self.completion_callbacks: Dict[str, List[Callable]] = {}
        self.error_callbacks: Dict[str, List[Callable]] = {}
        self.lock = threading.RLock()
        
        logger.info("StateTracker initialized")
    
    def create_workflow(self, context: ContentContext) -> ProcessingWorkflow:
        """
        Create a new processing workflow for a ContentContext.
        
        Args:
            context: ContentContext to create workflow for
            
        Returns:
            Created ProcessingWorkflow
        """
        with self.lock:
            workflow_id = f"workflow_{context.project_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            workflow = ProcessingWorkflow(
                workflow_id=workflow_id,
                context_id=context.project_id
            )
            
            # Create standard processing tasks based on content type
            self._create_standard_tasks(workflow, context)
            
            self.workflows[workflow_id] = workflow
            
            logger.info(f"Created workflow {workflow_id} for context {context.project_id}")
            return workflow
    
    def _create_standard_tasks(self, workflow: ProcessingWorkflow, context: ContentContext):
        """Create standard processing tasks for the workflow."""
        
        # Input processing tasks
        audio_analysis = ProcessingTask(
            task_id="audio_analysis",
            name="Audio Analysis",
            module_name="input_processing",
            stage="audio_analysis",
            priority=TaskPriority.HIGH
        )
        workflow.add_task(audio_analysis)
        
        video_analysis = ProcessingTask(
            task_id="video_analysis",
            name="Video Analysis",
            module_name="input_processing",
            stage="video_analysis",
            priority=TaskPriority.HIGH
        )
        workflow.add_task(video_analysis)
        
        content_analysis = ProcessingTask(
            task_id="content_analysis",
            name="Content Analysis",
            module_name="input_processing",
            stage="content_analysis",
            dependencies=["audio_analysis"],
            priority=TaskPriority.HIGH
        )
        workflow.add_task(content_analysis)
        
        # Intelligence layer tasks
        keyword_research = ProcessingTask(
            task_id="keyword_research",
            name="Keyword Research",
            module_name="intelligence_layer",
            stage="keyword_research",
            dependencies=["content_analysis"],
            priority=TaskPriority.NORMAL
        )
        workflow.add_task(keyword_research)
        
        competitor_analysis = ProcessingTask(
            task_id="competitor_analysis",
            name="Competitor Analysis",
            module_name="intelligence_layer",
            stage="competitor_analysis",
            dependencies=["content_analysis"],
            priority=TaskPriority.LOW
        )
        workflow.add_task(competitor_analysis)
        
        # Output generation tasks (can run in parallel after intelligence layer)
        thumbnail_generation = ProcessingTask(
            task_id="thumbnail_generation",
            name="Thumbnail Generation",
            module_name="output_generation",
            stage="thumbnail_generation",
            dependencies=["keyword_research", "video_analysis"],
            priority=TaskPriority.HIGH
        )
        workflow.add_task(thumbnail_generation)
        
        metadata_generation = ProcessingTask(
            task_id="metadata_generation",
            name="Metadata Generation",
            module_name="output_generation",
            stage="metadata_generation",
            dependencies=["keyword_research", "content_analysis"],
            priority=TaskPriority.HIGH
        )
        workflow.add_task(metadata_generation)
        
        # Video processing (can run in parallel)
        video_processing = ProcessingTask(
            task_id="video_processing",
            name="Video Processing",
            module_name="video_processing",
            stage="video_processing",
            dependencies=["video_analysis", "content_analysis"],
            priority=TaskPriority.NORMAL
        )
        workflow.add_task(video_processing)
        
        # Final synchronization task
        output_synchronization = ProcessingTask(
            task_id="output_synchronization",
            name="Output Synchronization",
            module_name="output_generation",
            stage="synchronization",
            dependencies=["thumbnail_generation", "metadata_generation"],
            priority=TaskPriority.CRITICAL
        )
        workflow.add_task(output_synchronization)
    
    def get_workflow(self, workflow_id: str) -> Optional[ProcessingWorkflow]:
        """
        Get workflow by ID.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            ProcessingWorkflow or None if not found
        """
        with self.lock:
            return self.workflows.get(workflow_id)
    
    def get_workflow_by_context(self, context_id: str) -> Optional[ProcessingWorkflow]:
        """
        Get active workflow for a context.
        
        Args:
            context_id: ContentContext project ID
            
        Returns:
            Active ProcessingWorkflow or None if not found
        """
        with self.lock:
            for workflow in self.workflows.values():
                if workflow.context_id == context_id and workflow.workflow_id in self.active_workflows:
                    return workflow
            return None
    
    def start_workflow(self, workflow_id: str) -> bool:
        """
        Start a workflow.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            True if started successfully, False otherwise
        """
        with self.lock:
            workflow = self.workflows.get(workflow_id)
            if not workflow:
                logger.error(f"Workflow not found: {workflow_id}")
                return False
            
            workflow.state = ProcessingState.RUNNING
            workflow.started_at = datetime.now()
            self.active_workflows.add(workflow_id)
            
            logger.info(f"Started workflow: {workflow_id}")
            return True
    
    def start_task(self, workflow_id: str, task_id: str) -> bool:
        """
        Start a specific task in a workflow.
        
        Args:
            workflow_id: Workflow identifier
            task_id: Task identifier
            
        Returns:
            True if started successfully, False otherwise
        """
        with self.lock:
            workflow = self.workflows.get(workflow_id)
            if not workflow:
                logger.error(f"Workflow not found: {workflow_id}")
                return False
            
            task = workflow.tasks.get(task_id)
            if not task:
                logger.error(f"Task not found: {task_id}")
                return False
            
            if not task.is_ready:
                logger.error(f"Task not ready to start: {task_id}")
                return False
            
            task.start()
            
            # Trigger progress callbacks
            self._trigger_progress_callbacks(workflow_id, task_id, 0.0)
            
            logger.info(f"Started task {task_id} in workflow {workflow_id}")
            return True
    
    def complete_task(self, workflow_id: str, task_id: str) -> bool:
        """
        Mark a task as completed.
        
        Args:
            workflow_id: Workflow identifier
            task_id: Task identifier
            
        Returns:
            True if completed successfully, False otherwise
        """
        with self.lock:
            workflow = self.workflows.get(workflow_id)
            if not workflow:
                logger.error(f"Workflow not found: {workflow_id}")
                return False
            
            task = workflow.tasks.get(task_id)
            if not task:
                logger.error(f"Task not found: {task_id}")
                return False
            
            task.complete()
            
            # Trigger progress callbacks
            self._trigger_progress_callbacks(workflow_id, task_id, 1.0)
            
            # Check if workflow is completed
            if workflow.is_completed:
                self._complete_workflow(workflow_id)
            
            logger.info(f"Completed task {task_id} in workflow {workflow_id}")
            return True
    
    def fail_task(self, workflow_id: str, task_id: str, error_message: str) -> bool:
        """
        Mark a task as failed.
        
        Args:
            workflow_id: Workflow identifier
            task_id: Task identifier
            error_message: Error message
            
        Returns:
            True if marked as failed successfully, False otherwise
        """
        with self.lock:
            workflow = self.workflows.get(workflow_id)
            if not workflow:
                logger.error(f"Workflow not found: {workflow_id}")
                return False
            
            task = workflow.tasks.get(task_id)
            if not task:
                logger.error(f"Task not found: {task_id}")
                return False
            
            task.fail(error_message)
            
            # Trigger error callbacks
            self._trigger_error_callbacks(workflow_id, task_id, error_message)
            
            logger.error(f"Failed task {task_id} in workflow {workflow_id}: {error_message}")
            return True
    
    def update_task_progress(self, workflow_id: str, task_id: str, progress: float) -> bool:
        """
        Update task progress.
        
        Args:
            workflow_id: Workflow identifier
            task_id: Task identifier
            progress: Progress value (0.0 to 1.0)
            
        Returns:
            True if updated successfully, False otherwise
        """
        with self.lock:
            workflow = self.workflows.get(workflow_id)
            if not workflow:
                return False
            
            task = workflow.tasks.get(task_id)
            if not task:
                return False
            
            task.update_progress(progress)
            
            # Trigger progress callbacks
            self._trigger_progress_callbacks(workflow_id, task_id, progress)
            
            return True
    
    def get_next_tasks(self, workflow_id: str, max_tasks: int = 1) -> List[ProcessingTask]:
        """
        Get next tasks ready to run.
        
        Args:
            workflow_id: Workflow identifier
            max_tasks: Maximum number of tasks to return
            
        Returns:
            List of ready tasks
        """
        with self.lock:
            workflow = self.workflows.get(workflow_id)
            if not workflow:
                return []
            
            ready_tasks = workflow.get_ready_tasks()
            return ready_tasks[:max_tasks]
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get comprehensive workflow status.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            Dictionary with workflow status information
        """
        with self.lock:
            workflow = self.workflows.get(workflow_id)
            if not workflow:
                return {'error': 'Workflow not found'}
            
            return {
                'workflow_id': workflow_id,
                'context_id': workflow.context_id,
                'state': workflow.state.value,
                'total_progress': workflow.total_progress,
                'created_at': workflow.created_at.isoformat(),
                'started_at': workflow.started_at.isoformat() if workflow.started_at else None,
                'completed_at': workflow.completed_at.isoformat() if workflow.completed_at else None,
                'total_tasks': len(workflow.tasks),
                'completed_tasks': len(workflow.get_completed_tasks()),
                'running_tasks': len(workflow.get_running_tasks()),
                'failed_tasks': len(workflow.get_failed_tasks()),
                'ready_tasks': len(workflow.get_ready_tasks()),
                'is_completed': workflow.is_completed,
                'has_failures': workflow.has_failures,
                'tasks': {task_id: {
                    'name': task.name,
                    'state': task.state.value,
                    'progress': task.progress,
                    'module_name': task.module_name,
                    'stage': task.stage,
                    'dependencies': task.dependencies,
                    'error_message': task.error_message
                } for task_id, task in workflow.tasks.items()}
            }
    
    def _complete_workflow(self, workflow_id: str):
        """Complete a workflow."""
        workflow = self.workflows[workflow_id]
        workflow.state = ProcessingState.COMPLETED
        workflow.completed_at = datetime.now()
        
        if workflow_id in self.active_workflows:
            self.active_workflows.remove(workflow_id)
        
        # Trigger completion callbacks
        self._trigger_completion_callbacks(workflow_id)
        
        logger.info(f"Completed workflow: {workflow_id}")
    
    def _trigger_progress_callbacks(self, workflow_id: str, task_id: str, progress: float):
        """Trigger progress callbacks."""
        callbacks = self.progress_callbacks.get(workflow_id, [])
        for callback in callbacks:
            try:
                callback(workflow_id, task_id, progress)
            except Exception as e:
                logger.error(f"Progress callback failed: {str(e)}")
    
    def _trigger_completion_callbacks(self, workflow_id: str):
        """Trigger completion callbacks."""
        callbacks = self.completion_callbacks.get(workflow_id, [])
        for callback in callbacks:
            try:
                callback(workflow_id)
            except Exception as e:
                logger.error(f"Completion callback failed: {str(e)}")
    
    def _trigger_error_callbacks(self, workflow_id: str, task_id: str, error_message: str):
        """Trigger error callbacks."""
        callbacks = self.error_callbacks.get(workflow_id, [])
        for callback in callbacks:
            try:
                callback(workflow_id, task_id, error_message)
            except Exception as e:
                logger.error(f"Error callback failed: {str(e)}")
    
    def add_progress_callback(self, workflow_id: str, callback: Callable):
        """Add progress callback for a workflow."""
        if workflow_id not in self.progress_callbacks:
            self.progress_callbacks[workflow_id] = []
        self.progress_callbacks[workflow_id].append(callback)
    
    def add_completion_callback(self, workflow_id: str, callback: Callable):
        """Add completion callback for a workflow."""
        if workflow_id not in self.completion_callbacks:
            self.completion_callbacks[workflow_id] = []
        self.completion_callbacks[workflow_id].append(callback)
    
    def add_error_callback(self, workflow_id: str, callback: Callable):
        """Add error callback for a workflow."""
        if workflow_id not in self.error_callbacks:
            self.error_callbacks[workflow_id] = []
        self.error_callbacks[workflow_id].append(callback)
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """
        Cancel a workflow.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        with self.lock:
            workflow = self.workflows.get(workflow_id)
            if not workflow:
                return False
            
            # Cancel all pending and running tasks
            for task in workflow.tasks.values():
                if task.state in [ProcessingState.PENDING, ProcessingState.RUNNING]:
                    task.cancel()
            
            workflow.state = ProcessingState.CANCELLED
            workflow.completed_at = datetime.now()
            
            if workflow_id in self.active_workflows:
                self.active_workflows.remove(workflow_id)
            
            logger.info(f"Cancelled workflow: {workflow_id}")
            return True
    
    def cleanup_completed_workflows(self, max_age_hours: int = 24) -> int:
        """
        Clean up old completed workflows.
        
        Args:
            max_age_hours: Maximum age in hours for completed workflows
            
        Returns:
            Number of workflows cleaned up
        """
        with self.lock:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            workflows_to_remove = []
            
            for workflow_id, workflow in self.workflows.items():
                if (workflow.state in [ProcessingState.COMPLETED, ProcessingState.CANCELLED] and
                    workflow.completed_at and workflow.completed_at < cutoff_time):
                    workflows_to_remove.append(workflow_id)
            
            for workflow_id in workflows_to_remove:
                del self.workflows[workflow_id]
                
                # Clean up callbacks
                self.progress_callbacks.pop(workflow_id, None)
                self.completion_callbacks.pop(workflow_id, None)
                self.error_callbacks.pop(workflow_id, None)
            
            logger.info(f"Cleaned up {len(workflows_to_remove)} old workflows")
            return len(workflows_to_remove)
    
    def get_active_workflows(self) -> List[str]:
        """Get list of active workflow IDs."""
        with self.lock:
            return list(self.active_workflows)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get state tracker statistics.
        
        Returns:
            Dictionary with statistics
        """
        with self.lock:
            total_workflows = len(self.workflows)
            active_workflows = len(self.active_workflows)
            completed_workflows = sum(1 for w in self.workflows.values() 
                                    if w.state == ProcessingState.COMPLETED)
            failed_workflows = sum(1 for w in self.workflows.values() 
                                 if w.has_failures)
            
            total_tasks = sum(len(w.tasks) for w in self.workflows.values())
            completed_tasks = sum(len(w.get_completed_tasks()) for w in self.workflows.values())
            running_tasks = sum(len(w.get_running_tasks()) for w in self.workflows.values())
            failed_tasks = sum(len(w.get_failed_tasks()) for w in self.workflows.values())
            
            return {
                'total_workflows': total_workflows,
                'active_workflows': active_workflows,
                'completed_workflows': completed_workflows,
                'failed_workflows': failed_workflows,
                'total_tasks': total_tasks,
                'completed_tasks': completed_tasks,
                'running_tasks': running_tasks,
                'failed_tasks': failed_tasks,
                'completion_rate': completed_tasks / total_tasks if total_tasks > 0 else 0.0,
                'failure_rate': failed_tasks / total_tasks if total_tasks > 0 else 0.0
            }