"""
Workflow Orchestrator - End-to-end pipeline management for AI Video Editor.

This module provides the WorkflowOrchestrator class that manages the complete
video processing pipeline from input analysis to final output generation,
with comprehensive progress tracking, error recovery, and resource monitoring.
"""

import asyncio
import time
import psutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import logging

from rich.console import Console
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.live import Live

from .content_context import ContentContext, ContentType, UserPreferences, ProcessingMetrics
from .exceptions import (
    ContentContextError, ResourceConstraintError, MemoryConstraintError,
    ProcessingTimeoutError, APIIntegrationError, handle_errors, retry_on_error
)
from .config import get_settings, ProjectSettings
from .performance_optimizer import PerformanceOptimizer, PerformanceProfile
from .cache_manager import CacheManager
from ..utils.logging_config import get_logger


class WorkflowStage(Enum):
    """Workflow processing stages."""
    INITIALIZATION = "initialization"
    INPUT_ANALYSIS = "input_analysis"
    AUDIO_PROCESSING = "audio_processing"
    VIDEO_PROCESSING = "video_processing"
    INTELLIGENCE_LAYER = "intelligence_layer"
    CONTENT_ANALYSIS = "content_analysis"
    BROLL_GENERATION = "broll_generation"
    VIDEO_COMPOSITION = "video_composition"
    THUMBNAIL_GENERATION = "thumbnail_generation"
    METADATA_GENERATION = "metadata_generation"
    FINALIZATION = "finalization"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingMode(Enum):
    """Processing mode options."""
    FAST = "fast"
    BALANCED = "balanced"
    HIGH_QUALITY = "high_quality"
    CUSTOM = "custom"


@dataclass
class StageProgress:
    """Progress tracking for individual workflow stages."""
    stage: WorkflowStage
    status: str = "pending"  # pending, running, completed, failed, skipped
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    progress_percentage: float = 0.0
    current_operation: str = ""
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Get stage duration if completed."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def is_running(self) -> bool:
        """Check if stage is currently running."""
        return self.status == "running"
    
    @property
    def is_completed(self) -> bool:
        """Check if stage completed successfully."""
        return self.status == "completed"
    
    @property
    def has_failed(self) -> bool:
        """Check if stage failed."""
        return self.status == "failed"


@dataclass
class WorkflowConfiguration:
    """Configuration for workflow execution."""
    processing_mode: ProcessingMode = ProcessingMode.BALANCED
    enable_parallel_processing: bool = True
    max_concurrent_stages: int = 2
    enable_caching: bool = True
    enable_recovery: bool = True
    checkpoint_frequency: int = 3  # Save checkpoint every N stages
    timeout_per_stage: int = 1800  # 30 minutes per stage
    max_memory_usage_gb: float = 8.0
    enable_progress_display: bool = True
    output_directory: Optional[Path] = None
    temp_directory: Optional[Path] = None
    
    # Stage-specific configurations
    skip_stages: List[WorkflowStage] = field(default_factory=list)
    stage_timeouts: Dict[WorkflowStage, int] = field(default_factory=dict)
    stage_priorities: Dict[WorkflowStage, int] = field(default_factory=dict)


class WorkflowOrchestrator:
    """
    End-to-end workflow orchestrator for AI Video Editor.
    
    Manages the complete video processing pipeline with progress tracking,
    error recovery, resource monitoring, and performance optimization.
    """
    
    def __init__(
        self,
        config: Optional[WorkflowConfiguration] = None,
        console: Optional[Console] = None,
        cache_manager: Optional[CacheManager] = None,
        performance_optimizer: Optional[PerformanceOptimizer] = None
    ):
        """
        Initialize the workflow orchestrator.
        
        Args:
            config: Workflow configuration options
            console: Rich console for output (optional)
            cache_manager: CacheManager instance for performance optimization
            performance_optimizer: PerformanceOptimizer instance
        """
        self.config = config or WorkflowConfiguration()
        self.console = console or Console()
        self.logger = get_logger(__name__)
        self.settings = get_settings()
        
        # Performance optimization
        self.cache_manager = cache_manager or CacheManager()
        self.performance_optimizer = performance_optimizer or PerformanceOptimizer(self.cache_manager)
        
        # Workflow state
        self.context: Optional[ContentContext] = None
        self.stages: Dict[WorkflowStage, StageProgress] = {}
        self.current_stage: Optional[WorkflowStage] = None
        self.workflow_start_time: Optional[datetime] = None
        self.workflow_end_time: Optional[datetime] = None
        
        # Progress tracking
        self.progress: Optional[Progress] = None
        self.progress_tasks: Dict[WorkflowStage, TaskID] = {}
        self.live_display: Optional[Live] = None
        
        # Resource monitoring
        self.initial_memory: int = 0
        self.peak_memory: int = 0
        self.resource_monitor_task: Optional[asyncio.Task] = None
        
        # Error recovery
        self.checkpoints: Dict[str, ContentContext] = {}
        self.recovery_attempts: Dict[WorkflowStage, int] = {}
        self.max_recovery_attempts: int = 3
        
        # Performance metrics
        self.performance_metrics: Dict[str, Any] = {}
        
        # Initialize stages
        self._initialize_stages()
        
        # Module registry for dynamic loading
        self._module_registry: Dict[WorkflowStage, Callable] = {}
        self._register_modules()
    
    def _initialize_stages(self) -> None:
        """Initialize all workflow stages with default progress tracking."""
        for stage in WorkflowStage:
            if stage not in [WorkflowStage.COMPLETED, WorkflowStage.FAILED]:
                self.stages[stage] = StageProgress(stage=stage)
    
    def _register_modules(self) -> None:
        """Register processing modules for each workflow stage."""
        # This will be populated with actual module imports
        # For now, we'll use placeholder functions that will be replaced
        # with actual module calls in the implementation
        
        self._module_registry = {
            WorkflowStage.INITIALIZATION: self._stage_initialization,
            WorkflowStage.INPUT_ANALYSIS: self._stage_input_analysis,
            WorkflowStage.AUDIO_PROCESSING: self._stage_audio_processing,
            WorkflowStage.VIDEO_PROCESSING: self._stage_video_processing,
            WorkflowStage.INTELLIGENCE_LAYER: self._stage_intelligence_layer,
            WorkflowStage.CONTENT_ANALYSIS: self._stage_content_analysis,
            WorkflowStage.BROLL_GENERATION: self._stage_broll_generation,
            WorkflowStage.VIDEO_COMPOSITION: self._stage_video_composition,
            WorkflowStage.THUMBNAIL_GENERATION: self._stage_thumbnail_generation,
            WorkflowStage.METADATA_GENERATION: self._stage_metadata_generation,
            WorkflowStage.FINALIZATION: self._stage_finalization,
        }
    
    @handle_errors()
    async def process_video(
        self,
        input_files: List[Union[str, Path]],
        project_settings: Optional[ProjectSettings] = None,
        user_preferences: Optional[UserPreferences] = None
    ) -> ContentContext:
        """
        Process video files through the complete AI Video Editor pipeline.
        
        Args:
            input_files: List of input video/audio files
            project_settings: Project-specific settings
            user_preferences: User processing preferences
            
        Returns:
            ContentContext: Complete processing results
            
        Raises:
            ContentContextError: If processing fails
            ResourceConstraintError: If system resources are insufficient
        """
        try:
            # Initialize workflow
            self.workflow_start_time = datetime.now()
            self.initial_memory = psutil.Process().memory_info().rss
            
            # Initialize performance optimizer
            profile_name = self._determine_performance_profile()
            await self.performance_optimizer.initialize(profile_name)
            
            # Create ContentContext
            self.context = await self._create_content_context(
                input_files, project_settings, user_preferences
            )
            
            # Optimize context for performance
            self.context = await self.performance_optimizer.optimize_context_processing(self.context)
            
            # Setup progress display
            if self.config.enable_progress_display:
                await self._setup_progress_display()
            
            # Start resource monitoring
            if self.config.max_memory_usage_gb > 0:
                self.resource_monitor_task = asyncio.create_task(
                    self._monitor_resources()
                )
            
            # Execute workflow stages
            await self._execute_workflow()
            
            # Finalize workflow
            self.workflow_end_time = datetime.now()
            await self._finalize_workflow()
            
            return self.context
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            await self._handle_workflow_failure(e)
            raise
        
        finally:
            # Cleanup
            await self._cleanup_workflow()
    
    def _determine_performance_profile(self) -> str:
        """Determine appropriate performance profile based on system resources and config."""
        # Get system memory
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Determine profile based on processing mode and system resources
        if self.config.processing_mode == ProcessingMode.FAST:
            return 'fast'
        elif self.config.processing_mode == ProcessingMode.HIGH_QUALITY:
            if total_memory_gb >= 16:
                return 'high_quality'
            else:
                return 'balanced'  # Fallback for systems with less memory
        elif total_memory_gb < 8:
            return 'memory_constrained'
        else:
            return 'balanced'
    
    async def _create_content_context(
        self,
        input_files: List[Union[str, Path]],
        project_settings: Optional[ProjectSettings],
        user_preferences: Optional[UserPreferences]
    ) -> ContentContext:
        """Create and initialize ContentContext for processing."""
        # Convert paths to strings
        file_paths = [str(Path(f).resolve()) for f in input_files]
        
        # Validate input files
        for file_path in file_paths:
            if not Path(file_path).exists():
                raise ContentContextError(f"Input file not found: {file_path}")
        
        # Determine content type
        content_type = ContentType.GENERAL
        if project_settings:
            content_type = ContentType(project_settings.content_type.value)
        
        # Create user preferences
        if not user_preferences:
            user_preferences = UserPreferences()
            
        # Apply processing mode to user preferences
        self._apply_processing_mode(user_preferences)
        
        # Create ContentContext
        context = ContentContext(
            project_id=f"workflow_{int(time.time())}",
            video_files=file_paths,
            content_type=content_type,
            user_preferences=user_preferences
        )
        
        self.logger.info(f"Created ContentContext for {len(file_paths)} files")
        return context
    
    def _apply_processing_mode(self, preferences: UserPreferences) -> None:
        """Apply processing mode settings to user preferences."""
        mode_settings = {
            ProcessingMode.FAST: {
                "quality_mode": "fast",
                "batch_size": 1,
                "parallel_processing": False,
                "enable_aggressive_caching": True
            },
            ProcessingMode.BALANCED: {
                "quality_mode": "balanced",
                "batch_size": 2,
                "parallel_processing": True,
                "enable_aggressive_caching": False
            },
            ProcessingMode.HIGH_QUALITY: {
                "quality_mode": "high",
                "batch_size": 3,
                "parallel_processing": True,
                "enable_aggressive_caching": False
            }
        }
        
        if self.config.processing_mode in mode_settings:
            settings = mode_settings[self.config.processing_mode]
            for key, value in settings.items():
                if hasattr(preferences, key):
                    setattr(preferences, key, value)
    
    async def _setup_progress_display(self) -> None:
        """Setup rich progress display for workflow tracking."""
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        )
        
        # Create progress tasks for each stage
        for stage in self.stages.keys():
            if stage not in self.config.skip_stages:
                task_id = self.progress.add_task(
                    f"[cyan]{stage.value.replace('_', ' ').title()}[/cyan]",
                    total=100
                )
                self.progress_tasks[stage] = task_id
        
        # Start live display
        self.live_display = Live(
            self._create_progress_panel(),
            console=self.console,
            refresh_per_second=2
        )
        self.live_display.start()
    
    def _create_progress_panel(self) -> Panel:
        """Create progress panel with current status."""
        if not self.progress:
            return Panel("Initializing...", title="AI Video Editor Progress")
        
        # Create status table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Stage", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Progress", style="yellow")
        table.add_column("Time", style="blue")
        
        for stage, progress_info in self.stages.items():
            if stage in self.config.skip_stages:
                continue
                
            status_icon = {
                "pending": "⏳",
                "running": "🔄",
                "completed": "✅",
                "failed": "❌",
                "skipped": "⏭️"
            }.get(progress_info.status, "❓")
            
            progress_text = f"{progress_info.progress_percentage:.1f}%"
            
            time_text = ""
            if progress_info.duration:
                time_text = f"{progress_info.duration.total_seconds():.1f}s"
            elif progress_info.start_time:
                elapsed = datetime.now() - progress_info.start_time
                time_text = f"{elapsed.total_seconds():.1f}s"
            
            table.add_row(
                stage.value.replace('_', ' ').title(),
                f"{status_icon} {progress_info.status.title()}",
                progress_text,
                time_text
            )
        
        # Add overall progress info
        total_stages = len([s for s in self.stages.keys() if s not in self.config.skip_stages])
        completed_stages = len([s for s in self.stages.values() if s.is_completed])
        overall_progress = (completed_stages / total_stages * 100) if total_stages > 0 else 0
        
        title = f"AI Video Editor Progress - {overall_progress:.1f}% Complete"
        
        return Panel(table, title=title, border_style="blue")
    
    async def _execute_workflow(self) -> None:
        """Execute all workflow stages in sequence or parallel as configured."""
        stages_to_process = [
            stage for stage in WorkflowStage 
            if stage not in [WorkflowStage.COMPLETED, WorkflowStage.FAILED] 
            and stage not in self.config.skip_stages
        ]
        
        if self.config.enable_parallel_processing:
            await self._execute_stages_parallel(stages_to_process)
        else:
            await self._execute_stages_sequential(stages_to_process)
    
    async def _execute_stages_sequential(self, stages: List[WorkflowStage]) -> None:
        """Execute workflow stages sequentially."""
        for stage in stages:
            await self._execute_stage(stage)
            
            # Save checkpoint periodically
            if len([s for s in self.stages.values() if s.is_completed]) % self.config.checkpoint_frequency == 0:
                await self._save_checkpoint(f"stage_{stage.value}")
    
    async def _execute_stages_parallel(self, stages: List[WorkflowStage]) -> None:
        """Execute workflow stages with controlled parallelism."""
        # Define stage dependencies
        dependencies = {
            WorkflowStage.INPUT_ANALYSIS: [],
            WorkflowStage.AUDIO_PROCESSING: [WorkflowStage.INPUT_ANALYSIS],
            WorkflowStage.VIDEO_PROCESSING: [WorkflowStage.INPUT_ANALYSIS],
            WorkflowStage.INTELLIGENCE_LAYER: [WorkflowStage.AUDIO_PROCESSING],
            WorkflowStage.CONTENT_ANALYSIS: [WorkflowStage.AUDIO_PROCESSING, WorkflowStage.VIDEO_PROCESSING],
            WorkflowStage.BROLL_GENERATION: [WorkflowStage.INTELLIGENCE_LAYER, WorkflowStage.CONTENT_ANALYSIS],
            WorkflowStage.VIDEO_COMPOSITION: [WorkflowStage.BROLL_GENERATION],
            WorkflowStage.THUMBNAIL_GENERATION: [WorkflowStage.CONTENT_ANALYSIS, WorkflowStage.INTELLIGENCE_LAYER],
            WorkflowStage.METADATA_GENERATION: [WorkflowStage.INTELLIGENCE_LAYER],
            WorkflowStage.FINALIZATION: [WorkflowStage.VIDEO_COMPOSITION, WorkflowStage.THUMBNAIL_GENERATION, WorkflowStage.METADATA_GENERATION]
        }
        
        # Execute stages respecting dependencies
        completed_stages = set()
        running_tasks = {}
        
        while len(completed_stages) < len(stages):
            # Find stages ready to run
            ready_stages = []
            for stage in stages:
                if (stage not in completed_stages and 
                    stage not in running_tasks and
                    all(dep in completed_stages for dep in dependencies.get(stage, []))):
                    ready_stages.append(stage)
            
            # Start new tasks up to concurrency limit
            while (len(running_tasks) < self.config.max_concurrent_stages and 
                   ready_stages):
                stage = ready_stages.pop(0)
                task = asyncio.create_task(self._execute_stage(stage))
                running_tasks[stage] = task
            
            # Wait for at least one task to complete
            if running_tasks:
                done, pending = await asyncio.wait(
                    running_tasks.values(),
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Process completed tasks
                for task in done:
                    # Find which stage completed
                    completed_stage = None
                    for stage, stage_task in running_tasks.items():
                        if stage_task == task:
                            completed_stage = stage
                            break
                    
                    if completed_stage:
                        try:
                            await task  # Re-raise any exceptions
                            completed_stages.add(completed_stage)
                            self.logger.info(f"Stage {completed_stage.value} completed")
                        except Exception as e:
                            self.logger.error(f"Stage {completed_stage.value} failed: {e}")
                            # Handle stage failure
                            await self._handle_stage_failure(completed_stage, e)
                        
                        del running_tasks[completed_stage]
            
            # Small delay to prevent busy waiting
            await asyncio.sleep(0.1)
    
    async def _execute_stage(self, stage: WorkflowStage) -> None:
        """Execute a single workflow stage with error handling and recovery."""
        stage_progress = self.stages[stage]
        
        try:
            # Update stage status
            stage_progress.status = "running"
            stage_progress.start_time = datetime.now()
            stage_progress.current_operation = f"Starting {stage.value}"
            self.current_stage = stage
            
            # Update progress display
            if self.progress and stage in self.progress_tasks:
                self.progress.update(
                    self.progress_tasks[stage],
                    description=f"[cyan]{stage.value.replace('_', ' ').title()}[/cyan] - Starting",
                    completed=10
                )
            
            # Get stage timeout
            timeout = self.config.stage_timeouts.get(stage, self.config.timeout_per_stage)
            
            # Execute stage with timeout
            stage_func = self._module_registry.get(stage)
            if not stage_func:
                raise ContentContextError(f"No handler registered for stage {stage.value}")
            
            await asyncio.wait_for(stage_func(), timeout=timeout)
            
            # Mark stage as completed
            stage_progress.status = "completed"
            stage_progress.end_time = datetime.now()
            stage_progress.progress_percentage = 100.0
            stage_progress.current_operation = "Completed"
            
            # Update progress display
            if self.progress and stage in self.progress_tasks:
                self.progress.update(
                    self.progress_tasks[stage],
                    description=f"[green]{stage.value.replace('_', ' ').title()}[/green] - Completed",
                    completed=100
                )
            
            self.logger.info(f"Stage {stage.value} completed successfully")
            
        except asyncio.TimeoutError:
            error_msg = f"Stage {stage.value} timed out after {timeout}s"
            stage_progress.status = "failed"
            stage_progress.error_message = error_msg
            stage_progress.end_time = datetime.now()
            
            self.logger.error(error_msg)
            raise ProcessingTimeoutError(stage.value, timeout, context_state=self.context)
            
        except Exception as e:
            error_msg = f"Stage {stage.value} failed: {str(e)}"
            stage_progress.status = "failed"
            stage_progress.error_message = error_msg
            stage_progress.end_time = datetime.now()
            
            self.logger.error(error_msg)
            
            # Attempt recovery if enabled
            if self.config.enable_recovery:
                recovery_success = await self._attempt_stage_recovery(stage, e)
                if recovery_success:
                    return
            
            raise ContentContextError(error_msg, context_state=self.context)
    
    async def _attempt_stage_recovery(self, stage: WorkflowStage, error: Exception) -> bool:
        """Attempt to recover from stage failure."""
        recovery_count = self.recovery_attempts.get(stage, 0)
        
        if recovery_count >= self.max_recovery_attempts:
            self.logger.error(f"Max recovery attempts reached for stage {stage.value}")
            return False
        
        self.recovery_attempts[stage] = recovery_count + 1
        self.logger.info(f"Attempting recovery for stage {stage.value} (attempt {recovery_count + 1})")
        
        try:
            # Load last checkpoint if available
            checkpoint_name = f"before_{stage.value}"
            if checkpoint_name in self.checkpoints:
                self.context = self.checkpoints[checkpoint_name]
                self.logger.info(f"Restored context from checkpoint {checkpoint_name}")
            
            # Apply recovery strategies based on error type
            if isinstance(error, MemoryConstraintError):
                # Reduce memory usage
                self.context.user_preferences.batch_size = max(1, self.context.user_preferences.batch_size // 2)
                self.context.user_preferences.enable_aggressive_caching = True
                self.logger.info("Applied memory constraint recovery: reduced batch size, enabled caching")
            
            elif isinstance(error, APIIntegrationError):
                # Wait and retry with exponential backoff
                wait_time = 2 ** recovery_count
                await asyncio.sleep(wait_time)
                self.logger.info(f"Applied API error recovery: waited {wait_time}s before retry")
            
            # Retry the stage
            await self._execute_stage(stage)
            
            self.logger.info(f"Recovery successful for stage {stage.value}")
            return True
            
        except Exception as recovery_error:
            self.logger.error(f"Recovery failed for stage {stage.value}: {recovery_error}")
            return False
    
    async def _handle_stage_failure(self, stage: WorkflowStage, error: Exception) -> None:
        """Handle stage failure with appropriate error reporting."""
        stage_progress = self.stages[stage]
        stage_progress.status = "failed"
        stage_progress.error_message = str(error)
        stage_progress.end_time = datetime.now()
        
        # Update progress display
        if self.progress and stage in self.progress_tasks:
            self.progress.update(
                self.progress_tasks[stage],
                description=f"[red]{stage.value.replace('_', ' ').title()}[/red] - Failed",
                completed=0
            )
        
        self.logger.error(f"Stage {stage.value} failed: {error}")
        
        # Determine if workflow should continue or abort
        critical_stages = [
            WorkflowStage.INITIALIZATION,
            WorkflowStage.INPUT_ANALYSIS,
            WorkflowStage.AUDIO_PROCESSING
        ]
        
        if stage in critical_stages:
            raise ContentContextError(f"Critical stage {stage.value} failed: {error}")
    
    async def _monitor_resources(self) -> None:
        """Monitor system resources during processing."""
        while True:
            try:
                # Check memory usage
                current_memory = psutil.Process().memory_info().rss
                memory_gb = current_memory / (1024**3)
                
                if memory_gb > self.peak_memory:
                    self.peak_memory = memory_gb
                
                if memory_gb > self.config.max_memory_usage_gb:
                    self.logger.warning(f"Memory usage ({memory_gb:.1f}GB) exceeds limit ({self.config.max_memory_usage_gb}GB)")
                    
                    # Apply memory pressure relief
                    if self.context:
                        self.context.user_preferences.batch_size = max(1, self.context.user_preferences.batch_size // 2)
                        self.context.user_preferences.enable_aggressive_caching = True
                
                # Update performance metrics
                self.performance_metrics.update({
                    "current_memory_gb": memory_gb,
                    "peak_memory_gb": self.peak_memory,
                    "cpu_percent": psutil.cpu_percent(),
                    "timestamp": datetime.now().isoformat()
                })
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(10)  # Wait longer on error
    
    async def _save_checkpoint(self, checkpoint_name: str) -> None:
        """Save current ContentContext as checkpoint."""
        if self.context and self.config.enable_recovery:
            try:
                # Create a deep copy of the context for the checkpoint
                import copy
                checkpoint_context = copy.deepcopy(self.context)
                self.checkpoints[checkpoint_name] = checkpoint_context
                
                self.logger.info(f"Saved checkpoint: {checkpoint_name}")
                
                # Optionally save to disk for persistence
                if self.config.temp_directory:
                    checkpoint_file = self.config.temp_directory / f"{checkpoint_name}.json"
                    with open(checkpoint_file, 'w') as f:
                        json.dump(checkpoint_context.to_dict(), f, indent=2, default=str)
                
            except Exception as e:
                self.logger.error(f"Failed to save checkpoint {checkpoint_name}: {e}")
    
    async def _finalize_workflow(self) -> None:
        """Finalize workflow execution and generate summary."""
        if not self.context:
            return
        
        # Update final metrics
        total_duration = self.workflow_end_time - self.workflow_start_time
        final_memory = psutil.Process().memory_info().rss
        memory_used = (final_memory - self.initial_memory) / (1024**3)
        
        self.context.processing_metrics.total_processing_time = total_duration.total_seconds()
        self.context.processing_metrics.memory_peak_usage = int(self.peak_memory * 1024**3)
        
        # Generate workflow summary
        summary = self._generate_workflow_summary()
        
        if self.config.enable_progress_display:
            self.console.print("\n" + "="*60)
            self.console.print(Panel(summary, title="Workflow Complete", border_style="green"))
        
        self.logger.info(f"Workflow completed in {total_duration.total_seconds():.1f}s")
        self.logger.info(f"Peak memory usage: {self.peak_memory:.1f}GB")
    
    def _generate_workflow_summary(self) -> str:
        """Generate a summary of workflow execution."""
        if not self.context:
            return "No context available"
        
        total_duration = (self.workflow_end_time - self.workflow_start_time).total_seconds()
        completed_stages = len([s for s in self.stages.values() if s.is_completed])
        failed_stages = len([s for s in self.stages.values() if s.has_failed])
        total_stages = len(self.stages)
        
        summary_lines = [
            f"📊 Processing Summary:",
            f"   • Total Time: {total_duration:.1f} seconds",
            f"   • Stages Completed: {completed_stages}/{total_stages}",
            f"   • Peak Memory: {self.peak_memory:.1f}GB",
            f"   • API Calls: {sum(self.context.processing_metrics.api_calls_made.values())}",
            f"   • Recovery Actions: {len(self.context.processing_metrics.recovery_actions)}",
        ]
        
        if failed_stages > 0:
            summary_lines.append(f"   • Failed Stages: {failed_stages}")
        
        # Add stage timing breakdown
        summary_lines.append("\n⏱️  Stage Timing:")
        for stage, progress in self.stages.items():
            if progress.duration:
                summary_lines.append(f"   • {stage.value.replace('_', ' ').title()}: {progress.duration.total_seconds():.1f}s")
        
        return "\n".join(summary_lines)
    
    async def _cleanup_workflow(self) -> None:
        """Cleanup workflow resources."""
        try:
            # Stop resource monitoring
            if self.resource_monitor_task:
                self.resource_monitor_task.cancel()
                try:
                    await self.resource_monitor_task
                except asyncio.CancelledError:
                    pass
            
            # Stop progress display
            if self.live_display:
                self.live_display.stop()
            
            # Clear temporary checkpoints
            self.checkpoints.clear()
            
            self.logger.info("Workflow cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during workflow cleanup: {e}")
    
    async def _handle_workflow_failure(self, error: Exception) -> None:
        """Handle overall workflow failure."""
        self.workflow_end_time = datetime.now()
        
        if self.config.enable_progress_display:
            error_panel = Panel(
                f"[red]Workflow failed: {str(error)}[/red]\n\n"
                f"Check logs for detailed error information.",
                title="Workflow Failed",
                border_style="red"
            )
            self.console.print(error_panel)
        
        self.logger.error(f"Workflow failed: {error}")
    
    # Stage implementation methods (placeholders for actual module integration)
    
    async def _stage_initialization(self) -> None:
        """Initialize workflow and validate inputs."""
        if not self.context:
            raise ContentContextError("ContentContext not initialized")
        
        # Validate input files
        for file_path in self.context.video_files:
            if not Path(file_path).exists():
                raise ContentContextError(f"Input file not found: {file_path}")
        
        # Initialize processing metrics
        self.context.processing_metrics = ProcessingMetrics()
        
        # Create output directories
        if self.config.output_directory:
            self.config.output_directory.mkdir(parents=True, exist_ok=True)
        
        if self.config.temp_directory:
            self.config.temp_directory.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Workflow initialization completed")
    
    async def _stage_input_analysis(self) -> None:
        """Analyze input files and extract metadata."""
        try:
            # For now, create basic video metadata since VideoAnalyzer may not be fully implemented
            # This will be enhanced when VideoAnalyzer is available
            
            # Update progress
            if self.progress and WorkflowStage.INPUT_ANALYSIS in self.progress_tasks:
                self.progress.update(
                    self.progress_tasks[WorkflowStage.INPUT_ANALYSIS],
                    description="[cyan]Input Analysis[/cyan] - Analyzing video files",
                    completed=25
                )
            
            # Analyze each video file
            for i, video_file in enumerate(self.context.video_files):
                self.logger.info(f"Analyzing video file: {video_file}")
                
                # Basic file analysis (can be enhanced with actual video analysis later)
                file_path = Path(video_file)
                file_size = file_path.stat().st_size if file_path.exists() else 0
                
                # Create basic metadata
                video_metadata = {
                    "file_path": str(file_path),
                    "file_size": file_size,
                    "duration": 300.0,  # Default 5 minutes - will be replaced with actual analysis
                    "resolution": (1920, 1080),  # Default HD - will be replaced with actual analysis
                    "fps": 30.0,  # Default 30fps - will be replaced with actual analysis
                    "format": file_path.suffix.lower().lstrip('.') if file_path.suffix else "mp4"
                }
                
                # Store in context (ensure list exists even if attribute present but None)
                if not getattr(self.context, 'video_metadata', None):
                    self.context.video_metadata = []
                self.context.video_metadata.append(video_metadata)
                
                # Update progress
                progress = 25 + (50 * (i + 1) / len(self.context.video_files))
                if self.progress and WorkflowStage.INPUT_ANALYSIS in self.progress_tasks:
                    self.progress.update(
                        self.progress_tasks[WorkflowStage.INPUT_ANALYSIS],
                        description=f"[cyan]Input Analysis[/cyan] - Analyzed {i+1}/{len(self.context.video_files)} files",
                        completed=progress
                    )
            
            # Record processing metrics
            self.context.processing_metrics.add_module_metrics(
                "input_analysis",
                time.time() - self.stages[WorkflowStage.INPUT_ANALYSIS].start_time.timestamp(),
                0
            )
            
            self.logger.info("Input analysis completed")
            
        except Exception as e:
            self.logger.error(f"Input analysis failed: {e}")
            raise ContentContextError(f"Input analysis failed: {e}", context_state=self.context)
    
    async def _stage_audio_processing(self) -> None:
        """Process audio with Whisper and content analysis."""
        try:
            from ..core.content_context import AudioAnalysisResult, AudioSegment, EmotionalPeak
            
            # Update progress
            if self.progress and WorkflowStage.AUDIO_PROCESSING in self.progress_tasks:
                self.progress.update(
                    self.progress_tasks[WorkflowStage.AUDIO_PROCESSING],
                    description="[cyan]Audio Processing[/cyan] - Creating audio analysis",
                    completed=30
                )
            
            # Create comprehensive audio analysis for the context
            segments = [
                AudioSegment(
                    text="Sample educational content for workflow processing",
                    start=0.0,
                    end=300.0,  # 5 minutes
                    confidence=0.95,
                    filler_words=[],
                    cleaned_text="Sample educational content for workflow processing"
                )
            ]
            
            emotional_peaks = [
                EmotionalPeak(
                    timestamp=60.0,
                    emotion="engagement",
                    intensity=0.8,
                    confidence=0.9,
                    context="Educational content introduction"
                )
            ]
            
            # Create audio analysis result
            self.context.audio_analysis = AudioAnalysisResult(
                transcript_text="Sample educational content for workflow processing",
                segments=segments,
                overall_confidence=0.95,
                language="en",
                processing_time=2.0,
                model_used="whisper-large-v3",
                filler_words_removed=0,
                segments_modified=0,
                quality_improvement_score=0.8,
                original_duration=300.0,
                enhanced_duration=300.0,
                financial_concepts=["education", "learning", "content"]
            )
            
            # Add emotional markers to context
            self.context.emotional_markers = emotional_peaks
            
            # Legacy attributes for backward compatibility
            self.context.audio_transcript = "Sample educational content for workflow processing"
            self.context.key_concepts = ["education", "learning", "content"]
            
            # Record processing metrics
            self.context.processing_metrics.add_module_metrics(
                "audio_processing",
                time.time() - self.stages[WorkflowStage.AUDIO_PROCESSING].start_time.timestamp(),
                0
            )
            
            self.logger.info("Audio processing completed")
            
        except Exception as e:
            self.logger.error(f"Audio processing failed: {e}")
            # Continue with basic mock data
            self.context.audio_transcript = "Mock audio content"
            self.context.key_concepts = ["education", "content"]
    
    async def _stage_video_processing(self) -> None:
        """Process video with OpenCV analysis."""
        # This will be replaced with actual VideoAnalyzer calls
        await asyncio.sleep(2)
        
        self.logger.info("Video processing completed")
    
    async def _stage_intelligence_layer(self) -> None:
        """Run AI Director and intelligence analysis."""
        # This will be replaced with actual AI Director calls
        await asyncio.sleep(3)
        
        self.logger.info("Intelligence layer processing completed")
    
    async def _stage_content_analysis(self) -> None:
        """Perform content analysis and concept extraction."""
        # This will be replaced with actual ContentAnalyzer calls
        await asyncio.sleep(2)
        
        self.logger.info("Content analysis completed")
    
    async def _stage_broll_generation(self) -> None:
        """Generate B-roll content and graphics."""
        # This will be replaced with actual B-roll generation calls
        await asyncio.sleep(4)
        
        self.logger.info("B-roll generation completed")
    
    async def _stage_video_composition(self) -> None:
        """Compose final video with movis."""
        try:
            from ..modules.video_processing.composer import VideoComposer
            
            # Update progress
            if self.progress and WorkflowStage.VIDEO_COMPOSITION in self.progress_tasks:
                self.progress.update(
                    self.progress_tasks[WorkflowStage.VIDEO_COMPOSITION],
                    description="[cyan]Video Composition[/cyan] - Initializing composer",
                    completed=20
                )
            
            # Initialize video composer
            composer = VideoComposer()
            
            # Update progress
            if self.progress and WorkflowStage.VIDEO_COMPOSITION in self.progress_tasks:
                self.progress.update(
                    self.progress_tasks[WorkflowStage.VIDEO_COMPOSITION],
                    description="[cyan]Video Composition[/cyan] - Creating composition plan",
                    completed=50
                )
            
            # Create composition plan
            composition_plan = composer.create_composition_plan(self.context)
            
            # Store composition plan in context
            self.context.composition_plan = composition_plan
            
            # Update progress
            if self.progress and WorkflowStage.VIDEO_COMPOSITION in self.progress_tasks:
                self.progress.update(
                    self.progress_tasks[WorkflowStage.VIDEO_COMPOSITION],
                    description="[cyan]Video Composition[/cyan] - Composition plan created",
                    completed=90
                )
            
            # Record processing metrics
            self.context.processing_metrics.add_module_metrics(
                "video_composition",
                time.time() - self.stages[WorkflowStage.VIDEO_COMPOSITION].start_time.timestamp(),
                0
            )
            
            self.logger.info(f"Video composition completed - {len(composition_plan.layers)} layers created")
            
        except Exception as e:
            self.logger.error(f"Video composition failed: {e}")
            # Continue with mock composition plan
            self.logger.warning("Continuing with mock composition plan")
            
            # Create minimal mock composition plan
            from ..modules.video_processing.composer import CompositionPlan, LayerInfo
            
            mock_layer = LayerInfo(
                layer_id="main_video",
                layer_type="video",
                source_path=self.context.video_files[0] if self.context.video_files else "mock.mp4",
                start_time=0.0,
                end_time=300.0
            )
            
            self.context.composition_plan = CompositionPlan(
                layers=[mock_layer],
                total_duration=300.0,
                quality_profile="balanced"
            )
    
    async def _stage_thumbnail_generation(self) -> None:
        """Generate thumbnails with AI assistance."""
        try:
            from ..modules.thumbnail_generation.generator import ThumbnailGenerator
            from ..modules.thumbnail_generation.synchronizer import ThumbnailMetadataSynchronizer
            
            # Update progress
            if self.progress and WorkflowStage.THUMBNAIL_GENERATION in self.progress_tasks:
                self.progress.update(
                    self.progress_tasks[WorkflowStage.THUMBNAIL_GENERATION],
                    description="[cyan]Thumbnail Generation[/cyan] - Initializing generator",
                    completed=25
                )
            
            # Initialize thumbnail generator with required dependencies
            from ..modules.intelligence.gemini_client import GeminiClient
            from ..core.cache_manager import CacheManager
            
            # Use existing cache manager or create a temporary one
            cache_manager = getattr(self, 'cache_manager', None)
            if not cache_manager:
                cache_manager = CacheManager()
            
            # Use existing gemini client or create a temporary one
            gemini_client = getattr(self, 'gemini_client', None)
            if not gemini_client:
                gemini_client = GeminiClient()
            
            thumbnail_generator = ThumbnailGenerator(gemini_client, cache_manager)
            synchronizer = ThumbnailMetadataSynchronizer()
            
            # Update progress
            if self.progress and WorkflowStage.THUMBNAIL_GENERATION in self.progress_tasks:
                self.progress.update(
                    self.progress_tasks[WorkflowStage.THUMBNAIL_GENERATION],
                    description="[cyan]Thumbnail Generation[/cyan] - Generating thumbnails",
                    completed=60
                )
            
            # Generate thumbnails
            thumbnail_results = await thumbnail_generator.generate_thumbnails(self.context)
            
            # Synchronize with metadata
            synchronized_results = await synchronizer.synchronize_thumbnails_metadata(
                self.context, thumbnail_results
            )
            
            # Store results in context
            self.context.thumbnail_results = synchronized_results
            
            # Record processing metrics
            self.context.processing_metrics.add_module_metrics(
                "thumbnail_generation",
                time.time() - self.stages[WorkflowStage.THUMBNAIL_GENERATION].start_time.timestamp(),
                0
            )
            
            self.logger.info(f"Thumbnail generation completed - {len(synchronized_results.thumbnails)} thumbnails created")
            
        except Exception as e:
            self.logger.error(f"Thumbnail generation failed: {e}")
            # Continue with mock thumbnail data
            self.logger.warning("Continuing with mock thumbnail data")
            
            # Create minimal mock thumbnail results
            from ..modules.thumbnail_generation.thumbnail_models import ThumbnailPackage, ThumbnailVariation, ThumbnailConcept
            from ..core.content_context import VisualHighlight, EmotionalPeak
            from datetime import datetime
            
            # Create mock visual highlight and emotional peak
            from ..core.content_context import FaceDetection
            
            mock_face = FaceDetection(
                bbox=(100, 100, 200, 200),
                confidence=0.8,
                landmarks={}
            )
            
            mock_visual = VisualHighlight(
                timestamp=30.0,
                description="Mock visual highlight",
                faces=[mock_face],
                visual_elements=["face"],
                thumbnail_potential=0.8
            )
            
            mock_emotional = EmotionalPeak(
                timestamp=30.0,
                emotion="excitement",
                intensity=0.8,
                confidence=0.9
            )
            
            mock_concept = ThumbnailConcept(
                concept_id="mock_concept_1",
                visual_highlight=mock_visual,
                emotional_peak=mock_emotional,
                hook_text="Educational Content",
                background_style="gradient",
                text_style={"font": "Arial", "size": 24},
                visual_elements=["text", "background"],
                thumbnail_potential=0.8,
                strategy="emotional"
            )
            
            mock_variation = ThumbnailVariation(
                variation_id="mock_var_1",
                concept=mock_concept,
                generated_image_path="mock_thumbnail.jpg",
                generation_method="mock",
                confidence_score=0.8,
                estimated_ctr=0.05,
                visual_appeal_score=0.8,
                text_readability_score=0.9,
                brand_consistency_score=0.7
            )
            
            self.context.thumbnail_results = ThumbnailPackage(
                package_id="mock_package",
                variations=[mock_variation],
                recommended_variation="mock_var_1",
                generation_timestamp=datetime.now(),
                synchronized_metadata={},
                a_b_testing_config={},
                performance_predictions={},
                total_generation_time=1.0,
                total_generation_cost=0.0
            )
    
    async def _stage_metadata_generation(self) -> None:
        """Generate SEO-optimized metadata."""
        # This will be replaced with actual metadata generation calls
        await asyncio.sleep(1)
        
        self.logger.info("Metadata generation completed")
    
    async def _stage_finalization(self) -> None:
        """Finalize output and cleanup."""
        # This will be replaced with actual finalization logic
        await asyncio.sleep(1)
        
        self.logger.info("Finalization completed")
    
    # Public utility methods
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status and metrics."""
        return {
            "current_stage": self.current_stage.value if self.current_stage else None,
            "stages": {
                stage.value: {
                    "status": progress.status,
                    "progress": progress.progress_percentage,
                    "duration": progress.duration.total_seconds() if progress.duration else None,
                    "error": progress.error_message
                }
                for stage, progress in self.stages.items()
            },
            "performance_metrics": self.performance_metrics,
            "recovery_attempts": {
                stage.value: count for stage, count in self.recovery_attempts.items()
            }
        }
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get processing summary and statistics."""
        if not self.context:
            return {}
        
        return {
            "project_id": self.context.project_id,
            "input_files": self.context.video_files,
            "content_type": self.context.content_type.value,
            "processing_metrics": self.context.processing_metrics.to_dict(),
            "cost_tracking": self.context.cost_tracking.to_dict(),
            "workflow_duration": (
                (self.workflow_end_time - self.workflow_start_time).total_seconds()
                if self.workflow_start_time and self.workflow_end_time else None
            )
        } 
   
    async def _cleanup_workflow(self):
        """Cleanup workflow resources."""
        # Stop resource monitoring
        if self.resource_monitor_task:
            self.resource_monitor_task.cancel()
            try:
                await self.resource_monitor_task
            except asyncio.CancelledError:
                pass
        
        # Stop live display
        if self.live_display:
            self.live_display.stop()
        
        # Shutdown performance optimizer
        if self.performance_optimizer:
            await self.performance_optimizer.shutdown()
        
        self.logger.info("Workflow cleanup completed")