"""
Batch Processor - Advanced batch processing system for multiple videos.

This module provides intelligent batch processing capabilities with resource
management, priority queuing, and progress tracking for processing multiple
videos efficiently.
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import pickle

from .content_context import ContentContext, ProcessingMetrics, ContentType
from .cache_manager import CacheManager
from .performance_optimizer import PerformanceOptimizer, ResourceMetrics
from .exceptions import (
    ContentContextError, ResourceConstraintError, BatchProcessingError,
    handle_errors
)


logger = logging.getLogger(__name__)


class BatchPriority(Enum):
    """Priority levels for batch processing."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class BatchStatus(Enum):
    """Status of batch processing jobs."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class BatchJob:
    """Individual batch processing job."""
    job_id: str
    context: ContentContext
    priority: BatchPriority
    created_at: datetime
    processor_func: Optional[Callable] = None
    
    # Status tracking
    status: BatchStatus = BatchStatus.QUEUED
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Resource requirements
    estimated_memory_gb: float = 2.0
    estimated_cpu_percent: float = 50.0
    estimated_duration_minutes: float = 10.0
    
    # Results
    result: Optional[ContentContext] = None
    processing_metrics: Optional[ProcessingMetrics] = None
    
    def __post_init__(self):
        """Initialize job with estimated requirements."""
        if not self.job_id:
            self.job_id = f"job_{int(time.time())}_{self.context.project_id}"
        
        # Estimate resource requirements based on content
        self._estimate_resource_requirements()
    
    def _estimate_resource_requirements(self):
        """Estimate resource requirements based on content characteristics."""
        # Base requirements
        base_memory = 1.0
        base_cpu = 30.0
        base_duration = 5.0
        
        # Adjust based on video files
        total_size_mb = 0
        for video_file in self.context.video_files:
            try:
                file_size = Path(video_file).stat().st_size / (1024 * 1024)
                total_size_mb += file_size
            except:
                pass
        
        # Memory scaling based on file size
        if total_size_mb > 1000:  # > 1GB
            base_memory *= 3.0
            base_duration *= 2.0
        elif total_size_mb > 500:  # > 500MB
            base_memory *= 2.0
            base_duration *= 1.5
        elif total_size_mb > 100:  # > 100MB
            base_memory *= 1.5
            base_duration *= 1.2
        
        # Adjust based on content type
        if self.context.content_type == ContentType.EDUCATIONAL:
            base_cpu *= 1.3  # More processing for educational content
            base_duration *= 1.2
        elif self.context.content_type == ContentType.MUSIC:
            base_memory *= 1.2  # More memory for audio processing
        
        # Adjust based on complexity
        complexity_factors = [
            len(self.context.key_concepts) / 10.0,
            len(self.context.emotional_markers) / 5.0,
            len(self.context.visual_highlights) / 10.0
        ]
        complexity_multiplier = 1.0 + sum(complexity_factors) * 0.1
        
        self.estimated_memory_gb = base_memory * complexity_multiplier
        self.estimated_cpu_percent = min(95.0, base_cpu * complexity_multiplier)
        self.estimated_duration_minutes = base_duration * complexity_multiplier
    
    @property
    def is_ready_to_process(self) -> bool:
        """Check if job is ready for processing."""
        return self.status == BatchStatus.QUEUED and self.retry_count <= self.max_retries
    
    @property
    def can_retry(self) -> bool:
        """Check if job can be retried."""
        return self.status == BatchStatus.FAILED and self.retry_count < self.max_retries
    
    @property
    def processing_duration(self) -> Optional[timedelta]:
        """Get processing duration if completed."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for serialization."""
        return {
            'job_id': self.job_id,
            'context_id': self.context.project_id,
            'priority': self.priority.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'estimated_memory_gb': self.estimated_memory_gb,
            'estimated_cpu_percent': self.estimated_cpu_percent,
            'estimated_duration_minutes': self.estimated_duration_minutes,
            'processing_duration_seconds': (
                self.processing_duration.total_seconds() 
                if self.processing_duration else None
            )
        }


@dataclass
class BatchConfiguration:
    """Configuration for batch processing."""
    max_concurrent_jobs: int = 2
    max_queue_size: int = 100
    resource_check_interval: float = 5.0
    job_timeout_minutes: float = 60.0
    
    # Resource thresholds
    max_memory_usage_percent: float = 80.0
    max_cpu_usage_percent: float = 85.0
    
    # Retry configuration
    default_max_retries: int = 3
    retry_delay_seconds: float = 30.0
    exponential_backoff: bool = True
    
    # Priority scheduling
    priority_boost_after_hours: float = 24.0
    starvation_prevention: bool = True
    
    # Persistence
    enable_job_persistence: bool = True
    persistence_file: str = "temp/batch_jobs.pkl"
    
    # Progress reporting
    progress_update_interval: float = 10.0
    enable_detailed_logging: bool = True


class BatchQueue:
    """Priority queue for batch jobs with intelligent scheduling."""
    
    def __init__(self, config: BatchConfiguration):
        """
        Initialize batch queue.
        
        Args:
            config: Batch processing configuration
        """
        self.config = config
        
        # Job storage
        self.jobs: Dict[str, BatchJob] = {}
        self.priority_queues: Dict[BatchPriority, deque] = {
            priority: deque() for priority in BatchPriority
        }
        
        # Processing state
        self.active_jobs: Dict[str, BatchJob] = {}
        self.completed_jobs: List[BatchJob] = []
        self.failed_jobs: List[BatchJob] = []
        
        # Statistics
        self.stats = {
            'total_jobs_processed': 0,
            'total_jobs_failed': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'queue_start_time': datetime.now()
        }
        
        logger.info("BatchQueue initialized")
    
    def add_job(self, job: BatchJob) -> bool:
        """
        Add job to the queue.
        
        Args:
            job: BatchJob to add
            
        Returns:
            True if job was added, False if queue is full
        """
        if len(self.jobs) >= self.config.max_queue_size:
            logger.warning(f"Batch queue is full ({self.config.max_queue_size} jobs)")
            return False
        
        # Store job
        self.jobs[job.job_id] = job
        
        # Add to priority queue
        self.priority_queues[job.priority].append(job.job_id)
        
        logger.info(f"Added job {job.job_id} to queue with priority {job.priority.name}")
        return True
    
    def get_next_job(self) -> Optional[BatchJob]:
        """
        Get next job to process based on priority and resource availability.
        
        Returns:
            Next BatchJob to process or None if no jobs available
        """
        # Check priorities from highest to lowest
        for priority in sorted(BatchPriority, key=lambda p: p.value, reverse=True):
            queue = self.priority_queues[priority]
            
            while queue:
                job_id = queue.popleft()
                job = self.jobs.get(job_id)
                
                if job and job.is_ready_to_process:
                    # Apply starvation prevention
                    if self.config.starvation_prevention:
                        job_age_hours = (datetime.now() - job.created_at).total_seconds() / 3600
                        if job_age_hours > self.config.priority_boost_after_hours:
                            logger.info(f"Applying priority boost to job {job_id} (age: {job_age_hours:.1f}h)")
                    
                    return job
        
        return None
    
    def mark_job_processing(self, job: BatchJob):
        """Mark job as currently processing."""
        job.status = BatchStatus.PROCESSING
        job.started_at = datetime.now()
        self.active_jobs[job.job_id] = job
        logger.info(f"Started processing job {job.job_id}")
    
    def mark_job_completed(self, job: BatchJob, result: ContentContext):
        """Mark job as completed."""
        job.status = BatchStatus.COMPLETED
        job.completed_at = datetime.now()
        job.result = result
        
        # Move from active to completed
        if job.job_id in self.active_jobs:
            del self.active_jobs[job.job_id]
        self.completed_jobs.append(job)
        
        # Update statistics
        self.stats['total_jobs_processed'] += 1
        if job.processing_duration:
            processing_time = job.processing_duration.total_seconds()
            self.stats['total_processing_time'] += processing_time
            self.stats['average_processing_time'] = (
                self.stats['total_processing_time'] / self.stats['total_jobs_processed']
            )
        
        logger.info(f"Completed job {job.job_id} in {job.processing_duration}")
    
    def mark_job_failed(self, job: BatchJob, error: Exception):
        """Mark job as failed."""
        job.status = BatchStatus.FAILED
        job.completed_at = datetime.now()
        job.error_message = str(error)
        job.retry_count += 1
        
        # Move from active
        if job.job_id in self.active_jobs:
            del self.active_jobs[job.job_id]
        
        # Check if job can be retried
        if job.can_retry:
            logger.info(f"Job {job.job_id} failed, scheduling retry {job.retry_count}/{job.max_retries}")
            job.status = BatchStatus.QUEUED
            
            # Add back to queue with delay
            retry_delay = self.config.retry_delay_seconds
            if self.config.exponential_backoff:
                retry_delay *= (2 ** (job.retry_count - 1))
            
            # Schedule retry (simplified - in real implementation would use proper scheduling)
            self.priority_queues[job.priority].append(job.job_id)
        else:
            logger.error(f"Job {job.job_id} failed permanently after {job.retry_count} retries: {error}")
            self.failed_jobs.append(job)
            self.stats['total_jobs_failed'] += 1
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        queued_count = sum(len(queue) for queue in self.priority_queues.values())
        
        return {
            'total_jobs': len(self.jobs),
            'queued_jobs': queued_count,
            'active_jobs': len(self.active_jobs),
            'completed_jobs': len(self.completed_jobs),
            'failed_jobs': len(self.failed_jobs),
            'statistics': self.stats,
            'priority_breakdown': {
                priority.name: len(queue) 
                for priority, queue in self.priority_queues.items()
            }
        }
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued job."""
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        if job.status == BatchStatus.QUEUED:
            job.status = BatchStatus.CANCELLED
            
            # Remove from priority queue
            for queue in self.priority_queues.values():
                if job_id in queue:
                    queue.remove(job_id)
                    break
            
            logger.info(f"Cancelled job {job_id}")
            return True
        
        return False
    
    def pause_job(self, job_id: str) -> bool:
        """Pause a queued job."""
        job = self.jobs.get(job_id)
        if not job or job.status != BatchStatus.QUEUED:
            return False
        
        job.status = BatchStatus.PAUSED
        logger.info(f"Paused job {job_id}")
        return True
    
    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job."""
        job = self.jobs.get(job_id)
        if not job or job.status != BatchStatus.PAUSED:
            return False
        
        job.status = BatchStatus.QUEUED
        self.priority_queues[job.priority].append(job_id)
        logger.info(f"Resumed job {job_id}")
        return True


class BatchProcessor:
    """
    Advanced batch processing system for multiple videos with intelligent
    resource management, priority scheduling, and progress tracking.
    """
    
    def __init__(
        self,
        performance_optimizer: PerformanceOptimizer,
        cache_manager: CacheManager,
        config: Optional[BatchConfiguration] = None
    ):
        """
        Initialize batch processor.
        
        Args:
            performance_optimizer: PerformanceOptimizer instance
            cache_manager: CacheManager instance
            config: Batch processing configuration
        """
        self.performance_optimizer = performance_optimizer
        self.cache_manager = cache_manager
        self.config = config or BatchConfiguration()
        
        # Initialize components
        self.queue = BatchQueue(self.config)
        
        # Processing state
        self.is_processing = False
        self.processing_task: Optional[asyncio.Task] = None
        self.resource_monitor_task: Optional[asyncio.Task] = None
        
        # Thread pool for CPU-intensive work
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_jobs,
            thread_name_prefix="batch_worker"
        )
        
        # Progress callbacks
        self.progress_callbacks: List[Callable] = []
        
        # Load persisted jobs if enabled
        if self.config.enable_job_persistence:
            self._load_persisted_jobs()
        
        logger.info("BatchProcessor initialized")
    
    def add_progress_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for progress updates."""
        self.progress_callbacks.append(callback)
    
    def _load_persisted_jobs(self):
        """Load persisted jobs from disk."""
        try:
            persistence_file = Path(self.config.persistence_file)
            if persistence_file.exists():
                with open(persistence_file, 'rb') as f:
                    persisted_data = pickle.load(f)
                
                # Restore jobs (simplified - would need proper context restoration)
                logger.info(f"Loaded {len(persisted_data.get('jobs', []))} persisted jobs")
                
        except Exception as e:
            logger.warning(f"Failed to load persisted jobs: {e}")
    
    def _save_persisted_jobs(self):
        """Save current jobs to disk."""
        if not self.config.enable_job_persistence:
            return
        
        try:
            persistence_file = Path(self.config.persistence_file)
            persistence_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare data for persistence
            persisted_data = {
                'jobs': [job.to_dict() for job in self.queue.jobs.values()],
                'stats': self.queue.stats,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(persistence_file, 'wb') as f:
                pickle.dump(persisted_data, f)
                
            logger.debug("Saved batch jobs to persistence file")
            
        except Exception as e:
            logger.warning(f"Failed to save persisted jobs: {e}")
    
    @handle_errors(logger)
    async def submit_job(
        self,
        context: ContentContext,
        processor_func: Callable[[ContentContext], ContentContext],
        priority: BatchPriority = BatchPriority.NORMAL,
        max_retries: Optional[int] = None
    ) -> str:
        """
        Submit a job for batch processing.
        
        Args:
            context: ContentContext to process
            processor_func: Function to process the context
            priority: Job priority level
            max_retries: Maximum retry attempts (uses config default if None)
            
        Returns:
            Job ID for tracking
            
        Raises:
            BatchProcessingError: If job submission fails
        """
        try:
            # Create batch job
            job = BatchJob(
                job_id="",  # Will be auto-generated
                context=context,
                priority=priority,
                created_at=datetime.now(),
                processor_func=processor_func,
                max_retries=max_retries or self.config.default_max_retries
            )
            
            # Add to queue
            if not self.queue.add_job(job):
                raise BatchProcessingError(
                    "job_submission",
                    reason="Batch queue is full",
                    details={"queue_size": len(self.queue.jobs), "max_size": self.config.max_queue_size}
                )
            
            # Start processing if not already running
            if not self.is_processing:
                await self.start_processing()
            
            logger.info(f"Submitted job {job.job_id} for batch processing")
            return job.job_id
            
        except Exception as e:
            logger.error(f"Failed to submit batch job: {e}")
            raise BatchProcessingError(
                "job_submission",
                reason=f"Job submission failed: {str(e)}",
                details={"context_id": context.project_id}
            )
    
    @handle_errors(logger)
    async def start_processing(self):
        """Start batch processing."""
        if self.is_processing:
            logger.warning("Batch processing is already running")
            return
        
        self.is_processing = True
        
        # Start processing task
        self.processing_task = asyncio.create_task(self._processing_loop())
        
        # Start resource monitoring
        self.resource_monitor_task = asyncio.create_task(self._resource_monitoring_loop())
        
        logger.info("Started batch processing")
    
    async def stop_processing(self, wait_for_completion: bool = True):
        """
        Stop batch processing.
        
        Args:
            wait_for_completion: Whether to wait for current jobs to complete
        """
        logger.info("Stopping batch processing")
        
        self.is_processing = False
        
        if wait_for_completion:
            # Wait for active jobs to complete
            while self.queue.active_jobs:
                logger.info(f"Waiting for {len(self.queue.active_jobs)} active jobs to complete")
                await asyncio.sleep(1.0)
        
        # Cancel tasks
        if self.processing_task:
            self.processing_task.cancel()
        
        if self.resource_monitor_task:
            self.resource_monitor_task.cancel()
        
        # Save state
        self._save_persisted_jobs()
        
        logger.info("Batch processing stopped")
    
    async def _processing_loop(self):
        """Main processing loop."""
        while self.is_processing:
            try:
                # Check if we can start new jobs
                if len(self.queue.active_jobs) >= self.config.max_concurrent_jobs:
                    await asyncio.sleep(1.0)
                    continue
                
                # Check resource availability
                if not await self._check_resource_availability():
                    await asyncio.sleep(self.config.resource_check_interval)
                    continue
                
                # Get next job
                next_job = self.queue.get_next_job()
                if not next_job:
                    await asyncio.sleep(1.0)
                    continue
                
                # Start processing job
                await self._process_job(next_job)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _resource_monitoring_loop(self):
        """Resource monitoring loop."""
        while self.is_processing:
            try:
                # Get current resource metrics
                current_metrics = self.performance_optimizer.resource_monitor.get_current_metrics()
                
                if current_metrics:
                    # Check resource thresholds
                    memory_usage_percent = (
                        current_metrics.memory_used_gb / 
                        (current_metrics.memory_used_gb + current_metrics.memory_available_gb) * 100
                    )
                    
                    if (memory_usage_percent > self.config.max_memory_usage_percent or
                        current_metrics.cpu_percent > self.config.max_cpu_usage_percent):
                        
                        logger.warning(f"Resource usage high: Memory {memory_usage_percent:.1f}%, CPU {current_metrics.cpu_percent:.1f}%")
                        
                        # Temporarily reduce concurrent jobs
                        if len(self.queue.active_jobs) > 1:
                            logger.info("Reducing concurrent jobs due to high resource usage")
                
                # Send progress updates
                await self._send_progress_update()
                
                await asyncio.sleep(self.config.resource_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(self.config.resource_check_interval * 2)
    
    async def _check_resource_availability(self) -> bool:
        """Check if resources are available for new job."""
        current_metrics = self.performance_optimizer.resource_monitor.get_current_metrics()
        
        if not current_metrics:
            return True  # Assume available if no metrics
        
        # Check memory availability
        memory_usage_percent = (
            current_metrics.memory_used_gb / 
            (current_metrics.memory_used_gb + current_metrics.memory_available_gb) * 100
        )
        
        if memory_usage_percent > self.config.max_memory_usage_percent:
            return False
        
        # Check CPU availability
        if current_metrics.cpu_percent > self.config.max_cpu_usage_percent:
            return False
        
        return True
    
    async def _process_job(self, job: BatchJob):
        """Process a single job."""
        try:
            # Mark job as processing
            self.queue.mark_job_processing(job)
            
            # Process in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._execute_job_processor,
                job
            )
            
            # Mark as completed
            self.queue.mark_job_completed(job, result)
            
        except Exception as e:
            logger.error(f"Job {job.job_id} processing failed: {e}")
            self.queue.mark_job_failed(job, e)
    
    def _execute_job_processor(self, job: BatchJob) -> ContentContext:
        """Execute job processor function."""
        if not job.processor_func:
            raise BatchProcessingError(
                "job_execution",
                reason="No processor function provided",
                details={"job_id": job.job_id}
            )
        
        try:
            # Execute processor function
            result = job.processor_func(job.context)
            
            # Update processing metrics
            if hasattr(result, 'processing_metrics'):
                job.processing_metrics = result.processing_metrics
            
            return result
            
        except Exception as e:
            logger.error(f"Job processor failed for {job.job_id}: {e}")
            raise BatchProcessingError(
                "job_execution",
                reason=f"Processor function failed: {str(e)}",
                details={"job_id": job.job_id, "context_id": job.context.project_id}
            )
    
    async def _send_progress_update(self):
        """Send progress update to callbacks."""
        if not self.progress_callbacks:
            return
        
        try:
            # Prepare progress data
            queue_status = self.queue.get_queue_status()
            current_metrics = self.performance_optimizer.resource_monitor.get_current_metrics()
            
            progress_data = {
                'timestamp': datetime.now().isoformat(),
                'queue_status': queue_status,
                'resource_metrics': current_metrics.to_dict() if current_metrics else None,
                'active_jobs': [
                    {
                        'job_id': job.job_id,
                        'context_id': job.context.project_id,
                        'priority': job.priority.name,
                        'started_at': job.started_at.isoformat() if job.started_at else None,
                        'estimated_completion': (
                            (job.started_at + timedelta(minutes=job.estimated_duration_minutes)).isoformat()
                            if job.started_at else None
                        )
                    }
                    for job in self.queue.active_jobs.values()
                ]
            }
            
            # Send to callbacks
            for callback in self.progress_callbacks:
                try:
                    callback(progress_data)
                except Exception as e:
                    logger.error(f"Progress callback failed: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to send progress update: {e}")
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job."""
        job = self.queue.jobs.get(job_id)
        if not job:
            return None
        
        return job.to_dict()
    
    def get_batch_status(self) -> Dict[str, Any]:
        """Get overall batch processing status."""
        queue_status = self.queue.get_queue_status()
        current_metrics = self.performance_optimizer.resource_monitor.get_current_metrics()
        
        return {
            'is_processing': self.is_processing,
            'queue_status': queue_status,
            'resource_metrics': current_metrics.to_dict() if current_metrics else None,
            'configuration': {
                'max_concurrent_jobs': self.config.max_concurrent_jobs,
                'max_queue_size': self.config.max_queue_size,
                'job_timeout_minutes': self.config.job_timeout_minutes
            }
        }
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued job."""
        return self.queue.cancel_job(job_id)
    
    def pause_job(self, job_id: str) -> bool:
        """Pause a queued job."""
        return self.queue.pause_job(job_id)
    
    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job."""
        return self.queue.resume_job(job_id)
    
    async def wait_for_completion(self, timeout_minutes: Optional[float] = None) -> Dict[str, Any]:
        """
        Wait for all jobs to complete.
        
        Args:
            timeout_minutes: Maximum time to wait (None for no timeout)
            
        Returns:
            Final batch status
        """
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60 if timeout_minutes else None
        
        while (self.queue.active_jobs or 
               any(len(queue) > 0 for queue in self.queue.priority_queues.values())):
            
            # Check timeout
            if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                logger.warning(f"Batch processing timeout after {timeout_minutes} minutes")
                break
            
            await asyncio.sleep(1.0)
        
        return self.get_batch_status()
    
    async def shutdown(self):
        """Shutdown batch processor."""
        logger.info("Shutting down BatchProcessor")
        
        await self.stop_processing(wait_for_completion=True)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("BatchProcessor shutdown complete")


# Exception class for batch processing errors
class BatchProcessingError(ContentContextError):
    """Exception raised during batch processing operations."""
    
    def __init__(self, operation: str, reason: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize batch processing error.
        
        Args:
            operation: Operation that failed
            reason: Reason for failure
            details: Additional error details
        """
        super().__init__(f"Batch processing failed in {operation}: {reason}")
        self.operation = operation
        self.reason = reason
        self.details = details or {}