"""
Performance Optimizer - Advanced performance monitoring and optimization system.

This module provides comprehensive performance optimization including intelligent
caching strategies, resource monitoring, batch processing, and API cost optimization.
"""

import asyncio
import logging
import psutil
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import pickle
from functools import wraps

from .content_context import ContentContext, ProcessingMetrics, CostMetrics
from .cache_manager import CacheManager
from .exceptions import (
    ContentContextError, ResourceConstraintError, MemoryConstraintError,
    ProcessingTimeoutError, handle_errors
)


logger = logging.getLogger(__name__)


@dataclass
class ResourceMetrics:
    """Real-time resource usage metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    gpu_utilization: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_used_gb': self.memory_used_gb,
            'memory_available_gb': self.memory_available_gb,
            'disk_io_read_mb': self.disk_io_read_mb,
            'disk_io_write_mb': self.disk_io_write_mb,
            'network_sent_mb': self.network_sent_mb,
            'network_recv_mb': self.network_recv_mb,
            'gpu_utilization': self.gpu_utilization,
            'gpu_memory_used_gb': self.gpu_memory_used_gb
        }


@dataclass
class PerformanceProfile:
    """Performance profile for different processing modes."""
    name: str
    max_memory_gb: float
    max_cpu_percent: float
    batch_size: int
    parallel_workers: int
    cache_aggressiveness: float  # 0.0 to 1.0
    api_rate_limit: int  # requests per minute
    quality_level: str  # "fast", "balanced", "high"
    
    # Processing timeouts
    stage_timeout_multiplier: float = 1.0
    api_timeout_seconds: float = 30.0
    
    # Resource thresholds
    memory_warning_threshold: float = 0.8
    memory_critical_threshold: float = 0.9
    cpu_warning_threshold: float = 80.0
    cpu_critical_threshold: float = 95.0


class ResourceMonitor:
    """Real-time resource monitoring and alerting system."""
    
    def __init__(self, monitoring_interval: float = 1.0, history_size: int = 300):
        """
        Initialize resource monitor.
        
        Args:
            monitoring_interval: Seconds between resource checks
            history_size: Number of historical metrics to keep
        """
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        
        # Resource history
        self.resource_history: deque = deque(maxlen=history_size)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Alert callbacks
        self.alert_callbacks: List[Callable] = []
        
        # Baseline metrics
        self.baseline_metrics: Optional[ResourceMetrics] = None
        
        # GPU monitoring (optional)
        self.gpu_available = self._check_gpu_availability()
        
        logger.info(f"ResourceMonitor initialized (GPU available: {self.gpu_available})")
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            import GPUtil
            return len(GPUtil.getGPUs()) > 0
        except ImportError:
            return False
    
    def _get_current_metrics(self) -> ResourceMetrics:
        """Get current system resource metrics."""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_used_gb = (memory.total - memory.available) / (1024**3)
        memory_available_gb = memory.available / (1024**3)
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_read_mb = disk_io.read_bytes / (1024**2) if disk_io else 0.0
        disk_write_mb = disk_io.write_bytes / (1024**2) if disk_io else 0.0
        
        # Network I/O
        network_io = psutil.net_io_counters()
        network_sent_mb = network_io.bytes_sent / (1024**2) if network_io else 0.0
        network_recv_mb = network_io.bytes_recv / (1024**2) if network_io else 0.0
        
        # GPU metrics (if available)
        gpu_utilization = None
        gpu_memory_used_gb = None
        
        if self.gpu_available:
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    gpu_utilization = gpu.load * 100
                    gpu_memory_used_gb = gpu.memoryUsed / 1024
            except Exception as e:
                logger.debug(f"GPU monitoring failed: {e}")
        
        return ResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_used_gb=memory_used_gb,
            memory_available_gb=memory_available_gb,
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb,
            gpu_utilization=gpu_utilization,
            gpu_memory_used_gb=gpu_memory_used_gb
        )
    
    def add_alert_callback(self, callback: Callable[[str, ResourceMetrics], None]):
        """Add callback for resource alerts."""
        self.alert_callbacks.append(callback)
    
    def _check_resource_alerts(self, metrics: ResourceMetrics, profile: PerformanceProfile):
        """Check for resource alerts and trigger callbacks."""
        alerts = []
        
        # Memory alerts
        memory_usage_ratio = metrics.memory_used_gb / (metrics.memory_used_gb + metrics.memory_available_gb)
        if memory_usage_ratio >= profile.memory_critical_threshold:
            alerts.append(f"CRITICAL: Memory usage at {memory_usage_ratio:.1%}")
        elif memory_usage_ratio >= profile.memory_warning_threshold:
            alerts.append(f"WARNING: Memory usage at {memory_usage_ratio:.1%}")
        
        # CPU alerts
        if metrics.cpu_percent >= profile.cpu_critical_threshold:
            alerts.append(f"CRITICAL: CPU usage at {metrics.cpu_percent:.1f}%")
        elif metrics.cpu_percent >= profile.cpu_warning_threshold:
            alerts.append(f"WARNING: CPU usage at {metrics.cpu_percent:.1f}%")
        
        # GPU alerts (if available)
        if metrics.gpu_utilization and metrics.gpu_utilization >= 95.0:
            alerts.append(f"WARNING: GPU utilization at {metrics.gpu_utilization:.1f}%")
        
        # Trigger alert callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert, metrics)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
    
    async def start_monitoring(self, profile: PerformanceProfile):
        """Start resource monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.baseline_metrics = self._get_current_metrics()
        
        logger.info("Started resource monitoring")
        
        while self.is_monitoring:
            try:
                metrics = self._get_current_metrics()
                self.resource_history.append(metrics)
                
                # Check for alerts
                self._check_resource_alerts(metrics, profile)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(self.monitoring_interval * 2)
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.is_monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
        logger.info("Stopped resource monitoring")
    
    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """Get most recent resource metrics."""
        return self.resource_history[-1] if self.resource_history else None
    
    def get_average_metrics(self, duration_minutes: int = 5) -> Optional[ResourceMetrics]:
        """Get average metrics over specified duration."""
        if not self.resource_history:
            return None
        
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        recent_metrics = [m for m in self.resource_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return None
        
        # Calculate averages
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_used_gb for m in recent_metrics) / len(recent_metrics)
        avg_memory_available = sum(m.memory_available_gb for m in recent_metrics) / len(recent_metrics)
        
        return ResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=avg_cpu,
            memory_used_gb=avg_memory,
            memory_available_gb=avg_memory_available,
            disk_io_read_mb=0.0,  # Not meaningful for averages
            disk_io_write_mb=0.0,
            network_sent_mb=0.0,
            network_recv_mb=0.0
        )
    
    def get_peak_metrics(self, duration_minutes: int = 5) -> Optional[ResourceMetrics]:
        """Get peak resource usage over specified duration."""
        if not self.resource_history:
            return None
        
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        recent_metrics = [m for m in self.resource_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return None
        
        # Find peaks
        peak_cpu = max(m.cpu_percent for m in recent_metrics)
        peak_memory = max(m.memory_used_gb for m in recent_metrics)
        min_memory_available = min(m.memory_available_gb for m in recent_metrics)
        
        return ResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=peak_cpu,
            memory_used_gb=peak_memory,
            memory_available_gb=min_memory_available,
            disk_io_read_mb=0.0,
            disk_io_write_mb=0.0,
            network_sent_mb=0.0,
            network_recv_mb=0.0
        )


class BatchProcessor:
    """Intelligent batch processing for multiple videos."""
    
    def __init__(self, performance_optimizer: 'PerformanceOptimizer'):
        """
        Initialize batch processor.
        
        Args:
            performance_optimizer: Parent performance optimizer
        """
        self.performance_optimizer = performance_optimizer
        self.batch_queue: List[ContentContext] = []
        self.processing_results: Dict[str, Any] = {}
        self.failed_contexts: List[Tuple[ContentContext, Exception]] = []
        
        logger.info("BatchProcessor initialized")
    
    def add_to_batch(self, context: ContentContext):
        """Add context to batch processing queue."""
        self.batch_queue.append(context)
        logger.debug(f"Added context {context.project_id} to batch queue")
    
    async def process_batch(
        self,
        processor_func: Callable[[ContentContext], ContentContext],
        max_concurrent: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process batch of contexts with intelligent resource management.
        
        Args:
            processor_func: Function to process each context
            max_concurrent: Maximum concurrent processing (auto-determined if None)
            
        Returns:
            Dictionary with processing results and statistics
        """
        if not self.batch_queue:
            return {'processed': 0, 'failed': 0, 'results': {}}
        
        start_time = time.time()
        
        # Determine optimal concurrency
        if max_concurrent is None:
            max_concurrent = self._calculate_optimal_concurrency()
        
        logger.info(f"Processing batch of {len(self.batch_queue)} contexts with {max_concurrent} concurrent workers")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_context(context: ContentContext) -> Tuple[str, Any]:
            """Process single context with resource monitoring."""
            async with semaphore:
                try:
                    # Monitor resources before processing
                    pre_metrics = self.performance_optimizer.resource_monitor.get_current_metrics()
                    
                    # Process context
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, processor_func, context
                    )
                    
                    # Monitor resources after processing
                    post_metrics = self.performance_optimizer.resource_monitor.get_current_metrics()
                    
                    # Calculate resource usage for this context
                    if pre_metrics and post_metrics:
                        resource_usage = {
                            'cpu_delta': post_metrics.cpu_percent - pre_metrics.cpu_percent,
                            'memory_delta_gb': post_metrics.memory_used_gb - pre_metrics.memory_used_gb,
                            'processing_time': time.time() - start_time
                        }
                        result.processing_metrics.module_processing_times['batch_processing'] = resource_usage
                    
                    return context.project_id, result
                    
                except Exception as e:
                    logger.error(f"Batch processing failed for context {context.project_id}: {e}")
                    self.failed_contexts.append((context, e))
                    return context.project_id, None
        
        # Process all contexts concurrently
        tasks = [process_single_context(context) for context in self.batch_queue]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        processed_count = 0
        failed_count = 0
        
        for project_id, result in results:
            if isinstance(result, Exception):
                failed_count += 1
                logger.error(f"Batch processing exception for {project_id}: {result}")
            elif result is not None:
                processed_count += 1
                self.processing_results[project_id] = result
            else:
                failed_count += 1
        
        total_time = time.time() - start_time
        
        batch_stats = {
            'processed': processed_count,
            'failed': failed_count,
            'total_time': total_time,
            'average_time_per_context': total_time / len(self.batch_queue) if self.batch_queue else 0,
            'concurrency_used': max_concurrent,
            'results': self.processing_results
        }
        
        logger.info(f"Batch processing completed: {processed_count} processed, {failed_count} failed in {total_time:.2f}s")
        
        # Clear batch queue
        self.batch_queue.clear()
        
        return batch_stats
    
    def _calculate_optimal_concurrency(self) -> int:
        """Calculate optimal concurrency based on system resources."""
        # Get current resource metrics
        current_metrics = self.performance_optimizer.resource_monitor.get_current_metrics()
        
        if not current_metrics:
            return 2  # Conservative default
        
        # Base concurrency on available resources
        cpu_cores = psutil.cpu_count()
        available_memory_gb = current_metrics.memory_available_gb
        
        # Calculate based on CPU
        cpu_based_concurrency = max(1, int(cpu_cores * 0.8))  # Use 80% of cores
        
        # Calculate based on memory (assume 2GB per concurrent process)
        memory_based_concurrency = max(1, int(available_memory_gb / 2.0))
        
        # Use the more conservative estimate
        optimal_concurrency = min(cpu_based_concurrency, memory_based_concurrency)
        
        # Apply profile constraints
        profile = self.performance_optimizer.current_profile
        if profile:
            optimal_concurrency = min(optimal_concurrency, profile.parallel_workers)
        
        return max(1, optimal_concurrency)


class APIRateLimiter:
    """Intelligent API rate limiting and cost optimization."""
    
    def __init__(self, cache_manager: CacheManager):
        """
        Initialize API rate limiter.
        
        Args:
            cache_manager: CacheManager for response caching
        """
        self.cache_manager = cache_manager
        
        # Rate limiting state
        self.api_call_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.api_costs: Dict[str, float] = defaultdict(float)
        self.api_quotas: Dict[str, Dict[str, Any]] = {}
        
        # Cost tracking
        self.total_cost_today = 0.0
        self.cost_budget_daily = 50.0  # Default daily budget
        
        logger.info("APIRateLimiter initialized")
    
    def set_api_quota(self, service: str, requests_per_minute: int, cost_per_request: float):
        """Set API quota for a service."""
        self.api_quotas[service] = {
            'requests_per_minute': requests_per_minute,
            'cost_per_request': cost_per_request,
            'last_reset': datetime.now()
        }
        logger.info(f"Set API quota for {service}: {requests_per_minute} req/min, ${cost_per_request:.4f}/req")
    
    async def acquire_api_slot(self, service: str, estimated_cost: float = 0.0) -> bool:
        """
        Acquire API slot with rate limiting and cost control.
        
        Args:
            service: API service name
            estimated_cost: Estimated cost of the API call
            
        Returns:
            True if slot acquired, False if rate limited
        """
        current_time = datetime.now()
        
        # Check cost budget
        if self.total_cost_today + estimated_cost > self.cost_budget_daily:
            logger.warning(f"Daily cost budget exceeded: ${self.total_cost_today:.2f} + ${estimated_cost:.2f} > ${self.cost_budget_daily:.2f}")
            return False
        
        # Check rate limits
        if service in self.api_quotas:
            quota = self.api_quotas[service]
            call_history = self.api_call_history[service]
            
            # Remove calls older than 1 minute
            cutoff_time = current_time - timedelta(minutes=1)
            while call_history and call_history[0] < cutoff_time:
                call_history.popleft()
            
            # Check if we're at the rate limit
            if len(call_history) >= quota['requests_per_minute']:
                wait_time = (call_history[0] + timedelta(minutes=1) - current_time).total_seconds()
                logger.info(f"Rate limit reached for {service}, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
        
        # Record the API call
        self.api_call_history[service].append(current_time)
        self.api_costs[service] += estimated_cost
        self.total_cost_today += estimated_cost
        
        return True
    
    def get_api_stats(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        stats = {
            'total_cost_today': self.total_cost_today,
            'cost_budget_daily': self.cost_budget_daily,
            'budget_remaining': self.cost_budget_daily - self.total_cost_today,
            'services': {}
        }
        
        for service, cost in self.api_costs.items():
            call_history = self.api_call_history[service]
            recent_calls = len([call for call in call_history 
                              if call > datetime.now() - timedelta(minutes=1)])
            
            stats['services'][service] = {
                'total_cost': cost,
                'recent_calls_per_minute': recent_calls,
                'total_calls': len(call_history)
            }
        
        return stats


class PerformanceOptimizer:
    """
    Advanced performance optimization system with intelligent caching,
    resource monitoring, batch processing, and API cost optimization.
    """
    
    def __init__(
        self,
        cache_manager: CacheManager,
        performance_profiles: Optional[Dict[str, PerformanceProfile]] = None,
        monitoring_interval: float = 1.0
    ):
        """
        Initialize performance optimizer.
        
        Args:
            cache_manager: CacheManager instance
            performance_profiles: Custom performance profiles
            monitoring_interval: Resource monitoring interval in seconds
        """
        self.cache_manager = cache_manager
        self.monitoring_interval = monitoring_interval
        
        # Initialize components
        self.resource_monitor = ResourceMonitor(monitoring_interval)
        self.batch_processor = BatchProcessor(self)
        self.api_rate_limiter = APIRateLimiter(cache_manager)
        
        # Performance profiles
        self.performance_profiles = performance_profiles or self._create_default_profiles()
        self.current_profile: Optional[PerformanceProfile] = None
        
        # Optimization state
        self.optimization_history: List[Dict[str, Any]] = []
        self.performance_benchmarks: Dict[str, float] = {}
        
        # Thread pools for different types of work
        self.io_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="io_worker")
        self.cpu_executor = ProcessPoolExecutor(max_workers=2)
        
        logger.info("PerformanceOptimizer initialized")
    
    def _create_default_profiles(self) -> Dict[str, PerformanceProfile]:
        """Create default performance profiles."""
        return {
            'fast': PerformanceProfile(
                name='fast',
                max_memory_gb=4.0,
                max_cpu_percent=60.0,
                batch_size=1,
                parallel_workers=1,
                cache_aggressiveness=1.0,
                api_rate_limit=30,
                quality_level='fast',
                stage_timeout_multiplier=0.5,
                api_timeout_seconds=15.0
            ),
            'balanced': PerformanceProfile(
                name='balanced',
                max_memory_gb=8.0,
                max_cpu_percent=80.0,
                batch_size=2,
                parallel_workers=2,
                cache_aggressiveness=0.7,
                api_rate_limit=60,
                quality_level='balanced',
                stage_timeout_multiplier=1.0,
                api_timeout_seconds=30.0
            ),
            'high_quality': PerformanceProfile(
                name='high_quality',
                max_memory_gb=16.0,
                max_cpu_percent=95.0,
                batch_size=3,
                parallel_workers=4,
                cache_aggressiveness=0.5,
                api_rate_limit=120,
                quality_level='high',
                stage_timeout_multiplier=2.0,
                api_timeout_seconds=60.0
            ),
            'memory_constrained': PerformanceProfile(
                name='memory_constrained',
                max_memory_gb=2.0,
                max_cpu_percent=50.0,
                batch_size=1,
                parallel_workers=1,
                cache_aggressiveness=1.0,
                api_rate_limit=15,
                quality_level='fast',
                stage_timeout_multiplier=0.8,
                api_timeout_seconds=20.0,
                memory_warning_threshold=0.6,
                memory_critical_threshold=0.8
            )
        }
    
    async def initialize(self, profile_name: str = 'balanced'):
        """
        Initialize optimizer with specified performance profile.
        
        Args:
            profile_name: Name of performance profile to use
        """
        if profile_name not in self.performance_profiles:
            raise ValueError(f"Unknown performance profile: {profile_name}")
        
        self.current_profile = self.performance_profiles[profile_name]
        
        # Configure API rate limits
        self.api_rate_limiter.set_api_quota('gemini', self.current_profile.api_rate_limit, 0.001)
        self.api_rate_limiter.set_api_quota('imagen', self.current_profile.api_rate_limit // 2, 0.01)
        
        # Start resource monitoring
        self.resource_monitor.add_alert_callback(self._handle_resource_alert)
        await self.resource_monitor.start_monitoring(self.current_profile)
        
        logger.info(f"PerformanceOptimizer initialized with profile: {profile_name}")
    
    def _handle_resource_alert(self, alert: str, metrics: ResourceMetrics):
        """Handle resource alerts by adjusting performance parameters."""
        logger.warning(f"Resource alert: {alert}")
        
        # Apply automatic optimizations based on alert type
        if "CRITICAL: Memory" in alert:
            self._apply_memory_pressure_relief()
        elif "CRITICAL: CPU" in alert:
            self._apply_cpu_pressure_relief()
        elif "WARNING: Memory" in alert:
            self._apply_gentle_memory_optimization()
    
    def _apply_memory_pressure_relief(self):
        """Apply aggressive memory optimization."""
        logger.info("Applying memory pressure relief")
        
        # Reduce batch sizes
        if self.current_profile:
            self.current_profile.batch_size = max(1, self.current_profile.batch_size // 2)
            self.current_profile.parallel_workers = max(1, self.current_profile.parallel_workers // 2)
            self.current_profile.cache_aggressiveness = min(1.0, self.current_profile.cache_aggressiveness + 0.2)
        
        # Clear non-essential caches
        self.cache_manager.clear_expired()
        
        # Force garbage collection
        import gc
        gc.collect()
    
    def _apply_cpu_pressure_relief(self):
        """Apply CPU optimization."""
        logger.info("Applying CPU pressure relief")
        
        if self.current_profile:
            self.current_profile.parallel_workers = max(1, self.current_profile.parallel_workers - 1)
            self.current_profile.api_rate_limit = max(10, self.current_profile.api_rate_limit // 2)
    
    def _apply_gentle_memory_optimization(self):
        """Apply gentle memory optimization."""
        logger.info("Applying gentle memory optimization")
        
        # Enable more aggressive caching
        if self.current_profile:
            self.current_profile.cache_aggressiveness = min(1.0, self.current_profile.cache_aggressiveness + 0.1)
        
        # Clean up expired cache entries
        self.cache_manager.clear_expired()
    
    @handle_errors(logger)
    async def optimize_context_processing(self, context: ContentContext) -> ContentContext:
        """
        Optimize processing for a single context.
        
        Args:
            context: ContentContext to optimize
            
        Returns:
            Optimized ContentContext
        """
        start_time = time.time()
        
        try:
            # Apply user preferences based on current profile
            if self.current_profile:
                context.user_preferences.batch_size = self.current_profile.batch_size
                context.user_preferences.parallel_processing = self.current_profile.parallel_workers > 1
                context.user_preferences.enable_aggressive_caching = self.current_profile.cache_aggressiveness > 0.7
                context.user_preferences.quality_mode = self.current_profile.quality_level
            
            # Optimize based on content type and size
            await self._optimize_for_content_characteristics(context)
            
            # Apply intelligent caching strategies
            await self._apply_intelligent_caching(context)
            
            # Monitor and record optimization
            optimization_time = time.time() - start_time
            context.processing_metrics.add_module_metrics(
                "performance_optimizer", optimization_time, 0
            )
            
            logger.info(f"Context optimization completed in {optimization_time:.2f}s")
            
            return context
            
        except Exception as e:
            logger.error(f"Context optimization failed: {e}")
            raise ContentContextError(f"Performance optimization failed: {e}", context_state=context)
    
    async def _optimize_for_content_characteristics(self, context: ContentContext):
        """Optimize based on content characteristics."""
        # Estimate content complexity
        complexity_score = self._calculate_content_complexity(context)
        
        if complexity_score > 0.8:  # High complexity
            logger.info("High complexity content detected, adjusting parameters")
            if self.current_profile:
                self.current_profile.stage_timeout_multiplier *= 1.5
                self.current_profile.api_timeout_seconds *= 1.2
        elif complexity_score < 0.3:  # Low complexity
            logger.info("Low complexity content detected, enabling fast processing")
            if self.current_profile:
                self.current_profile.batch_size = min(self.current_profile.batch_size + 1, 5)
    
    def _calculate_content_complexity(self, context: ContentContext) -> float:
        """Calculate content complexity score (0.0 to 1.0)."""
        complexity_score = 0.0
        
        # Video file size factor
        total_size_mb = 0
        for video_file in context.video_files:
            try:
                file_size = Path(video_file).stat().st_size / (1024 * 1024)
                total_size_mb += file_size
            except:
                pass
        
        if total_size_mb > 1000:  # > 1GB
            complexity_score += 0.3
        elif total_size_mb > 500:  # > 500MB
            complexity_score += 0.2
        elif total_size_mb > 100:  # > 100MB
            complexity_score += 0.1
        
        # Content analysis complexity
        if len(context.key_concepts) > 10:
            complexity_score += 0.2
        
        if len(context.emotional_markers) > 5:
            complexity_score += 0.1
        
        if len(context.visual_highlights) > 10:
            complexity_score += 0.2
        
        # Audio analysis complexity
        if context.audio_analysis:
            if len(context.audio_analysis.segments) > 100:
                complexity_score += 0.2
        
        return min(1.0, complexity_score)
    
    async def _apply_intelligent_caching(self, context: ContentContext):
        """Apply intelligent caching strategies."""
        if not self.current_profile:
            return
        
        cache_aggressiveness = self.current_profile.cache_aggressiveness
        
        # Adjust cache TTLs based on aggressiveness
        if cache_aggressiveness > 0.8:
            # Very aggressive caching
            self.cache_manager.default_ttls['api_response'] = 7200  # 2 hours
            self.cache_manager.default_ttls['keyword_research'] = 172800  # 2 days
            self.cache_manager.default_ttls['content_analysis'] = 7200  # 2 hours
        elif cache_aggressiveness > 0.5:
            # Moderate caching
            self.cache_manager.default_ttls['api_response'] = 3600  # 1 hour
            self.cache_manager.default_ttls['keyword_research'] = 86400  # 1 day
            self.cache_manager.default_ttls['content_analysis'] = 3600  # 1 hour
        else:
            # Conservative caching
            self.cache_manager.default_ttls['api_response'] = 1800  # 30 minutes
            self.cache_manager.default_ttls['keyword_research'] = 43200  # 12 hours
            self.cache_manager.default_ttls['content_analysis'] = 1800  # 30 minutes
        
        # Pre-cache common operations
        await self._precache_common_operations(context)
    
    async def _precache_common_operations(self, context: ContentContext):
        """Pre-cache common operations for faster processing."""
        # This would pre-cache common API responses, keyword research, etc.
        # Implementation would depend on specific use patterns
        logger.debug("Pre-caching common operations")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        current_metrics = self.resource_monitor.get_current_metrics()
        average_metrics = self.resource_monitor.get_average_metrics(5)
        peak_metrics = self.resource_monitor.get_peak_metrics(5)
        
        cache_stats = self.cache_manager.get_stats()
        api_stats = self.api_rate_limiter.get_api_stats()
        
        return {
            'current_profile': self.current_profile.name if self.current_profile else None,
            'resource_metrics': {
                'current': current_metrics.to_dict() if current_metrics else None,
                'average_5min': average_metrics.to_dict() if average_metrics else None,
                'peak_5min': peak_metrics.to_dict() if peak_metrics else None
            },
            'cache_performance': cache_stats,
            'api_usage': api_stats,
            'optimization_history_count': len(self.optimization_history),
            'benchmarks': self.performance_benchmarks
        }
    
    def benchmark_operation(self, operation_name: str, duration: float):
        """Record benchmark for an operation."""
        self.performance_benchmarks[operation_name] = duration
        logger.debug(f"Benchmarked {operation_name}: {duration:.2f}s")
    
    async def shutdown(self):
        """Shutdown performance optimizer and cleanup resources."""
        logger.info("Shutting down PerformanceOptimizer")
        
        # Stop resource monitoring
        self.resource_monitor.stop_monitoring()
        
        # Shutdown thread pools
        self.io_executor.shutdown(wait=True)
        self.cpu_executor.shutdown(wait=True)
        
        logger.info("PerformanceOptimizer shutdown complete")


def performance_monitor(operation_name: str):
    """Decorator for monitoring operation performance."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                logger.debug(f"Operation {operation_name} completed in {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Operation {operation_name} failed after {duration:.2f}s: {e}")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.debug(f"Operation {operation_name} completed in {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Operation {operation_name} failed after {duration:.2f}s: {e}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator