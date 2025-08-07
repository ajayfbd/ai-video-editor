"""
Performance Optimization Example - Demonstrates advanced performance optimization features.

This example shows how to use the PerformanceOptimizer for intelligent caching,
resource monitoring, batch processing, and API cost optimization.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import AI Video Editor components
from ai_video_editor.core.performance_optimizer import (
    PerformanceOptimizer, PerformanceProfile, performance_monitor
)
from ai_video_editor.core.batch_processor import (
    BatchProcessor, BatchPriority, BatchConfiguration
)
from ai_video_editor.core.cache_manager import CacheManager
from ai_video_editor.core.content_context import ContentContext, ContentType, UserPreferences


class PerformanceOptimizationDemo:
    """Demonstrates performance optimization features."""
    
    def __init__(self):
        """Initialize the demo."""
        self.cache_manager = None
        self.performance_optimizer = None
        self.batch_processor = None
    
    async def setup(self):
        """Set up the performance optimization system."""
        logger.info("Setting up performance optimization system...")
        
        # Initialize cache manager
        self.cache_manager = CacheManager(
            cache_dir="temp/performance_demo_cache",
            max_memory_entries=100
        )
        
        # Create custom performance profiles
        custom_profiles = {
            'demo_fast': PerformanceProfile(
                name='demo_fast',
                max_memory_gb=4.0,
                max_cpu_percent=60.0,
                batch_size=1,
                parallel_workers=1,
                cache_aggressiveness=1.0,
                api_rate_limit=30,
                quality_level='fast',
                stage_timeout_multiplier=0.5
            ),
            'demo_balanced': PerformanceProfile(
                name='demo_balanced',
                max_memory_gb=8.0,
                max_cpu_percent=80.0,
                batch_size=2,
                parallel_workers=2,
                cache_aggressiveness=0.7,
                api_rate_limit=60,
                quality_level='balanced'
            ),
            'demo_high_quality': PerformanceProfile(
                name='demo_high_quality',
                max_memory_gb=16.0,
                max_cpu_percent=95.0,
                batch_size=4,
                parallel_workers=4,
                cache_aggressiveness=0.5,
                api_rate_limit=120,
                quality_level='high',
                stage_timeout_multiplier=2.0
            )
        }
        
        # Initialize performance optimizer
        self.performance_optimizer = PerformanceOptimizer(
            cache_manager=self.cache_manager,
            performance_profiles=custom_profiles,
            monitoring_interval=1.0
        )
        
        # Initialize with balanced profile
        await self.performance_optimizer.initialize('demo_balanced')
        
        # Set up batch processor
        batch_config = BatchConfiguration(
            max_concurrent_jobs=2,
            max_queue_size=20,
            resource_check_interval=2.0,
            job_timeout_minutes=30.0,
            enable_job_persistence=True,
            persistence_file="temp/demo_batch_jobs.pkl"
        )
        
        self.batch_processor = BatchProcessor(
            performance_optimizer=self.performance_optimizer,
            cache_manager=self.cache_manager,
            config=batch_config
        )
        
        # Add progress callback
        self.batch_processor.add_progress_callback(self._progress_callback)
        
        logger.info("Performance optimization system ready!")
    
    def _progress_callback(self, progress_data):
        """Handle batch processing progress updates."""
        queue_status = progress_data.get('queue_status', {})
        active_jobs = progress_data.get('active_jobs', [])
        
        logger.info(f"Batch Progress - Queued: {queue_status.get('queued_jobs', 0)}, "
                   f"Active: {len(active_jobs)}, "
                   f"Completed: {queue_status.get('completed_jobs', 0)}")
    
    def _resource_alert_callback(self, alert: str, metrics):
        """Handle resource alerts."""
        logger.warning(f"Resource Alert: {alert}")
        logger.info(f"Current metrics - CPU: {metrics.cpu_percent:.1f}%, "
                   f"Memory: {metrics.memory_used_gb:.1f}GB used, "
                   f"{metrics.memory_available_gb:.1f}GB available")
    
    async def demonstrate_caching_optimization(self):
        """Demonstrate intelligent caching strategies."""
        logger.info("\n=== Caching Optimization Demo ===")
        
        # Create test contexts
        contexts = []
        for i in range(3):
            context = ContentContext(
                project_id=f'cache_demo_{i}',
                video_files=[f'demo_video_{i}.mp4'],
                content_type=ContentType.EDUCATIONAL,
                user_preferences=UserPreferences()
            )
            context.key_concepts = ['finance', 'education', 'tutorial']
            contexts.append(context)
        
        # Test different cache aggressiveness levels
        profiles = ['demo_fast', 'demo_balanced', 'demo_high_quality']
        
        for profile_name in profiles:
            logger.info(f"\nTesting caching with profile: {profile_name}")
            
            # Switch to profile
            await self.performance_optimizer.initialize(profile_name)
            
            # Optimize contexts
            for i, context in enumerate(contexts):
                start_time = time.time()
                optimized_context = await self.performance_optimizer.optimize_context_processing(context)
                processing_time = time.time() - start_time
                
                logger.info(f"Context {i+1} optimized in {processing_time:.3f}s")
                logger.info(f"Cache aggressiveness: {self.performance_optimizer.current_profile.cache_aggressiveness}")
        
        # Show cache statistics
        cache_stats = self.cache_manager.get_stats()
        logger.info(f"\nCache Statistics:")
        logger.info(f"  Hit rate: {cache_stats['hit_rate']:.2%}")
        logger.info(f"  Total hits: {cache_stats['hits']}")
        logger.info(f"  Total misses: {cache_stats['misses']}")
        logger.info(f"  Memory cache size: {cache_stats['memory_cache_size']}")
    
    async def demonstrate_resource_monitoring(self):
        """Demonstrate resource monitoring and alerts."""
        logger.info("\n=== Resource Monitoring Demo ===")
        
        # Add resource alert callback
        self.performance_optimizer.resource_monitor.add_alert_callback(
            self._resource_alert_callback
        )
        
        # Get current resource metrics
        current_metrics = self.performance_optimizer.resource_monitor.get_current_metrics()
        if current_metrics:
            logger.info(f"Current Resource Usage:")
            logger.info(f"  CPU: {current_metrics.cpu_percent:.1f}%")
            logger.info(f"  Memory: {current_metrics.memory_used_gb:.1f}GB used, "
                       f"{current_metrics.memory_available_gb:.1f}GB available")
            logger.info(f"  Disk I/O: {current_metrics.disk_io_read_mb:.1f}MB read, "
                       f"{current_metrics.disk_io_write_mb:.1f}MB write")
        
        # Get average metrics over last 5 minutes
        avg_metrics = self.performance_optimizer.resource_monitor.get_average_metrics(5)
        if avg_metrics:
            logger.info(f"\nAverage Resource Usage (5 min):")
            logger.info(f"  CPU: {avg_metrics.cpu_percent:.1f}%")
            logger.info(f"  Memory: {avg_metrics.memory_used_gb:.1f}GB")
        
        # Get peak metrics
        peak_metrics = self.performance_optimizer.resource_monitor.get_peak_metrics(5)
        if peak_metrics:
            logger.info(f"\nPeak Resource Usage (5 min):")
            logger.info(f"  CPU: {peak_metrics.cpu_percent:.1f}%")
            logger.info(f"  Memory: {peak_metrics.memory_used_gb:.1f}GB")
        
        # Simulate resource-intensive operation
        logger.info("\nSimulating resource-intensive operation...")
        await self._simulate_cpu_intensive_task()
    
    async def _simulate_cpu_intensive_task(self):
        """Simulate a CPU-intensive task to trigger resource monitoring."""
        @performance_monitor("cpu_intensive_simulation")
        async def cpu_task():
            # Simulate CPU work
            total = 0
            for i in range(1000000):
                total += i * i
            await asyncio.sleep(0.1)
            return total
        
        result = await cpu_task()
        logger.info(f"CPU task completed with result: {result}")
    
    async def demonstrate_api_rate_limiting(self):
        """Demonstrate API rate limiting and cost optimization."""
        logger.info("\n=== API Rate Limiting Demo ===")
        
        # Configure API quotas
        api_limiter = self.performance_optimizer.api_rate_limiter
        api_limiter.set_api_quota('demo_gemini', 10, 0.001)  # 10 req/min, $0.001/req
        api_limiter.set_api_quota('demo_imagen', 5, 0.01)    # 5 req/min, $0.01/req
        
        # Simulate API calls
        logger.info("Simulating API calls...")
        
        for i in range(15):  # More than the rate limit
            success = await api_limiter.acquire_api_slot('demo_gemini', 0.001)
            if success:
                logger.info(f"API call {i+1} approved")
            else:
                logger.warning(f"API call {i+1} rejected (rate limited or budget exceeded)")
            
            await asyncio.sleep(0.1)  # Small delay between calls
        
        # Show API statistics
        api_stats = api_limiter.get_api_stats()
        logger.info(f"\nAPI Usage Statistics:")
        logger.info(f"  Total cost today: ${api_stats['total_cost_today']:.4f}")
        logger.info(f"  Budget remaining: ${api_stats['budget_remaining']:.4f}")
        
        for service, stats in api_stats['services'].items():
            logger.info(f"  {service}: ${stats['total_cost']:.4f}, "
                       f"{stats['recent_calls_per_minute']} calls/min")
    
    async def demonstrate_batch_processing(self):
        """Demonstrate intelligent batch processing."""
        logger.info("\n=== Batch Processing Demo ===")
        
        # Create multiple contexts for batch processing
        contexts = []
        for i in range(5):
            context = ContentContext(
                project_id=f'batch_demo_{i}',
                video_files=[f'batch_video_{i}.mp4'],
                content_type=ContentType.EDUCATIONAL if i % 2 == 0 else ContentType.GENERAL,
                user_preferences=UserPreferences()
            )
            context.key_concepts = [f'concept_{i}', 'batch_processing']
            contexts.append(context)
        
        # Mock processor function
        @performance_monitor("batch_video_processing")
        def mock_video_processor(context: ContentContext) -> ContentContext:
            """Mock video processing function."""
            # Simulate processing time based on content complexity
            processing_time = 0.5 + len(context.key_concepts) * 0.1
            time.sleep(processing_time)
            
            # Add some processing results
            context.content_themes = ['processed', 'batch_result']
            context.processing_metrics.add_module_metrics(
                'mock_processor', processing_time, 100 * 1024 * 1024  # 100MB
            )
            
            return context
        
        # Submit jobs with different priorities
        job_ids = []
        priorities = [BatchPriority.LOW, BatchPriority.NORMAL, BatchPriority.HIGH, 
                     BatchPriority.NORMAL, BatchPriority.URGENT]
        
        for i, (context, priority) in enumerate(zip(contexts, priorities)):
            job_id = await self.batch_processor.submit_job(
                context=context,
                processor_func=mock_video_processor,
                priority=priority,
                max_retries=2
            )
            job_ids.append(job_id)
            logger.info(f"Submitted job {i+1} with priority {priority.name}")
        
        # Monitor batch processing
        logger.info("\nMonitoring batch processing...")
        
        # Wait for completion with timeout
        final_status = await self.batch_processor.wait_for_completion(timeout_minutes=2.0)
        
        # Show final results
        queue_status = final_status['queue_status']
        logger.info(f"\nBatch Processing Results:")
        logger.info(f"  Total jobs: {queue_status['total_jobs']}")
        logger.info(f"  Completed: {queue_status['completed_jobs']}")
        logger.info(f"  Failed: {queue_status['failed_jobs']}")
        logger.info(f"  Statistics: {queue_status['statistics']}")
        
        # Show individual job results
        for i, job_id in enumerate(job_ids):
            job_status = self.batch_processor.get_job_status(job_id)
            if job_status:
                logger.info(f"  Job {i+1} ({job_id}): {job_status['status']}")
                if job_status.get('processing_duration_seconds'):
                    logger.info(f"    Processing time: {job_status['processing_duration_seconds']:.2f}s")
    
    async def demonstrate_performance_profiles(self):
        """Demonstrate different performance profiles."""
        logger.info("\n=== Performance Profiles Demo ===")
        
        # Test context
        context = ContentContext(
            project_id='profile_demo',
            video_files=['profile_test.mp4'],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        context.key_concepts = ['performance', 'optimization', 'profiles']
        
        # Test each profile
        profiles = ['demo_fast', 'demo_balanced', 'demo_high_quality']
        
        for profile_name in profiles:
            logger.info(f"\nTesting profile: {profile_name}")
            
            # Switch to profile
            await self.performance_optimizer.initialize(profile_name)
            profile = self.performance_optimizer.current_profile
            
            logger.info(f"  Max memory: {profile.max_memory_gb}GB")
            logger.info(f"  Max CPU: {profile.max_cpu_percent}%")
            logger.info(f"  Batch size: {profile.batch_size}")
            logger.info(f"  Parallel workers: {profile.parallel_workers}")
            logger.info(f"  Cache aggressiveness: {profile.cache_aggressiveness}")
            logger.info(f"  API rate limit: {profile.api_rate_limit} req/min")
            logger.info(f"  Quality level: {profile.quality_level}")
            
            # Optimize context with this profile
            start_time = time.time()
            optimized_context = await self.performance_optimizer.optimize_context_processing(context)
            processing_time = time.time() - start_time
            
            logger.info(f"  Optimization time: {processing_time:.3f}s")
            logger.info(f"  User preferences updated:")
            logger.info(f"    Batch size: {optimized_context.user_preferences.batch_size}")
            logger.info(f"    Parallel processing: {optimized_context.user_preferences.parallel_processing}")
            logger.info(f"    Aggressive caching: {optimized_context.user_preferences.enable_aggressive_caching}")
    
    async def demonstrate_performance_stats(self):
        """Demonstrate performance statistics and monitoring."""
        logger.info("\n=== Performance Statistics Demo ===")
        
        # Add some benchmark data
        self.performance_optimizer.benchmark_operation('video_analysis', 5.2)
        self.performance_optimizer.benchmark_operation('audio_processing', 3.1)
        self.performance_optimizer.benchmark_operation('thumbnail_generation', 1.8)
        
        # Get comprehensive performance stats
        stats = self.performance_optimizer.get_performance_stats()
        
        logger.info(f"Performance Statistics:")
        logger.info(f"  Current profile: {stats['current_profile']}")
        
        # Resource metrics
        if stats['resource_metrics']['current']:
            current = stats['resource_metrics']['current']
            logger.info(f"  Current CPU: {current['cpu_percent']:.1f}%")
            logger.info(f"  Current Memory: {current['memory_used_gb']:.1f}GB used")
        
        # Cache performance
        cache_perf = stats['cache_performance']
        logger.info(f"  Cache hit rate: {cache_perf['hit_rate']:.2%}")
        logger.info(f"  Cache size: {cache_perf['memory_cache_size']}")
        
        # API usage
        api_usage = stats['api_usage']
        logger.info(f"  API cost today: ${api_usage['total_cost_today']:.4f}")
        logger.info(f"  Budget remaining: ${api_usage['budget_remaining']:.4f}")
        
        # Benchmarks
        benchmarks = stats['benchmarks']
        logger.info(f"  Benchmarks:")
        for operation, duration in benchmarks.items():
            logger.info(f"    {operation}: {duration:.2f}s")
    
    async def cleanup(self):
        """Clean up resources."""
        logger.info("\nCleaning up...")
        
        if self.batch_processor:
            await self.batch_processor.shutdown()
        
        if self.performance_optimizer:
            await self.performance_optimizer.shutdown()
        
        logger.info("Cleanup complete!")


async def main():
    """Run the performance optimization demonstration."""
    logger.info("Starting Performance Optimization Demo")
    
    demo = PerformanceOptimizationDemo()
    
    try:
        # Set up the system
        await demo.setup()
        
        # Run demonstrations
        await demo.demonstrate_caching_optimization()
        await demo.demonstrate_resource_monitoring()
        await demo.demonstrate_api_rate_limiting()
        await demo.demonstrate_performance_profiles()
        await demo.demonstrate_batch_processing()
        await demo.demonstrate_performance_stats()
        
        logger.info("\n=== Demo Complete ===")
        logger.info("All performance optimization features demonstrated successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
    
    finally:
        # Always clean up
        await demo.cleanup()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())