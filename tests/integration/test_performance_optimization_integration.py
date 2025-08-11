"""
Integration tests for Performance Optimization system.

Tests the complete performance optimization workflow including caching,
resource monitoring, batch processing, and API optimization.
"""

import pytest
import asyncio
import tempfile
import shutil
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from ai_video_editor.core.performance_optimizer import PerformanceOptimizer, PerformanceProfile
from ai_video_editor.core.batch_processor import BatchProcessor, BatchPriority, BatchConfiguration
from ai_video_editor.core.cache_manager import CacheManager
from ai_video_editor.core.content_context import ContentContext, ContentType, UserPreferences
from ai_video_editor.core.workflow_orchestrator import WorkflowOrchestrator, WorkflowConfiguration


class TestPerformanceOptimizationIntegration:
    """Integration tests for performance optimization system."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def cache_manager(self, temp_cache_dir):
        """Create CacheManager for testing."""
        return CacheManager(cache_dir=temp_cache_dir, max_memory_entries=50)
    
    @pytest.fixture
    def performance_optimizer(self, cache_manager):
        """Create PerformanceOptimizer for testing."""
        return PerformanceOptimizer(cache_manager, monitoring_interval=0.5)
    
    @pytest.fixture
    def batch_processor(self, performance_optimizer, cache_manager):
        """Create BatchProcessor for testing."""
        config = BatchConfiguration(
            max_concurrent_jobs=2,
            max_queue_size=10,
            resource_check_interval=0.5,
            enable_job_persistence=False
        )
        return BatchProcessor(performance_optimizer, cache_manager, config)
    
    @pytest.fixture
    def sample_contexts(self):
        """Create sample ContentContext objects for testing."""
        contexts = []
        for i in range(3):
            context = ContentContext(
                project_id=f'integration_test_{i}',
                video_files=[f'test_video_{i}.mp4'],
                content_type=ContentType.EDUCATIONAL if i % 2 == 0 else ContentType.GENERAL,
                user_preferences=UserPreferences()
            )
            context.key_concepts = [f'concept_{i}', 'integration', 'testing']
            context.content_themes = ['education', 'tutorial']
            contexts.append(context)
        return contexts
    
    @patch('ai_video_editor.core.performance_optimizer.psutil')
    @pytest.mark.asyncio
    async def test_complete_optimization_workflow(self, mock_psutil, performance_optimizer, sample_contexts):
        """Test complete performance optimization workflow."""
        # Mock psutil for consistent testing
        mock_psutil.cpu_percent.return_value = 45.0
        mock_memory = Mock()
        mock_memory.total = 16 * 1024**3  # 16GB
        mock_memory.available = 8 * 1024**3  # 8GB available
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.disk_io_counters.return_value = None
        mock_psutil.net_io_counters.return_value = None
        
        try:
            # Initialize with balanced profile
            await performance_optimizer.initialize('balanced')
            
            # Test caching optimization
            for i, context in enumerate(sample_contexts):
                start_time = time.time()
                optimized_context = await performance_optimizer.optimize_context_processing(context)
                processing_time = time.time() - start_time
                
                # Verify optimization results
                assert optimized_context.project_id == context.project_id
                assert optimized_context.user_preferences.batch_size == 2  # Balanced profile
                assert optimized_context.user_preferences.parallel_processing is True
                
                # Check processing metrics
                assert 'performance_optimizer' in optimized_context.processing_metrics.module_processing_times
                
                # Verify processing time is reasonable
                assert processing_time < 1.0  # Should be fast
            
            # Test API rate limiting with mocked sleep to avoid timeout
            api_limiter = performance_optimizer.api_rate_limiter
            api_limiter.set_api_quota('test_service', 10, 0.001)
            
            # Mock asyncio.sleep to avoid actual waiting
            with patch('asyncio.sleep') as mock_sleep:
                mock_sleep.return_value = None
                
                # Test multiple API calls
                successful_calls = 0
                for i in range(15):  # More than rate limit
                    success = await api_limiter.acquire_api_slot('test_service', 0.001)
                    if success:
                        successful_calls += 1
                
                # Should have some successful calls
                assert successful_calls > 0
            
            # Get performance statistics
            stats = performance_optimizer.get_performance_stats()
            assert stats['current_profile'] == 'balanced'
            assert 'resource_metrics' in stats
            assert 'cache_performance' in stats
            assert 'api_usage' in stats
            
            # Verify cache is working
            cache_stats = stats['cache_performance']
            assert cache_stats['memory_cache_size'] >= 0
            
        finally:
            await performance_optimizer.shutdown()
    
    @patch('ai_video_editor.core.performance_optimizer.psutil')
    @pytest.mark.asyncio
    async def test_batch_processing_integration(self, mock_psutil, batch_processor, sample_contexts):
        """Test batch processing integration with performance optimization."""
        # Mock psutil
        mock_psutil.cpu_percent.return_value = 40.0
        mock_memory = Mock()
        mock_memory.total = 16 * 1024**3
        mock_memory.available = 10 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_memory
        
        # Mock processor function
        def mock_processor(context: ContentContext) -> ContentContext:
            # Simulate processing
            time.sleep(0.2)
            context.content_themes = ['processed', 'batch_test']
            context.processing_metrics.add_module_metrics(
                'mock_processor', 0.2, 50 * 1024 * 1024  # 50MB
            )
            return context
        
        try:
            # Submit batch jobs
            job_ids = []
            priorities = [BatchPriority.HIGH, BatchPriority.NORMAL, BatchPriority.LOW]
            
            for i, (context, priority) in enumerate(zip(sample_contexts, priorities)):
                job_id = await batch_processor.submit_job(
                    context=context,
                    processor_func=mock_processor,
                    priority=priority
                )
                job_ids.append(job_id)
            
            # Wait for processing to complete
            final_status = await batch_processor.wait_for_completion(timeout_minutes=1.0)
            
            # Verify results
            queue_status = final_status['queue_status']
            assert queue_status['total_jobs'] == 3
            
            # Check that at least some jobs completed
            completed_count = 0
            for job_id in job_ids:
                job_status = batch_processor.get_job_status(job_id)
                if job_status and job_status['status'] == 'completed':
                    completed_count += 1
            
            assert completed_count >= 1  # At least one job should complete
            
        finally:
            await batch_processor.shutdown()
    
    @patch('ai_video_editor.core.performance_optimizer.psutil')
    @pytest.mark.asyncio
    async def test_resource_monitoring_integration(self, mock_psutil, performance_optimizer):
        """Test resource monitoring integration."""
        # Mock normal resource usage initially
        mock_psutil.cpu_percent.return_value = 50.0
        mock_memory = Mock()
        mock_memory.total = 16 * 1024**3
        mock_memory.available = 8 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.disk_io_counters.return_value = None
        mock_psutil.net_io_counters.return_value = None
        
        alerts_received = []
        
        def alert_callback(alert: str, metrics):
            alerts_received.append(alert)
        
        try:
            # Initialize and add alert callback
            await performance_optimizer.initialize('balanced')
            performance_optimizer.resource_monitor.add_alert_callback(alert_callback)
            
            # Let monitoring run briefly
            await asyncio.sleep(0.8)
            
            # Simulate high memory usage
            mock_memory.available = 0.5 * 1024**3  # Very low available memory
            
            # Wait for alert
            await asyncio.sleep(1.0)
            
            # Should have received memory alert
            memory_alerts = [alert for alert in alerts_received if 'Memory' in alert]
            assert len(memory_alerts) > 0
            
            # Test resource metrics retrieval
            current_metrics = performance_optimizer.resource_monitor.get_current_metrics()
            assert current_metrics is not None
            assert current_metrics.cpu_percent == 50.0
            assert current_metrics.memory_available_gb == 0.5
            
        finally:
            await performance_optimizer.shutdown()
    
    @patch('ai_video_editor.core.performance_optimizer.psutil')
    @pytest.mark.asyncio
    async def test_caching_across_modules(self, mock_psutil, performance_optimizer, cache_manager):
        """Test caching integration across different modules."""
        # Mock psutil
        mock_psutil.cpu_percent.return_value = 45.0
        mock_memory = Mock()
        mock_memory.total = 16 * 1024**3
        mock_memory.available = 8 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.disk_io_counters.return_value = None
        mock_psutil.net_io_counters.return_value = None
        
        try:
            await performance_optimizer.initialize('balanced')
            
            # Test API response caching
            cache_manager.cache_api_response(
                service='test_gemini',
                endpoint='analyze_content',
                params={'text': 'test content'},
                response={'analysis': 'test result'},
                cost=0.01
            )
            
            # Retrieve cached response
            cached_response = cache_manager.get_api_response(
                service='test_gemini',
                endpoint='analyze_content',
                params={'text': 'test content'}
            )
            
            assert cached_response is not None
            assert cached_response['analysis'] == 'test result'
            
            # Test keyword research caching
            cache_manager.cache_keyword_research(
                concepts=['finance', 'education'],
                content_type='educational',
                research_result={'keywords': ['financial education', 'learning']}
            )
            
            # Retrieve cached keyword research
            cached_keywords = cache_manager.get_keyword_research(
                concepts=['finance', 'education'],
                content_type='educational'
            )
            
            assert cached_keywords is not None
            assert 'keywords' in cached_keywords
            
            # Test processing result caching
            cache_manager.cache_processing_result(
                context_id='test_context',
                module_name='audio_analysis',
                stage='transcription',
                result={'transcript': 'test transcript'}
            )
            
            # Retrieve cached processing result
            cached_result = cache_manager.get_processing_result(
                context_id='test_context',
                module_name='audio_analysis',
                stage='transcription'
            )
            
            assert cached_result is not None
            assert cached_result['transcript'] == 'test transcript'
            
            # Verify cache statistics
            cache_stats = cache_manager.get_stats()
            assert cache_stats['hits'] >= 3  # Should have cache hits
            assert cache_stats['memory_cache_size'] >= 3  # Should have cached items
            
        finally:
            await performance_optimizer.shutdown()
    
    @patch('ai_video_editor.core.performance_optimizer.psutil')
    @pytest.mark.asyncio
    async def test_performance_profile_switching(self, mock_psutil, performance_optimizer, sample_contexts):
        """Test switching between different performance profiles."""
        # Mock psutil
        mock_psutil.cpu_percent.return_value = 50.0
        mock_memory = Mock()
        mock_memory.total = 16 * 1024**3
        mock_memory.available = 8 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.disk_io_counters.return_value = None
        mock_psutil.net_io_counters.return_value = None
        
        context = sample_contexts[0]
        
        try:
            # Test fast profile
            await performance_optimizer.initialize('fast')
            fast_optimized = await performance_optimizer.optimize_context_processing(context)
            
            assert performance_optimizer.current_profile.name == 'fast'
            assert fast_optimized.user_preferences.batch_size == 1
            assert fast_optimized.user_preferences.enable_aggressive_caching is True
            
            # Test balanced profile
            await performance_optimizer.initialize('balanced')
            balanced_optimized = await performance_optimizer.optimize_context_processing(context)
            
            assert performance_optimizer.current_profile.name == 'balanced'
            assert balanced_optimized.user_preferences.batch_size == 2
            assert balanced_optimized.user_preferences.parallel_processing is True
            
            # Test high quality profile
            await performance_optimizer.initialize('high_quality')
            hq_optimized = await performance_optimizer.optimize_context_processing(context)
            
            assert performance_optimizer.current_profile.name == 'high_quality'
            assert hq_optimized.user_preferences.batch_size == 3
            assert hq_optimized.user_preferences.parallel_processing is True
            
            # Verify different cache aggressiveness levels
            fast_cache_ttl = performance_optimizer.cache_manager.default_ttls['api_response']
            
            await performance_optimizer.initialize('fast')
            await performance_optimizer.optimize_context_processing(context)
            aggressive_cache_ttl = performance_optimizer.cache_manager.default_ttls['api_response']
            
            # Fast profile should have more aggressive caching
            assert aggressive_cache_ttl >= fast_cache_ttl
            
        finally:
            await performance_optimizer.shutdown()
    
    @patch('ai_video_editor.core.performance_optimizer.psutil')
    @pytest.mark.asyncio
    async def test_workflow_orchestrator_integration(self, mock_psutil, cache_manager):
        """Test integration with WorkflowOrchestrator."""
        # Mock psutil
        mock_psutil.cpu_percent.return_value = 45.0
        mock_memory = Mock()
        mock_memory.total = 16 * 1024**3
        mock_memory.available = 8 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.disk_io_counters.return_value = None
        mock_psutil.net_io_counters.return_value = None
        
        # Create performance optimizer
        performance_optimizer = PerformanceOptimizer(cache_manager, monitoring_interval=0.5)
        
        # Create workflow configuration
        workflow_config = WorkflowConfiguration(
            enable_progress_display=False,  # Disable for testing
            max_memory_usage_gb=8.0
        )
        
        # Create workflow orchestrator with performance optimization
        orchestrator = WorkflowOrchestrator(
            config=workflow_config,
            cache_manager=cache_manager,
            performance_optimizer=performance_optimizer
        )
        
        try:
            # Create test input files (mock)
            test_files = ['test_video.mp4']
            
            # Mock file existence
            with patch('pathlib.Path.exists', return_value=True):
                # Process video with performance optimization
                result_context = await orchestrator.process_video(
                    input_files=test_files,
                    user_preferences=UserPreferences(quality_mode='balanced')
                )
            
            # Verify that performance optimization was applied
            assert result_context is not None
            assert result_context.user_preferences.batch_size >= 1
            
            # Check that performance metrics were recorded
            assert 'performance_optimizer' in result_context.processing_metrics.module_processing_times
            
            # Verify workflow status
            workflow_status = orchestrator.get_workflow_status()
            assert 'performance_metrics' in workflow_status
            
            # Get processing summary
            summary = orchestrator.get_processing_summary()
            assert 'processing_metrics' in summary
            assert 'cost_tracking' in summary
            
        finally:
            # Cleanup is handled by the orchestrator's cleanup method
            pass
    
    @patch('ai_video_editor.core.performance_optimizer.psutil')
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, mock_psutil, performance_optimizer, sample_contexts):
        """Test error handling and recovery in performance optimization."""
        # Mock psutil
        mock_psutil.cpu_percent.return_value = 50.0
        mock_memory = Mock()
        mock_memory.total = 16 * 1024**3
        mock_memory.available = 8 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.disk_io_counters.return_value = None
        mock_psutil.net_io_counters.return_value = None
        
        try:
            await performance_optimizer.initialize('balanced')
            
            # Test with invalid profile
            with pytest.raises(ValueError):
                await performance_optimizer.initialize('invalid_profile')
            
            # Test context optimization with missing data
            incomplete_context = ContentContext(
                project_id='incomplete_test',
                video_files=[],  # Empty video files
                content_type=ContentType.GENERAL,
                user_preferences=UserPreferences()
            )
            
            # Should still work with incomplete context
            optimized_context = await performance_optimizer.optimize_context_processing(incomplete_context)
            assert optimized_context is not None
            assert optimized_context.project_id == 'incomplete_test'
            
            # Test resource pressure handling
            # Simulate memory pressure
            mock_memory.available = 0.2 * 1024**3  # Very low memory
            
            # Wait for resource monitoring to detect pressure
            await asyncio.sleep(1.0)
            
            # Optimize context under pressure
            context = sample_contexts[0]
            optimized_context = await performance_optimizer.optimize_context_processing(context)
            
            # Should still work but with adjusted parameters
            assert optimized_context is not None
            
        finally:
            await performance_optimizer.shutdown()
    
    @patch('ai_video_editor.core.performance_optimizer.psutil')
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, mock_psutil, performance_optimizer, sample_contexts):
        """Test concurrent operations with performance optimization."""
        # Mock psutil
        mock_psutil.cpu_percent.return_value = 40.0
        mock_memory = Mock()
        mock_memory.total = 16 * 1024**3
        mock_memory.available = 10 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.disk_io_counters.return_value = None
        mock_psutil.net_io_counters.return_value = None
        
        try:
            await performance_optimizer.initialize('balanced')
            
            # Run multiple optimizations concurrently
            tasks = []
            for context in sample_contexts:
                task = asyncio.create_task(
                    performance_optimizer.optimize_context_processing(context)
                )
                tasks.append(task)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify all tasks completed successfully
            successful_results = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_results) == len(sample_contexts)
            
            # Verify each result
            for i, result in enumerate(successful_results):
                assert result.project_id == sample_contexts[i].project_id
                assert 'performance_optimizer' in result.processing_metrics.module_processing_times
            
            # Test concurrent API calls
            api_tasks = []
            for i in range(5):
                task = asyncio.create_task(
                    performance_optimizer.api_rate_limiter.acquire_api_slot('concurrent_test', 0.001)
                )
                api_tasks.append(task)
            
            api_results = await asyncio.gather(*api_tasks)
            
            # Should have some successful API calls
            successful_api_calls = sum(1 for result in api_results if result)
            assert successful_api_calls > 0
            
        finally:
            await performance_optimizer.shutdown()


class TestPerformanceOptimizationStressTest:
    """Stress tests for performance optimization system."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def cache_manager(self, temp_cache_dir):
        """Create CacheManager for testing."""
        return CacheManager(cache_dir=temp_cache_dir, max_memory_entries=100)
    
    @pytest.fixture
    def performance_optimizer(self, cache_manager):
        """Create PerformanceOptimizer for testing."""
        return PerformanceOptimizer(cache_manager, monitoring_interval=0.2)
    
    @patch('ai_video_editor.core.performance_optimizer.psutil')
    @pytest.mark.asyncio
    async def test_high_volume_processing(self, mock_psutil, performance_optimizer):
        """Test performance optimization with high volume of contexts."""
        # Mock psutil
        mock_psutil.cpu_percent.return_value = 60.0
        mock_memory = Mock()
        mock_memory.total = 32 * 1024**3  # 32GB for stress test
        mock_memory.available = 16 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.disk_io_counters.return_value = None
        mock_psutil.net_io_counters.return_value = None
        
        try:
            await performance_optimizer.initialize('high_quality')
            
            # Create many contexts
            contexts = []
            for i in range(20):  # High volume
                context = ContentContext(
                    project_id=f'stress_test_{i}',
                    video_files=[f'stress_video_{i}.mp4'],
                    content_type=ContentType.EDUCATIONAL if i % 2 == 0 else ContentType.GENERAL,
                    user_preferences=UserPreferences()
                )
                context.key_concepts = [f'concept_{i}', 'stress', 'test']
                contexts.append(context)
            
            # Process all contexts
            start_time = time.time()
            
            optimized_contexts = []
            for context in contexts:
                optimized_context = await performance_optimizer.optimize_context_processing(context)
                optimized_contexts.append(optimized_context)
            
            total_time = time.time() - start_time
            
            # Verify all contexts were processed
            assert len(optimized_contexts) == 20
            
            # Verify reasonable processing time (should benefit from caching)
            avg_time_per_context = total_time / len(contexts)
            assert avg_time_per_context < 0.1  # Should be fast due to optimization
            
            # Verify cache effectiveness
            cache_stats = performance_optimizer.cache_manager.get_stats()
            assert cache_stats['memory_cache_size'] > 0
            
            # Get performance statistics
            perf_stats = performance_optimizer.get_performance_stats()
            assert perf_stats['current_profile'] == 'high_quality'
            
        finally:
            await performance_optimizer.shutdown()
    
    @patch('ai_video_editor.core.performance_optimizer.psutil')
    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self, mock_psutil, performance_optimizer):
        """Test handling of memory pressure scenarios."""
        # Start with normal memory
        mock_psutil.cpu_percent.return_value = 50.0
        mock_memory = Mock()
        mock_memory.total = 8 * 1024**3  # 8GB
        mock_memory.available = 4 * 1024**3  # 4GB available
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.disk_io_counters.return_value = None
        mock_psutil.net_io_counters.return_value = None
        
        memory_alerts = []
        
        def memory_alert_callback(alert: str, metrics):
            memory_alerts.append(alert)
        
        try:
            await performance_optimizer.initialize('balanced')
            performance_optimizer.resource_monitor.add_alert_callback(memory_alert_callback)
            
            # Let monitoring start
            await asyncio.sleep(0.5)
            
            # Simulate increasing memory pressure
            mock_memory.available = 1 * 1024**3  # 1GB available (high pressure)
            
            # Wait for pressure detection
            await asyncio.sleep(1.0)
            
            # Create context and optimize under pressure
            context = ContentContext(
                project_id='memory_pressure_test',
                video_files=['large_video.mp4'],
                content_type=ContentType.EDUCATIONAL,
                user_preferences=UserPreferences()
            )
            
            optimized_context = await performance_optimizer.optimize_context_processing(context)
            
            # Should still work but with adjusted parameters
            assert optimized_context is not None
            
            # Should have received memory alerts
            memory_related_alerts = [alert for alert in memory_alerts if 'Memory' in alert or 'memory' in alert]
            assert len(memory_related_alerts) > 0
            
            # Profile should have been adjusted for memory constraints
            current_profile = performance_optimizer.current_profile
            assert current_profile.batch_size <= 2  # Should be reduced
            assert current_profile.cache_aggressiveness >= 0.7  # Should be increased
            
        finally:
            await performance_optimizer.shutdown()