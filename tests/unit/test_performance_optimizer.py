"""
Unit tests for PerformanceOptimizer.

Tests the PerformanceOptimizer class for intelligent caching, resource monitoring,
and performance optimization with comprehensive mocking strategies.
"""

import pytest
import asyncio
import time
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from pathlib import Path

from ai_video_editor.core.performance_optimizer import (
    PerformanceOptimizer, ResourceMonitor, BatchProcessor, APIRateLimiter,
    PerformanceProfile, ResourceMetrics, performance_monitor
)
from ai_video_editor.core.cache_manager import CacheManager
from ai_video_editor.core.content_context import ContentContext, ContentType, UserPreferences
from ai_video_editor.core.exceptions import ContentContextError, ResourceConstraintError


class TestResourceMetrics:
    """Test ResourceMetrics functionality."""
    
    def test_resource_metrics_creation(self):
        """Test creating ResourceMetrics."""
        metrics = ResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=45.5,
            memory_used_gb=8.2,
            memory_available_gb=7.8,
            disk_io_read_mb=150.0,
            disk_io_write_mb=75.0,
            network_sent_mb=25.0,
            network_recv_mb=50.0,
            gpu_utilization=80.0,
            gpu_memory_used_gb=4.0
        )
        
        assert metrics.cpu_percent == 45.5
        assert metrics.memory_used_gb == 8.2
        assert metrics.gpu_utilization == 80.0
        assert metrics.gpu_memory_used_gb == 4.0
    
    def test_resource_metrics_serialization(self):
        """Test ResourceMetrics serialization."""
        metrics = ResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=45.5,
            memory_used_gb=8.2,
            memory_available_gb=7.8,
            disk_io_read_mb=150.0,
            disk_io_write_mb=75.0,
            network_sent_mb=25.0,
            network_recv_mb=50.0
        )
        
        metrics_dict = metrics.to_dict()
        
        assert metrics_dict['cpu_percent'] == 45.5
        assert metrics_dict['memory_used_gb'] == 8.2
        assert 'timestamp' in metrics_dict
        assert isinstance(metrics_dict['timestamp'], str)


class TestPerformanceProfile:
    """Test PerformanceProfile functionality."""
    
    def test_performance_profile_creation(self):
        """Test creating PerformanceProfile."""
        profile = PerformanceProfile(
            name='test_profile',
            max_memory_gb=8.0,
            max_cpu_percent=80.0,
            batch_size=2,
            parallel_workers=2,
            cache_aggressiveness=0.7,
            api_rate_limit=60,
            quality_level='balanced'
        )
        
        assert profile.name == 'test_profile'
        assert profile.max_memory_gb == 8.0
        assert profile.batch_size == 2
        assert profile.cache_aggressiveness == 0.7
    
    def test_performance_profile_defaults(self):
        """Test PerformanceProfile default values."""
        profile = PerformanceProfile(
            name='minimal',
            max_memory_gb=4.0,
            max_cpu_percent=60.0,
            batch_size=1,
            parallel_workers=1,
            cache_aggressiveness=0.5,
            api_rate_limit=30,
            quality_level='fast'
        )
        
        # Check default values
        assert profile.stage_timeout_multiplier == 1.0
        assert profile.api_timeout_seconds == 30.0
        assert profile.memory_warning_threshold == 0.8
        assert profile.memory_critical_threshold == 0.9


class TestResourceMonitor:
    """Test ResourceMonitor functionality."""
    
    @pytest.fixture
    def resource_monitor(self):
        """Create ResourceMonitor for testing."""
        return ResourceMonitor(monitoring_interval=0.1, history_size=10)
    
    @patch('ai_video_editor.core.performance_optimizer.psutil')
    def test_resource_monitor_initialization(self, mock_psutil, resource_monitor):
        """Test ResourceMonitor initialization."""
        assert resource_monitor.monitoring_interval == 0.1
        assert resource_monitor.history_size == 10
        assert not resource_monitor.is_monitoring
        assert len(resource_monitor.resource_history) == 0
    
    @patch('ai_video_editor.core.performance_optimizer.psutil')
    def test_get_current_metrics(self, mock_psutil, resource_monitor):
        """Test getting current resource metrics."""
        # Mock psutil responses
        mock_psutil.cpu_percent.return_value = 45.5
        
        mock_memory = Mock()
        mock_memory.total = 16 * 1024**3  # 16GB
        mock_memory.available = 8 * 1024**3  # 8GB available
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_disk_io = Mock()
        mock_disk_io.read_bytes = 150 * 1024**2  # 150MB
        mock_disk_io.write_bytes = 75 * 1024**2   # 75MB
        mock_psutil.disk_io_counters.return_value = mock_disk_io
        
        mock_network_io = Mock()
        mock_network_io.bytes_sent = 25 * 1024**2  # 25MB
        mock_network_io.bytes_recv = 50 * 1024**2  # 50MB
        mock_psutil.net_io_counters.return_value = mock_network_io
        
        # Get metrics
        metrics = resource_monitor._get_current_metrics()
        
        assert metrics.cpu_percent == 45.5
        assert metrics.memory_used_gb == 8.0  # 16GB - 8GB available
        assert metrics.memory_available_gb == 8.0
        assert metrics.disk_io_read_mb == 150.0
        assert metrics.network_sent_mb == 25.0
    
    @patch('ai_video_editor.core.performance_optimizer.psutil')
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, mock_psutil, resource_monitor):
        """Test starting and stopping resource monitoring."""
        # Mock psutil
        mock_psutil.cpu_percent.return_value = 50.0
        mock_memory = Mock()
        mock_memory.total = 16 * 1024**3
        mock_memory.available = 8 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.disk_io_counters.return_value = None
        mock_psutil.net_io_counters.return_value = None
        
        # Create a simple profile for testing
        profile = PerformanceProfile(
            name='test',
            max_memory_gb=8.0,
            max_cpu_percent=80.0,
            batch_size=1,
            parallel_workers=1,
            cache_aggressiveness=0.5,
            api_rate_limit=30,
            quality_level='fast'
        )
        
        # Start monitoring
        monitor_task = asyncio.create_task(resource_monitor.start_monitoring(profile))
        
        # Let it run briefly
        await asyncio.sleep(0.2)
        
        # Check that monitoring started
        assert resource_monitor.is_monitoring
        assert len(resource_monitor.resource_history) > 0
        
        # Stop monitoring
        resource_monitor.stop_monitoring()
        monitor_task.cancel()
        
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        
        assert not resource_monitor.is_monitoring
    
    def test_add_alert_callback(self, resource_monitor):
        """Test adding alert callbacks."""
        callback_called = False
        
        def test_callback(alert: str, metrics: ResourceMetrics):
            nonlocal callback_called
            callback_called = True
        
        resource_monitor.add_alert_callback(test_callback)
        assert len(resource_monitor.alert_callbacks) == 1
        
        # Test callback execution
        profile = PerformanceProfile(
            name='test',
            max_memory_gb=4.0,
            max_cpu_percent=50.0,
            batch_size=1,
            parallel_workers=1,
            cache_aggressiveness=0.5,
            api_rate_limit=30,
            quality_level='fast',
            memory_critical_threshold=0.5  # Low threshold for testing
        )
        
        metrics = ResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=30.0,
            memory_used_gb=8.0,  # High memory usage
            memory_available_gb=2.0,  # Low available memory
            disk_io_read_mb=0.0,
            disk_io_write_mb=0.0,
            network_sent_mb=0.0,
            network_recv_mb=0.0
        )
        
        resource_monitor._check_resource_alerts(metrics, profile)
        assert callback_called
    
    def test_get_average_metrics(self, resource_monitor):
        """Test getting average metrics."""
        # Add some test metrics
        for i in range(5):
            metrics = ResourceMetrics(
                timestamp=datetime.now() - timedelta(minutes=i),
                cpu_percent=50.0 + i * 10,
                memory_used_gb=4.0 + i,
                memory_available_gb=8.0 - i,
                disk_io_read_mb=0.0,
                disk_io_write_mb=0.0,
                network_sent_mb=0.0,
                network_recv_mb=0.0
            )
            resource_monitor.resource_history.append(metrics)
        
        # Get average
        avg_metrics = resource_monitor.get_average_metrics(10)  # 10 minutes window
        
        assert avg_metrics is not None
        assert avg_metrics.cpu_percent == 70.0  # Average of 50, 60, 70, 80, 90
        assert avg_metrics.memory_used_gb == 6.0  # Average of 4, 5, 6, 7, 8
    
    def test_get_peak_metrics(self, resource_monitor):
        """Test getting peak metrics."""
        # Add some test metrics
        for i in range(5):
            metrics = ResourceMetrics(
                timestamp=datetime.now() - timedelta(minutes=i),
                cpu_percent=50.0 + i * 10,
                memory_used_gb=4.0 + i,
                memory_available_gb=8.0 - i,
                disk_io_read_mb=0.0,
                disk_io_write_mb=0.0,
                network_sent_mb=0.0,
                network_recv_mb=0.0
            )
            resource_monitor.resource_history.append(metrics)
        
        # Get peaks
        peak_metrics = resource_monitor.get_peak_metrics(10)  # 10 minutes window
        
        assert peak_metrics is not None
        assert peak_metrics.cpu_percent == 90.0  # Max CPU
        assert peak_metrics.memory_used_gb == 8.0  # Max memory used
        assert peak_metrics.memory_available_gb == 4.0  # Min available memory


class TestAPIRateLimiter:
    """Test APIRateLimiter functionality."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def cache_manager(self, temp_cache_dir):
        """Create CacheManager for testing."""
        return CacheManager(cache_dir=temp_cache_dir, max_memory_entries=10)
    
    @pytest.fixture
    def api_rate_limiter(self, cache_manager):
        """Create APIRateLimiter for testing."""
        return APIRateLimiter(cache_manager)
    
    def test_api_rate_limiter_initialization(self, api_rate_limiter):
        """Test APIRateLimiter initialization."""
        assert api_rate_limiter.cache_manager is not None
        assert api_rate_limiter.total_cost_today == 0.0
        assert api_rate_limiter.cost_budget_daily == 50.0
    
    def test_set_api_quota(self, api_rate_limiter):
        """Test setting API quota."""
        api_rate_limiter.set_api_quota('test_service', 60, 0.01)
        
        assert 'test_service' in api_rate_limiter.api_quotas
        quota = api_rate_limiter.api_quotas['test_service']
        assert quota['requests_per_minute'] == 60
        assert quota['cost_per_request'] == 0.01
    
    @pytest.mark.asyncio
    async def test_acquire_api_slot_success(self, api_rate_limiter):
        """Test successful API slot acquisition."""
        api_rate_limiter.set_api_quota('test_service', 60, 0.01)
        
        # Should succeed
        success = await api_rate_limiter.acquire_api_slot('test_service', 0.01)
        assert success is True
        
        # Check that call was recorded
        assert len(api_rate_limiter.api_call_history['test_service']) == 1
        assert api_rate_limiter.api_costs['test_service'] == 0.01
        assert api_rate_limiter.total_cost_today == 0.01
    
    @pytest.mark.asyncio
    async def test_acquire_api_slot_budget_exceeded(self, api_rate_limiter):
        """Test API slot acquisition when budget is exceeded."""
        api_rate_limiter.cost_budget_daily = 1.0
        api_rate_limiter.total_cost_today = 0.9
        
        # Should fail due to budget
        success = await api_rate_limiter.acquire_api_slot('test_service', 0.2)
        assert success is False
    
    @pytest.mark.asyncio
    async def test_acquire_api_slot_rate_limited(self, api_rate_limiter):
        """Test API slot acquisition when rate limited."""
        # Set very low rate limit for testing
        api_rate_limiter.set_api_quota('test_service', 1, 0.01)  # 1 request per minute
        
        # First request should succeed
        success1 = await api_rate_limiter.acquire_api_slot('test_service', 0.01)
        assert success1 is True
        
        # Second request should be delayed (we'll mock the sleep)
        with patch('asyncio.sleep') as mock_sleep:
            success2 = await api_rate_limiter.acquire_api_slot('test_service', 0.01)
            assert success2 is True
            mock_sleep.assert_called_once()  # Should have waited
    
    def test_get_api_stats(self, api_rate_limiter):
        """Test getting API statistics."""
        # Add some usage
        api_rate_limiter.api_costs['service1'] = 1.5
        api_rate_limiter.api_costs['service2'] = 0.8
        api_rate_limiter.total_cost_today = 2.3
        
        # Add some call history
        current_time = datetime.now()
        api_rate_limiter.api_call_history['service1'].extend([
            current_time - timedelta(seconds=30),
            current_time - timedelta(seconds=10)
        ])
        
        stats = api_rate_limiter.get_api_stats()
        
        assert stats['total_cost_today'] == 2.3
        assert stats['budget_remaining'] == 47.7  # 50.0 - 2.3
        assert 'services' in stats
        assert 'service1' in stats['services']
        assert stats['services']['service1']['total_cost'] == 1.5


class TestPerformanceOptimizer:
    """Test PerformanceOptimizer functionality."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def cache_manager(self, temp_cache_dir):
        """Create CacheManager for testing."""
        return CacheManager(cache_dir=temp_cache_dir, max_memory_entries=10)
    
    @pytest.fixture
    def performance_optimizer(self, cache_manager):
        """Create PerformanceOptimizer for testing."""
        return PerformanceOptimizer(cache_manager, monitoring_interval=0.1)
    
    def test_performance_optimizer_initialization(self, performance_optimizer):
        """Test PerformanceOptimizer initialization."""
        assert performance_optimizer.cache_manager is not None
        assert performance_optimizer.resource_monitor is not None
        assert performance_optimizer.batch_processor is not None
        assert performance_optimizer.api_rate_limiter is not None
        assert len(performance_optimizer.performance_profiles) > 0
    
    def test_default_performance_profiles(self, performance_optimizer):
        """Test default performance profiles."""
        profiles = performance_optimizer.performance_profiles
        
        assert 'fast' in profiles
        assert 'balanced' in profiles
        assert 'high_quality' in profiles
        assert 'memory_constrained' in profiles
        
        # Check fast profile characteristics
        fast_profile = profiles['fast']
        assert fast_profile.max_memory_gb == 4.0
        assert fast_profile.batch_size == 1
        assert fast_profile.cache_aggressiveness == 1.0
        
        # Check high quality profile characteristics
        hq_profile = profiles['high_quality']
        assert hq_profile.max_memory_gb == 16.0
        assert hq_profile.parallel_workers == 4
        assert hq_profile.cache_aggressiveness == 0.5
    
    @patch('ai_video_editor.core.performance_optimizer.psutil')
    @pytest.mark.asyncio
    async def test_initialize_with_profile(self, mock_psutil, performance_optimizer):
        """Test initializing with a specific profile."""
        # Mock psutil for resource monitoring
        mock_psutil.cpu_percent.return_value = 50.0
        mock_memory = Mock()
        mock_memory.total = 16 * 1024**3
        mock_memory.available = 8 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.disk_io_counters.return_value = None
        mock_psutil.net_io_counters.return_value = None
        
        # Initialize with balanced profile
        await performance_optimizer.initialize('balanced')
        
        assert performance_optimizer.current_profile is not None
        assert performance_optimizer.current_profile.name == 'balanced'
        assert performance_optimizer.resource_monitor.is_monitoring
        
        # Cleanup
        await performance_optimizer.shutdown()
    
    @pytest.mark.asyncio
    async def test_initialize_invalid_profile(self, performance_optimizer):
        """Test initializing with invalid profile."""
        with pytest.raises(ValueError, match="Unknown performance profile"):
            await performance_optimizer.initialize('invalid_profile')
    
    @patch('ai_video_editor.core.performance_optimizer.psutil')
    @pytest.mark.asyncio
    async def test_optimize_context_processing(self, mock_psutil, performance_optimizer):
        """Test optimizing context processing."""
        # Mock psutil
        mock_psutil.cpu_percent.return_value = 50.0
        mock_memory = Mock()
        mock_memory.total = 16 * 1024**3
        mock_memory.available = 8 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.disk_io_counters.return_value = None
        mock_psutil.net_io_counters.return_value = None
        
        # Initialize optimizer
        await performance_optimizer.initialize('balanced')
        
        # Create test context
        context = ContentContext(
            project_id='test_project',
            video_files=['test_video.mp4'],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        # Add some test data
        context.key_concepts = ['finance', 'investment', 'education']
        
        # Optimize context
        optimized_context = await performance_optimizer.optimize_context_processing(context)
        
        assert optimized_context is not None
        assert optimized_context.project_id == 'test_project'
        
        # Check that user preferences were updated
        assert optimized_context.user_preferences.batch_size == 2  # Balanced profile
        assert optimized_context.user_preferences.parallel_processing is True
        
        # Check processing metrics
        assert 'performance_optimizer' in optimized_context.processing_metrics.module_processing_times
        
        # Cleanup
        await performance_optimizer.shutdown()
    
    def test_calculate_content_complexity(self, performance_optimizer):
        """Test content complexity calculation."""
        # Create context with various complexity factors
        context = ContentContext(
            project_id='test_project',
            video_files=['test_video.mp4'],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        # Add complexity factors
        context.key_concepts = ['concept' + str(i) for i in range(15)]  # Many concepts
        context.emotional_markers = [Mock() for _ in range(8)]  # Many emotional markers
        context.visual_highlights = [Mock() for _ in range(12)]  # Many visual highlights
        
        complexity = performance_optimizer._calculate_content_complexity(context)
        
        assert 0.0 <= complexity <= 1.0
        assert complexity > 0.5  # Should be high complexity
    
    def test_memory_pressure_relief(self, performance_optimizer):
        """Test memory pressure relief."""
        # Set up a profile
        performance_optimizer.current_profile = performance_optimizer.performance_profiles['balanced']
        original_batch_size = performance_optimizer.current_profile.batch_size
        original_workers = performance_optimizer.current_profile.parallel_workers
        
        # Apply memory pressure relief
        performance_optimizer._apply_memory_pressure_relief()
        
        # Check that parameters were reduced
        assert performance_optimizer.current_profile.batch_size <= original_batch_size
        assert performance_optimizer.current_profile.parallel_workers <= original_workers
        assert performance_optimizer.current_profile.cache_aggressiveness > 0.7
    
    def test_cpu_pressure_relief(self, performance_optimizer):
        """Test CPU pressure relief."""
        # Set up a profile
        performance_optimizer.current_profile = performance_optimizer.performance_profiles['balanced']
        original_workers = performance_optimizer.current_profile.parallel_workers
        original_rate_limit = performance_optimizer.current_profile.api_rate_limit
        
        # Apply CPU pressure relief
        performance_optimizer._apply_cpu_pressure_relief()
        
        # Check that parameters were reduced
        assert performance_optimizer.current_profile.parallel_workers <= original_workers
        assert performance_optimizer.current_profile.api_rate_limit <= original_rate_limit
    
    @patch('ai_video_editor.core.performance_optimizer.psutil')
    def test_get_performance_stats(self, mock_psutil, performance_optimizer):
        """Test getting performance statistics."""
        # Mock psutil
        mock_psutil.cpu_percent.return_value = 50.0
        mock_memory = Mock()
        mock_memory.total = 16 * 1024**3
        mock_memory.available = 8 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.disk_io_counters.return_value = None
        mock_psutil.net_io_counters.return_value = None
        
        # Add some test data
        performance_optimizer.current_profile = performance_optimizer.performance_profiles['balanced']
        performance_optimizer.performance_benchmarks['test_operation'] = 5.2
        
        stats = performance_optimizer.get_performance_stats()
        
        assert 'current_profile' in stats
        assert stats['current_profile'] == 'balanced'
        assert 'resource_metrics' in stats
        assert 'cache_performance' in stats
        assert 'api_usage' in stats
        assert 'benchmarks' in stats
        assert stats['benchmarks']['test_operation'] == 5.2
    
    def test_benchmark_operation(self, performance_optimizer):
        """Test benchmarking operations."""
        performance_optimizer.benchmark_operation('test_op', 3.5)
        
        assert 'test_op' in performance_optimizer.performance_benchmarks
        assert performance_optimizer.performance_benchmarks['test_op'] == 3.5
    
    @patch('ai_video_editor.core.performance_optimizer.psutil')
    @pytest.mark.asyncio
    async def test_shutdown(self, mock_psutil, performance_optimizer):
        """Test shutting down performance optimizer."""
        # Mock psutil
        mock_psutil.cpu_percent.return_value = 50.0
        mock_memory = Mock()
        mock_memory.total = 16 * 1024**3
        mock_memory.available = 8 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.disk_io_counters.return_value = None
        mock_psutil.net_io_counters.return_value = None
        
        # Initialize first
        await performance_optimizer.initialize('fast')
        
        # Should not raise any exceptions
        await performance_optimizer.shutdown()
        
        # Check that monitoring stopped
        assert not performance_optimizer.resource_monitor.is_monitoring


class TestPerformanceMonitorDecorator:
    """Test performance_monitor decorator."""
    
    @pytest.mark.asyncio
    async def test_async_function_monitoring(self):
        """Test monitoring async functions."""
        @performance_monitor("test_async_operation")
        async def test_async_func(x, y):
            await asyncio.sleep(0.1)
            return x + y
        
        result = await test_async_func(2, 3)
        assert result == 5
    
    def test_sync_function_monitoring(self):
        """Test monitoring sync functions."""
        @performance_monitor("test_sync_operation")
        def test_sync_func(x, y):
            time.sleep(0.1)
            return x * y
        
        result = test_sync_func(3, 4)
        assert result == 12
    
    @pytest.mark.asyncio
    async def test_async_function_error_monitoring(self):
        """Test monitoring async functions that raise errors."""
        @performance_monitor("test_error_operation")
        async def test_error_func():
            await asyncio.sleep(0.05)
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            await test_error_func()
    
    def test_sync_function_error_monitoring(self):
        """Test monitoring sync functions that raise errors."""
        @performance_monitor("test_error_operation")
        def test_error_func():
            time.sleep(0.05)
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            test_error_func()


class TestIntegration:
    """Integration tests for performance optimization components."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def cache_manager(self, temp_cache_dir):
        """Create CacheManager for testing."""
        return CacheManager(cache_dir=temp_cache_dir, max_memory_entries=10)
    
    @pytest.fixture
    def performance_optimizer(self, cache_manager):
        """Create PerformanceOptimizer for testing."""
        return PerformanceOptimizer(cache_manager, monitoring_interval=0.1)
    
    @patch('ai_video_editor.core.performance_optimizer.psutil')
    @pytest.mark.asyncio
    async def test_full_optimization_workflow(self, mock_psutil, performance_optimizer):
        """Test complete optimization workflow."""
        # Mock psutil
        mock_psutil.cpu_percent.return_value = 45.0
        mock_memory = Mock()
        mock_memory.total = 16 * 1024**3
        mock_memory.available = 10 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.disk_io_counters.return_value = None
        mock_psutil.net_io_counters.return_value = None
        
        try:
            # Initialize optimizer
            await performance_optimizer.initialize('balanced')
            
            # Create test context
            context = ContentContext(
                project_id='integration_test',
                video_files=['test1.mp4', 'test2.mp4'],
                content_type=ContentType.EDUCATIONAL,
                user_preferences=UserPreferences()
            )
            
            # Add complexity
            context.key_concepts = ['finance', 'investment', 'education', 'analysis']
            context.content_themes = ['learning', 'tutorial']
            
            # Optimize context
            optimized_context = await performance_optimizer.optimize_context_processing(context)
            
            # Verify optimization results
            assert optimized_context.project_id == 'integration_test'
            assert optimized_context.user_preferences.batch_size == 2
            assert optimized_context.user_preferences.parallel_processing is True
            
            # Check that processing metrics were recorded
            assert 'performance_optimizer' in optimized_context.processing_metrics.module_processing_times
            
            # Get performance stats
            stats = performance_optimizer.get_performance_stats()
            assert stats['current_profile'] == 'balanced'
            assert 'resource_metrics' in stats
            
            # Test API rate limiting
            api_success = await performance_optimizer.api_rate_limiter.acquire_api_slot('test_api', 0.01)
            assert api_success is True
            
        finally:
            # Cleanup
            await performance_optimizer.shutdown()
    
    @patch('ai_video_editor.core.performance_optimizer.psutil')
    @pytest.mark.asyncio
    async def test_resource_pressure_handling(self, mock_psutil, performance_optimizer):
        """Test handling resource pressure scenarios."""
        # Mock high resource usage
        mock_psutil.cpu_percent.return_value = 95.0  # High CPU
        mock_memory = Mock()
        mock_memory.total = 8 * 1024**3
        mock_memory.available = 0.5 * 1024**3  # Low available memory
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.disk_io_counters.return_value = None
        mock_psutil.net_io_counters.return_value = None
        
        try:
            # Initialize with memory constrained profile
            await performance_optimizer.initialize('memory_constrained')
            
            # Create context
            context = ContentContext(
                project_id='pressure_test',
                video_files=['large_video.mp4'],
                content_type=ContentType.GENERAL,
                user_preferences=UserPreferences()
            )
            
            # Simulate resource pressure by adding alert callback
            alert_received = False
            
            def pressure_callback(alert: str, metrics: ResourceMetrics):
                nonlocal alert_received
                alert_received = True
            
            performance_optimizer.resource_monitor.add_alert_callback(pressure_callback)
            
            # Let monitoring run briefly to detect high resource usage
            await asyncio.sleep(0.2)
            
            # Optimize context under pressure
            optimized_context = await performance_optimizer.optimize_context_processing(context)
            
            # Verify that optimization adapted to constraints
            assert optimized_context.user_preferences.batch_size == 1  # Reduced batch size
            assert optimized_context.user_preferences.enable_aggressive_caching is True
            
        finally:
            await performance_optimizer.shutdown()
    
    @patch('ai_video_editor.core.performance_optimizer.psutil')
    @pytest.mark.asyncio
    async def test_caching_optimization_integration(self, mock_psutil, performance_optimizer):
        """Test integration between performance optimization and caching."""
        # Mock normal resource usage
        mock_psutil.cpu_percent.return_value = 50.0
        mock_memory = Mock()
        mock_memory.total = 16 * 1024**3
        mock_memory.available = 8 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.disk_io_counters.return_value = None
        mock_psutil.net_io_counters.return_value = None
        
        try:
            # Initialize with high quality profile (less aggressive caching)
            await performance_optimizer.initialize('high_quality')
            
            # Check initial cache TTLs
            initial_api_ttl = performance_optimizer.cache_manager.default_ttls['api_response']
            
            # Create context and optimize
            context = ContentContext(
                project_id='cache_test',
                video_files=['test.mp4'],
                content_type=ContentType.EDUCATIONAL,
                user_preferences=UserPreferences()
            )
            
            await performance_optimizer.optimize_context_processing(context)
            
            # Verify cache settings were adjusted based on profile
            # High quality profile should have moderate caching
            assert performance_optimizer.cache_manager.default_ttls['api_response'] <= initial_api_ttl
            
            # Test with fast profile (more aggressive caching)
            await performance_optimizer.initialize('fast')
            await performance_optimizer.optimize_context_processing(context)
            
            # Fast profile should have more aggressive caching
            assert performance_optimizer.cache_manager.default_ttls['api_response'] >= 7200  # 2 hours
            
        finally:
            await performance_optimizer.shutdown()