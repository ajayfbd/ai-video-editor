"""
Performance Benchmarks for Video Quality Assessment.

This module implements performance benchmarking following the guidelines
in .kiro/steering/performance-guidelines.md for video quality assessment.
"""

import time
import psutil
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class QualityAssessmentBenchmark:
    """Performance benchmark for video quality assessment."""
    video_duration: float  # Duration in seconds
    video_resolution: tuple  # (width, height)
    frames_analyzed: int
    processing_time: float  # Time in seconds
    memory_peak_usage: int  # Peak memory in bytes
    memory_average_usage: int  # Average memory in bytes
    cpu_usage_percent: float  # Average CPU usage
    frames_per_second: float  # Processing rate
    quality_score_accuracy: float  # Accuracy of quality assessment
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'video_duration': self.video_duration,
            'video_resolution': list(self.video_resolution),
            'frames_analyzed': self.frames_analyzed,
            'processing_time': self.processing_time,
            'memory_peak_usage': self.memory_peak_usage,
            'memory_average_usage': self.memory_average_usage,
            'cpu_usage_percent': self.cpu_usage_percent,
            'frames_per_second': self.frames_per_second,
            'quality_score_accuracy': self.quality_score_accuracy
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QualityAssessmentBenchmark':
        data['video_resolution'] = tuple(data['video_resolution'])
        return cls(**data)


@dataclass
class PerformanceTargets:
    """Performance targets for video quality assessment."""
    # Processing time targets (seconds per minute of video)
    educational_content_target: float = 40.0  # 15+ min videos should process in under 10 min
    music_video_target: float = 60.0  # 5-6 min videos should process in under 5 min
    general_content_target: float = 60.0  # 3 min videos should process in under 3 min
    
    # Memory usage targets
    max_memory_usage_gb: float = 16.0  # Stay under 16GB peak usage
    
    # Quality assessment targets
    min_frames_per_second: float = 5.0  # Minimum processing rate
    min_quality_accuracy: float = 0.85  # Minimum quality assessment accuracy
    
    # Resource utilization targets
    max_cpu_usage_percent: float = 80.0  # Maximum CPU usage
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'educational_content_target': self.educational_content_target,
            'music_video_target': self.music_video_target,
            'general_content_target': self.general_content_target,
            'max_memory_usage_gb': self.max_memory_usage_gb,
            'min_frames_per_second': self.min_frames_per_second,
            'min_quality_accuracy': self.min_quality_accuracy,
            'max_cpu_usage_percent': self.max_cpu_usage_percent
        }


class QualityAssessmentProfiler:
    """Performance profiler for video quality assessment."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        self.memory_samples = []
        self.cpu_samples = []
        self.targets = PerformanceTargets()
    
    def start_profiling(self):
        """Start performance profiling."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss
        self.memory_samples = [self.start_memory]
        self.cpu_samples = []
        logger.info("Performance profiling started")
    
    def sample_resources(self):
        """Sample current resource usage."""
        if self.start_time is None:
            return
        
        try:
            memory_usage = self.process.memory_info().rss
            cpu_percent = self.process.cpu_percent()
            
            self.memory_samples.append(memory_usage)
            self.cpu_samples.append(cpu_percent)
            
        except Exception as e:
            logger.debug(f"Failed to sample resources: {e}")
    
    def end_profiling(self, video_duration: float, video_resolution: tuple, 
                     frames_analyzed: int, quality_score_accuracy: float = 0.0) -> QualityAssessmentBenchmark:
        """End profiling and create benchmark."""
        if self.start_time is None:
            raise ValueError("Profiling not started")
        
        end_time = time.time()
        processing_time = end_time - self.start_time
        
        # Calculate metrics
        memory_peak = max(self.memory_samples) if self.memory_samples else self.start_memory
        memory_average = sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else self.start_memory
        cpu_average = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0.0
        frames_per_second = frames_analyzed / processing_time if processing_time > 0 else 0.0
        
        benchmark = QualityAssessmentBenchmark(
            video_duration=video_duration,
            video_resolution=video_resolution,
            frames_analyzed=frames_analyzed,
            processing_time=processing_time,
            memory_peak_usage=memory_peak,
            memory_average_usage=memory_average,
            cpu_usage_percent=cpu_average,
            frames_per_second=frames_per_second,
            quality_score_accuracy=quality_score_accuracy
        )
        
        logger.info(f"Performance profiling completed: {processing_time:.2f}s, {frames_per_second:.1f} fps")
        return benchmark
    
    def check_performance_targets(self, benchmark: QualityAssessmentBenchmark, 
                                content_type: str = "general") -> Dict[str, bool]:
        """Check if benchmark meets performance targets."""
        results = {}
        
        # Processing time target
        target_time_per_minute = getattr(self.targets, f"{content_type}_content_target", self.targets.general_content_target)
        expected_time = (benchmark.video_duration / 60.0) * target_time_per_minute
        results['processing_time_target'] = benchmark.processing_time <= expected_time
        
        # Memory usage target
        memory_gb = benchmark.memory_peak_usage / (1024**3)
        results['memory_usage_target'] = memory_gb <= self.targets.max_memory_usage_gb
        
        # Processing rate target
        results['frames_per_second_target'] = benchmark.frames_per_second >= self.targets.min_frames_per_second
        
        # Quality accuracy target
        results['quality_accuracy_target'] = benchmark.quality_score_accuracy >= self.targets.min_quality_accuracy
        
        # CPU usage target
        results['cpu_usage_target'] = benchmark.cpu_usage_percent <= self.targets.max_cpu_usage_percent
        
        return results


class BenchmarkManager:
    """Manager for storing and analyzing performance benchmarks."""
    
    def __init__(self, benchmark_file: Optional[str] = None):
        self.benchmark_file = benchmark_file or "performance_benchmarks.json"
        self.benchmarks: List[QualityAssessmentBenchmark] = []
        self.load_benchmarks()
    
    def load_benchmarks(self):
        """Load existing benchmarks from file."""
        try:
            if Path(self.benchmark_file).exists():
                with open(self.benchmark_file, 'r') as f:
                    data = json.load(f)
                    self.benchmarks = [QualityAssessmentBenchmark.from_dict(b) for b in data.get('benchmarks', [])]
                logger.info(f"Loaded {len(self.benchmarks)} benchmarks")
        except Exception as e:
            logger.warning(f"Failed to load benchmarks: {e}")
            self.benchmarks = []
    
    def save_benchmarks(self):
        """Save benchmarks to file."""
        try:
            data = {
                'benchmarks': [b.to_dict() for b in self.benchmarks],
                'targets': PerformanceTargets().to_dict()
            }
            with open(self.benchmark_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.benchmarks)} benchmarks")
        except Exception as e:
            logger.error(f"Failed to save benchmarks: {e}")
    
    def add_benchmark(self, benchmark: QualityAssessmentBenchmark):
        """Add new benchmark."""
        self.benchmarks.append(benchmark)
        self.save_benchmarks()
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics from all benchmarks."""
        if not self.benchmarks:
            return {}
        
        processing_times = [b.processing_time for b in self.benchmarks]
        memory_peaks = [b.memory_peak_usage / (1024**3) for b in self.benchmarks]  # Convert to GB
        fps_values = [b.frames_per_second for b in self.benchmarks]
        cpu_values = [b.cpu_usage_percent for b in self.benchmarks]
        
        return {
            'total_benchmarks': len(self.benchmarks),
            'processing_time': {
                'mean': sum(processing_times) / len(processing_times),
                'min': min(processing_times),
                'max': max(processing_times)
            },
            'memory_usage_gb': {
                'mean': sum(memory_peaks) / len(memory_peaks),
                'min': min(memory_peaks),
                'max': max(memory_peaks)
            },
            'frames_per_second': {
                'mean': sum(fps_values) / len(fps_values),
                'min': min(fps_values),
                'max': max(fps_values)
            },
            'cpu_usage_percent': {
                'mean': sum(cpu_values) / len(cpu_values),
                'min': min(cpu_values),
                'max': max(cpu_values)
            }
        }
    
    def get_regression_analysis(self) -> Dict[str, Any]:
        """Analyze performance regression over time."""
        if len(self.benchmarks) < 2:
            return {'status': 'insufficient_data'}
        
        # Compare recent benchmarks with older ones
        recent_count = min(5, len(self.benchmarks) // 2)
        recent_benchmarks = self.benchmarks[-recent_count:]
        older_benchmarks = self.benchmarks[:-recent_count] if len(self.benchmarks) > recent_count else []
        
        if not older_benchmarks:
            return {'status': 'insufficient_historical_data'}
        
        # Calculate averages
        recent_avg_time = sum(b.processing_time for b in recent_benchmarks) / len(recent_benchmarks)
        older_avg_time = sum(b.processing_time for b in older_benchmarks) / len(older_benchmarks)
        
        recent_avg_memory = sum(b.memory_peak_usage for b in recent_benchmarks) / len(recent_benchmarks)
        older_avg_memory = sum(b.memory_peak_usage for b in older_benchmarks) / len(older_benchmarks)
        
        recent_avg_fps = sum(b.frames_per_second for b in recent_benchmarks) / len(recent_benchmarks)
        older_avg_fps = sum(b.frames_per_second for b in older_benchmarks) / len(older_benchmarks)
        
        # Calculate changes
        time_change = ((recent_avg_time - older_avg_time) / older_avg_time) * 100 if older_avg_time > 0 else 0
        memory_change = ((recent_avg_memory - older_avg_memory) / older_avg_memory) * 100 if older_avg_memory > 0 else 0
        fps_change = ((recent_avg_fps - older_avg_fps) / older_avg_fps) * 100 if older_avg_fps > 0 else 0
        
        return {
            'status': 'analysis_complete',
            'recent_benchmarks': len(recent_benchmarks),
            'historical_benchmarks': len(older_benchmarks),
            'processing_time_change_percent': time_change,
            'memory_usage_change_percent': memory_change,
            'fps_change_percent': fps_change,
            'performance_trend': 'improving' if time_change < -5 and fps_change > 5 else 
                               'degrading' if time_change > 5 or fps_change < -5 else 'stable'
        }


def create_quality_assessment_profiler() -> QualityAssessmentProfiler:
    """Factory function to create quality assessment profiler."""
    return QualityAssessmentProfiler()


def create_benchmark_manager(benchmark_file: Optional[str] = None) -> BenchmarkManager:
    """Factory function to create benchmark manager."""
    return BenchmarkManager(benchmark_file)