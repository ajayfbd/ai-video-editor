# API Integration Guide

Complete guide to integrating the AI Video Editor into your applications and workflows programmatically.

## 🧭 Navigation

**New to the API?** Review the [**Developer Guide**](../../../developer/README.md) and [**API Reference**](../../developer/api-reference.md) for technical foundations.

**Need workflow mastery first?** Complete the [**First Video Tutorial**](../first-video.md) and [**content workflows**](../workflows/) before API integration.

**Looking for performance optimization?** Check the [**Performance Tuning Guide**](performance-tuning.md) for API efficiency strategies.

## 🎯 Overview

The AI Video Editor provides comprehensive APIs for:
- **Python API integration**
- **REST API endpoints**
- **Webhook integrations**
- **Custom workflow automation**
- **Third-party platform integration**

## 🚀 Quick Start

### Python API Integration

```python
import asyncio
from ai_video_editor.core.workflow_orchestrator import WorkflowOrchestrator
from ai_video_editor.core.content_context import ContentType
from ai_video_editor.core.config import ProjectSettings, VideoQuality

async def process_video_api():
    # Create orchestrator
    orchestrator = WorkflowOrchestrator()
    
    # Configure processing
    settings = ProjectSettings(
        content_type=ContentType.EDUCATIONAL,
        quality=VideoQuality.HIGH
    )
    
    # Process video
    result = await orchestrator.process_video(
        input_files=["video.mp4"],
        project_settings=settings
    )
    
    return result

# Run processing
result = asyncio.run(process_video_api())
```

### REST API Usage

```bash
# Start API server
python -m ai_video_editor.api.server --port 8000

# Process video via REST API
curl -X POST "http://localhost:8000/api/v1/process" \
  -H "Content-Type: multipart/form-data" \
  -F "video=@video.mp4" \
  -F "content_type=educational" \
  -F "quality=high"
```

## 🐍 Python API Reference

### Core Classes

#### WorkflowOrchestrator

**Main orchestration class for video processing:**

```python
from ai_video_editor.core.workflow_orchestrator import (
    WorkflowOrchestrator,
    WorkflowConfiguration,
    ProcessingMode
)

class WorkflowOrchestrator:
    def __init__(self, config: WorkflowConfiguration = None):
        """Initialize workflow orchestrator."""
        
    async def process_video(
        self,
        input_files: List[str],
        project_settings: ProjectSettings,
        output_dir: str = None
    ) -> ProcessingResult:
        """Process video through complete pipeline."""
        
    async def process_batch(
        self,
        input_files: List[str],
        project_settings: ProjectSettings,
        batch_config: BatchConfiguration = None
    ) -> BatchProcessingResult:
        """Process multiple videos in batch."""
        
    def get_processing_status(self, job_id: str) -> ProcessingStatus:
        """Get status of processing job."""
```

#### ContentContext

**Core data structure for video processing:**

```python
from ai_video_editor.core.content_context import (
    ContentContext,
    ContentType,
    ProcessingStage
)

class ContentContext:
    def __init__(
        self,
        project_id: str,
        video_files: List[str],
        content_type: ContentType
    ):
        """Initialize content context."""
        
    def add_analysis_result(
        self,
        stage: ProcessingStage,
        result: Dict[str, Any]
    ) -> None:
        """Add analysis result to context."""
        
    def get_analysis_result(
        self,
        stage: ProcessingStage
    ) -> Optional[Dict[str, Any]]:
        """Get analysis result from context."""
```

### Processing Configuration

#### ProjectSettings

```python
from ai_video_editor.core.config import (
    ProjectSettings,
    VideoQuality,
    ProcessingMode
)

settings = ProjectSettings(
    content_type=ContentType.EDUCATIONAL,
    quality=VideoQuality.HIGH,
    processing_mode=ProcessingMode.BALANCED,
    enable_parallel_processing=True,
    max_memory_usage_gb=12,
    output_format="mp4",
    custom_config={
        "filler_word_removal": True,
        "concept_detection_sensitivity": 0.8,
        "thumbnail_strategies": ["authority", "curiosity"]
    }
)
```

#### WorkflowConfiguration

```python
from ai_video_editor.core.workflow_orchestrator import WorkflowConfiguration

config = WorkflowConfiguration(
    processing_mode=ProcessingMode.HIGH_QUALITY,
    enable_parallel_processing=True,
    max_concurrent_operations=4,
    resource_monitoring=True,
    error_handling_strategy="continue_on_error",
    checkpoint_frequency=30,  # seconds
    timeout_seconds=3600
)
```

## 🌐 REST API Reference

### API Endpoints

#### Process Video

**POST /api/v1/process**

Process a single video file.

```bash
curl -X POST "http://localhost:8000/api/v1/process" \
  -H "Content-Type: multipart/form-data" \
  -F "video=@video.mp4" \
  -F "content_type=educational" \
  -F "quality=high" \
  -F "processing_mode=balanced"
```

**Response:**
```json
{
  "job_id": "job_20240107_143022",
  "status": "processing",
  "estimated_completion": "2024-01-07T14:38:22Z",
  "progress_url": "/api/v1/status/job_20240107_143022"
}
```

#### Batch Process

**POST /api/v1/batch**

Process multiple videos in batch.

```bash
curl -X POST "http://localhost:8000/api/v1/batch" \
  -H "Content-Type: multipart/form-data" \
  -F "videos=@video1.mp4" \
  -F "videos=@video2.mp4" \
  -F "content_type=educational" \
  -F "coordinate_series=true"
```

#### Get Processing Status

**GET /api/v1/status/{job_id}**

Get current processing status.

```bash
curl "http://localhost:8000/api/v1/status/job_20240107_143022"
```

**Response:**
```json
{
  "job_id": "job_20240107_143022",
  "status": "processing",
  "progress": 0.65,
  "current_stage": "asset_generation",
  "elapsed_time": 420,
  "estimated_remaining": 180,
  "results_available": false
}
```

#### Get Results

**GET /api/v1/results/{job_id}**

Get processing results.

```bash
curl "http://localhost:8000/api/v1/results/job_20240107_143022"
```

**Response:**
```json
{
  "job_id": "job_20240107_143022",
  "status": "completed",
  "results": {
    "video_url": "/api/v1/download/job_20240107_143022/video.mp4",
    "thumbnails": [
      "/api/v1/download/job_20240107_143022/thumbnail_authority.jpg",
      "/api/v1/download/job_20240107_143022/thumbnail_curiosity.jpg"
    ],
    "metadata": {
      "title": "Complete Guide to Financial Planning",
      "description": "Learn financial planning with practical examples...",
      "tags": ["finance", "education", "planning"]
    }
  }
}
```

### API Authentication

**API Key Authentication:**

```bash
# Set API key in header
curl -X POST "http://localhost:8000/api/v1/process" \
  -H "Authorization: Bearer your_api_key_here" \
  -H "Content-Type: multipart/form-data" \
  -F "video=@video.mp4"
```

**JWT Token Authentication:**

```bash
# Get JWT token
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass"}'

# Use JWT token
curl -X POST "http://localhost:8000/api/v1/process" \
  -H "Authorization: Bearer jwt_token_here" \
  -F "video=@video.mp4"
```

## 🔗 Webhook Integration

### Webhook Configuration

**Configure webhooks for processing events:**

```python
from ai_video_editor.api.webhooks import WebhookManager

webhook_manager = WebhookManager()

# Register webhook for processing completion
webhook_manager.register_webhook(
    event="processing_completed",
    url="https://your-app.com/webhooks/video-processed",
    headers={"Authorization": "Bearer your_webhook_secret"}
)

# Register webhook for processing errors
webhook_manager.register_webhook(
    event="processing_failed",
    url="https://your-app.com/webhooks/video-error",
    headers={"Authorization": "Bearer your_webhook_secret"}
)
```

### Webhook Events

**Available webhook events:**

- `processing_started`: Video processing has begun
- `processing_progress`: Processing progress update (every 10%)
- `processing_completed`: Video processing completed successfully
- `processing_failed`: Video processing failed with error
- `batch_completed`: Batch processing completed
- `resource_warning`: System resource usage warning

**Webhook payload example:**

```json
{
  "event": "processing_completed",
  "job_id": "job_20240107_143022",
  "timestamp": "2024-01-07T14:45:22Z",
  "data": {
    "input_file": "video.mp4",
    "processing_time": 480,
    "output_files": {
      "video": "enhanced_video.mp4",
      "thumbnails": ["thumb1.jpg", "thumb2.jpg"],
      "metadata": "metadata.json"
    },
    "metrics": {
      "memory_peak": 8.5,
      "api_cost": 1.25,
      "quality_score": 0.92
    }
  }
}
```

## 🔧 Custom Integration Examples

### Flask Application Integration

```python
from flask import Flask, request, jsonify
import asyncio
from ai_video_editor.core.workflow_orchestrator import WorkflowOrchestrator

app = Flask(__name__)
orchestrator = WorkflowOrchestrator()

@app.route('/process-video', methods=['POST'])
def process_video():
    # Get uploaded file
    video_file = request.files['video']
    content_type = request.form.get('content_type', 'general')
    
    # Save uploaded file
    video_path = f"uploads/{video_file.filename}"
    video_file.save(video_path)
    
    # Process video asynchronously
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    result = loop.run_until_complete(
        orchestrator.process_video(
            input_files=[video_path],
            project_settings=ProjectSettings(
                content_type=ContentType(content_type),
                quality=VideoQuality.HIGH
            )
        )
    )
    
    return jsonify({
        'status': 'completed',
        'job_id': result.job_id,
        'output_files': result.output_files
    })

if __name__ == '__main__':
    app.run(debug=True)
```

### Django Integration

```python
# views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import asyncio
from ai_video_editor.core.workflow_orchestrator import WorkflowOrchestrator

@csrf_exempt
@require_http_methods(["POST"])
def process_video_view(request):
    # Handle file upload
    video_file = request.FILES['video']
    content_type = request.POST.get('content_type', 'general')
    
    # Save file
    with open(f'media/uploads/{video_file.name}', 'wb+') as destination:
        for chunk in video_file.chunks():
            destination.write(chunk)
    
    # Process video
    orchestrator = WorkflowOrchestrator()
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    result = loop.run_until_complete(
        orchestrator.process_video(
            input_files=[f'media/uploads/{video_file.name}'],
            project_settings=ProjectSettings(
                content_type=ContentType(content_type),
                quality=VideoQuality.HIGH
            )
        )
    )
    
    return JsonResponse({
        'status': 'success',
        'job_id': result.job_id,
        'results': result.to_dict()
    })
```

### FastAPI Integration

```python
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import asyncio
from ai_video_editor.core.workflow_orchestrator import WorkflowOrchestrator

app = FastAPI()
orchestrator = WorkflowOrchestrator()

@app.post("/process-video")
async def process_video(
    video: UploadFile = File(...),
    content_type: str = Form("general"),
    quality: str = Form("high")
):
    # Save uploaded file
    video_path = f"uploads/{video.filename}"
    with open(video_path, "wb") as buffer:
        content = await video.read()
        buffer.write(content)
    
    # Process video
    result = await orchestrator.process_video(
        input_files=[video_path],
        project_settings=ProjectSettings(
            content_type=ContentType(content_type),
            quality=VideoQuality(quality)
        )
    )
    
    return JSONResponse({
        "status": "completed",
        "job_id": result.job_id,
        "output_files": result.output_files,
        "metadata": result.metadata
    })

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    status = orchestrator.get_processing_status(job_id)
    return JSONResponse(status.to_dict())
```

## 🔄 Workflow Automation

### Automated Processing Pipeline

```python
import asyncio
import os
from pathlib import Path
from ai_video_editor.core.workflow_orchestrator import WorkflowOrchestrator
from ai_video_editor.utils.file_watcher import FileWatcher

class AutomatedProcessor:
    def __init__(self, watch_dir: str, output_dir: str):
        self.watch_dir = Path(watch_dir)
        self.output_dir = Path(output_dir)
        self.orchestrator = WorkflowOrchestrator()
        self.file_watcher = FileWatcher(watch_dir)
        
    async def start_watching(self):
        """Start watching for new video files."""
        self.file_watcher.on_file_added = self.process_new_file
        await self.file_watcher.start()
        
    async def process_new_file(self, file_path: str):
        """Process newly added video file."""
        if not file_path.endswith(('.mp4', '.mov', '.avi')):
            return
            
        # Determine content type from filename or directory
        content_type = self.detect_content_type(file_path)
        
        # Process video
        result = await self.orchestrator.process_video(
            input_files=[file_path],
            project_settings=ProjectSettings(
                content_type=content_type,
                quality=VideoQuality.HIGH
            ),
            output_dir=str(self.output_dir)
        )
        
        # Send notification
        await self.send_completion_notification(result)
        
    def detect_content_type(self, file_path: str) -> ContentType:
        """Detect content type from file path."""
        path_lower = file_path.lower()
        
        if 'educational' in path_lower or 'tutorial' in path_lower:
            return ContentType.EDUCATIONAL
        elif 'music' in path_lower or 'performance' in path_lower:
            return ContentType.MUSIC
        else:
            return ContentType.GENERAL
            
    async def send_completion_notification(self, result):
        """Send notification when processing completes."""
        # Implementation depends on your notification system
        # Could be email, Slack, webhook, etc.
        pass

# Usage
async def main():
    processor = AutomatedProcessor(
        watch_dir="/path/to/input/videos",
        output_dir="/path/to/output"
    )
    await processor.start_watching()

if __name__ == "__main__":
    asyncio.run(main())
```

### Scheduled Processing

```python
import schedule
import time
import asyncio
from ai_video_editor.core.workflow_orchestrator import WorkflowOrchestrator

class ScheduledProcessor:
    def __init__(self):
        self.orchestrator = WorkflowOrchestrator()
        
    async def process_daily_batch(self):
        """Process daily batch of videos."""
        input_dir = "/path/to/daily/videos"
        video_files = [
            f for f in os.listdir(input_dir) 
            if f.endswith(('.mp4', '.mov', '.avi'))
        ]
        
        if not video_files:
            return
            
        # Process batch
        result = await self.orchestrator.process_batch(
            input_files=[os.path.join(input_dir, f) for f in video_files],
            project_settings=ProjectSettings(
                content_type=ContentType.EDUCATIONAL,
                quality=VideoQuality.HIGH
            )
        )
        
        # Generate report
        await self.generate_batch_report(result)
        
    def generate_batch_report(self, result):
        """Generate processing report."""
        # Implementation for report generation
        pass

# Schedule processing
processor = ScheduledProcessor()

# Schedule daily processing at 2 AM
schedule.every().day.at("02:00").do(
    lambda: asyncio.run(processor.process_daily_batch())
)

# Keep the scheduler running
while True:
    schedule.run_pending()
    time.sleep(60)
```

## 📊 Monitoring and Analytics

### Processing Metrics Collection

```python
from ai_video_editor.utils.metrics import MetricsCollector

class ProcessingMonitor:
    def __init__(self):
        self.metrics = MetricsCollector()
        
    async def monitor_processing(self, job_id: str):
        """Monitor processing job and collect metrics."""
        while True:
            status = orchestrator.get_processing_status(job_id)
            
            # Collect metrics
            self.metrics.record_metric(
                name="processing_progress",
                value=status.progress,
                tags={"job_id": job_id, "stage": status.current_stage}
            )
            
            self.metrics.record_metric(
                name="memory_usage",
                value=status.memory_usage_gb,
                tags={"job_id": job_id}
            )
            
            if status.status in ["completed", "failed"]:
                break
                
            await asyncio.sleep(10)
            
    def generate_analytics_report(self):
        """Generate analytics report from collected metrics."""
        return self.metrics.generate_report()
```

## 📚 Best Practices

### API Integration Best Practices

1. **Error Handling**: Implement comprehensive error handling
2. **Rate Limiting**: Respect API rate limits
3. **Async Processing**: Use asynchronous processing for better performance
4. **Resource Management**: Monitor and manage system resources
5. **Security**: Implement proper authentication and authorization

### Performance Optimization

1. **Connection Pooling**: Use connection pooling for database connections
2. **Caching**: Implement caching for frequently accessed data
3. **Batch Processing**: Use batch processing for multiple videos
4. **Resource Monitoring**: Monitor system resources during processing
5. **Graceful Degradation**: Handle resource constraints gracefully

### Security Considerations

1. **Input Validation**: Validate all input parameters
2. **File Upload Security**: Secure file upload handling
3. **API Authentication**: Implement proper API authentication
4. **Data Encryption**: Encrypt sensitive data in transit and at rest
5. **Access Control**: Implement proper access control mechanisms

---

*Integrate AI Video Editor seamlessly into your applications and workflows*