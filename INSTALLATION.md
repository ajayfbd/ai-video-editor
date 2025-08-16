# Installation Guide

This guide provides detailed installation instructions for AI Video Editor across different platforms and use cases.

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.9 or higher
- **RAM**: 8GB (16GB recommended)
- **Storage**: 5GB free space
- **Internet**: Required for AI services
- **OS**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 20.04+)

### Recommended Requirements
- **CPU**: Intel i7 11th gen or equivalent
- **RAM**: 32GB for large video processing
- **GPU**: 2GB VRAM for acceleration (optional)
- **Storage**: SSD with 20GB+ free space

## 🚀 Quick Installation

### 1. Install Python
Ensure Python 3.9+ is installed:
```bash
python --version  # Should show 3.9 or higher
```

**Install Python if needed:**
- **Windows**: Download from [python.org](https://python.org)
- **macOS**: `brew install python` or download from python.org
- **Linux**: `sudo apt install python3.9 python3.9-pip`

### 2. Clone Repository
```bash
git clone https://github.com/your-username/ai-video-editor.git
cd ai-video-editor
```

### 3. Install AI Video Editor
```bash
# Install with pip (recommended)
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### 4. Configure Environment
```bash
# Copy configuration template
cp .env.example .env

# Edit .env with your API keys
# Required: AI_VIDEO_EDITOR_GEMINI_API_KEY
```

### 5. Verify Installation
```bash
# Test the installation
python -c "import ai_video_editor; print('Installation successful!')"

# Run basic tests
python -m pytest tests/unit/ -v
```

## 🔧 Detailed Installation Options

### Option 1: Standard Installation (Recommended)
```bash
# Clone and install
git clone https://github.com/your-username/ai-video-editor.git
cd ai-video-editor
pip install -e .

# Configure
cp .env.example .env
# Edit .env with your API keys
```

### Option 2: Development Installation
```bash
# Clone and install with dev dependencies
git clone https://github.com/your-username/ai-video-editor.git
cd ai-video-editor
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run full test suite
python -m pytest tests/ -v
```

### Option 3: Virtual Environment (Isolated)
```bash
# Create virtual environment
python -m venv ai-video-editor-env

# Activate environment
# Windows:
ai-video-editor-env\Scripts\activate
# macOS/Linux:
source ai-video-editor-env/bin/activate

# Install
pip install -e .
```

### Option 4: Docker Installation (Future)
```bash
# Docker support coming soon
# docker pull ai-video-editor:latest
# docker run -it ai-video-editor
```

## 🔑 API Configuration

### Required: Gemini API Key
1. **Get API Key**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. **Create Key**: Follow the setup instructions
3. **Configure**: Add to `.env` file:
   ```bash
   AI_VIDEO_EDITOR_GEMINI_API_KEY=your_actual_api_key_here
   ```

### Optional: Additional APIs
```bash
# Future integrations (not currently used)
# AI_VIDEO_EDITOR_IMAGEN_API_KEY=your_imagen_key
# AI_VIDEO_EDITOR_GOOGLE_CLOUD_PROJECT=your_project_id
```

## 🖥️ Platform-Specific Instructions

### Windows Installation
```powershell
# Install Python from python.org
# Open PowerShell as Administrator

# Install Git (if not installed)
winget install Git.Git

# Clone and install
git clone https://github.com/your-username/ai-video-editor.git
cd ai-video-editor
python -m pip install --upgrade pip
pip install -e .

# Configure
copy .env.example .env
# Edit .env with notepad or your preferred editor
```

### macOS Installation
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and Git
brew install python git

# Clone and install
git clone https://github.com/your-username/ai-video-editor.git
cd ai-video-editor
pip3 install -e .

# Configure
cp .env.example .env
# Edit .env with nano, vim, or your preferred editor
```

### Linux Installation (Ubuntu/Debian)
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install python3.9 python3.9-pip python3.9-venv git ffmpeg -y

# Clone and install
git clone https://github.com/your-username/ai-video-editor.git
cd ai-video-editor
python3.9 -m pip install -e .

# Configure
cp .env.example .env
# Edit .env with nano, vim, or your preferred editor
```

### Linux Installation (CentOS/RHEL/Fedora)
```bash
# Install dependencies
sudo dnf install python3.9 python3-pip git ffmpeg -y
# or for older versions: sudo yum install python3 python3-pip git

# Clone and install
git clone https://github.com/your-username/ai-video-editor.git
cd ai-video-editor
python3 -m pip install -e .

# Configure
cp .env.example .env
```

## 🧪 Verification and Testing

### Basic Verification
```bash
# Test Python import
python -c "import ai_video_editor; print('✅ Import successful')"

# Test CLI access
ai-ve --help

# Test configuration
python -c "from ai_video_editor.core.config import get_config; print('✅ Config loaded')"
```

### Run Test Suite
```bash
# Quick tests (unit tests only)
python -m pytest tests/unit/ -v

# Full test suite
python -m pytest tests/ -v

# With coverage report
python -m pytest tests/ --cov=ai_video_editor --cov-report=html
```

### Performance Test
```bash
# Test system performance
python -c "
import psutil
print(f'CPU cores: {psutil.cpu_count()}')
print(f'RAM: {psutil.virtual_memory().total // (1024**3)}GB')
print(f'Available RAM: {psutil.virtual_memory().available // (1024**3)}GB')
"
```

## 🔧 Troubleshooting Installation

### Common Issues

#### Python Version Issues
```bash
# Check Python version
python --version
python3 --version

# Use specific Python version
python3.9 -m pip install -e .
```

#### Permission Issues (Linux/macOS)
```bash
# Use user installation
pip install --user -e .

# Or fix permissions
sudo chown -R $USER:$USER ~/.local/
```

#### FFmpeg Not Found
```bash
# Windows: Download from https://ffmpeg.org/
# macOS: brew install ffmpeg
# Linux: sudo apt install ffmpeg
```

#### Memory Issues
```bash
# Reduce memory usage in .env
AI_VIDEO_EDITOR_MAX_MEMORY_USAGE_GB=4.0
AI_VIDEO_EDITOR_MAX_CONCURRENT_PROCESSES=1
```

#### API Key Issues
```bash
# Verify API key format
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
key = os.getenv('AI_VIDEO_EDITOR_GEMINI_API_KEY')
print(f'API key configured: {bool(key and key != \"your_gemini_api_key_here\")}')
"
```

### Getting Help
1. **Check Documentation**: [docs/](docs/)
2. **Search Issues**: [GitHub Issues](https://github.com/your-username/ai-video-editor/issues)
3. **Create Issue**: Use bug report template
4. **Community Support**: GitHub Discussions

## 🚀 Next Steps

After successful installation:

1. **Quick Start**: Follow [Quick Start Guide](quick-start.md)
2. **First Video**: Complete [First Video Tutorial](docs/tutorials/first-video.md)
3. **Explore Features**: Check [User Guide](docs/user-guide/README.md)
4. **Join Community**: Contribute via [Contributing Guide](CONTRIBUTING.md)

## 📊 Installation Verification Checklist

- [ ] Python 3.9+ installed and accessible
- [ ] Repository cloned successfully
- [ ] AI Video Editor package installed
- [ ] Environment configuration completed
- [ ] API keys configured (at minimum Gemini)
- [ ] Basic import test passes
- [ ] CLI commands accessible
- [ ] Unit tests pass
- [ ] System meets minimum requirements

## 🔄 Updating Installation

### Update to Latest Version
```bash
# Pull latest changes
git pull origin main

# Reinstall package
pip install -e .

# Run tests to verify
python -m pytest tests/unit/ -v
```

### Update Dependencies
```bash
# Update all dependencies
pip install --upgrade -r requirements.txt

# Or reinstall with latest versions
pip uninstall ai-video-editor
pip install -e .
```

---

**Need help?** Check our [Troubleshooting Guide](docs/support/troubleshooting-unified.md) or create an issue on GitHub.