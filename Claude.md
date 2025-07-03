# XVerse Installation Guide

This guide documents the complete setup process to successfully run the XVerse Gradio app, based on our actual installation experience and debugging process.

## Overview

XVerse is a consistent multi-subject control image generation app using DiT (Diffusion Transformer) modulation. The setup involves several critical steps including environment creation, dependency installation, system packages, Git LFS handling, and code fixes for PyTorch 2.7+ compatibility.

## Prerequisites

- Ubuntu 22.04 (or similar Linux distribution)
- NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
- Git and Git LFS
- Miniconda/Anaconda
- Sufficient disk space (~50GB+ for all models)
- Good internet connection (models are large)

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://huggingface.co/spaces/alexnasa/XVerse
cd XVerse
```

### 2. Install System Dependencies

Install required OpenGL and multimedia libraries first:

```bash
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 ffmpeg
```

**Why needed**: OpenCV requires OpenGL libraries, and the app will fail with `libGL.so.1: cannot open shared object file` without these.

### 3. Install and Configure Git LFS

The repository uses Git LFS for model files and sample images:

```bash
# Install Git LFS
sudo apt-get install -y git-lfs

# Initialize Git LFS for this repository
git lfs install

# Pull all LFS files (models and sample images)
git lfs pull
```

**Critical**: This step is essential! Many files appear as tiny pointer files (132 bytes) until you run `git lfs pull`. You'll know it worked when:
- `checkpoints/model_ir_se50.pth` is ~175MB (not 132 bytes)
- Sample images in `sample/` are proper image files
- `checkpoints/sam2.1_hiera_large.pt` is ~898MB

### 4. Create Conda Environment

```bash
# Create Python 3.10 environment
conda create -n xverse python=3.10 -y

# Activate the environment (you'll need to do this every time)
conda activate xverse
```

### 5. Install Python Dependencies

Install from the existing requirements.txt:

```bash
pip install -r requirements.txt
```

### 6. Install Missing Dependencies

The original requirements.txt is missing some critical packages:

```bash
pip install gradio spaces protobuf
```

**Or update requirements.txt permanently**:
```bash
echo "protobuf" >> requirements.txt
# gradio and spaces are typically installed separately for Hugging Face Spaces
```

### 7. Fix PyTorch 2.7+ Compatibility Issue

**Critical Fix**: PyTorch 2.7+ changed security defaults. Edit `eval/tools/face_id.py` line 59:

**From:**
```python
self.model.load_state_dict(torch.load(face_model_path, map_location=device))
```

**To:**
```python
self.model.load_state_dict(torch.load(face_model_path, map_location=device, weights_only=False))
```

**Why needed**: Without this, you'll get a `WeightsUnpickler error: Unsupported operand` or similar pickle/unpickling error.

### 8. Enable Public Sharing (Optional)

To create a public shareable link, modify `app.py` line 721:

**From:**
```python
demo.launch()
```

**To:**
```python
demo.launch(share=True)
```

### 9. Create Required Directories

```bash
mkdir -p proprocess_data checkpoints
```

### 10. Run the Application

```bash
# Always activate environment first
conda activate xverse

# Run the app
python app.py
```

## What Happens on First Run

The app will automatically download several large models (~25GB total):

1. **FLUX.1-schnell** - Main diffusion model (~24GB)
2. **Florence-2-large** - Vision-language model (~1.5GB)  
3. **CLIP ViT-Large** - Image-text embeddings (~1.7GB)
4. **DINO ViT-s16** - Vision transformer (~87MB)
5. **mPLUG VQA** - Visual question answering (~3GB)
6. **XVerse modulation adapters** - Custom control modules (~1.2GB)

Plus these from Git LFS:
7. **SAM2.1** - Segment Anything Model (~898MB)
8. **Face ID model** - Face recognition (~175MB)

## Troubleshooting Common Issues

### 1. `ImportError: libGL.so.1: cannot open shared object file`
**Problem**: Missing OpenGL libraries
**Solution**: 
```bash
sudo apt-get install -y libgl1-mesa-glx
```

### 2. `protobuf library but it was not found`
**Problem**: Missing protobuf dependency
**Solution**: 
```bash
pip install protobuf
```

### 3. Model files are tiny (132 bytes) 
**Problem**: Git LFS files not downloaded
**Solution**: 
```bash
git lfs install
git lfs pull
```
**Verify**: Check file sizes with `ls -la checkpoints/`

### 4. `torch.load` fails with pickle/weights_only error
**Problem**: PyTorch 2.7+ security changes
**Solution**: Add `weights_only=False` to torch.load calls (see step 7)

### 5. `cannot identify image file` during example caching
**Problem**: Sample images are Git LFS pointers, not actual images
**Solution**: Ensure `git lfs pull` completed successfully

### 6. CUDA/GPU errors
**Problem**: Insufficient VRAM or CUDA issues
**Solution**: 
- Ensure you have 8GB+ VRAM
- The app automatically handles CUDA versions
- Try reducing batch size or image resolution

### 7. App starts but examples don't work
**Problem**: Sample images not properly downloaded
**Solution**: 
```bash
git lfs pull
ls -la sample/  # Should show proper file sizes, not 132 bytes
```

## File Structure After Installation

```
XVerse/
├── app.py                     # Main Gradio app (modified for share=True)
├── requirements.txt           # Dependencies (modified to add protobuf)
├── eval/tools/face_id.py     # Face recognition (modified for PyTorch 2.7+)
├── sample/                   # Example images (from Git LFS)
├── checkpoints/              # Model files (Git LFS + auto-downloaded)
│   ├── model_ir_se50.pth    # Face model (~175MB)
│   ├── sam2.1_hiera_large.pt # SAM model (~898MB)
│   ├── FLUX.1-schnell/      # Main diffusion model
│   ├── Florence-2-large/    # Vision-language model
│   └── ...                  # Other models
├── proprocess_data/          # Temporary processing (created at runtime)
└── Claude.md                # This guide
```

## Performance and Hardware Notes

- **GPU**: 8GB+ VRAM minimum, 16GB+ recommended
- **RAM**: 16GB+ system RAM for large models
- **Storage**: 50GB+ free space (models are large)
- **First run**: 10-15 minutes for model downloads
- **Generation time**: 15-30 seconds per image
- **Concurrent users**: Limited by GPU memory

## Success Indicators

When everything works correctly:

1. ✅ All models load without errors
2. ✅ Gradio interface starts on `http://127.0.0.1:7860`  
3. ✅ Example caching completes successfully
4. ✅ Public sharing link generated (if enabled)
5. ✅ Sample images display properly in examples
6. ✅ Image generation works without CUDA errors

## Key Commands Summary

```bash
# Full installation from scratch
git clone https://huggingface.co/spaces/alexnasa/XVerse
cd XVerse
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 ffmpeg git-lfs
git lfs install
git lfs pull
conda create -n xverse python=3.10 -y
conda activate xverse
pip install -r requirements.txt
pip install gradio spaces protobuf

# Edit eval/tools/face_id.py line 59 to add weights_only=False
# Edit app.py line 721 to add share=True (optional)

python app.py
```

## Maintenance and Updates

- **Disk cleanup**: Monitor `proprocess_data/` and `.gradio/` cache directories
- **Dependencies**: Update with `pip install -r requirements.txt --upgrade`
- **Model updates**: Check XVerse repository for newer model versions
- **Git LFS**: Periodically run `git lfs pull` to get updated models

---

*This guide documents our complete debugging journey from initial clone to successful deployment, including all the gotchas and fixes we discovered.*
