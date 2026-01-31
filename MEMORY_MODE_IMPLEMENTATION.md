# Memory Mode Implementation Plan

## Overview

This document outlines the current memory-optimized setup (working on RTX4090/24GB VRAM) and provides an implementation plan for adding a switch to enable high-VRAM mode for beefier machines.

---

## Current Setup (Lightweight Mode - WORKING)

### Configuration
- **Target Hardware**: RTX4090 (24GB VRAM) or similar constrained systems
- **Status**: ✅ **Tested and working**
- **Git Commit**: `f519462` ("Progress")

### Memory Optimizations Applied

#### 1. Pipeline Memory Management
- **Location**: `batch_infer_flash_pro.py`, line 334
- **Implementation**: `pipeline.enable_sequential_cpu_offload()`
- **Effect**: Models are automatically moved to CPU when not in use, reducing VRAM usage
- **Alternative (disabled)**: `pipeline.to(device=device)` - commented out

#### 2. Sequential Weight Loading
- **Location**: `batch_infer_flash_pro.py`, lines 264, 298
- **Implementation**: 
  - Transformer: `low_cpu_mem_usage=True` (line 264)
  - Text Encoder: `low_cpu_mem_usage=True` (line 298)
- **Effect**: Loads model weights sequentially to avoid RAM spikes during loading
- **Trade-off**: Slower initial loading, but prevents OOM errors

#### 3. Audio Encoder on CPU
- **Location**: `batch_infer_flash_pro.py`, line 250
- **Implementation**: `audio_encoder = Wav2Vec2Model.from_pretrained(...).to('cpu')`
- **Effect**: Wav2Vec2 model stays on CPU, reducing VRAM usage
- **Note**: Audio features are moved to GPU only when needed (line 396)

#### 4. Memory Fragmentation Control
- **Location**: `run_batch_flash_pro.sh`, line 6
- **Implementation**: `export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`
- **Effect**: Limits CUDA memory fragmentation by capping split sizes
- **Trade-off**: Prevents fragmentation but may limit allocation efficiency

#### 5. TeaCache Configuration
- **Location**: `batch_infer_flash_pro.py`, line 344
- **Implementation**: `offload=False` (cache stays on GPU)
- **Effect**: TeaCache coefficients remain on GPU for faster access

### Current Code Structure

```python
# In load_models() function:
# 1. Audio encoder → CPU
audio_encoder = Wav2Vec2Model.from_pretrained(...).to('cpu')

# 2. Transformer → Sequential loading
transformer = WanTransformer.from_pretrained(
    ...,
    low_cpu_mem_usage=True,  # Sequential weight loading
    torch_dtype=weight_dtype,
)

# 3. Text encoder → Sequential loading
text_encoder = WanT5EncoderModel.from_pretrained(
    ...,
    low_cpu_mem_usage=True,  # Sequential weight loading
    torch_dtype=weight_dtype,
)

# 4. Pipeline → Sequential CPU offload
pipeline.enable_sequential_cpu_offload()  # Instead of pipeline.to(device)
```

### Shell Script Configuration

```bash
# run_batch_flash_pro.sh
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Fragmentation control
python batch_infer_flash_pro.py [args...]
```

---

## Proposed Implementation: Memory Mode Switch

### Goal
Add a single `--memory_mode` flag that toggles between:
- **`lightweight`** (default): Current optimizations for constrained VRAM
- **`beefy`**: Direct GPU loading for high-VRAM systems

### Implementation Plan

#### Step 1: Add Command-Line Argument

**File**: `batch_infer_flash_pro.py`

**Location**: In `parse_args()` function, around line 102 (GPU and memory section)

```python
parser.add_argument(
    "--memory_mode", 
    type=str, 
    default="lightweight",
    choices=["lightweight", "beefy"],
    help="Memory management mode: 'lightweight' for constrained VRAM (RTX4090), 'beefy' for high-VRAM systems"
)
```

#### Step 2: Modify `load_models()` Function

**File**: `batch_infer_flash_pro.py`

**Location**: `load_models()` function (lines 240-359)

**Changes needed**:

1. **Audio Encoder Placement** (line 250):
   ```python
   # Current (lightweight):
   audio_encoder = Wav2Vec2Model.from_pretrained(...).to('cpu')
   
   # Beefy mode:
   audio_encoder = Wav2Vec2Model.from_pretrained(...).to(device)
   ```

2. **Transformer Loading** (line 264):
   ```python
   # Current (lightweight):
   low_cpu_mem_usage=True
   
   # Beefy mode:
   low_cpu_mem_usage=False  # Faster parallel loading
   ```

3. **Text Encoder Loading** (line 298):
   ```python
   # Current (lightweight):
   low_cpu_mem_usage=True
   
   # Beefy mode:
   low_cpu_mem_usage=False  # Faster parallel loading
   ```

4. **Pipeline Memory Mode** (line 334):
   ```python
   # Current (lightweight):
   pipeline.enable_sequential_cpu_offload()
   
   # Beefy mode:
   pipeline.to(device=device)  # All models stay on GPU
   ```

#### Step 3: Update Shell Script

**File**: `run_batch_flash_pro.sh`

**Changes**:

1. **Conditional CUDA Allocator Config**:
   ```bash
   # Lightweight mode (default)
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   
   # Beefy mode (if --memory_mode beefy is passed)
   # Either remove the export or increase the limit:
   # export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048
   ```

2. **Add Memory Mode Flag** (optional, for convenience):
   ```bash
   # Add to the python command:
   --memory_mode "lightweight"  # or "beefy"
   ```

#### Step 4: Implementation Logic

**Recommended approach**: Use conditional logic in `load_models()`

```python
def load_models(args):
    # ... existing code ...
    
    memory_mode = args.memory_mode  # "lightweight" or "beefy"
    is_lightweight = (memory_mode == "lightweight")
    
    # Audio encoder placement
    if is_lightweight:
        audio_encoder = Wav2Vec2Model.from_pretrained(...).to('cpu')
    else:
        audio_encoder = Wav2Vec2Model.from_pretrained(...).to(device)
    
    # Transformer loading
    transformer = WanTransformer.from_pretrained(
        ...,
        low_cpu_mem_usage=is_lightweight,  # True for lightweight, False for beefy
        torch_dtype=weight_dtype,
    )
    
    # Text encoder loading
    text_encoder = WanT5EncoderModel.from_pretrained(
        ...,
        low_cpu_mem_usage=is_lightweight,  # True for lightweight, False for beefy
        torch_dtype=weight_dtype,
    )
    
    # Pipeline memory mode
    if is_lightweight:
        pipeline.enable_sequential_cpu_offload()
        print("  ✓ Pipeline created with sequential CPU offload (lightweight mode)")
    else:
        pipeline.to(device=device)
        print("  ✓ Pipeline created with direct GPU loading (beefy mode)")
    
    # ... rest of the code ...
```

---

## Mode Comparison

| Feature | Lightweight Mode (Current) | Beefy Mode (Proposed) |
|---------|---------------------------|----------------------|
| **Pipeline Memory** | `enable_sequential_cpu_offload()` | `pipeline.to(device)` |
| **Model Loading** | `low_cpu_mem_usage=True` | `low_cpu_mem_usage=False` |
| **Audio Encoder** | CPU (`.to('cpu')`) | GPU (`.to(device)`) |
| **CUDA Allocator** | `max_split_size_mb:512` | `max_split_size_mb:2048` or removed |
| **TeaCache Offload** | `offload=False` | `offload=False` |
| **VRAM Usage** | Lower (~12-16GB) | Higher (~20-24GB+) |
| **Inference Speed** | Slower (CPU-GPU transfers) | Faster (all on GPU) |
| **Loading Speed** | Slower (sequential) | Faster (parallel) |
| **Target Hardware** | RTX4090, 24GB VRAM | 48GB+ VRAM systems |

---

## Testing Checklist

### Lightweight Mode (Default)
- [ ] Verify models load sequentially
- [ ] Verify audio encoder on CPU
- [ ] Verify pipeline uses sequential CPU offload
- [ ] Verify VRAM usage stays within limits
- [ ] Test batch processing with multiple videos

### Beefy Mode
- [ ] Verify all models load to GPU directly
- [ ] Verify audio encoder on GPU
- [ ] Verify pipeline uses direct GPU loading
- [ ] Verify faster inference times
- [ ] Test batch processing with multiple videos
- [ ] Monitor VRAM usage (should be higher but acceptable)

---

## Backward Compatibility

- **Default behavior**: `--memory_mode` defaults to `"lightweight"`, preserving current behavior
- **No breaking changes**: Existing scripts and workflows continue to work unchanged
- **Opt-in**: Beefy mode must be explicitly requested

---

## Future Enhancements (Optional)

1. **Auto-detection**: Automatically detect available VRAM and suggest appropriate mode
2. **Model Compilation**: Add `torch.compile()` option for beefy mode
3. **Batch Parallelism**: Process multiple videos concurrently in beefy mode
4. **Memory Profiling**: Add memory usage reporting for both modes

---

## Notes

- The current setup (lightweight mode) is **guaranteed to work** on RTX4090
- Beefy mode is an optimization for systems with abundant VRAM
- The switch is simple and clean - just one flag controls all memory-related optimizations
- All changes are isolated to `load_models()` function and argument parsing

---

## References

- Git commit with optimizations: `f519462` ("Progress")
- Original optimization: Changed from `pipeline.to(device)` to `pipeline.enable_sequential_cpu_offload()`
- Related commits: `ac6f5c5` ("Release unused memory"), `95ea4b0` ("12G VRAM is all U need")

