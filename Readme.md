# Video Analysis Pipeline with Vision Language Models

A unified pipeline for analyzing videos using Vision Language Models (Qwen, InternVL) with vLLM inference.

## Quick Start

### 1. Installation

```bash
# Create virtual environment
pip install uv
uv venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Preprocessing (One-Time Setup)

Extract frames and transcribe audio before inference:

```bash
python preprocess.py data/configs/config_preprocess.json
```

### 3. Run Inference

```bash
# Qwen models
python main.py data/configs/config_qwen.json

# InternVL models
python main.py data/configs/config_internvl.json
```

## Running Different Models

### Qwen Models
```json
{
  "MODEL": {
    "MODEL_NAME": "Qwen/Qwen2.5-VL-32B-Instruct",
    "MODEL_FAMILY": "qwen",
    "MODEL_SUFFIX": "qwenvl_32b",
    "DTYPE": "half",
    "MAX_TOKENS": 2048,
    "TEMPERATURE": 0.1,
    "TOP_P": 0.9,
    "REPETITION_PENALTY": 1.05
  }
}
```

**Available models:**
- `Qwen/Qwen2.5-VL-7B-Instruct`
- `Qwen/Qwen2.5-VL-32B-Instruct`
- `Qwen/Qwen2.5-VL-32B-Instruct-AWQ` (quantized)

### InternVL Models
```json
{
  "MODEL": {
    "MODEL_NAME": "OpenGVLab/InternVL2_5-26B",
    "MODEL_FAMILY": "internvl",
    "MODEL_SUFFIX": "internvl_26b",
    "DTYPE": "half",
    "MAX_TOKENS": 2048,
    "TEMPERATURE": 0.1,
    "TOP_P": 0.9,
    "REPETITION_PENALTY": 1.05
  }
}
```

**Available models:**
- `OpenGVLab/InternVL2_5-8B`
- `OpenGVLab/InternVL2_5-26B`
- `OpenGVLab/InternVL2_5-78B`

## Running on New Datasets

### 1. Update Config Paths
```json
{
  "PATHS": {
    "VIDEO_FOLDER": "path/to/your/videos/",
    "PROMPT_FILE": "path/to/your/prompt.txt",
    "AUDIO_TRANSCRIPT_FOLDER": "data/inputs/audio_transcripts/",
    "KEY_FRAMES_FOLDER": "data/inputs/key_frames/",
    "UNIFORM_FRAMES_FOLDER": "data/inputs/uniform_frames/",
    "GROUND_TRUTH_FILE": "path/to/ground_truth.csv"  // Optional
  },
  "OUTPUT": {
    "OUTPUT_FOLDER": "data/outputs/your_experiment"
  }
}
```

### 2. Update System Prompt
Edit your `PROMPT_FILE` to define what information the model should extract from videos.

### 3. Run Preprocessing
```bash
python preprocess.py config_preprocess.json
```

### 4. Run Inference
```bash
python main.py config_your_model.json
```

## Adding New Models

### 1. Create Model Loader
Create `utils_{model_family}/model_loader.py`:

```python
def load_your_model(config):
    from vllm import LLM, SamplingParams
    
    llm = LLM(
        model=config["MODEL"]["MODEL_NAME"],
        trust_remote_code=True,
        dtype=config["MODEL"]["DTYPE"],
        # Add model-specific parameters
    )
    
    sampling = SamplingParams(
        temperature=config["MODEL"]["TEMPERATURE"],
        top_p=config["MODEL"]["TOP_P"],
        max_tokens=config["MODEL"]["MAX_TOKENS"]
    )
    
    return llm, sampling
```

### 2. Update main.py

Add to `load_model_for_family()`:
```python
elif model_family == 'your_model':
    from utils_your_model.model_loader import load_your_model
    llm, sampling_params = load_your_model(config)
    return llm, None, sampling_params, model_family, True
```

Add to `prepare_multimodal_input()`:
```python
elif model_family == 'your_model':
    return prepare_your_model_input(images, audio_text, system_prompt)
```

Create `prepare_your_model_input()` function:
```python
def prepare_your_model_input(images, audio_text, system_prompt):
    # Format prompt according to your model's requirements
    prompt = f"Your model-specific prompt format with {len(images)} images"
    
    return {
        "prompt": prompt,
        "multi_modal_data": {"image": images}
    }
```

### 3. Create Config File
```json
{
  "MODEL": {
    "MODEL_NAME": "your-org/your-model",
    "MODEL_FAMILY": "your_model",
    "MODEL_SUFFIX": "your_model_suffix",
    "DTYPE": "half",
    "MAX_TOKENS": 2048
  }
}
```

## Configuration Reference

### Core Settings

**PATHS**
- `VIDEO_FOLDER`: Input videos directory
- `PROMPT_FILE`: System instruction prompt
- `AUDIO_TRANSCRIPT_FOLDER`: Audio transcripts location
- `KEY_FRAMES_FOLDER`: Extracted key frames location
- `UNIFORM_FRAMES_FOLDER`: Uniform frames location
- `GROUND_TRUTH_FILE`: Ground truth for evaluation (optional)

**MODEL**
- `MODEL_NAME`: HuggingFace model ID
- `MODEL_FAMILY`: `"qwen"` or `"internvl"`
- `MODEL_SUFFIX`: Short name for output folders
- `DTYPE`: `"half"` (fp16) or `"bfloat16"`
- `MAX_TOKENS`: Maximum generation length
- `TEMPERATURE`: Sampling temperature (0.0-1.0)
- `TOP_P`: Nucleus sampling threshold
- `REPETITION_PENALTY`: Repetition penalty (1.0 = no penalty)

**VIDEO_PROCESSING**
- `NUM_SEGMENTS`: Number of frames per video (recommended: 30)
- `INPUT_SIZE`: Frame resize dimension (recommended: 360 or 224)

**FEATURES**
- `USE_KEY_FRAMES`: Use scene-based key frames (variable count)
- `USE_UNIFORM_FRAMES`: Use uniformly sampled frames (fixed count)
- If both false: Extract frames on-the-fly

**SAMPLING**
- `SAMPLE`: Process subset for testing
- `SAMPLE_SIZE`: Number of videos to process

**EVALUATION**
- `CALCULATE_METRICS`: Enable/disable evaluation against ground truth

### Preprocessing Configuration

```json
{
  "FEATURES": {
    "TRANSCRIBE_AUDIO": true,
    "EXTRACT_KEY_FRAMES": false,
    "EXTRACT_UNIFORM_FRAMES": true
  },
  "AUDIO_TRANSCRIPTION": {
    "WHISPER_MODEL": "openai/whisper-large-v3",
    "DEVICE_ID": 0,
    "CHUNK_LENGTH_S": 30,
    "BATCH_SIZE": 16
  },
  "KEY_FRAME_EXTRACTION": {
    "DETECTION_MODE": "scenedetect",
    "MAX_WORKERS": 8,
    "JPEG_QUALITY": 85,
    "THRESHOLD": 30.0,
    "MIN_SCENE_LEN": 15
  }
}
```

## Output Structure

```
data/outputs/{model_suffix}_inputsize{size}_numframes{frames}/
├── json/              # Per-video JSON responses
├── csv/
│   └── response.csv   # Aggregated results
├── statistics/
│   ├── inference_statistics.txt
│   └── evaluation.json
└── missed_videos.txt  # Failed videos with errors
```

## Memory Requirements

- **Qwen 32B**: ~150GB GPU memory
- **InternVL 26B**: ~145GB GPU memory
- **InternVL 8B**: ~35GB GPU memory

**Optimization tips:**
- Reduce `NUM_SEGMENTS` to lower memory usage
- Use key frames for variable frame counts
- Set `SAMPLE: true` for testing

## Troubleshooting

**Out of Memory**
- Reduce `NUM_SEGMENTS`
- Use smaller model variant
- Use key frames with lower threshold

**Model Not Found**
- Verify `MODEL_NAME` matches HuggingFace model ID
- Set `HF_TOKEN` for gated models: `export HF_TOKEN=your_token`

**Evaluation Skipped**
- Evaluation only runs if `CALCULATE_METRICS: true` AND `GROUND_TRUTH_FILE` exists
- Missing ground truth is not an error

**Failed Videos**
- Check `missed_videos.txt` for error details
- Common issues: corrupted videos, missing audio, unsupported formats

## Environment Variables

```bash
export HF_HOME=/path/to/cache          # HuggingFace cache
export HF_TOKEN=your_token             # For gated models
export CUDA_VISIBLE_DEVICES=0,1,2,3    # GPU selection
```

## Requirements

- Python 3.11+
- CUDA-capable GPU(s)
- Sufficient GPU memory (see Memory Requirements)
- Disk space for videos, frames, and outputs