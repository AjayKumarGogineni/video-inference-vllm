# Video Analysis Pipeline with Vision Language Models

A modular pipeline for analyzing videos using Vision Language Models (VLMs). Supports key frame extraction, audio transcription, and structured JSON output generation. The current pipeline uses Qwen2.5-VL-32B-Instruct by default.

## Project Structure

```
.
├── config.json              # Configuration file
├── main.py                  # Main entry point
├── requirements.txt         # Python dependencies
├── utils/
│   ├── config_loader.py    # Configuration loading
│   ├── model_loader.py     # VLM model initialization
│   ├── video_processor.py  # Video frame extraction
│   ├── audio_processor.py  # Audio extraction & transcription
│   ├── key_frame_extractor.py  # Key frame detection
│   ├── gpu_monitor.py      # GPU memory monitoring
│   └── statistics.py       # Statistics logging
└── README.md
```

## Features

- **Flexible Frame Loading**: Extract frames uniformly or use pre-extracted key frames
- **Audio Transcription**: Automatic audio extraction and Whisper-based transcription
- **Key Frame Extraction**: Scene-based key frame detection using PySceneDetect or OpenCV
- **Structured Output**: JSON schema-enforced output using Pydantic models
- **GPU Monitoring**: Track memory usage during inference
- **Comprehensive Statistics**: Detailed timing and performance metrics


## Clone the Repository

```bash
git clone <repository_url>
```
## Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```
Use the uv package manager for fast, reliable Python package installation and environment management.

```bash
pip install uv
uv venv
source .venv/bin/activate
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.json` to customize the pipeline:

### Key Configuration Options

**PATHS**: Input/output directory paths
- `VIDEO_FOLDER`: Source videos directory
- `PROMPT_FILE`: System instruction prompt
- `OUTPUT_FOLDER`: Generated JSON outputs (set via MODE)

**MODEL**: VLM configuration
- `MODEL_NAME`: HuggingFace model identifier
- `MAX_TOKENS`: Maximum generation length
- `TEMPERATURE`, `TOP_P`: Sampling parameters

**VIDEO_PROCESSING**: Frame extraction settings
- `NUM_SEGMENTS`: Number of frames to extract per video
- `INPUT_SIZE`: Frame resize dimension

**FEATURES**: Toggle functionality
- `TRANSCRIBE_AUDIO`: Enable audio transcription
- `EXTRACT_KEY_FRAMES`: Run key frame extraction
- `USE_KEY_FRAMES`: Use pre-extracted key frames instead of uniform sampling

**SAMPLING**: Test on subset
- `SAMPLE`: Enable sampling mode
- `SAMPLE_SIZE`: Number of videos to process

## Usage

### Create ground truth using Gemini model

Update the .env file with your Gemini API key.

```bash
python create_ground_truth_gemini.py
```

### Create ground truth using Openai model

The openai models do not take the video as input directly. You need to extract the key frames and audio transcript first using the pipeline and then use those to create the ground truth using the openai models. Documentation: https://platform.openai.com/docs/models/gpt-4o-mini
Reference code: https://platform.openai.com/docs/guides/images-vision?api-mode=responses&format=base64-encoded#analyze-images

Update the .env file with your Openai API key.

Run the create_ground_truth_openai.py script to create the ground truth using openai models.

```bash
python create_ground_truth_openai.py
```


### Basic Usage

```bash
python main.py
```

### Custom Config

```bash
python main.py path/to/custom_config.json
```

### Workflow Steps

1. **[Required the first time the videos are loaded, can be set to false once the transcripts are generated] Transcribe Audio**: If `TRANSCRIBE_AUDIO: true`
   - Runs BEFORE model loading
   - Extracts audio as MP3 from videos
   - Transcribes using Whisper Large V3
   - Skips videos with existing transcripts

2. **[Optional] Extract Key Frames**: If `EXTRACT_KEY_FRAMES: true`
   - Runs BEFORE model loading
   - Detects scene changes using PySceneDetect or OpenCV
   - Saves frames to `KEY_FRAMES_FOLDER`
   - Skips already processed videos

3. **Load Model**: Initialize VLM with tensor parallelism

4. **Load System Instructions**: Read instructions from `PROMPT_FILE`

5. **Process Videos**:
   - Load frames (uniform sampling or key frames)
   - Load audio transcript
   - Append the frames and the transcript to the system instructions to create a prompt
   - Run inference: Pass the prompt to the Vision language model for generating response
   - Save the generations as a JSON file, check the generations and update the system instructions if needed

6. **Save Statistics**: Timing and memory metrics saved to `STATISTICS_FILE`

## Output for Qwen and Intern vl scripts

### Generated Files

- `json/{VIDEO_ID}.json`: Structured analysis for each video
- `csv/file_name.csv`: CSV summary of all videos
- `missed_videos.txt`: List of failed videos
- Statistics file: Comprehensive performance metrics
- Bleu score for the generated summaries by comparing them with ground truth summaries and classification accuracy of the generated categories.

### JSON Schema

```python
{
    "summary": str,
    "category": str,
    "color_composition": str,
    "keywords": str,
    "creator_demographic": str,
    "political_leaning": str,
    "misinformation_indicators": str,
    "sentiment": str,
    "tone": str,
    "hateful_content": str,
    "hateful_explanation": str,
    "hateful_target": str,
    "hateful_severity": str,
    "graphic_content": str,
    "threatening_content": str,
    "illicit_content": str,
    "self_harm": str
}
```

### Things required to change the output schema:
- Change the system prompt file to include/exclude fields or modify instructions
- Update the OutputJson class in utils/model_loader.py to reflect any schema changes

## Performance Tracking

The pipeline tracks:
- Model load time
- Video load time per video
- Inference time per video
- GPU memory usage per video
- Total processing time

Statistics are logged to console and saved to the configured statistics file.

## Requirements

- Python 3.11
- CUDA-capable GPU
- HuggingFace account (for model access) -> Optional
- Sufficient disk space for videos and outputs

## Environment Variables

```bash
export HF_HOME=/path/to/huggingface/cache  # Optional: Custom HF cache location
```

## Notes

- Audio transcription uses Whisper Large V3 by default
- Key frame extraction supports parallel processing
- Model uses tensor parallelism across all available GPUs
- All paths in config.json use absolute paths for cluster compatibility
- The Qwen 32B model requires 150 GB of total GPU memory out of which 66 GB is needed to load the model and around 80GB is required for the KV cache used by VLLM during inference.

## Troubleshooting

**Out of Memory**: Reduce `NUM_SEGMENTS` or enable key frames with lower frame count

**Missing Audio**: Some videos may not have audio tracks (logged as warnings)

**Failed Videos**: Check `missed_videos.txt` for the list of videos that failed during processing

**Slow Processing**: Adjust `MAX_WORKERS` for key frame extraction or check GPU utilization