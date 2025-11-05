import time
import os
import sys
import torch
import sglang as sgl

from utils.config_loader import load_config
from utils.video_processor import load_video_frames, load_key_frames
from utils.audio_processor import extract_and_transcribe_all
from utils.key_frame_extractor import extract_key_frames_batch
from utils.gpu_monitor import get_gpu_memory
from utils.statistics import save_statistics, print_summary

def get_video_list(config):
    """Get list of videos to process."""
    video_folder = config['PATHS']['VIDEO_FOLDER']
    video_ids = [
        vid for vid in os.listdir(video_folder) 
        if vid.endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ]
    
    if config['SAMPLING']['SAMPLE']:
        video_ids = video_ids[:config['SAMPLING']['SAMPLE_SIZE']]
    
    return video_ids

def get_audio_transcript(video_id, config):
    """Load existing audio transcript."""
    audio_file = os.path.join(config['PATHS']['AUDIO_TRANSCRIPT_FOLDER'], f'{video_id}.txt')
    if os.path.exists(audio_file):
        with open(audio_file, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# def prepare_multimodal_input(video_id, video_path, config, system_instruction):
#     """Prepare messages for SGLang inference (matches vLLM style)."""
#     # Load frames
#     if config['FEATURES']['USE_KEY_FRAMES']:
#         images = load_key_frames(video_id, config['PATHS']['KEY_FRAMES_FOLDER'])
#         if not images:
#             print(f"No key frames found for {video_id}, falling back to video frames")
#             images = load_video_frames(
#                 video_path,
#                 config['VIDEO_PROCESSING']['NUM_SEGMENTS'],
#                 config['VIDEO_PROCESSING']['INPUT_SIZE']
#             )
#     else:
#         images = load_video_frames(
#             video_path,
#             config['VIDEO_PROCESSING']['NUM_SEGMENTS'],
#             config['VIDEO_PROCESSING']['INPUT_SIZE']
#         )
    
#     # Get audio transcript
#     audio_transcript = get_audio_transcript(video_id, config)
    
#     # Build messages
#     messages = [
#         {"role": "system", "content": system_instruction},
#         {"role": "user", "content": [
#             *[{"type": "image", "image": img} for img in images],
#             {"type": "text", "text": f"Audio transcript: {audio_transcript.strip()}"}
#         ]}
#     ]
    
#     return messages

def prepare_multimodal_input(video_id, video_path, config, system_instruction):
    """Prepare messages for SGLang inference correctly."""
    # Load frames
    if config['FEATURES']['USE_KEY_FRAMES']:
        images = load_key_frames(video_id, config['PATHS']['KEY_FRAMES_FOLDER'])
        if not images:
            images = load_video_frames(
                video_path,
                config['VIDEO_PROCESSING']['NUM_SEGMENTS'],
                config['VIDEO_PROCESSING']['INPUT_SIZE']
            )
    else:
        images = load_video_frames(
            video_path,
            config['VIDEO_PROCESSING']['NUM_SEGMENTS'],
            config['VIDEO_PROCESSING']['INPUT_SIZE']
        )
    audio_transcript = get_audio_transcript(video_id, config)
    # Build content as a simple list - SGLang handles the conversion
    content_parts = []
    # Add images
    for img in images:
        content_parts.append({"type": "image", "image": img})
    # Add text
    content_parts.append({
        "type": "text", 
        "text": f"Audio transcript: {audio_transcript.strip()}"
    })
    # Build SGLang messages
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": content_parts}
    ]
    
    return messages


@sgl.function
def video_analysis_prompt(s, messages, sampling_params):
    # Pass the system message
    s += sgl.system(messages[0]["content"])
    
    # Pass the user message content (list of dicts)
    s += sgl.user(messages[1]["content"])
    
    # Generate response
    s += sgl.assistant(
        sgl.gen(
            "response",
            temperature=sampling_params["temperature"],
            top_p=sampling_params["top_p"],
            max_tokens=sampling_params["max_tokens"],
            json_schema=sampling_params.get("json_schema")
        )
    )


def run_inference(config):
    """Main inference pipeline using SGLang, updated to vLLM structure."""
    start_time = time.time()
    extract_frames_time = None
    extract_audio_time = None
    
    video_ids = get_video_list(config)
    print(f"Found {len(video_ids)} videos to process")
    step = 1
    
    # STEP 1: Extract key frames
    if config['FEATURES']['EXTRACT_KEY_FRAMES']:
        print("\n" + "="*60)
        print(f"STEP {step}: Extracting key frames from videos...")
        print("="*60)
        extract_start = time.time()
        extract_key_frames_batch(
            config['PATHS']['VIDEO_FOLDER'],
            config['PATHS']['KEY_FRAMES_FOLDER'],
            config
        )
        extract_frames_time = time.time() - extract_start
        print(f"Key frame extraction complete in {extract_frames_time:.2f}s\n")
        step += 1
    
    # STEP 2: Extract and transcribe audio
    if config['FEATURES']['TRANSCRIBE_AUDIO']:
        print("="*60)
        print(f"STEP {step}: Extracting and transcribing audio...")
        print("="*60)
        extract_audio_start = time.time()
        extract_and_transcribe_all(video_ids, config['PATHS']['VIDEO_FOLDER'], config)
        extract_audio_time = time.time() - extract_audio_start
        print(f"Audio transcription complete in {extract_audio_time:.2f}s\n")
        step += 1
    
    # STEP 3: Initialize SGLang runtime
    print("="*60)
    print(f"STEP {step}: Initializing SGLang runtime...")
    print("="*60)
    model_load_start = time.time()
    num_gpus = torch.cuda.device_count()
    runtime = sgl.Runtime(
        model_path=config['MODEL']['MODEL_NAME'],
        tokenizer_path=config['MODEL']['MODEL_NAME'],
        tp_size=num_gpus if num_gpus > 0 else 1,
        dtype=config['MODEL']['DTYPE']
    )
    sgl.set_default_backend(runtime)
    model_load_time = time.time() - model_load_start
    model_memory = get_gpu_memory()
    print(f"SGLang runtime initialized in {model_load_time:.2f}s")
    print(f"Model GPU memory: {model_memory:.2f} GB\n")
    step += 1
    
    # STEP 4: Load system prompt
    print("="*60)
    print(f"STEP {step}: Loading system prompt...")
    print("="*60)
    with open(config['PATHS']['PROMPT_FILE'], 'r') as f:
        system_instruction = f.read()
    print("System prompt loaded\n")
    step += 1
    
    # STEP 5: Prepare sampling parameters
    sampling_params = {
        "temperature": config['MODEL']['TEMPERATURE'],
        "top_p": config['MODEL']['TOP_P'],
        "max_tokens": config['MODEL']['MAX_TOKENS']
    }
    try:
        from utils.model_loader import OutputJson
        sampling_params["json_schema"] = OutputJson.model_json_schema()
    except:
        print("Warning: Could not load JSON schema, proceeding without structured output")
    
    # STEP 6: Process videos
    print("="*60)
    print(f"STEP {step}: Processing {len(video_ids)} videos...")
    print("="*60)
    
    video_load_times, inference_times, video_mem_usages = [], [], []
    video_folder = config['PATHS']['VIDEO_FOLDER']
    output_folder = config['OUTPUT']['OUTPUT_FOLDER']
    statistics_file = config['OUTPUT']['STATISTICS_FILE']
    
    if config["FEATURES"]["EXTRACT_KEY_FRAMES"]:
        output_folder += "_keyframes"
        statistics_file = statistics_file.replace(".txt", "_keyframes.txt")
    output_folder += '/'
    
    for idx, vid in enumerate(video_ids, 1):
        video_id = vid.split('.')[0]
        video_path = os.path.join(video_folder, vid)
        print(f"\n[{idx}/{len(video_ids)}] Processing {video_id}...")
        mem_before = get_gpu_memory()
        
        try:
            # Prepare multimodal input
            video_load_start = time.time()
            messages = prepare_multimodal_input(video_id, video_path, config, system_instruction)
            video_load_time = time.time() - video_load_start
            video_load_times.append(video_load_time)
            print(f"  → Video loaded in {video_load_time:.2f}s")
            
            # Run inference
            inference_start = time.time()
            state = video_analysis_prompt.run(messages=messages, sampling_params=sampling_params)
            response = state["response"].replace('```json', '').replace('```', '').strip()
            inference_time = time.time() - inference_start
            inference_times.append(inference_time)
            
            # Save output
            output_path = os.path.join(output_folder, f'{video_id}.json')
            with open(output_path, 'w') as f:
                f.write(response)
            
            mem_after = get_gpu_memory()
            video_mem_usages.append(mem_after - mem_before)
            
            print(f"  → Inference completed in {inference_time:.2f}s")
            print(f"  → Output saved to {output_path}")
            
        except Exception as e:
            print(f'  ✗ Error processing video {video_id}: {e}')
            with open(os.path.join(output_folder, 'missed_videos.txt'), 'a') as f:
                f.write(f'{video_id}\n')
            continue
    
    # Shutdown runtime
    runtime.shutdown()
    
    total_time = time.time() - start_time
    
    # Save and print statistics
    print("\n" + "="*60)
    print("Saving statistics...")
    print("="*60)
    save_statistics(
        config, config['MODEL']['MODEL_NAME'], model_load_time, model_memory,
        video_load_times, inference_times, [], video_mem_usages, total_time,
        extract_frames_time, extract_audio_time, statistics_file
    )
    
    print_summary(
        video_load_times, inference_times, [],
        video_mem_usages, model_memory, total_time, extract_frames_time, extract_audio_time
    )
    
    print(f"\nAll outputs saved to: {output_folder}")
    print(f"Statistics saved to: {statistics_file}")

def main():
    """Entry point."""
    config_path = 'config_sglang.json'
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    print("="*60)
    print("Video Analysis Pipeline (SGLang)")
    print("="*60)
    config = load_config(config_path)
    print(f"Configuration loaded from: {config_path}\n")
    
    run_inference(config)

if __name__ == "__main__":
    main()
