import time
import os
import sys

from utils.config_loader import load_config
from utils.model_loader import load_model
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
        sample_size = config['SAMPLING']['SAMPLE_SIZE']
        video_ids = video_ids[:sample_size]
    
    return video_ids

def get_audio_transcript(video_id, config):
    """Load existing audio transcript."""
    audio_transcript_file = os.path.join(
        config['PATHS']['AUDIO_TRANSCRIPT_FOLDER'], 
        f'{video_id}.txt'
    )
    
    if os.path.exists(audio_transcript_file):
        with open(audio_transcript_file, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

def prepare_multimodal_input(video_id, video_path, config, processor, system_instruction):
    """Prepare multimodal input for model inference."""
    # Load frames (either from video or pre-extracted key frames)
    if config['FEATURES']['USE_KEY_FRAMES']:
        images = load_key_frames(video_id, config['PATHS']['KEY_FRAMES_FOLDER'])
        if not images:
            print(f"No key frames found for {video_id}, falling back to video frames")
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
    
    # Get audio transcript
    audio_transcript = get_audio_transcript(video_id, config)
    
    # Build conversation
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
        {"role": "user", "content": (
            [{"type": "image"} for _ in images] +
            [{"type": "text", "text": f"Audio transcript: {audio_transcript.strip()}"}]
        )}
    ]
    
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    multimodal_input = {
        "prompt": text_prompt,
        "multi_modal_data": {"image": images}
    }
    
    return multimodal_input

def run_inference(config):
    """Main inference pipeline."""
    start_time = time.time()
    extract_frames_time = None
    extract_audio_time = None
    
    # Get video list first
    video_ids = get_video_list(config)
    print(f"Found {len(video_ids)} videos to process")
    step = 1
    extract_frames_start_time = time.time()
    # STEP 1: Extract key frames if enabled (before model loading)
    if config['FEATURES']['EXTRACT_KEY_FRAMES']:
        print("\n" + "="*60)
        print(f"STEP {step}: Extracting key frames from videos...")
        print("="*60)
        extract_key_frames_batch(
            config['PATHS']['VIDEO_FOLDER'],
            config['PATHS']['KEY_FRAMES_FOLDER'],
            config
        )
        print("Key frame extraction complete!\n")
        step += 1
        extract_frames_end_time = time.time()
        extract_frames_time = extract_frames_end_time - extract_frames_start_time
        print(f"Key frame extraction took {extract_frames_time:.2f}s\n")
    
    # STEP 2: Transcribe audio if enabled (before model loading)
    extract_audio_start_time = time.time()
    if config['FEATURES']['TRANSCRIBE_AUDIO']:
        print("="*60)
        print(f"STEP {step}: Extracting and transcribing audio...")
        print("="*60)
        extract_and_transcribe_all(
            video_ids,
            config['PATHS']['VIDEO_FOLDER'],
            config
        )
        print("Audio transcription complete!\n")
        step += 1
        extract_audio_end_time = time.time()
        extract_audio_time = extract_audio_end_time - extract_audio_start_time
        print(f"Audio extraction and transcription took {extract_audio_time:.2f}s\n")
    
    # STEP 3: Load model
    print("="*60)
    print(f"STEP {step}: Loading model...")
    print("="*60)
    model_load_start = time.time()
    llm, processor, sampling_params = load_model(config)
    model_load_time = time.time() - model_load_start
    model_memory = get_gpu_memory()
    print(f"Model loaded in {model_load_time:.2f}s")
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
    
    # STEP 5: Process videos
    print("="*60)
    print(f"STEP {step}: Processing {len(video_ids)} videos...")
    print("="*60)
    step += 1
    
    # Initialize timing trackers
    video_load_times = []
    inference_times = []
    video_mem_usages = []
    
    video_folder = config['PATHS']['VIDEO_FOLDER']
    output_folder = config['OUTPUT']['OUTPUT_FOLDER']
    statistics_file = config['OUTPUT']['STATISTICS_FILE']

    if config["FEATURES"]["EXTRACT_KEY_FRAMES"]:
        output_folder += "_keyframes"
        statistics_file = statistics_file.replace(".txt", "_keyframes.txt")
    output_folder += '/'


    
    # Process each video
    for idx, vid in enumerate(video_ids, 1):
        video_id = vid.split('.')[0]
        video_path = os.path.join(video_folder, vid)
        
        print(f"\n[{idx}/{len(video_ids)}] Processing {video_id}...")
        
        # Track memory before processing
        mem_before = get_gpu_memory()
        
        try:
            # Load video and prepare input
            video_load_start = time.time()
            multimodal_input = prepare_multimodal_input(
                video_id, video_path, config, processor, system_instruction
            )
            video_load_time = time.time() - video_load_start
            video_load_times.append(video_load_time)
            print(f"  → Video loaded in {video_load_time:.2f}s")
            
            # Run inference
            inference_start = time.time()
            outputs = llm.generate([multimodal_input], sampling_params)
            response = outputs[0].outputs[0].text.replace('```json', '').replace('```', '')
            inference_time = time.time() - inference_start
            inference_times.append(inference_time)
            
            # Save output
            output_path = os.path.join(output_folder, f'{video_id}.json')
            with open(output_path, 'w') as f:
                f.write(response)
            
            # Track memory after processing
            mem_after = get_gpu_memory().copy()
            video_mem_usages.append(mem_after - mem_before)
            
            print(f"  → Inference completed in {inference_time:.2f}s")
            print(f"  → Output saved to {output_path}")
            
        except Exception as e:
            print(f'  ✗ Error processing video {video_id}: {e}')
            with open(os.path.join(output_folder, 'missed_videos.txt'), 'a') as f:
                f.write(f'{video_id}\n')
            continue
    
    total_time = time.time() - start_time
    
    # Save and print statistics
    print("\n" + "="*60)
    print("Saving statistics...")
    print("="*60)
    save_statistics(
        config, config['MODEL']['MODEL_NAME'], model_load_time, model_memory,
        video_load_times, inference_times, [], video_mem_usages, total_time, extract_frames_time, extract_audio_time, statistics_file
    )
    
    print_summary(
        video_load_times, inference_times, [],
        video_mem_usages, model_memory, total_time, extract_frames_time, extract_audio_time
    )
    
    print(f"\nAll outputs saved to: {output_folder}")
    print(f"Statistics saved to: {statistics_file}")

def main():
    """Entry point."""
    config_path = 'config.json'
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    print("="*60)
    print("Video Analysis Pipeline")
    print("="*60)
    config = load_config(config_path)
    print(f"Configuration loaded from: {config_path}\n")
    
    run_inference(config)

if __name__ == "__main__":
    main()