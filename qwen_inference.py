import time
import os
import sys
import json
import pandas as pd

from utils.config_loader import load_config
from utils.model_loader import load_model
from utils.video_processor import load_video_frames, load_key_frames
from utils.gpu_monitor import get_gpu_memory
from utils.statistics import save_statistics, print_summary
from utils.evaluation_utils import evaluate_with_stats

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

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
    """Load existing audio transcript (created in preprocessing)."""
    audio_path = os.path.join(
        config['PATHS']['AUDIO_TRANSCRIPT_FOLDER'],
        f"{video_id}.txt"
    )
    if os.path.exists(audio_path):
        with open(audio_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def prepare_multimodal_input(video_id, video_path, config, processor, system_prompt):
    """Prepare data for LLM inference."""
    # Load frames or keyframes
    if config['FEATURES']['USE_KEY_FRAMES']:
        images = load_key_frames(
            video_id,
            config['PATHS']['KEY_FRAMES_FOLDER'],
            config['VIDEO_PROCESSING']['INPUT_SIZE']
        )
        if not images:
            images = load_video_frames(
                video_path,
                config['VIDEO_PROCESSING']['NUM_SEGMENTS'],
                config['VIDEO_PROCESSING']['INPUT_SIZE']
            )
        # config[]
    else:
        images = load_video_frames(
            video_path,
            config['VIDEO_PROCESSING']['NUM_SEGMENTS'],
            config['VIDEO_PROCESSING']['INPUT_SIZE']
        )

    # Audio transcript loaded from preprocessing output
    audio_text = get_audio_transcript(video_id, config)

    # Build multimodal conversation
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": (
            [{"type": "image"} for _ in images] +
            [{"type": "text", "text": f"Audio transcript: {audio_text.strip()}"}]
        )}
    ]

    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    return {
        "prompt": text_prompt,
        "multi_modal_data": {"image": images}
    }


def run_inference(config):
    start_time = time.time()

    # Load video list
    video_ids = get_video_list(config)
    print(f"Found {len(video_ids)} videos to run inference on")

    # STEP 1 — Load model
    print("="*60)
    print("Loading model...")
    print("="*60)
    load_start = time.time()
    llm, processor, sampling_params = load_model(config)
    model_load_time = time.time() - load_start
    model_memory = get_gpu_memory()
    print(f"Model loaded in {model_load_time:.2f}s")
    print(f"GPU memory usage: {model_memory:.2f} GB\n")

    # STEP 2 — Load system prompt
    with open(config['PATHS']['PROMPT_FILE'], "r") as f:
        system_prompt = f.read()

    video_folder = config['PATHS']['VIDEO_FOLDER']

    # Prepare output folders
    output_folder = config['OUTPUT']['OUTPUT_FOLDER']
    input_size = config['VIDEO_PROCESSING']['INPUT_SIZE']
    num_frames = config['VIDEO_PROCESSING']['NUM_SEGMENTS']

    if config['FEATURES']['USE_KEY_FRAMES']:
        output_folder += "_keyframes_"
    else:
        output_folder += "_all_frames_"

    output_folder += f"inputsize{input_size}_numframes{num_frames}/"

    json_folder = output_folder + "json/"
    csv_folder = output_folder + "csv/"
    stats_folder = output_folder + "statistics/"

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(json_folder, exist_ok=True)
    os.makedirs(csv_folder, exist_ok=True)
    os.makedirs(stats_folder, exist_ok=True)

    responses = []
    inference_times = []
    video_load_times = []
    video_mem_usages = []

    # Loop all videos
    for idx, vid in enumerate(video_ids, 1):
        video_id = vid.split('.')[0]
        video_path = os.path.join(video_folder, vid)
        print(f"\n[{idx}/{len(video_ids)}] Processing {video_id}...")

        mem_before = get_gpu_memory()
        try:
            # Prepare input
            load_start = time.time()
            multimodal_input = prepare_multimodal_input(
                video_id, video_path, config, processor, system_prompt
            )
            load_t = time.time() - load_start
            video_load_times.append(load_t)

            # Inference
            inf_start = time.time()
            outputs = llm.generate([multimodal_input], sampling_params)
            response = outputs[0].outputs[0].text.replace("```json", "").replace("```", "")
            inf_t = time.time() - inf_start
            inference_times.append(inf_t)

            # Save JSON
            out_path = os.path.join(json_folder, f"{video_id}.json")
            with open(out_path, "w") as f:
                f.write(response)

            responses.append({"video_id": video_id, **json.loads(response)})
            mem_after = get_gpu_memory()
            video_mem_usages.append(mem_after - mem_before)

            print(f"  ✅ Loaded in {load_t:.2f}s | Inference {inf_t:.2f}s")

        except Exception as e:
            print(f"✗ Error processing {video_id}: {e}")
            with open(os.path.join(output_folder, 'missed_videos.txt'), 'a') as f:
                f.write(f"{video_id}\n")
            continue

    # Save CSV
    df = pd.DataFrame(responses)
    df.to_csv(os.path.join(csv_folder, "response.csv"), index=False)

    # Evaluation
    ground_truth = pd.read_csv(config['PATHS']['GROUND_TRUTH_FILE'])
    common = set(ground_truth.video_id.astype(str)) & set(df.video_id.astype(str))
    gt = ground_truth[ground_truth.video_id.astype(str).isin(common)]
    df = df[df.video_id.astype(str).isin(common)]

    gt = gt.sort_values("video_id").reset_index(drop=True)
    df = df.sort_values("video_id").reset_index(drop=True)

    eval_results = evaluate_with_stats(gt, df)
    with open(os.path.join(stats_folder, "evaluation.json"), "w") as f:
        json.dump(eval_results, f, indent=4)

    # Stats
    save_statistics(
        config, config['MODEL']['MODEL_NAME'], model_load_time, model_memory,
        video_load_times, inference_times, [],
        video_mem_usages, time.time() - start_time,
        extract_frames_time=None, extract_audio_time=None,
        statistics_file=os.path.join(stats_folder, "inference_statistics.txt")
    )

    print("\n✅ Inference complete!")


def main():
    config_path = "config_qwen.json"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    config = load_config(config_path)
    run_inference(config)


if __name__ == "__main__":
    main()
