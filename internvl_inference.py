import time
import os
import sys
import json
import pandas as pd
import torch

from utils.config_loader import load_config
from utils.gpu_monitor import get_gpu_memory
from utils.statistics import save_statistics, print_summary
from utils.evaluation_utils import evaluate_with_stats

from utils_internvl.model_loader import load_internvl_model
from utils_internvl.video_processor import load_video_frames, load_key_frames

from utils.audio_processor import extract_and_transcribe_all
from utils.key_frame_extractor import extract_key_frames_batch


def get_video_list(config):
    """Return list of videos matching config options."""
    video_folder = config['PATHS']['VIDEO_FOLDER']
    video_ids = [
        vid for vid in os.listdir(video_folder)
        if vid.endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]

    if config['SAMPLING']['SAMPLE']:
        video_ids = video_ids[:config['SAMPLING']['SAMPLE_SIZE']]

    return video_ids


def get_audio_transcript(video_id, config):
    """Load transcript from TXT file."""
    fpath = os.path.join(
        config["PATHS"]["AUDIO_TRANSCRIPT_FOLDER"],
        f"{video_id}.txt"
    )
    if os.path.exists(fpath):
        return open(fpath, "r").read()
    return ""


def prepare_multimodal_input(video_id, video_path, config, system_prompt):
    """Load frames + transcript and prepare vLLM multimodal input."""

    # Key frames or linear sampling
    if config["FEATURES"]["USE_KEY_FRAMES"]:
        images = load_key_frames(
            video_id,
            config["PATHS"]["KEY_FRAMES_FOLDER"],
            config["VIDEO_PROCESSING"]["INPUT_SIZE"]
        )
        if not images:
            images = load_video_frames(
                video_path,
                config["VIDEO_PROCESSING"]["NUM_SEGMENTS"],
                config["VIDEO_PROCESSING"]["INPUT_SIZE"]
            )
    else:
        images = load_video_frames(
            video_path,
            config["VIDEO_PROCESSING"]["NUM_SEGMENTS"],
            config["VIDEO_PROCESSING"]["INPUT_SIZE"]
        )

    audio_transcript = get_audio_transcript(video_id, config)

    # InternVL uses MANUAL prompt (NOT chat template)
    image_tokens = " ".join(["<image>"] * len(images))
    prompt = (
        f"{image_tokens}\n"
        f"Audio transcript: {audio_transcript.strip()}\n"
        f"The video has {len(images)} frames. Use both visuals and audio.\n"
        + system_prompt
    )

    # vLLM multimodal format
    return {
        "prompt": prompt,
        "multi_modal_data": {"image": images}
    }


def run_inference(config):
    """The full InternVL pipeline, parallel to the Qwen pipeline."""
    start_time = time.time()

    # 1. Video list
    video_ids = get_video_list(config)
    print(f"Found {len(video_ids)} videos\n")

    # 2. Key-frame extraction (optional)
    extract_frames_time = None
    if config["FEATURES"]["EXTRACT_KEY_FRAMES"]:
        t0 = time.time()
        extract_key_frames_batch(
            config["PATHS"]["VIDEO_FOLDER"],
            config["PATHS"]["KEY_FRAMES_FOLDER"],
            config
        )
        extract_frames_time = time.time() - t0

    # 3. Audio transcription (optional)
    extract_audio_time = None
    if config["FEATURES"]["TRANSCRIBE_AUDIO"]:
        t0 = time.time()
        extract_and_transcribe_all(
            video_ids,
            config["PATHS"]["VIDEO_FOLDER"],
            config
        )
        extract_audio_time = time.time() - t0

    # 4. Load InternVL model (vLLM)
    llm, sampling_params = load_internvl_model(config)
    model_memory = get_gpu_memory()
    print(f"Model loaded. GPU={model_memory:.2f} GB\n")

    # 5. Load system prompt
    with open(config["PATHS"]["PROMPT_FILE"], "r") as f:
        system_prompt = f.read()

    # 6. Output folder organization (same as Qwen)
    model_suffix = config["MODEL"]["MODEL_SUFFIX"]
    base_out = config["OUTPUT"]["MODES"][model_suffix]["OUTPUT_FOLDER"]

    input_size = config["VIDEO_PROCESSING"]["INPUT_SIZE"]
    num_frames = config["VIDEO_PROCESSING"]["NUM_SEGMENTS"]

    output_root = f"{base_out}_inputsize{input_size}_numframes{num_frames}"
    if config["FEATURES"]["EXTRACT_KEY_FRAMES"]:
        output_root += "_keyframes"

    json_folder = f"{output_root}/json/"
    csv_folder = f"{output_root}/csv/"
    stats_folder = f"{output_root}/statistics/"

    os.makedirs(json_folder, exist_ok=True)
    os.makedirs(csv_folder, exist_ok=True)
    os.makedirs(stats_folder, exist_ok=True)

    statistics_file = f"{stats_folder}/inference_statistics.txt"

    # Logs
    video_load_times = []
    inference_times = []
    mem_usages = []
    responses_list = []

    video_folder = config["PATHS"]["VIDEO_FOLDER"]

    # 7. Inference
    for idx, vid in enumerate(video_ids, 1):
        video_id = vid.split(".")[0]
        video_path = os.path.join(video_folder, vid)

        print(f"\n[{idx}/{len(video_ids)}] Processing {video_id}")
        mem_before = get_gpu_memory()

        try:
            # Load video + build prompt
            t0 = time.time()
            multimodal_input = prepare_multimodal_input(
                video_id,
                video_path,
                config,
                system_prompt
            )
            load_time = time.time() - t0
            video_load_times.append(load_time)

            # Generate
            t0 = time.time()
            out = llm.generate([multimodal_input], sampling_params)
            text = out[0].outputs[0].text
            # text = text.replace("\njson", "").replace("\n", "")
            text = text.replace("```json", "").replace("```", "")
            infer_time = time.time() - t0
            inference_times.append(infer_time)

            # Save JSON
            with open(f"{json_folder}/{video_id}.json", "w") as f:
                f.write(text)

            responses_list.append(
                {"video_id": video_id, **json.loads(text)}
            )
            mem_usages.append(get_gpu_memory() - mem_before)

        except Exception as e:
            print(f"ERROR: {e}")
            with open(f"{output_root}/missed_videos.txt", "a") as f:
                f.write(f"{video_id}\n")

    # 8. Save CSV
    df_resp = pd.DataFrame(responses_list)
    df_resp.to_csv(f"{csv_folder}/response.csv", index=False)

    # 9. Evaluation
    gt_file = config["PATHS"]["GROUND_TRUTH_FILE"]
    gt = pd.read_csv(gt_file)

    gt_vids = set(gt["video_id"].astype(str))
    out_vids = set(df_resp["video_id"].astype(str))
    common = sorted(list(gt_vids.intersection(out_vids)))

    gt = gt[gt["video_id"].astype(str).isin(common)].reset_index(drop=True)
    df_resp = df_resp[df_resp["video_id"].astype(str).isin(common)].reset_index(drop=True)

    evaluation_results = evaluate_with_stats(gt, df_resp)

    with open(f"{stats_folder}/evaluation_results_{model_suffix}.json", "w") as f:
        json.dump(evaluation_results, f, indent=4)

    # 10. Statistics
    total_time = time.time() - start_time

    save_statistics(
        config,
        config["MODEL"]["MODEL_NAME"],
        0,
        model_memory,
        video_load_times,
        inference_times,
        [],
        mem_usages,
        total_time,
        extract_frames_time,
        extract_audio_time,
        statistics_file
    )

    print_summary(
        video_load_times,
        inference_times,
        [],
        mem_usages,
        model_memory,
        total_time,
        extract_frames_time,
        extract_audio_time
    )

    print(f"\nAll outputs saved in: {output_root}")
    print(f"Statistics saved to: {statistics_file}")


def main():
    cfg = "config_internvl.json"
    if len(sys.argv) > 1:
        cfg = sys.argv[1]

    config = load_config(cfg)
    print("InternVL Pipeline\n")
    run_inference(config)


if __name__ == "__main__":
    main()