import time
import os
import sys
import json
import pandas as pd

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'

# from utils.config_loader import load_config
from utils.gpu_monitor import get_gpu_memory
from utils.statistics import save_statistics, print_summary
from utils.evaluation_utils import evaluate_with_stats

def get_video_list(config):
    video_folder = config['PATHS']['VIDEO_FOLDER']
    video_ids = [
        vid for vid in os.listdir(video_folder)
        if vid.endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ]
    
    if config['SAMPLING']['SAMPLE']:
        video_ids = video_ids[:config['SAMPLING']['SAMPLE_SIZE']]
    
    return video_ids


def get_audio_transcript(video_id, config):
    audio_path = os.path.join(
        config['PATHS']['AUDIO_TRANSCRIPT_FOLDER'],
        f"{video_id}.txt"
    )
    if os.path.exists(audio_path):
        with open(audio_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def load_model_for_family(config):
    model_family = config['MODEL']['MODEL_FAMILY']
    
    if model_family == 'qwen':
        from utils.model_loader import load_model
        llm, processor, sampling_params = load_model(config)
        return llm, processor, sampling_params, model_family, True
    
    elif model_family == 'internvl':
        from utils_internvl.model_loader import load_internvl_model
        llm, sampling_params = load_internvl_model(config)
        return llm, None, sampling_params, model_family, True
    
    else:
        raise ValueError(f"Unknown model family: {model_family}")


def load_frames(video_id, video_path, config, model_family):
    if model_family == 'qwen':
        from utils.video_processor import load_video_frames, load_key_frames, load_uniform_frames
    elif model_family == 'internvl':
        from utils.video_processor import load_video_frames, load_key_frames, load_uniform_frames
    else:
        raise ValueError(f"Unknown model family: {model_family}")
    
    if config['FEATURES'].get('USE_KEY_FRAMES', False):
        images = load_key_frames(
            video_id,
            config['PATHS']['KEY_FRAMES_FOLDER'],
            config['VIDEO_PROCESSING']['INPUT_SIZE']
        )
        if not images:
            print(f"  ⚠️  No key frames found for {video_id}, falling back to uniform sampling")
            images = load_video_frames(
                video_path,
                config['VIDEO_PROCESSING']['NUM_SEGMENTS'],
                config['VIDEO_PROCESSING']['INPUT_SIZE']
            )
    elif config['FEATURES'].get('USE_UNIFORM_FRAMES', False):
        images = load_uniform_frames(
            video_id,
            config['PATHS']['UNIFORM_FRAMES_FOLDER'],
            config['VIDEO_PROCESSING']['INPUT_SIZE']
        )
        if not images:
            print(f"  ⚠️  No uniform frames found for {video_id}, extracting on-the-fly")
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
    
    return images


def prepare_qwen_input(images, audio_text, system_prompt, processor):
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


def prepare_internvl_input(images, audio_text, system_prompt):
    image_tokens = " ".join(["<image>"] * len(images))
    prompt = (
        f"{image_tokens}\n"
        f"Audio transcript: {audio_text.strip()}\n"
        f"The video has {len(images)} frames. Use both visuals and audio.\n"
        + system_prompt
    )
    
    # vLLM multimodal format
    return {
        "prompt": prompt,
        "multi_modal_data": {"image": images}
    }


def prepare_multimodal_input(video_id, video_path, config, system_prompt, processor, model_family):
    images = load_frames(video_id, video_path, config, model_family)
    audio_text = get_audio_transcript(video_id, config)
    
    if model_family == 'qwen':
        return prepare_qwen_input(images, audio_text, system_prompt, processor)
    elif model_family == 'internvl':
        return prepare_internvl_input(images, audio_text, system_prompt)
    
    raise ValueError(f"Unknown model family: {model_family}")


def setup_output_folders(config):
    model_suffix = config['MODEL']['MODEL_SUFFIX']
    base_out = config['OUTPUT']['OUTPUT_FOLDER']
    
    input_size = config['VIDEO_PROCESSING']['INPUT_SIZE']
    num_frames = config['VIDEO_PROCESSING']['NUM_SEGMENTS']
    
    output_root = f"{base_out}_inputsize{input_size}_numframes{num_frames}"
    
    if config['FEATURES'].get('USE_KEY_FRAMES', False):
        output_root += "_keyframes"
    elif config['FEATURES'].get('USE_UNIFORM_FRAMES', False):
        output_root += "_uniformframes"
    
    json_folder = f"{output_root}/json/"
    csv_folder = f"{output_root}/csv/"
    stats_folder = f"{output_root}/statistics/"
    
    os.makedirs(json_folder, exist_ok=True)
    os.makedirs(csv_folder, exist_ok=True)
    os.makedirs(stats_folder, exist_ok=True)
    
    return output_root, json_folder, csv_folder, stats_folder


def run_inference(config):
    start_time = time.time()
    
    video_ids = get_video_list(config)
    print(f"Found {len(video_ids)} videos to process\n")
    
    print("="*60)
    print(f"Loading {config['MODEL']['MODEL_NAME']}...")
    print("="*60)
    load_start = time.time()
    
    try:
        llm, processor, sampling_params, model_family, use_vllm = load_model_for_family(config)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    model_load_time = time.time() - load_start
    model_memory = get_gpu_memory()
    print(f"Model loaded in {model_load_time:.2f}s")
    print(f"GPU memory usage: {model_memory:.2f} GB\n")
    
    with open(config['PATHS']['PROMPT_FILE'], "r") as f:
        system_prompt = f.read()
    
    output_root, json_folder, csv_folder, stats_folder = setup_output_folders(config)
    
    video_folder = config['PATHS']['VIDEO_FOLDER']
    video_load_times = []
    inference_times = []
    video_mem_usages = []
    responses = []
    
    for idx, vid in enumerate(video_ids, 1):
        video_id = vid.split('.')[0]
        video_path = os.path.join(video_folder, vid)
        print(f"\n[{idx}/{len(video_ids)}] Processing {video_id}...")
        
        mem_before = get_gpu_memory()
        
        try:
            load_start = time.time()
            multimodal_input = prepare_multimodal_input(
                video_id, video_path, config, system_prompt, processor, model_family
            )
            load_t = time.time() - load_start
            video_load_times.append(load_t)
            
            inf_start = time.time()
            
            # Both Qwen and InternVL use vLLM in this setup
            outputs = llm.generate([multimodal_input], sampling_params)
            response = outputs[0].outputs[0].text.replace("```json", "").replace("```", "")
            
            inf_t = time.time() - inf_start
            inference_times.append(inf_t)
            
            out_path = os.path.join(json_folder, f"{video_id}.json")
            with open(out_path, "w") as f:
                f.write(response)
            
            responses.append({"video_id": video_id, **json.loads(response)})
            mem_after = get_gpu_memory()
            video_mem_usages.append(mem_after - mem_before)
            
            print(f"  ✅ Loaded in {load_t:.2f}s | Inference {inf_t:.2f}s")
            
        except Exception as e:
            print(f"❌ Error processing {video_id}: {e}")
            import traceback
            traceback.print_exc()
            with open(os.path.join(output_root, 'missed_videos.txt'), 'a') as f:
                f.write(f"{video_id}: {str(e)}\n")
            continue
    
    df = pd.DataFrame(responses)
    df.to_csv(os.path.join(csv_folder, "response.csv"), index=False)
    
    eval_results = None
    if config.get('EVALUATION', {}).get('CALCULATE_METRICS', False):
        ground_truth_file = config['PATHS'].get('GROUND_TRUTH_FILE')
        
        if ground_truth_file and os.path.exists(ground_truth_file):
            print("\n" + "="*60)
            print("Running Evaluation")
            print("="*60)
            try:
                ground_truth = pd.read_csv(ground_truth_file)
                common = set(ground_truth.video_id.astype(str)) & set(df.video_id.astype(str))
                
                if len(common) == 0:
                    print("⚠️  No common videos found between predictions and ground truth")
                else:
                    gt = ground_truth[ground_truth.video_id.astype(str).isin(common)]
                    df_eval = df[df.video_id.astype(str).isin(common)]
                    
                    gt = gt.sort_values("video_id").reset_index(drop=True)
                    df_eval = df_eval.sort_values("video_id").reset_index(drop=True)
                    
                    eval_results = evaluate_with_stats(gt, df_eval)
                    with open(os.path.join(stats_folder, "evaluation.json"), "w") as f:
                        json.dump(eval_results, f, indent=4)
                    print(f"✅ Evaluation complete: {len(common)} videos evaluated")
            except Exception as e:
                print(f"⚠️  Evaluation failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\n⚠️  Ground truth file not found or not specified. Skipping evaluation.")
    else:
        print("\nSkipping evaluation (CALCULATE_METRICS is False).")
    
    total_time = time.time() - start_time
    
    save_statistics(
        config, config['MODEL']['MODEL_NAME'], model_load_time, model_memory,
        video_load_times, inference_times, [],
        video_mem_usages, total_time,
        None, None,
        statistics_file=os.path.join(stats_folder, "inference_statistics.txt")
    )
    
    print_summary(
        video_load_times, inference_times, [],
        video_mem_usages, model_memory,
        total_time, None, None
    )
    
    print(f"\n✅ Inference complete!")
    print(f"All outputs saved in: {output_root}")


def main():
    config_path = "data/configs/config_qwen.json"
    # The config path can be over written by command line argument
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    # config = load_config(config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)
    run_inference(config)


if __name__ == "__main__":
    main()