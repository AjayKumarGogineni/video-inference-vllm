import time
import os
import sys
import json
import pandas as pd
import torch
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import math

from utils.gpu_monitor import get_gpu_memory
from utils.statistics import save_statistics, print_summary
from utils.evaluation_utils import evaluate_with_stats

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def load_video_frames(video_path, num_segments, input_size):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    
    # Evenly sample frames
    indices = np.linspace(0, max_frame, num=num_segments, dtype=int)
    
    all_pixel_values = []
    transform = build_transform(input_size=input_size)
    
    for idx in indices:
        # Get the frame
        frame = vr[idx].asnumpy()
        img = Image.fromarray(frame).convert('RGB')
        
        # Resize to the input size
        img_resized = img.resize((input_size, input_size), Image.BICUBIC)
        
        # Transform and add to list
        pixel_values = transform(img_resized)
        all_pixel_values.append(pixel_values)
    
    # Return list of tensors, not stacked
    return all_pixel_values


def load_uniform_frames(video_id, uniform_frames_folder, input_size):
    frames_path = os.path.join(uniform_frames_folder, video_id)
    if not os.path.exists(frames_path):
        return []
    
    frame_files = sorted([f for f in os.listdir(frames_path) if f.endswith(('.jpg', '.png'))])
    if not frame_files:
        return []
    
    all_pixel_values = []
    transform = build_transform(input_size=input_size)
    
    for frame_file in frame_files:
        frame_path = os.path.join(frames_path, frame_file)
        img = Image.open(frame_path).convert('RGB')
        img_resized = img.resize((input_size, input_size), Image.BICUBIC)
        pixel_values = transform(img_resized)
        all_pixel_values.append(pixel_values)
    
    # Return list of tensors, not stacked
    return all_pixel_values


def load_key_frames(video_id, key_frames_folder, input_size):
    frames_path = os.path.join(key_frames_folder, video_id)
    if not os.path.exists(frames_path):
        return []
    
    frame_files = sorted([f for f in os.listdir(frames_path) if f.endswith(('.jpg', '.png'))])
    if not frame_files:
        return []
    
    all_pixel_values = []
    transform = build_transform(input_size=input_size)
    
    for frame_file in frame_files:
        frame_path = os.path.join(frames_path, frame_file)
        img = Image.open(frame_path).convert('RGB')
        img_resized = img.resize((input_size, input_size), Image.BICUBIC)
        pixel_values = transform(img_resized)
        all_pixel_values.append(pixel_values)
    
    # Return list of tensors, not stacked
    return all_pixel_values


def load_frames(video_id, video_path, config):
    if config['FEATURES'].get('USE_KEY_FRAMES', False):
        images = load_key_frames(
            video_id,
            config['PATHS']['KEY_FRAMES_FOLDER'],
            config['VIDEO_PROCESSING']['INPUT_SIZE']
        )
        if len(images) == 0:
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
        if len(images) == 0:
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


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    name = model_name.split('/')[-1]
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80
    }[name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


def load_model_for_family(config):
    model_name = config['MODEL']['MODEL_NAME']
    
    print(f"Loading model: {model_name}")
    
    if config['MODEL'].get('USE_DEVICE_MAP', False):
        device_map = split_model(model_name)
    else:
        device_map = "auto"
    
    model = AutoModel.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        load_in_8bit=config['MODEL'].get('LOAD_IN_8BIT', True),
        device_map=device_map
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        use_fast=False
    )
    
    return model, tokenizer


def prepare_multimodal_input(video_id, video_path, config, system_prompt):
    images = load_frames(video_id, video_path, config)
    audio_text = get_audio_transcript(video_id, config)
    
    # Create question with video prefix
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(images))])
    question = f'Audio transcript: {audio_text.strip()}\n'
    question += f'Note that the video is divided into {len(images)} frames. Please use the accompanying audio transcript when generating your response, ensuring that all relevant information from both the visual frames and the transcript is accurately included.\n'
    question += system_prompt
    full_question = video_prefix + question
    
    return full_question, images


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
        llm, processor = load_model_for_family(config)
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
    
    generation_config = {
        'max_new_tokens': config['MODEL']['MAX_TOKENS'],
        'do_sample': True,
        'temperature': config['MODEL'].get('TEMPERATURE', 0.1),
        'top_p': config['MODEL'].get('TOP_P', 0.9)
    }
    
    for idx, vid in enumerate(video_ids, 1):
        video_id = vid.split('.')[0]
        video_path = os.path.join(video_folder, vid)
        print(f"\n[{idx}/{len(video_ids)}] Processing {video_id}...")
        
        mem_before = get_gpu_memory()
        
        try:
            load_start = time.time()
            question, images = prepare_multimodal_input(
                video_id, video_path, config, system_prompt
            )
            
            pixel_values = torch.stack(images).to(torch.float16).cuda()
            num_patches_list = [1] * len(images)
            
            load_t = time.time() - load_start
            video_load_times.append(load_t)
            
            inf_start = time.time()
            
            response = llm.chat(
                processor,
                pixel_values,
                question,
                generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=False
            )
            response = response.replace("```json", "").replace("```", "")
            
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
    if config['EVALUATION']['CALCULATE_METRICS'] and config['PATHS'].get('GROUND_TRUTH_FILE') and os.path.exists(config['PATHS']['GROUND_TRUTH_FILE']):
        print("\n" + "="*60)
        print("Running Evaluation")
        print("="*60)
        try:
            ground_truth = pd.read_csv(config['PATHS']['GROUND_TRUTH_FILE'])
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
        if not config['PATHS'].get('GROUND_TRUTH_FILE') or not os.path.exists(config['PATHS']['GROUND_TRUTH_FILE']):
            print("\n⚠️  Ground truth file not found or not specified. Skipping evaluation.")
        else:
            print("\nSkipping evaluation.")
    
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
    config_path = "data/configs/config_internvl_direct.json"
    # The config path can be overwritten by command line argument
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    run_inference(config)


if __name__ == "__main__":
    main()