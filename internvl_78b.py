import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import time
import math
import os

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_SEGMENTS = 45
INPUT_SIZE = 448

OUTPUT_FOLDER = "data/outputs/internvl_78b_v2"
AUDIO_FOLDER = "data/inputs/audio_transcripts/"
KEY_FRAMES_FOLDER = "data/inputs/key_frames/"

use_key_frames = False
sample = True
sample_size = 2

OUTPUT_FOLDER = OUTPUT_FOLDER + f"_inputsize_{INPUT_SIZE}_numframes_{NUM_SEGMENTS}"
if use_key_frames:
    OUTPUT_FOLDER = OUTPUT_FOLDER + "_key_frames/"
else:
    OUTPUT_FOLDER = OUTPUT_FOLDER + "_uniform/"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

json_folder = f"{OUTPUT_FOLDER}/json/"
csv_folder = f"{OUTPUT_FOLDER}/csv/"
stats_folder = f"{OUTPUT_FOLDER}/statistics/"

os.makedirs(json_folder, exist_ok=True)
os.makedirs(csv_folder, exist_ok=True)
os.makedirs(stats_folder, exist_ok=True)

with open('data/prompt.txt', 'r') as f:
    system_instruction = f.read()
video_base_path = 'data/inputs/videos/'

video_ids = os.listdir(video_base_path)
video_ids = [x for x in video_ids if '.DS_Store' not in x]

if sample:
    video_ids = video_ids[:sample_size]

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def load_video_simple(video_path, num_segments=NUM_SEGMENTS, input_size=448):
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
    
    # Stack all frames
    pixel_values = torch.stack(all_pixel_values)
    
    # Each frame has one patch
    num_patches_list = [1] * num_segments
    
    return pixel_values, num_patches_list


def load_key_frames(video_id, key_frames_folder, num_segments=NUM_SEGMENTS, input_size=448):
    """
    Reads uniformly sampled key frames from key_frames_folder/video_id/*.jpg
    and returns stacked pixel tensors similar to load_video_simple.
    """
    video_folder = os.path.join(key_frames_folder, str(video_id))
    frame_files = sorted(
        [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.lower().endswith(".jpg")]
    )

    if len(frame_files) == 0:
        raise ValueError(f"No key frames found for video_id: {video_id}")

    # Evenly sample frames if there are more than num_segments
    indices = np.linspace(0, len(frame_files) - 1, num=num_segments, dtype=int)
    sampled_frames = [frame_files[i] for i in indices]

    all_pixel_values = []
    transform = build_transform(input_size=input_size)

    for frame_path in sampled_frames:
        img = Image.open(frame_path).convert("RGB")
        img_resized = img.resize((input_size, input_size), Image.BICUBIC)
        pixel_values = transform(img_resized)
        all_pixel_values.append(pixel_values)

    # Stack all frames into a tensor [num_segments, C, H, W]
    pixel_values = torch.stack(all_pixel_values)

    # Each frame is treated as a single patch
    num_patches_list = [1] * num_segments

    return pixel_values, num_patches_list



# Initialize model
models = ["OpenGVLab/InternVL2_5-8B", "OpenGVLab/InternVL2_5-26B", "OpenGVLab/InternVL2_5-78B"]
model_name = models[2]

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    name = model_name.split('/')[-1]
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[name]
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

device_map = split_model(model_name)

model = AutoModel.from_pretrained(
    model_name,
    # torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    load_in_8bit=True,                          # Enables 8-bit loading
    device_map="auto",                          # Automatically maps layers to devices
    # device_map={"": 0},
    # device_map=device_map
).eval()#.cuda()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

# 8 bit
# bnb_config = BitsAndBytesConfig(
#     load_in_8bit=True,
#     llm_int8_threshold=6.0,
#     llm_int8_skip_modules=None,
#     llm_int8_enable_fp32_cpu_offload=False
# )


# # 4 bit
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16  # Match this with your input tensor dtype
# )

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     quantization_config=bnb_config,
#     device_map="auto",
#     trust_remote_code=True
# ).eval()




# Generate response
generation_config = dict(max_new_tokens=2048, do_sample=True)

start_time = time.time()


processed_videos = os.listdir(OUTPUT_FOLDER)
processed_videos_ids = [x.split('.json')[0] for x in processed_videos if '.DS_Store' not in x]

video_ids = [x for x in video_ids if x.split('.mp4')[0] not in processed_videos_ids]

times = []
# 7407540452611640606
for i in range(len(video_ids)):
    generation_start_time = time.time()
    video_id = video_ids[i].split('.')[0]
    # Load video
    video_path = f'{video_base_path}/{video_id}.mp4'
    if use_key_frames:
        # video_id = video_path.split('/')[-1].split('.mp4')[0]
        pixel_values, num_patches_list = load_key_frames(video_id, KEY_FRAMES_FOLDER, num_segments=NUM_SEGMENTS, input_size=INPUT_SIZE)
        
    else:
        # pixel_values, num_patches_list = load_video_simple(video_path)
        pixel_values, num_patches_list = load_video_simple(video_path, num_segments=NUM_SEGMENTS, input_size=INPUT_SIZE)
    # pixel_values = pixel_values.to(torch.bfloat16).cuda()
    pixel_values = pixel_values.to(torch.float16).cuda()

    audio_file = f'{AUDIO_FOLDER}/{video_id}.txt'
    if os.path.exists(audio_file):
        with open(audio_file, 'r') as f:
            lines = f.readlines()
            # print(lines)
            audio_transcript = ''.join(lines)
            print(f'Audio transcript: {audio_transcript}')
    else:
        print(f'Audio transcript not found for video ID: {video_id}')
        audio_transcript = ''
    
    # Create question with video prefix
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    print(f'Video: {video_id}')
    print(f'Number of image patches: {len(num_patches_list)}')
    question = 'Audio transcript: ' + audio_transcript + '\n'
    question += f'Note that the video is divided into {len(num_patches_list)} frames. Please use the accompanying audio transcript when generating your response, ensuring that all relevant information from both the visual frames and the transcript is accurately included.\n'
    question += system_instruction

    # question = system_instruction#'Describe this video in detail.'
    full_question = video_prefix + question
    try:
        response = model.chat(
                tokenizer, 
                pixel_values, 
                full_question, 
                generation_config,
                num_patches_list=num_patches_list, 
                history=None, 
                return_history=False#True -> response, history
            )
        response_text = response.replace('```json', '').replace('```', '')
        print(f"Response: {response_text}")
        with open(f'{json_folder}/{video_id}.json', 'w') as f:
            f.write(response_text)
    except Exception as e:
        print(f'Error processing video {video_id}: {e}')
        print(f"Saving error to {OUTPUT_FOLDER}/missed.txt")
        with open(f'{OUTPUT_FOLDER}/missed.txt', 'a') as f:
            f.write(video_id)
            f.write('\n')
            f.write(f"Error: {e}")
        continue

    generation_end_time = time.time()
    times.append(generation_end_time - generation_start_time)
    print(f'Assistant: {response}')

# Save the times list in a text file
with open(f'{OUTPUT_FOLDER}/generation_times_internvl_78b_keyframes.txt', 'w') as f:
    for t in times:
        f.write(f'{t}\n')
    # f.write(f'Average time: {sum(times)/len(times)}\n')

if len(times) > 0:
    print(f'Average time: {sum(times)/len(times)}')
end_time = time.time()
print(f'Time: {end_time - start_time}')

# save_statistics(
#         config, config['MODEL']['MODEL_NAME'], model_load_time, model_memory,
#         video_load_times, inference_times, [],
#         video_mem_usages, total_time,
#         None, None,
#         statistics_file=os.path.join(stats_folder, "inference_statistics.txt")
#     )


# Evaluation

import os
import pandas as pd
import json

from utils.evaluation_utils import evaluate_with_stats

GROUND_TRUTH_FILE = "data/outputs/gemini/csv/ground_truth_gemini.csv"

output_root = OUTPUT_FOLDER
# "data/outputs/internvl_78b_inputsize_448_numframes_45_uniform/"
# f"{base_out}_inputsize{input_size}_numframes{num_frames}"

    
json_folder = f"{output_root}/json/"
csv_folder = f"{output_root}/csv/"
stats_folder = f"{output_root}/statistics/"
os.makedirs(json_folder, exist_ok=True)
os.makedirs(csv_folder, exist_ok=True)
os.makedirs(stats_folder, exist_ok=True)

# Read all json files in the predictions folder and save as a dataframe
responses = []

for filename in os.listdir(json_folder):
    if filename.endswith(".json"):
        video_id = filename.split(".json")[0]
        with open(os.path.join(json_folder, filename), 'r') as f:
            response = f.read()
        responses.append({"video_id": video_id, **json.loads(response)})

df = pd.DataFrame(responses)
df.to_csv(os.path.join(csv_folder, "response.csv"), index=False)

ground_truth = pd.read_csv(GROUND_TRUTH_FILE)

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