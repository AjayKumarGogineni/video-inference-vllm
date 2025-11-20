import os

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import torch
import time

# # Define 8-bit quantization config
# bnb_config = BitsAndBytesConfig(
#     load_in_8bit=True,
#     llm_int8_threshold=6.0,      # optional tuning
#     llm_int8_skip_modules=None,  # or specify to skip some modules
# )
# Qwen/Qwen2.5-VL-32B-Instruct
# Load 32B model

models_list = ["Qwen/Qwen2.5-VL-32B-Instruct", 
                "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
               "Qwen/Qwen2.5-VL-72B-Instruct",
               "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
               ]
model_idx = 0
model_name = models_list[model_idx]

OUTPUT_FOLDER = 'data/outputs/qwen_2.5_32b/'
json_folder = OUTPUT_FOLDER + 'json/'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(json_folder, exist_ok=True)

video_base_path = 'data/inputs/videos/'

video_ids = os.listdir(video_base_path)
video_ids = video_ids[:2]

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name, 
    torch_dtype=torch.bfloat16,  # or torch.float16
    # device_map={"": 1},          # Use GPU 1
    # quantization_config=bnb_config,
    # torch_dtype="auto", 
    device_map="auto",
    # cache_dir=custom_cache_dir
)

# default processer
processor = AutoProcessor.from_pretrained(model_name,
                                            # cache_dir=custom_cache_dir
                                            )


def analyze_video(video_path: str, query: str):
    # Process video frames
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path},
            {"type": "text", "text": query}
        ]
    }]
    
    # Prepare inputs
    text = processor.apply_chat_template(messages, tokenize=False)
    image_inputs, video_inputs = process_vision_info(messages)
    # print(f'Video Inputs: {video_inputs}')
    inputs = processor(
        text=[text],
        videos=video_inputs,
        return_tensors="pt",
        padding=True
    ).to("cuda")

    # Generate analysis
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(f"Output text: {output_text}")
    return output_text, processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Video Processing Workflow
# 1. Frame Extraction

from decord import VideoReader, cpu
import numpy as np

def extract_key_frames(video_path: str, num_segments=45):
    vr = VideoReader(video_path, ctx=cpu(0))
    frame_indices = np.linspace(0, len(vr)-1, num_segments, dtype=int)
    return [vr[i].asnumpy() for i in frame_indices]

start_time = time.time()

# Example temporal query
with open('data/prompt.txt', 'r') as f:
    system_instruction = f.read()



# 7407540452611640606
for i in range(len(video_ids)):
    video_id = video_ids[i].split('.')[0]
    output_text, response = analyze_video(
        video_path = f'{video_base_path}/{video_id}.mp4',
        query=system_instruction#'Summarize the video'
    )

    with open(f'{json_folder}/{video_id}.json', 'w') as f:
        f.write(response)

    print(f'Video: {video_id} Response: {response}')

end_time = time.time()
print(f'Time: {end_time - start_time}')
