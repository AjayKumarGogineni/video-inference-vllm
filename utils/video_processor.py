import numpy as np
import torchvision.transforms as T
from PIL import Image
from decord import VideoReader, cpu
from torchvision.transforms.functional import InterpolationMode
import os
import cv2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def load_video_frames(video_path, num_segments, input_size):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    indices = np.linspace(0, max_frame, num=num_segments, dtype=int)
    transform = build_transform(input_size=input_size)
    frames = []
    for idx in indices:
        frame = vr[idx].asnumpy()
        img = Image.fromarray(frame).convert('RGB')
        img = transform(img)
        frames.append(img)
    return frames


def load_key_frames(video_id, key_frames_folder, input_size):
    video_frame_folder = os.path.join(key_frames_folder, video_id)
    if not os.path.exists(video_frame_folder):
        return []
    
    transform = build_transform(input_size=input_size)
    frame_files = sorted([
        f for f in os.listdir(video_frame_folder) 
        if f.endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    frames = []
    for frame_file in frame_files:
        frame_path = os.path.join(video_frame_folder, frame_file)
        img = Image.open(frame_path).convert('RGB')
        img = transform(img)
        frames.append(img)
    
    return frames


def load_uniform_frames(video_id, uniform_frames_folder, input_size):
    video_frame_folder = os.path.join(uniform_frames_folder, video_id)
    if not os.path.exists(video_frame_folder):
        return []
    
    transform = build_transform(input_size=input_size)
    frame_files = sorted([
        f for f in os.listdir(video_frame_folder) 
        if f.endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    frames = []
    for frame_file in frame_files:
        frame_path = os.path.join(video_frame_folder, frame_file)
        img = Image.open(frame_path).convert('RGB')
        img = transform(img)
        frames.append(img)
    
    return frames

# def extract_uniform_frames(video_ids, config):
#     print("="*60)
#     print("Extracting Uniform Frames")
#     print("="*60)
    
#     video_folder = config['PATHS']['VIDEO_FOLDER']
#     output_folder = config['PATHS']['UNIFORM_FRAMES_FOLDER']
#     num_segments = config['VIDEO_PROCESSING']['NUM_SEGMENTS']
#     input_size = config['VIDEO_PROCESSING']['INPUT_SIZE']
    
#     os.makedirs(output_folder, exist_ok=True)
    
#     success_count = 0
#     failed_videos = []
    
#     for idx, vid in enumerate(video_ids, 1):
#         video_id = vid.split('.')[0]
#         video_path = os.path.join(video_folder, vid)
#         video_output_folder = os.path.join(output_folder, video_id)
        
#         if os.path.exists(video_output_folder) and len(os.listdir(video_output_folder)) >= num_segments:
#             print(f"[{idx}/{len(video_ids)}] ⏭️  {video_id} - Already processed")
#             success_count += 1
#             continue
        
#         os.makedirs(video_output_folder, exist_ok=True)
        
#         try:
#             vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
#             max_frame = len(vr) - 1
#             indices = np.linspace(0, max_frame, num=num_segments, dtype=int)
            
#             for frame_idx, idx_in_video in enumerate(indices):
#                 frame = vr[idx_in_video].asnumpy()
#                 img = Image.fromarray(frame).convert('RGB')
#                 img_resized = img.resize((input_size, input_size), Image.BICUBIC)
                
#                 frame_path = os.path.join(video_output_folder, f"frame_{frame_idx:04d}.jpg")
#                 img_resized.save(frame_path, quality=config['UNIFORM_FRAME_EXTRACTION'].get('JPEG_QUALITY', 95))
            
#             print(f"[{idx}/{len(video_ids)}] ✅ {video_id} - Extracted {num_segments} frames")
#             success_count += 1
            
#         except Exception as e:
#             print(f"[{idx}/{len(video_ids)}] ❌ {video_id} - Error: {e}")
#             failed_videos.append((video_id, str(e)))
    
#     print(f"\n{'='*60}")
#     print(f"Uniform Frame Extraction Summary")
#     print(f"{'='*60}")
#     print(f"Total videos: {len(video_ids)}")
#     print(f"Successful: {success_count}")
#     print(f"Failed: {len(failed_videos)}")
    
#     if failed_videos:
#         failed_file = os.path.join(output_folder, 'failed_videos.txt')
#         with open(failed_file, 'w') as f:
#             for video_id, error in failed_videos:
#                 f.write(f"{video_id}: {error}\n")
#         print(f"Failed videos saved to: {failed_file}")

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
from decord import VideoReader, cpu

def get_max_frames(duration_sec):
    """Determine maximum frames based on video duration."""
    if duration_sec <= 15: return 5
    if duration_sec <= 30: return 8
    if duration_sec <= 60: return 12
    if duration_sec <= 120: return 18
    if duration_sec <= 240: return 24
    return 30

def process_uniform_frames_single(video_file, config, dynamic = False):
    """Process a single video and return (video_id, status, error_msg)."""

    video_folder = config['PATHS']['VIDEO_FOLDER']
    output_folder = config['PATHS']['UNIFORM_FRAMES_FOLDER']
    num_segments = config['VIDEO_PROCESSING']['NUM_SEGMENTS']
    input_size = config['VIDEO_PROCESSING']['INPUT_SIZE']

    video_id = video_file.split('.')[0]
    video_path = os.path.join(video_folder, video_file)
    video_output_folder = os.path.join(output_folder, video_id)

    # Skip if already processed
    if os.path.exists(video_output_folder) and \
       len(os.listdir(video_output_folder)) >= num_segments:
        return (video_id, "already_processed", None)

    os.makedirs(video_output_folder, exist_ok=True)

    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        # max_frames = len(vr) - 1

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        duration_sec = total_frames / fps if fps > 0 else 0
        max_frames = get_max_frames(duration_sec)

        indices = np.linspace(0, max_frames, num=num_segments, dtype=int)

        for frame_idx, idx_in_video in enumerate(indices):
            frame = vr[idx_in_video].asnumpy()
            img = Image.fromarray(frame).convert('RGB')
            img_resized = img.resize((input_size, input_size), Image.BICUBIC)

            frame_path = os.path.join(video_output_folder, f"frame_{frame_idx:04d}.jpg")
            img_resized.save(
                frame_path,
                quality=config['UNIFORM_FRAME_EXTRACTION'].get('JPEG_QUALITY', 95)
            )

        return (video_id, "success", None)

    except Exception as e:
        return (video_id, "error", str(e))



def extract_uniform_frames(video_ids, config, dynamic = False):
    print("="*60)
    print("Extracting Uniform Frames")
    print("="*60)

    output_folder = config['PATHS']['UNIFORM_FRAMES_FOLDER']
    os.makedirs(output_folder, exist_ok=True)

    max_workers = config['UNIFORM_FRAME_EXTRACTION'].get("MAX_WORKERS", 8)

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_uniform_frames_single, vid, config, dynamic = dynamic): vid
            for vid in video_ids
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
            results.append(future.result())

    # Summary
    success = sum(1 for r in results if r[1] == "success" or r[1] == "already_processed")
    failed = [r for r in results if r[1] == "error"]

    print(f"\n{'='*60}")
    print("Uniform Frame Extraction Summary")
    print(f"Total Videos      : {len(results)}")
    print(f"Successful/Skipped: {success}")
    print(f"Failed            : {len(failed)}")

    # Write failed list
    if failed:
        failed_file = os.path.join(output_folder, "failed_videos.txt")
        with open(failed_file, 'w') as f:
            for vid, _, err in failed:
                f.write(f"{vid}: {err}\n")
        print(f"Failed list saved to: {failed_file}")



def extract_uniform_frames_dynamic(video_ids, config):
    print("="*60)
    print("Extracting Uniform Frames")
    print("="*60)

    output_folder = config['PATHS']['UNIFORM_FRAMES_FOLDER']
    os.makedirs(output_folder, exist_ok=True)

    max_workers = config['UNIFORM_FRAME_EXTRACTION'].get("MAX_WORKERS", 8)

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_uniform_frames_single, vid, config): vid
            for vid in video_ids
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
            results.append(future.result())

    # Summary
    success = sum(1 for r in results if r[1] == "success" or r[1] == "already_processed")
    failed = [r for r in results if r[1] == "error"]

    print(f"\n{'='*60}")
    print("Uniform Frame Extraction Summary")
    print(f"Total Videos      : {len(results)}")
    print(f"Successful/Skipped: {success}")
    print(f"Failed            : {len(failed)}")

    # Write failed list
    if failed:
        failed_file = os.path.join(output_folder, "failed_videos.txt")
        with open(failed_file, 'w') as f:
            for vid, _, err in failed:
                f.write(f"{vid}: {err}\n")
        print(f"Failed list saved to: {failed_file}")

