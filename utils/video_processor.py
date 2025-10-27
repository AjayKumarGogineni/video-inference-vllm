import numpy as np
import torchvision.transforms as T
from PIL import Image
from decord import VideoReader, cpu
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    """Build image transformation pipeline."""
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def load_video_frames(video_path, num_segments, input_size):
    """Extract evenly spaced frames from video."""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    indices = np.linspace(0, max_frame, num=num_segments, dtype=int)

    frames = []
    for idx in indices:
        frame = vr[idx].asnumpy()
        img = Image.fromarray(frame).convert('RGB')
        frames.append(img)
    return frames

def load_key_frames(video_id, key_frames_folder):
    """Load pre-extracted key frames from folder."""
    import os
    from PIL import Image
    
    video_frame_folder = os.path.join(key_frames_folder, video_id)
    if not os.path.exists(video_frame_folder):
        return []
    
    frame_files = sorted([f for f in os.listdir(video_frame_folder) if f.endswith('.jpg')])
    frames = []
    for frame_file in frame_files:
        frame_path = os.path.join(video_frame_folder, frame_file)
        img = Image.open(frame_path).convert('RGB')
        frames.append(img)
    
    return frames