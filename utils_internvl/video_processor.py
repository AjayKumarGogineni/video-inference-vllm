import numpy as np
import os
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from decord import VideoReader, cpu


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(size):
    return T.Compose([
        T.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
        T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_video_frames(video_path, num_segments, input_size):
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr) - 1

    idxs = np.linspace(0, total, num_segments, dtype=int)
    transform = build_transform(input_size)

    images = []
    for i in idxs:
        frame = vr[i].asnumpy()
        img = Image.fromarray(frame).convert("RGB")
        img = img.resize((input_size, input_size), Image.BICUBIC)

        t = transform(img)

        # denormalize → PIL (vLLM wants PIL images)
        t = t * torch.tensor(IMAGENET_STD).view(3,1,1) + torch.tensor(IMAGENET_MEAN).view(3,1,1)
        img_np = (t.permute(1,2,0).numpy() * 255).astype("uint8")
        images.append(Image.fromarray(img_np))

    return images


def load_key_frames(video_id, key_frame_folder, input_size):
    """Load frames from pre-extracted JPG key frame folder."""
    folder = f"{key_frame_folder}/{video_id}/"

    if not os.path.exists(folder):
        return []

    files = sorted([f for f in os.listdir(folder) if f.endswith(".jpg")])

    transform = build_transform(input_size)
    images = []

    for f in files:
        img = Image.open(os.path.join(folder, f)).convert("RGB")
        img = transform(img)

        # denormalize
        img = img * torch.tensor(IMAGENET_STD).view(3,1,1) + torch.tensor(IMAGENET_MEAN).view(3,1,1)
        arr = (img.permute(1,2,0).numpy() * 255).astype("uint8")
        images.append(Image.fromarray(arr))

    return images
