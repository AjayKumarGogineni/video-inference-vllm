import os
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from scenedetect import detect, ContentDetector

def get_max_frames(duration_sec):
    """Determine maximum frames based on video duration."""
    if duration_sec <= 15: return 5
    if duration_sec <= 30: return 8
    if duration_sec <= 60: return 12
    if duration_sec <= 120: return 18
    if duration_sec <= 240: return 24
    return 30

def detect_scenes_opencv(video_path, threshold=30.0, min_scene_len=15):
    """Detect scene changes using frame differencing."""
    cap = cv2.VideoCapture(video_path)
    prev_frame, scene_list, frame_num, last_scene = None, [], 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            diff = cv2.absdiff(gray, prev_frame)
            mean_diff = np.mean(diff)
            if mean_diff > threshold and (frame_num - last_scene) > min_scene_len:
                scene_list.append(frame_num)
                last_scene = frame_num
        prev_frame = gray
        frame_num += 1

    cap.release()
    
    if not scene_list:
        return []
    scenes = [(scene_list[i - 1] if i > 0 else 0, f) for i, f in enumerate(scene_list)]
    scenes.append((scene_list[-1], frame_num))
    return scenes

def get_frame_indices(scene_list, total_frames, max_frames):
    """Calculate frame indices from scenes or evenly spaced."""
    if scene_list:
        frame_indices = [(s[0] + s[1]) // 2 for s in scene_list]
    else:
        frame_indices = np.linspace(0, total_frames - 1, num=max_frames, dtype=int).tolist()
    
    if len(frame_indices) > max_frames:
        idx = np.linspace(0, len(frame_indices) - 1, num=max_frames, dtype=int)
        frame_indices = [frame_indices[i] for i in idx]
    return frame_indices

def save_frames(video_path, output_dir, frame_indices, fps, video_name, jpeg_quality):
    """Extract and save selected frames."""
    cap = cv2.VideoCapture(video_path)
    saved_count = 0
    
    for j, frame_num in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num))
        ret, frame = cap.read()
        if not ret:
            continue
        timestamp_sec = frame_num / fps
        m, s = divmod(int(timestamp_sec), 60)
        timestamp = f"{m:02d}_{s:02d}"
        out_path = os.path.join(output_dir, f"{video_name}_frame_{j:03d}_{timestamp}.jpg")
        cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        saved_count += 1
    
    cap.release()
    return saved_count

def process_single_video(video_path, output_base, config):
    """Process a single video for key frame extraction."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_out_dir = os.path.join(output_base, video_name)

    # Skip if already processed
    if os.path.exists(video_out_dir) and len(os.listdir(video_out_dir)) > 0:
        return video_name, "skipped", 0

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    duration_sec = total_frames / fps if fps > 0 else 0
    max_frames = get_max_frames(duration_sec)

    detection_mode = config['KEY_FRAME_EXTRACTION']['DETECTION_MODE']
    
    if detection_mode == "scenedetect":
        scene_list = detect(video_path, ContentDetector(threshold=27.0, min_scene_len=15))
        scenes = [(s[0].frame_num, s[1].frame_num) for s in scene_list]
    else:
        threshold = config['KEY_FRAME_EXTRACTION']['THRESHOLD']
        min_scene_len = config['KEY_FRAME_EXTRACTION']['MIN_SCENE_LEN']
        scenes = detect_scenes_opencv(video_path, threshold=threshold, min_scene_len=min_scene_len)

    frame_indices = get_frame_indices(scenes, total_frames, max_frames)
    os.makedirs(video_out_dir, exist_ok=True)
    
    jpeg_quality = config['KEY_FRAME_EXTRACTION']['JPEG_QUALITY']
    count = save_frames(video_path, video_out_dir, frame_indices, fps, video_name, jpeg_quality)

    if not scenes:
        return video_name, "no_scenes", count
    return video_name, "ok", count

def extract_key_frames_batch(video_ids, output_folder, config):
    """Extract key frames from all videos in folder."""
    video_folder = config['PATHS']['VIDEO_FOLDER']

    video_files = video_ids#[f for f in os.listdir(video_folder) if not f.startswith('.')]
    results = []
    
    max_workers = config['KEY_FRAME_EXTRACTION']['MAX_WORKERS']

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_video, 
                os.path.join(video_folder, vf), 
                output_folder, 
                config
            ): vf
            for vf in video_files
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting key frames"):
            try:
                results.append(future.result())
            except Exception as e:
                results.append((futures[future], f"error: {e}", 0))

    # Save logs
    frame_count_file = os.path.join(output_folder, "video_frame_counts.txt")
    with open(frame_count_file, "w") as f:
        for name, status, cnt in results:
            f.write(f"{name}: {status} ({cnt} frames)\n")

    videos_with_no_scenes = [r[0] for r in results if r[1] == "no_scenes"]
    if videos_with_no_scenes:
        failed_list_file = os.path.join(output_folder, "videos_with_no_scenes.txt")
        with open(failed_list_file, "w") as f:
            f.write("\n".join(videos_with_no_scenes))
    
    print(f"Key frame extraction complete. Logs saved to {frame_count_file}")