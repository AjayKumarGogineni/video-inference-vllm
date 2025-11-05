import time
import os
import sys

from utils.config_loader import load_config
from utils.key_frame_extractor import extract_key_frames_batch
from utils.audio_processor import extract_and_transcribe_all
from utils.video_processor import load_video_frames


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


def preprocess_videos(config):
    """
    Preprocessing Pipeline:
      ✅ Key-frame extraction (optional)
      ✅ Audio transcription (optional)
      ✅ (Optional) Preload frames for caching (same behavior as before)
    """
    print("="*60)
    print("Video Pre-Processing Pipeline")
    print("="*60)

    # Load video list
    video_ids = get_video_list(config)
    print(f"Found {len(video_ids)} videos to preprocess\n")

    video_folder = config['PATHS']['VIDEO_FOLDER']

    # STEP 1: Extract key frames
    if config['FEATURES']['EXTRACT_KEY_FRAMES']:
        print("="*60)
        print("STEP 1: Extracting key frames...")
        print("="*60)
        start = time.time()

        extract_key_frames_batch(
            video_folder,
            config['PATHS']['KEY_FRAMES_FOLDER'],
            config
        )
        print(f"Key frame extraction completed in {time.time() - start:.2f}s\n")

    # STEP 2: Audio transcription
    if config['FEATURES']['TRANSCRIBE_AUDIO']:
        print("="*60)
        print("STEP 2: Extracting + transcribing audio...")
        print("="*60)
        start = time.time()

        extract_and_transcribe_all(
            video_ids,
            video_folder,
            config
        )
        print(f"Audio transcription completed in {time.time() - start:.2f}s\n")

    print("✅ Pre-processing complete!\n")


def main():
    config_path = "config_preprocess.json"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    config = load_config(config_path)
    preprocess_videos(config)


if __name__ == "__main__":
    main()
