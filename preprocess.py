import time
import os
import sys

from utils.config_loader import load_config
from utils.key_frame_extractor import extract_key_frames_batch
from utils.audio_processor import extract_and_transcribe_all
from utils.video_processor import load_video_frames, extract_uniform_frames


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

        key_frames_folder = config['PATHS']['KEY_FRAMES_FOLDER']
        os.makedirs(key_frames_folder, exist_ok=True)

        extract_key_frames_batch(
            video_ids,
            key_frames_folder,
            config
        )
        print(f"Key frame extraction completed in {time.time() - start:.2f}s\n")

    # STEP 1b: Extract uniform frames
    extract_uniform = config['FEATURES'].get('EXTRACT_UNIFORM_FRAMES', False)
    if extract_uniform:
        start = time.time()
        uniform_frames_folder = config['PATHS']['UNIFORM_FRAMES_FOLDER']
        os.makedirs(uniform_frames_folder, exist_ok=True)
        try:
            # dynamic = True updates the number of frames based on video duration
            extract_uniform_frames(video_ids, config, dynamic = True)
            print(f"✅ Uniform frame extraction completed in {time.time() - start:.2f}s\n")
        except Exception as e:
            print(f"❌ Uniform frame extraction failed: {e}\n")
            import traceback
            traceback.print_exc()
        print(f"✅ Uniform frame extraction completed in {time.time() - start:.2f}s\n")
    
    # Step 1c: Extract uniform frames (static)
    extract_uniform_dynamic = config['FEATURES'].get('EXTRACT_UNIFORM_FRAMES_DYNAMIC', False)
    if extract_uniform_dynamic:
        start = time.time()
        uniform_frames_dynamic_folder = config['PATHS']['UNIFORM_FRAMES_DYNAMIC_FOLDER']
        os.makedirs(uniform_frames_dynamic_folder, exist_ok=True)
        try:
            # dynamic = False uses fixed number of frames from config
            extract_uniform_frames(video_ids, config, dynamic = False)
            print(f"✅ Uniform frame extraction completed in {time.time() - start:.2f}s\n")
        except Exception as e:
            print(f"❌ Uniform frame extraction failed: {e}\n")
            import traceback
            traceback.print_exc()
        print(f"✅ Uniform frame extraction completed in {time.time() - start:.2f}s\n")

    # STEP 2: Audio transcription
    if config['FEATURES']['TRANSCRIBE_AUDIO']:
        print("="*60)
        print("STEP 2: Extracting + transcribing audio...")
        print("="*60)
        start = time.time()
        audio_folder = config['PATHS']['AUDIO_FOLDER']
        os.makedirs(audio_folder, exist_ok=True)
        audio_transcript_folder = config['PATHS']['AUDIO_TRANSCRIPT_FOLDER']
        os.makedirs(audio_transcript_folder, exist_ok=True)

        extract_and_transcribe_all(
            video_ids,
            video_folder,
            config
        )
        print(f"Audio transcription completed in {time.time() - start:.2f}s\n")

    print("✅ Pre-processing complete!\n")


def main():
    config_path = "data/configs/config_preprocess.json"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    config = load_config(config_path)
    preprocess_videos(config)


if __name__ == "__main__":
    main()

# # V2
# import time
# import os
# import sys
# import numpy as np
# from PIL import Image
# from decord import VideoReader, cpu

# from utils.config_loader import load_config
# from utils.key_frame_extractor import extract_key_frames_batch
# from utils.audio_processor import extract_and_transcribe_all


# def get_video_list(config):
#     video_folder = config['PATHS']['VIDEO_FOLDER']
#     video_ids = [
#         vid for vid in os.listdir(video_folder)
#         if vid.endswith(('.mp4', '.avi', '.mov', '.mkv'))
#     ]

#     if config['SAMPLING']['SAMPLE']:
#         sample_size = config['SAMPLING']['SAMPLE_SIZE']
#         video_ids = video_ids[:sample_size]

#     return video_ids


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


# def preprocess_videos(config):
#     print("="*60)
#     print("Video Preprocessing Pipeline")
#     print("="*60)

#     video_ids = get_video_list(config)
#     print(f"Found {len(video_ids)} videos to preprocess\n")

#     if not video_ids:
#         print("❌ No videos found in VIDEO_FOLDER")
#         return

#     video_folder = config['PATHS']['VIDEO_FOLDER']
    
#     extract_keyframes = config['FEATURES'].get('EXTRACT_KEY_FRAMES', False)
#     extract_uniform = config['FEATURES'].get('EXTRACT_UNIFORM_FRAMES', False)
#     transcribe_audio = config['FEATURES'].get('TRANSCRIBE_AUDIO', False)

#     if extract_keyframes:
#         print("="*60)
#         print("STEP 1: Extracting Key Frames")
#         print("="*60)
#         start = time.time()
#         try:
#             extract_key_frames_batch(
#                 video_folder,
#                 config['PATHS']['KEY_FRAMES_FOLDER'],
#                 config
#             )
#             print(f"✅ Key frame extraction completed in {time.time() - start:.2f}s\n")
#         except Exception as e:
#             print(f"❌ Key frame extraction failed: {e}\n")
#             import traceback
#             traceback.print_exc()
    
#     if extract_uniform:
#         start = time.time()
#         try:
#             extract_uniform_frames(video_ids, config)
#             print(f"✅ Uniform frame extraction completed in {time.time() - start:.2f}s\n")
#         except Exception as e:
#             print(f"❌ Uniform frame extraction failed: {e}\n")
#             import traceback
#             traceback.print_exc()

#     if transcribe_audio:
#         print("="*60)
#         print("STEP 2: Extracting and Transcribing Audio")
#         print("="*60)
#         start = time.time()
#         try:
#             extract_and_transcribe_all(
#                 video_ids,
#                 video_folder,
#                 config
#             )
#             print(f"✅ Audio transcription completed in {time.time() - start:.2f}s\n")
#         except Exception as e:
#             print(f"❌ Audio transcription failed: {e}\n")
#             import traceback
#             traceback.print_exc()

#     if not (extract_keyframes or extract_uniform or transcribe_audio):
#         print("⚠️  No preprocessing steps enabled in config")
#         print("   Set EXTRACT_KEY_FRAMES, EXTRACT_UNIFORM_FRAMES, or TRANSCRIBE_AUDIO to true")
#     else:
#         print("✅ Preprocessing complete!\n")


# def main():
#     config_path = "data/configs/config_preprocess.json"
#     if len(sys.argv) > 1:
#         config_path = sys.argv[1]

#     if not os.path.exists(config_path):
#         print(f"❌ Config file not found: {config_path}")
#         sys.exit(1)

#     try:
#         config = load_config(config_path)
#         preprocess_videos(config)
#     except Exception as e:
#         print(f"❌ Preprocessing failed: {e}")
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)


# if __name__ == "__main__":
#     main()