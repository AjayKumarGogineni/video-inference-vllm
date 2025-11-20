# import os
# from moviepy import VideoFileClip
# from transformers import pipeline
# import torch

# def extract_audio_mp3(video_path, video_id, output_folder):
#     """Extract audio from video and save as MP3."""
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     try:
#         video_clip = VideoFileClip(video_path)
        
#         if video_clip.audio is None:
#             print(f"  ⚠ Warning: Video has no audio track")
#             video_clip.close()
#             return False

#         output_audio_path = os.path.join(output_folder, f"{video_id}.mp3")
#         video_clip.audio.write_audiofile(output_audio_path, codec='mp3', logger=None)
        
#         video_clip.audio.close()
#         video_clip.close()
        
#         return True

#     except Exception as e:
#         print(f"  ✗ Error extracting audio: {e}")
#         if 'video_clip' in locals() and video_clip:
#             if video_clip.audio: 
#                 video_clip.audio.close()
#             video_clip.close()
#         return False

# def transcribe_audio_whisper(audio_path, video_id, output_folder, config, asr_pipeline):
#     """Transcribe audio using Whisper model."""
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
        
#     if not os.path.exists(audio_path):
#         print(f"  ✗ Error: Audio file not found")
#         return False

#     try:
#         result = asr_pipeline(
#             audio_path,
#             chunk_length_s=config['AUDIO_TRANSCRIPTION']['CHUNK_LENGTH_S'],
#             batch_size=config['AUDIO_TRANSCRIPTION']['BATCH_SIZE'],
#             return_timestamps=False
#         )
#         transcription = result["text"]

#         output_txt_path = os.path.join(output_folder, f"{video_id}.txt")
#         with open(output_txt_path, 'w', encoding='utf-8') as f:
#             f.write(transcription)
        
#         return True

#     except Exception as e:
#         print(f"  ✗ Error transcribing: {e}")
#         return False

# def extract_and_transcribe_all(video_ids, video_folder, config):
#     """Extract and transcribe audio for all videos."""
#     audio_folder = config['PATHS']['AUDIO_FOLDER']
#     transcript_folder = config['PATHS']['AUDIO_TRANSCRIPT_FOLDER']
    
#     # Initialize Whisper pipeline once
#     print("Initializing Whisper pipeline...")
#     asr_pipeline = pipeline(
#         "automatic-speech-recognition",
#         model=config['AUDIO_TRANSCRIPTION']['WHISPER_MODEL'],
#         torch_dtype=torch.float16,
#         device=f"cuda:{config['AUDIO_TRANSCRIPTION']['DEVICE_ID']}"
#     )
#     print("Whisper pipeline initialized\n")
    
#     successful = 0
#     skipped = 0
#     failed = 0
    
#     for idx, vid in enumerate(video_ids, 1):
#         video_id = vid.split('.')[0]
#         video_path = os.path.join(video_folder, vid)
#         transcript_file = os.path.join(transcript_folder, f'{video_id}.txt')
        
#         print(f"[{idx}/{len(video_ids)}] Processing audio for {video_id}...")
        
#         # Skip if transcript already exists
#         if os.path.exists(transcript_file):
#             print(f"  ✓ Transcript already exists, skipping")
#             skipped += 1
#             continue
        
#         # Extract audio
#         audio_path = os.path.join(audio_folder, f'{video_id}.mp3')
#         if not os.path.exists(audio_path):
#             print(f"  → Extracting audio...")
#             if not extract_audio_mp3(video_path, video_id, audio_folder):
#                 failed += 1
#                 continue
        
#         # Transcribe audio
#         print(f"  → Transcribing audio...")
#         if transcribe_audio_whisper(audio_path, video_id, transcript_folder, config, asr_pipeline):
#             print(f"  ✓ Transcription complete")
#             successful += 1
#         else:
#             failed += 1
    
#     print(f"\nAudio transcription summary:")
#     print(f"  ✓ Successful: {successful}")
#     print(f"  → Skipped (already exists): {skipped}")
#     print(f"  ✗ Failed: {failed}")

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from moviepy import VideoFileClip
from transformers import pipeline
import torch
from tqdm import tqdm


# ---------------------------
# AUDIO EXTRACTION (parallel)
# ---------------------------
def extract_audio_mp3(video_path, video_id, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    try:
        video_clip = VideoFileClip(video_path)
        if video_clip.audio is None:
            return False

        output_audio_path = os.path.join(output_folder, f"{video_id}.mp3")
        video_clip.audio.write_audiofile(output_audio_path, codec="mp3", logger=None)

        video_clip.audio.close()
        video_clip.close()
        return True

    except Exception:
        return False


def extract_audio_parallel(video_ids, video_folder, audio_folder, max_workers=8):
    tasks = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for vid in video_ids:
            video_id = vid.split(".")[0]
            video_path = os.path.join(video_folder, vid)
            futures[executor.submit(extract_audio_mp3, video_path, video_id, audio_folder)] = video_id

        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting audio"):
            vid = futures[future]
            success = future.result()
            tasks.append((vid, success))

    return tasks  # list of (video_id, True/False)


# ---------------------------
# WHISPER TRANSCRIPTION (sequential)
# ---------------------------
def transcribe_audio(audio_path, video_id, transcript_folder, asr_pipeline, config):
    os.makedirs(transcript_folder, exist_ok=True)

    if not os.path.exists(audio_path):
        return False

    try:
        result = asr_pipeline(
            audio_path,
            chunk_length_s=config["AUDIO_TRANSCRIPTION"]["CHUNK_LENGTH_S"],
            batch_size=config["AUDIO_TRANSCRIPTION"]["BATCH_SIZE"],
            return_timestamps=False,
        )

        text = result["text"]
        out_path = os.path.join(transcript_folder, f"{video_id}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

        return True

    except Exception as e:
        print(f"✗ Whisper error for {video_id}: {e}")
        return False


def extract_and_transcribe_all(video_ids, video_folder, config):
    audio_folder = config["PATHS"]["AUDIO_FOLDER"]
    transcript_folder = config["PATHS"]["AUDIO_TRANSCRIPT_FOLDER"]
    max_workers = config["AUDIO_TRANSCRIPTION"].get("MAX_WORKERS", 4)

    # 1️⃣ Extract Audio in Parallel
    audio_results = extract_audio_parallel(
        video_ids,
        video_folder,
        audio_folder,
        max_workers=max_workers,
    )

    # 2️⃣ Initialize Whisper once → safe
    print("\nLoading Whisper model ONCE on CUDA…")
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=config["AUDIO_TRANSCRIPTION"]["WHISPER_MODEL"],
        torch_dtype=torch.float16,
        device=f"cuda:{config['AUDIO_TRANSCRIPTION']['DEVICE_ID']}",
    )

    # 3️⃣ Transcribe sequentially (GPU stable)
    success_count = 0
    fail_count = 0

    print("\nTranscribing audio sequentially…")
    for vid, audio_ok in tqdm(audio_results, desc="Transcribing"):
        if not audio_ok:
            fail_count += 1
            continue

        audio_path = os.path.join(audio_folder, f"{vid}.mp3")
        ok = transcribe_audio(audio_path, vid, transcript_folder, asr_pipeline, config)

        if ok:
            success_count += 1
        else:
            fail_count += 1

    print("\nFinal Summary:")
    print(f"  ✓ Successful transcriptions: {success_count}")
    print(f"  ✗ Failed: {fail_count}")
