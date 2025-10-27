import os
from moviepy import VideoFileClip
from transformers import pipeline
import torch

def extract_audio_mp3(video_path, video_id, output_folder):
    """Extract audio from video and save as MP3."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        video_clip = VideoFileClip(video_path)
        
        if video_clip.audio is None:
            print(f"  ⚠ Warning: Video has no audio track")
            video_clip.close()
            return False

        output_audio_path = os.path.join(output_folder, f"{video_id}.mp3")
        video_clip.audio.write_audiofile(output_audio_path, codec='mp3', logger=None)
        
        video_clip.audio.close()
        video_clip.close()
        
        return True

    except Exception as e:
        print(f"  ✗ Error extracting audio: {e}")
        if 'video_clip' in locals() and video_clip:
            if video_clip.audio: 
                video_clip.audio.close()
            video_clip.close()
        return False

def transcribe_audio_whisper(audio_path, video_id, output_folder, config, asr_pipeline):
    """Transcribe audio using Whisper model."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    if not os.path.exists(audio_path):
        print(f"  ✗ Error: Audio file not found")
        return False

    try:
        result = asr_pipeline(
            audio_path,
            chunk_length_s=config['AUDIO_TRANSCRIPTION']['CHUNK_LENGTH_S'],
            batch_size=config['AUDIO_TRANSCRIPTION']['BATCH_SIZE'],
            return_timestamps=False
        )
        transcription = result["text"]

        output_txt_path = os.path.join(output_folder, f"{video_id}.txt")
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(transcription)
        
        return True

    except Exception as e:
        print(f"  ✗ Error transcribing: {e}")
        return False

def extract_and_transcribe_all(video_ids, video_folder, config):
    """Extract and transcribe audio for all videos."""
    audio_folder = config['PATHS']['AUDIO_FOLDER']
    transcript_folder = config['PATHS']['AUDIO_TRANSCRIPT_FOLDER']
    
    # Initialize Whisper pipeline once
    print("Initializing Whisper pipeline...")
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=config['AUDIO_TRANSCRIPTION']['WHISPER_MODEL'],
        torch_dtype=torch.float16,
        device=f"cuda:{config['AUDIO_TRANSCRIPTION']['DEVICE_ID']}"
    )
    print("Whisper pipeline initialized\n")
    
    successful = 0
    skipped = 0
    failed = 0
    
    for idx, vid in enumerate(video_ids, 1):
        video_id = vid.split('.')[0]
        video_path = os.path.join(video_folder, vid)
        transcript_file = os.path.join(transcript_folder, f'{video_id}.txt')
        
        print(f"[{idx}/{len(video_ids)}] Processing audio for {video_id}...")
        
        # Skip if transcript already exists
        if os.path.exists(transcript_file):
            print(f"  ✓ Transcript already exists, skipping")
            skipped += 1
            continue
        
        # Extract audio
        audio_path = os.path.join(audio_folder, f'{video_id}.mp3')
        if not os.path.exists(audio_path):
            print(f"  → Extracting audio...")
            if not extract_audio_mp3(video_path, video_id, audio_folder):
                failed += 1
                continue
        
        # Transcribe audio
        print(f"  → Transcribing audio...")
        if transcribe_audio_whisper(audio_path, video_id, transcript_folder, config, asr_pipeline):
            print(f"  ✓ Transcription complete")
            successful += 1
        else:
            failed += 1
    
    print(f"\nAudio transcription summary:")
    print(f"  ✓ Successful: {successful}")
    print(f"  → Skipped (already exists): {skipped}")
    print(f"  ✗ Failed: {failed}")