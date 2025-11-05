import os
import base64
import json
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# Helper: Encode image to base64
# -----------------------------
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# -----------------------------
# Folders & Setup
# -----------------------------
video_folder = "data/inputs/videos/"
key_frames_folder = "data/inputs/key_frames/"
audio_transcript_folder = "data/inputs/audio_transcripts/"
prompt_file = "data/prompt.txt"

output_folder = "data/outputs/openai/"
json_output_folder = os.path.join(output_folder, "json/")
csv_output_folder = os.path.join(output_folder, "csv/")
statistics_folder = os.path.join(output_folder, "statistics/")
failed_folder = os.path.join(output_folder, "failed_videos/")

os.makedirs(json_output_folder, exist_ok=True)
os.makedirs(csv_output_folder, exist_ok=True)
os.makedirs(statistics_folder, exist_ok=True)
os.makedirs(failed_folder, exist_ok=True)

# -----------------------------
# Sample Mode
# -----------------------------
sample = True
if sample:
    sample_size = 1
    sample_frames = 5
else:
    sample_size = len(os.listdir(video_folder))

# -----------------------------
# Load System Instructions
# -----------------------------
with open(prompt_file, 'r') as file:
    system_instructions = file.read()

# -----------------------------
# Get Video Files
# -----------------------------
video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
video_files = video_files[:sample_size]

# -----------------------------
# Stats & Tracking
# -----------------------------
res_dict = {}
failed_videos = []
times = []
max_retries = 3

print(f"Total videos to process: {len(video_files)}")
print(f"Sample video file names: {video_files}")

# ======================================================
# PROCESS ALL VIDEOS
# ======================================================
for idx, video_file in enumerate(video_files, start=1):

    video_id = video_file.split(".")[0]
    video_path = os.path.join(video_folder, video_file)
    print(f"\n\n========== Processing Video {idx}: {video_path} ==========")

    cur_video_start = time.time()
    retry_num = 0
    success_flag = False

    # --------------------------------------------------------------
    # Build key-frame list
    # --------------------------------------------------------------
    keyframe_dir = os.path.join(key_frames_folder, video_id)
    frame_names = os.listdir(keyframe_dir)
    frame_paths = [os.path.join(keyframe_dir, f) for f in frame_names]

    if sample:
        frame_paths = frame_paths[:sample_frames]

    print(f"Using {len(frame_paths)} frames for {video_id}")

    # --------------------------------------------------------------
    # Audio Transcript
    # --------------------------------------------------------------
    transcript_file = os.path.join(audio_transcript_folder, f"{video_id}.txt")
    if os.path.exists(transcript_file):
        with open(transcript_file, 'r') as f:
            audio_transcript = f.read()
    else:
        audio_transcript = ""

    # --------------------------------------------------------------
    # Build request input
    # --------------------------------------------------------------
    encoded_images = [f"data:image/jpeg;base64,{encode_image(p)}" for p in frame_paths]

    content_blocks = [
        {
            "type": "input_text",
            "text": f"{system_instructions}\n\nAudio transcript:\n{audio_transcript}"
        }
    ]

    for img in encoded_images:
        content_blocks.append({
            "type": "input_image",
            "image_url": img
        })

    # ======================================================
    # API CALL + RETRIES
    # ======================================================
    while retry_num < max_retries and not success_flag:
        try:
            response = client.responses.create(
                model="gpt-4o-mini-2024-07-18",
                input=[
                    {
                        "role": "user",
                        "content": content_blocks
                    }
                ]
            )

            response_text = response.output_text
            response_text = response_text.replace("```json", "").replace("```", "").strip()

            cur_response = json.loads(response_text)

            # Save in dictionary
            res_dict[video_id] = cur_response
            print(f"✅ Success for video {video_id}")

            success_flag = True

        except Exception as e:
            print(f"❌ Error for video {video_id}, retry {retry_num+1}/{max_retries}")
            print(e)
            retry_num += 1
            time.sleep(5)

    # --------------------------------------------------------------
    # Handle failures
    # --------------------------------------------------------------
    if not success_flag:
        failed_videos.append(video_path)
        print(f"❌ Failed permanently: {video_path}")
        continue

    # --------------------------------------------------------------
    # Save per-video JSON response (same as Gemini)
    # --------------------------------------------------------------
    json_path = os.path.join(json_output_folder, f"responses_{video_id}.json")
    with open(json_path, "w") as f:
        json.dump(res_dict[video_id], f, indent=4)

    print(f"✅ Saved JSON → {json_path}")

    # --------------------------------------------------------------
    # Timing Stats
    # --------------------------------------------------------------
    cur_video_end = time.time()
    elapsed = cur_video_end - cur_video_start
    times.append(elapsed)
    print(f"⏱ Time for video {video_id}: {elapsed:.2f} sec")

# ======================================================
# FINAL OUTPUT FILES (CSV + ALL JSON + FAILED + STATS)
# ======================================================

# ---- Save combined responses ----
final_json_path = os.path.join(output_folder, "responses_all.json")
with open(final_json_path, "w") as f:
    json.dump(res_dict, f, indent=4)
print(f"\n✅ Saved ALL responses → {final_json_path}")

# ---- Convert to CSV ----
csv_path = os.path.join(csv_output_folder, "ground_truth_openai.csv")
df = pd.DataFrame.from_dict(res_dict, orient="index").reset_index()
df.rename(columns={"index": "video_id"}, inplace=True)
df.to_csv(csv_path, index=False)
print(f"✅ Saved CSV → {csv_path}")

# ---- Failed videos ----
if failed_videos:
    failed_path = os.path.join(failed_folder, "failed_videos.txt")
    with open(failed_path, "a") as f:
        for v in failed_videos:
            f.write(v + "\n")
    print(f"⚠️ Failed videos saved → {failed_path}")

# ---- Statistics ----
stats_path = os.path.join(statistics_folder, "timing_stats.json")
stats = {
    "num_videos": len(video_files),
    "failed": len(failed_videos),
    "success": len(video_files) - len(failed_videos),
    "avg_time_sec": sum(times)/len(times) if times else 0,
    "max_time_sec": max(times) if times else 0,
    "min_time_sec": min(times) if times else 0,
}
with open(stats_path, "w") as f:
    json.dump(stats, f, indent=4)

print(f"✅ Saved statistics → {stats_path}")

print("\n✅✅ All processing complete ✅✅")
