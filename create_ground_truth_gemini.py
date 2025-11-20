import os
import pandas as pd
from google import genai
from google.genai import types
# import base64
from dotenv import load_dotenv
import time
import json
import ffmpeg

# Load environment variables from .env file
load_dotenv()

# Access environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
GEMINI_REQUEST_SLEEP_TIME = int(os.getenv("GEMINI_REQUEST_SLEEP_TIME", "30"))
print(f"GEMINI_API_KEY: {GEMINI_API_KEY}")
print(f"GEMINI_MODEL: {GEMINI_MODEL}")

video_folder = os.getenv("VIDEO_FOLDER", "data/inputs/videos/")
prompt_file = os.getenv("PROMPT_FILE", "data/prompt.txt")

video_files = [file for file in os.listdir(video_folder) if file.endswith('.mp4')]
with open(prompt_file, 'r') as file:
    prompt = file.read()

def get_video_metadata(video_path):
    try:
        # Probe video file to get all metadata
        probe = ffmpeg.probe(video_path)
        # Extract video and audio stream details
        video_stream = next((stream for stream in probe["streams"] if stream["codec_type"] == "video"), None)
        audio_stream = next((stream for stream in probe["streams"] if stream["codec_type"] == "audio"), None)
        
        if not video_stream:
            return {"Error": "No video stream found."}

        # Video properties
        width = video_stream.get("width", 0)
        height = video_stream.get("height", 0)
        resolution = f"{width}x{height}"
        frame_rate = eval(video_stream["r_frame_rate"]) if "r_frame_rate" in video_stream else None
        bit_rate_video = int(video_stream.get("bit_rate", 0)) if "bit_rate" in video_stream else None
        codec_video = video_stream.get("codec_name", "Unknown")
        
        # Determine quality category
        quality = (
            "SD" if height <= 720 else
            "HD" if height <= 1080 else
            "Full HD" if height <= 1920 else
            "4K+"
        )

        # Audio properties
        if audio_stream:
            sample_rate = int(audio_stream.get("sample_rate", 0))
            bit_rate_audio = int(audio_stream.get("bit_rate", 0)) if "bit_rate" in audio_stream else None
            channels = int(audio_stream.get("channels", 1))
            codec_audio = audio_stream.get("codec_name", "Unknown")
        else:
            sample_rate, bit_rate_audio, channels, codec_audio = None, None, None, None

        # Duration and size
        duration = float(probe["format"].get("duration", 0))
        file_size = int(probe["format"].get("size", 0))  # File size in bytes

        return {
            "Resolution": resolution,
            "Quality": quality,
            "Frame Rate (FPS)": frame_rate,
            "Video Bit Rate (bps)": bit_rate_video,
            "Video Codec": codec_video,
            "Audio Sample Rate (Hz)": sample_rate,
            "Audio Bit Rate (bps)": bit_rate_audio,
            "Clarity": (
                "High" if bit_rate_audio and bit_rate_audio >= 192000 
                else "Medium" if bit_rate_audio and bit_rate_audio >= 128000 
                else "Low" if bit_rate_audio 
                else "Unknown"
            ),
            "Audio Channels": channels,
            "Audio Codec": codec_audio,
            "Duration (sec)": duration,
            "File Size (MB)": round(file_size / (1024 * 1024), 2)  # Convert bytes to MB
        }
    except Exception as e:
        return {"Error": str(e)}

def generate(video_path, retry_flag):
    client = genai.Client(
        api_key=GEMINI_API_KEY
    )
    files = [
        client.files.upload(file=f"{video_path}")]

    time.sleep(GEMINI_REQUEST_SLEEP_TIME)
    model = GEMINI_MODEL

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=files[0].uri,
                    mime_type=files[0].mime_type,
                ),
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1.0,
        response_mime_type="text/plain",
    )

    try:
        # Generate content using the client
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        print(f'Response: {response}')
        # Return the response
        return response.text
    except Exception as e:
        print(f"Error generating content: {e}")
        return None

start_time = time.time()
res_dict = {}
failed_videos = []
max_retries = 3

times = []
OUTPUT_FOLDER = 'data/outputs/gemini/'
CSV_FOLDER = OUTPUT_FOLDER + 'csv/'
JSON_FOLDER = OUTPUT_FOLDER + 'json/'

if not os.path.exists('data/outputs/'):
    os.makedirs('data/outputs/')
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
if not os.path.exists(CSV_FOLDER):
    os.makedirs(CSV_FOLDER)
if not os.path.exists(JSON_FOLDER):
    os.makedirs(JSON_FOLDER)

SAMPLE = os.getenv("SAMPLE", "false")
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "5"))

if SAMPLE.lower() == "true":
    video_files = video_files[:SAMPLE_SIZE]
    
start = 0
end = len(video_files)
print(f"Total videos to process: {len(video_files)}")
print(f"Sample video file names: {video_files[0:5]}")

for i in range(start, end):
    video_id = video_files[i].split('.')[0]
    video_path = os.path.join(video_folder, video_files[i])
    cur_video_start = time.time()
    print(f"Processing {video_path}...")
    retry_num = 0
    flag = False
    while retry_num < max_retries and not flag:
        try:
            cur_response = generate(video_path, flag)
            # print(f'Video {i+1} response: {cur_response}')
            cur_response = cur_response.replace('```json', '').replace('```', '')
            cur_response = json.loads(cur_response)
            print(f'Keys: {cur_response.keys()}')
            res_dict[video_id] = cur_response
            time.sleep(1)
            flag = True
            continue
        except Exception as e:
            print(f'Retrying video {i+1}, retry number: {retry_num+1}')
            print(f'Error: {e}')
            retry_num += 1
            time.sleep(GEMINI_REQUEST_SLEEP_TIME)
    if retry_num == max_retries:
        failed_videos.append(video_path)

    cur_video_end = time.time()
    times.append(cur_video_end - cur_video_start)
    print(f'Time taken for video {i+1}: {cur_video_end - cur_video_start} seconds')
    if res_dict != {} and video_id in res_dict.keys():
        # for key in metadata.keys():
        #     res_dict[video_id][key] = metadata[key]
        print(f'Completed video {i+1}')
        with open(f'{JSON_FOLDER}/responses_{video_id}.json', 'w') as f:
            json.dump(res_dict[video_id], f, indent=4)
    time.sleep(GEMINI_REQUEST_SLEEP_TIME)

end_time = time.time()
print(f'Time taken: {end_time - start_time} seconds')
average_time = sum(times) / len(times)
print(f"Average time taken for all videos: {average_time} seconds")
print(f"Failed videos: {failed_videos}")

# Save the failed videos to a text file
if failed_videos != []:
    with open(f'data/outputs/gemini/failed_videos/failed_videos.txt', 'a') as f:
        for video in failed_videos:
            f.write(video + '\n')

# Save the dictionary as a JSON file
with open(f'{OUTPUT_FOLDER}/responses_all.json', 'w') as f:
    json.dump(res_dict, f, indent=4)

# Save the json as a csv file
df = pd.DataFrame.from_dict(res_dict, orient="index").reset_index()
df.rename(columns={"index": "video_id"}, inplace=True)
df.to_csv(f'{CSV_FOLDER}/ground_truth_gemini.csv', index = False)