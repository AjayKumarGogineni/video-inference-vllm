import numpy as np

def write_stats(f, name, values):
    """Write statistics for a list of timing values."""
    if not values:
        f.write(f"\n--- {name} Times ---\nNo data collected.\n")
        return
    
    f.write(f"\n--- {name} Times ---\n")
    for idx, t in enumerate(values, 1):
        f.write(f"{name} {idx}: {t:.2f}s\n")
    
    f.write(f"Average {name.lower()} time: {np.mean(values):.2f}s\n")
    f.write(f"Min {name.lower()} time: {np.min(values):.2f}s\n")
    f.write(f"Max {name.lower()} time: {np.max(values):.2f}s\n")
    f.write(f"Std {name.lower()} time: {np.std(values):.2f}s\n")

def save_statistics(config, model_name, model_load_time, model_memory, 
                   video_load_times, inference_times, batch_times, 
                   video_mem_usages, total_time, extract_frames_time, extract_audio_time, statistics_file):
    """Save all processing statistics to file."""
    stats_file = statistics_file
    #config['OUTPUT']['STATISTICS_FILE']
    
    with open(stats_file, 'w') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"==== Processing Statistics ====\n")
        f.write(f"{'='*50}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Model load time: {model_load_time:.2f}s\n")
        f.write(f"Model GPU memory: {model_memory:.2f} GB\n")

        if extract_frames_time is not None:
            f.write(f"Key frame extraction time: {extract_frames_time:.2f}s\n")
        if extract_audio_time is not None:
            f.write(f"Audio extraction and transcription time: {extract_audio_time:.2f}s\n")

        write_stats(f, "Video Load", video_load_times)
        write_stats(f, "Inference", inference_times)

        f.write(f"Average video load time: {np.mean(video_load_times):.2f}s\n")
        f.write(f"Average inference time: {np.mean(inference_times):.2f}s\n")

        f.write(f"\nTotal processing time: {total_time:.2f}s\n")

def print_summary(video_load_times, inference_times, batch_times, 
                 video_mem_usages, model_memory, total_time, extract_frames_time, extract_audio_time):
    """Print summary statistics to console."""
    print(f"\n{'='*50}")
    print(f"=== SUMMARY ===")
    print(f"{'='*50}")

    if extract_frames_time is not None:
        print(f"Key frame extraction time: {extract_frames_time:.2f}s")
    if extract_audio_time is not None:
        print(f"Audio extraction and transcription time: {extract_audio_time:.2f}s")
    
    if video_load_times:
        print(f"Video Load - avg: {np.mean(video_load_times):.2f}s, "
              f"min: {np.min(video_load_times):.2f}s, "
              f"max: {np.max(video_load_times):.2f}s, "
              f"std: {np.std(video_load_times):.2f}s")

    if inference_times:
        print(f"Inference - avg: {np.mean(inference_times):.2f}s, "
              f"min: {np.min(inference_times):.2f}s, "
              f"max: {np.max(inference_times):.2f}s, "
              f"std: {np.std(inference_times):.2f}s")

    if video_load_times:
        print(f"Video Load - avg: {np.mean(video_load_times):.2f}s,")
    if inference_times:
        print(f"Inference - avg: {np.mean(inference_times):.2f}s,")
    print(f"\nTotal GPU memory for model: {model_memory:.2f} GB")
    print(f"Total processing time: {total_time:.2f}s")
    print(f"{'='*50}\n")