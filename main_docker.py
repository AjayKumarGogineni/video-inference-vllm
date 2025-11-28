import json
import os
import statistics
import subprocess
import sys
import time
from base64 import b64encode
from io import BytesIO

import pandas as pd
import requests
from PIL import Image
from bert_score import score as bertscore_score
from sklearn.metrics import accuracy_score, f1_score


def run_cmd(cmd, verbose=True):
    if verbose:
        print("🚀 Running:", " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("❌ Error:", result.stderr)
    else:
        print("✅ Success:", result.stdout)
    return result.returncode == 0


def evaluate_with_stats(ground_truth, test_df):
    refs, cands = ground_truth["summary"].tolist(), test_df["summary"].tolist()
    P, R, F1 = bertscore_score(cands, refs, lang="en", model_type="xlm-roberta-large", verbose=True)

    y_true = ground_truth["category"].astype(str).str.strip()
    y_pred = test_df["category"].astype(str).str.strip().str.rstrip(".")
    acc = accuracy_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred, average="weighted")

    results = {
        "BERTScore": {
            "Precision": {"mean": float(P.mean()), "std": float(P.std())},
            "Recall": {"mean": float(R.mean()), "std": float(R.std())},
            "F1": {"mean": float(F1.mean()), "std": float(F1.std())},
        },
        "Category": {"Accuracy": acc, "F1_weighted": f1}
    }
    return results


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_model_config(config):
    model_section = config.get("MODEL", {})
    if "CONFIGS" in model_section:
        selected_key = model_section.get("SELECTED_KEY") or model_section.get("SELECTED_MODEL")
        options = model_section.get("CONFIGS", {})
        if not selected_key or selected_key not in options:
            raise ValueError("MODEL.CONFIGS must include the selected key")
        resolved = dict(options[selected_key])
        resolved.setdefault("MODEL_SUFFIX", selected_key)
        available = [cfg.get("MODEL_NAME") for cfg in options.values() if cfg.get("MODEL_NAME")]
        if available:
            resolved["AVAILABLE_MODELS"] = available
    else:
        resolved = dict(model_section)

    if not resolved.get("MODEL_NAME"):
        raise ValueError("MODEL configuration must include MODEL_NAME")

    request_cfg = dict(resolved.get("REQUEST", {})) or dict(config.get("REQUEST", {}))
    docker_cfg = dict(resolved.get("DOCKER", {})) or dict(config.get("DOCKER", {}))

    config["MODEL"] = resolved
    config["REQUEST"] = request_cfg
    config["DOCKER"] = docker_cfg

    return resolved


def get_gpu_memory_usage():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits"
            ],
            text=True
        )
        values = [float(line.strip()) for line in output.splitlines() if line.strip()]
        if not values:
            return None
        return sum(values) / 1024.0
    except Exception as exc:
        print(f"⚠️  Unable to query GPU memory: {exc}")
        return None


def summarize_times(values):
    if not values:
        return {"avg": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    return {
        "avg": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0
    }


def write_statistics(stats_folder, stats_payload, detailed=True):
    video_stats = summarize_times(stats_payload.get("video_load_times", []))
    inference_stats = summarize_times(stats_payload.get("inference_times", []))

    lines = [
        "=" * 50,
        "==== Processing Statistics ====",
        "=" * 50,
        f"Model: {stats_payload['model_name']}",
    ]

    if stats_payload.get("reused_server"):
        lines.append(f"Model load time: {stats_payload['model_load_time']:.2f}s (reused existing server)")
    else:
        lines.append(f"Model load time: {stats_payload['model_load_time']:.2f}s")

    gpu_mem = stats_payload.get("gpu_memory")
    if gpu_mem is not None:
        lines.append(f"Model GPU memory: {gpu_mem:.2f} GB")

    lines.append("\n--- Video Load Times ---")
    if detailed:
        for idx, value in enumerate(stats_payload.get("video_load_times", []), start=1):
            lines.append(f"Video Load {idx}: {value:.2f}s")
    lines.append(f"Average video load time: {video_stats['avg']:.2f}s")
    lines.append(f"Min video load time: {video_stats['min']:.2f}s")
    lines.append(f"Max video load time: {video_stats['max']:.2f}s")
    lines.append(f"Std video load time: {video_stats['std']:.2f}s")

    lines.append("\n--- Inference Times ---")
    if detailed:
        for idx, value in enumerate(stats_payload.get("inference_times", []), start=1):
            lines.append(f"Inference {idx}: {value:.2f}s")
    lines.append(f"Average inference time: {inference_stats['avg']:.2f}s")
    lines.append(f"Min inference time: {inference_stats['min']:.2f}s")
    lines.append(f"Max inference time: {inference_stats['max']:.2f}s")
    lines.append(f"Std inference time: {inference_stats['std']:.2f}s")

    lines.append("")
    lines.append(f"Average video load time: {video_stats['avg']:.2f}s")
    lines.append(f"Average inference time: {inference_stats['avg']:.2f}s")

    lines.append("")
    lines.append(f"Total processing time: {stats_payload['total_time']:.2f}s")

    stats_path = os.path.join(stats_folder, "inference_statistics.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"📊 Statistics saved to {stats_path}")


def setup_output_folders(config):
    model_suffix = config["MODEL"].get("MODEL_SUFFIX", "model")
    base_out = config["OUTPUT"]["OUTPUT_FOLDER"]
    input_size = config["VIDEO_PROCESSING"].get("INPUT_SIZE", 224)
    num_frames = config["VIDEO_PROCESSING"].get("NUM_SEGMENTS", 0)

    output_root = f"{base_out}_{model_suffix}_inputsize{input_size}"
    if num_frames:
        output_root += f"_numframes{num_frames}"

    features = config.get("FEATURES", {})
    if features.get("USE_KEY_FRAMES", False):
        output_root += "_keyframes"
    elif features.get("USE_UNIFORM_FRAMES", False):
        output_root += "_uniformframes"

    json_folder = os.path.join(output_root, "json")
    csv_folder = os.path.join(output_root, "csv")
    stats_folder = os.path.join(output_root, "statistics")

    os.makedirs(json_folder, exist_ok=True)
    os.makedirs(csv_folder, exist_ok=True)
    os.makedirs(stats_folder, exist_ok=True)

    return output_root, json_folder, csv_folder, stats_folder


def get_video_list(config):
    video_folder = config["PATHS"]["VIDEO_FOLDER"]
    video_files = [
        vid for vid in sorted(os.listdir(video_folder))
        if vid.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
    ]

    sampling_cfg = config.get("SAMPLING", {})
    if sampling_cfg.get("SAMPLE", False):
        sample_size = sampling_cfg.get("SAMPLE_SIZE", len(video_files))
        return video_files[:sample_size]
    return video_files


def get_audio_transcript(video_id, config):
    transcript_folder = config["PATHS"]["AUDIO_TRANSCRIPT_FOLDER"]
    transcript_path = os.path.join(transcript_folder, f"{video_id}.txt")
    if os.path.exists(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            return f.read()
    return "No audio transcript available."


def load_image_file_as_base64(img_path, input_size=None):
    with open(img_path, "rb") as img_file:
        img = Image.open(img_file).convert("RGB")
        if input_size:
            img = img.resize((input_size, input_size), Image.LANCZOS)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return b64encode(buffered.getvalue()).decode("utf-8")


def collect_frame_images(video_id, config):
    features = config.get("FEATURES", {})
    if features.get("USE_KEY_FRAMES", True):
        frames_root = config["PATHS"]["KEY_FRAMES_FOLDER"]
    elif features.get("USE_UNIFORM_FRAMES", False):
        frames_root = config["PATHS"]["UNIFORM_FRAMES_FOLDER"]
    else:
        raise ValueError("No frame extraction strategy enabled in config")

    frame_folder = os.path.join(frames_root, video_id)
    if not os.path.exists(frame_folder):
        return []

    input_size = config["VIDEO_PROCESSING"].get("INPUT_SIZE", 224)
    images_b64 = []

    for img_name in sorted(os.listdir(frame_folder)):
        if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        img_path = os.path.join(frame_folder, img_name)
        try:
            images_b64.append(load_image_file_as_base64(img_path, input_size))
        except Exception as exc:
            print(f"  ❌ Error loading image {img_name}: {exc}")
    return images_b64


def build_request_content(system_prompt, audio_text, images_b64):
    content = [{
        "type": "text",
        "text": f"{system_prompt.strip()}\n\nAudio Transcript:\n{audio_text.strip()}"
    }]

    for img_b64 in images_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
        })
    return content


def send_chat_completion(chat_data, base_url, request_cfg):
    endpoint = f"{base_url}/chat/completions"
    retries = request_cfg.get("MAX_RETRIES", 3)
    retry_sleep = request_cfg.get("RETRY_SLEEP_SECONDS", 5)
    timeout = request_cfg.get("TIMEOUT_SECONDS", 600)

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(endpoint, json=chat_data, timeout=timeout)
            if response.status_code == 200:
                return response.json()
            print(f"  ⚠️  Request failed ({response.status_code}): {response.text}")
        except requests.RequestException as exc:
            print(f"  ⚠️  Request error: {exc}")

        if attempt < retries:
            print(f"  🔁 Retrying in {retry_sleep}s...")
            time.sleep(retry_sleep)
    raise RuntimeError("Chat completion failed after maximum retries")


def wait_for_server(base_url, retries, sleep_seconds):
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(f"{base_url}/models", timeout=10)
            if resp.status_code == 200:
                print("🔥 vLLM is alive!")
                return True
        except requests.RequestException:
            pass
        if attempt < retries:
            print(f"  ⏳ Waiting for server ({attempt}/{retries})...")
            time.sleep(sleep_seconds)
    return False


def start_docker_container(config, model_name):
    docker_cfg = config.get("DOCKER", {})
    image_tar = config.get("DOCKER_IMAGE_TAR")
    if image_tar:
        if not os.path.exists(image_tar):
            raise FileNotFoundError(f"Docker image tar not found: {image_tar}")
        print("📦 Loading Docker image from tar…")
        load_ok = run_cmd(["docker", "load", "-i", image_tar, "-q"])
        if not load_ok:
            raise RuntimeError("Docker image load failed; aborting container launch.")
        time.sleep(docker_cfg.get("POST_LOAD_SLEEP_SECONDS", 180))

    print("🚀 Starting vLLM container…")
    container_name = docker_cfg.get("CONTAINER_NAME", "vllm-container")
    host_port = docker_cfg.get("HOST_PORT", 8800)
    container_port = docker_cfg.get("CONTAINER_PORT", 8000)

    cmd = [
        "docker", "run", "--rm",
        "--name", container_name,
        "--gpus", docker_cfg.get("GPUS", "all"),
        f"--ipc={docker_cfg.get('IPC', 'host')}",
        "-p", f"{host_port}:{container_port}",
    ]

    hf_home = config.get("HF_HOME")
    if hf_home:
        cmd.extend(["-v", f"{hf_home}:/root/.cache/huggingface"])
    else:
        default_hf = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        cmd.extend(["-v", f"{default_hf}:/root/.cache/huggingface"])

    cmd.append("-d")
    cmd.append(docker_cfg.get("IMAGE_NAME", "vllm/vllm-openai:latest"))

    model_cfg = config["MODEL"]
    cmd.extend(["--model", model_name])

    if model_cfg.get("TOKENIZER_MODE"):
        cmd.extend(["--tokenizer-mode", model_cfg["TOKENIZER_MODE"]])
    if model_cfg.get("CONFIG_FORMAT"):
        cmd.extend(["--config-format", model_cfg["CONFIG_FORMAT"]])
    if model_cfg.get("LOAD_FORMAT"):
        cmd.extend(["--load-format", model_cfg["LOAD_FORMAT"]])
    if model_cfg.get("TOOL_CALL_PARSER"):
        cmd.extend(["--tool-call-parser", model_cfg["TOOL_CALL_PARSER"]])
    if model_cfg.get("ENABLE_AUTO_TOOL_CHOICE", False):
        cmd.append("--enable-auto-tool-choice")

    cmd.extend(["--tensor-parallel-size", str(model_cfg.get("TENSOR_PARALLEL_SIZE", 1))])
    cmd.extend(["--dtype", model_cfg.get("DTYPE", "bfloat16")])
    cmd.extend(["--max-model-len", str(model_cfg.get("MAX_MODEL_LEN", 32768))])
    cmd.append("--trust-remote-code")
    cmd.extend(["--gpu-memory-utilization", str(model_cfg.get("GPU_MEMORY_UTILIZATION", 0.9))])
    cmd.extend(["--max-num-seqs", str(model_cfg.get("MAX_NUM_SEQS", 8))])
    cmd.extend(["--max-num-batched-tokens", str(model_cfg.get("MAX_NUM_BATCHED_TOKENS", 32768))])
    cmd.extend(["--kv-cache-memory-bytes", str(model_cfg.get("KV_CACHE_MEMORY_BYTES", 20_000_000_000))])

    extra_args = model_cfg.get("EXTRA_ARGS", [])
    if extra_args:
        cmd.extend(extra_args)

    run_cmd(cmd)
    time.sleep(docker_cfg.get("POST_RUN_SLEEP_SECONDS", 20))


def ensure_vllm_server(config, model_name):
    docker_cfg = config.get("DOCKER", {})
    base_url = f"http://localhost:{docker_cfg.get('HOST_PORT', 8800)}/v1"
    start_time = time.time()
    if wait_for_server(base_url, 1, 1):
        print("✅ Reusing existing vLLM server")
        return base_url, 0.0, True

    start_docker_container(config, model_name)

    retries = docker_cfg.get("HEALTHCHECK_RETRIES", 20)
    sleep_seconds = docker_cfg.get("HEALTHCHECK_SLEEP_SECONDS", 5)
    if not wait_for_server(base_url, retries, sleep_seconds):
        raise RuntimeError("vLLM server did not become ready in time")

    load_time = time.time() - start_time
    return base_url, load_time, False


def process_videos(config):
    total_start = time.time()
    model_cfg = resolve_model_config(config)
    model_name = model_cfg["MODEL_NAME"]
    base_url, model_load_time, reused_server = ensure_vllm_server(config, model_name)
    gpu_memory = get_gpu_memory_usage()

    system_prompt_file = config["PATHS"].get("PROMPT_FILE")
    if system_prompt_file and os.path.exists(system_prompt_file):
        with open(system_prompt_file, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    else:
        system_prompt = "You are a helpful assistant. Analyze the visuals and transcript."

    video_files = get_video_list(config)
    if not video_files:
        print("⚠️  No videos found to process.")
        return

    output_root, json_folder, csv_folder, stats_folder = setup_output_folders(config)
    request_cfg = config.get("REQUEST", {})
    stats_cfg = config.get("STATISTICS", {})
    detailed_stats = stats_cfg.get("SAVE_DETAILED", True)

    responses = []
    video_load_times = []
    inference_times = []

    for idx, video_file in enumerate(video_files, 1):
        video_id = os.path.splitext(video_file)[0]
        print(f"\n[{idx}/{len(video_files)}] Processing {video_id}…")

        load_start = time.time()
        images_b64 = collect_frame_images(video_id, config)
        if not images_b64:
            print(f"  ⚠️  No frames found for {video_id}, skipping.")
            continue

        audio_text = get_audio_transcript(video_id, config)
        content = build_request_content(system_prompt, audio_text, images_b64)
        load_time = time.time() - load_start

        chat_data = {
            "model": model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": request_cfg.get("MAX_TOKENS", 2048),
            "temperature": request_cfg.get("TEMPERATURE", 0.7)
        }

        inf_start = time.time()
        try:
            result = send_chat_completion(chat_data, base_url, request_cfg)
        except Exception as exc:
            print(f"  ❌ Request failed for {video_id}: {exc}")
            continue

        latency = time.time() - inf_start
        summary = result["choices"][0]["message"]["content"]
        cleaned = summary.replace("```json", "").replace("```", "").strip()

        try:
            summary_dict = json.loads(cleaned)
        except json.JSONDecodeError:
            print(f"  ⚠️  Failed to parse JSON for {video_id}, storing raw text.")
            summary_dict = {"summary": cleaned, "category": "Unknown"}

        summary_dict.setdefault("summary", cleaned)
        summary_dict.setdefault("category", "Unknown")
        # summary_dict.update({
        #     "video_id": video_id,
        #     "model": model_name,
        #     "response_time": latency,
        #     "frame_count": len(images_b64)
        # })

        responses.append(summary_dict)
        video_load_times.append(load_time)
        inference_times.append(latency)

        json_path = os.path.join(json_folder, f"{video_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2)

        print(f"  ✅ Loaded in {load_time:.2f}s | Inference {latency:.2f}s | Frames {len(images_b64)}")

    if not responses:
        print("❌ No videos were processed successfully.")
        return

    df = pd.DataFrame(responses)
    csv_path = os.path.join(csv_folder, f"video_summaries_{config['MODEL'].get('MODEL_SUFFIX', 'model')}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to {csv_path}")

    eval_cfg = config.get("EVALUATION", {})
    ground_truth_file = config["PATHS"].get("GROUND_TRUTH_FILE")
    if eval_cfg.get("CALCULATE_METRICS", False) and ground_truth_file and os.path.exists(ground_truth_file):
        print("\n" + "=" * 60)
        print("Running Evaluation")
        print("=" * 60)
        try:
            ground_truth = pd.read_csv(ground_truth_file)
            common = set(ground_truth.video_id.astype(str)) & set(df.video_id.astype(str))
            if not common:
                print("⚠️  No overlapping video IDs for evaluation.")
            else:
                gt = ground_truth[ground_truth.video_id.astype(str).isin(common)].sort_values("video_id").reset_index(drop=True)
                df_eval = df[df.video_id.astype(str).isin(common)].sort_values("video_id").reset_index(drop=True)
                eval_results = evaluate_with_stats(gt, df_eval)
                with open(os.path.join(stats_folder, "evaluation.json"), "w", encoding="utf-8") as f:
                    json.dump(eval_results, f, indent=2)
                print(f"✅ Evaluation complete for {len(common)} videos")
        except Exception as exc:
            print(f"⚠️  Evaluation failed: {exc}")
    else:
        print("\nSkipping evaluation (disabled or ground truth missing).")

    total_time = time.time() - total_start
    write_statistics(
        stats_folder,
        {
            "model_name": model_name,
            "model_load_time": model_load_time,
            "reused_server": reused_server,
            "gpu_memory": gpu_memory,
            "video_load_times": video_load_times,
            "inference_times": inference_times,
            "total_time": total_time
        },
        detailed_stats
    )

    print(f"\n🎉 DONE — Full pipeline completed! Outputs stored in {output_root}")

    stop_docker_container = config.get("STOP_DOCKER_CONTAINER_AFTER_INFERENCE", True)
    if stop_docker_container:
        docker_cfg = config.get("DOCKER", {})
        container_name = docker_cfg.get("CONTAINER_NAME", "qwenvl32b-vllm")
        print(f"🛑 Stopping Docker container {container_name}…")
        run_cmd(["docker", "stop", container_name])
    else:
        print("⚠️  Docker container left running after inference.")

def main():
    default_config = os.path.join(
        "data",
        "configs",
        "config_docker.json"
    )

    config_path = sys.argv[1] if len(sys.argv) > 1 else default_config
    print(f"📄 Loading config from {config_path}")
    config = load_config(config_path)
    process_videos(config)


if __name__ == "__main__":
    main()