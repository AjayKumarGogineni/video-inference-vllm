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

os.environ["VLLM_BATCH_INVARIANT"] = "1"

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

    # If the original MODEL section contained global metadata (e.g. paths
    # to docker image tar or HF cache), preserve those at top-level so other
    # functions (like start_docker_container) can find them even after we
    # replace config['MODEL'] with the per-model resolved config.
    for key in (
        "DOCKER_IMAGE_TAR",
        "HF_HOME",
        "LOAD_DOCKER_CONTAINER",
        "STOP_DOCKER_CONTAINER_AFTER_INFERENCE",
    ):
        # promote only if present in the original model_section and not
        # already set at top-level (so explicit top-level keys win)
        if key in model_section and key not in config:
            config[key] = model_section[key]

    # Merge top-level and model-level REQUEST and DOCKER settings.
    # Model-level values should override global/top-level defaults.
    request_cfg = dict(config.get("REQUEST", {}))
    model_req = resolved.get("REQUEST") or {}
    if isinstance(model_req, dict):
        request_cfg.update(model_req)

    docker_cfg = dict(config.get("DOCKER", {}))
    model_docker = resolved.get("DOCKER") or {}
    if isinstance(model_docker, dict):
        docker_cfg.update(model_docker)

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


def load_image_file_as_base64(img_path, input_size=None, img_format="PNG", quality=95):
    """Load image, optionally resize, and return base64-encoded image bytes.

    Parameters:
    - img_path: path to the local image file
    - input_size: if provided, resize to (input_size, input_size)
    - img_format: output format, e.g. 'PNG' or 'JPEG'
    - quality: JPEG quality (ignored for PNG)
    """
    with open(img_path, "rb") as img_file:
        img = Image.open(img_file).convert("RGB")
        if input_size:
            img = img.resize((input_size, input_size), Image.LANCZOS)
        buffered = BytesIO()
        fmt = (img_format or "PNG").upper()
        save_kwargs = {}
        if fmt == "JPEG":
            # reduce payload by using JPEG; quality configurable
            save_kwargs["quality"] = int(quality)
            save_kwargs["optimize"] = True
        img.save(buffered, format=fmt, **save_kwargs)
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
    # Return the list of frame file paths. Encoding (size/format/quality)
    # is done later so we can retry with smaller payloads if needed.
    image_paths = []
    for img_name in sorted(os.listdir(frame_folder)):
        if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        img_path = os.path.join(frame_folder, img_name)
        image_paths.append(img_path)
    return image_paths


def encode_images_for_request(image_paths, input_size, img_format, img_quality, max_images=None):
    """Encode up to max_images from image_paths into base64 strings using provided parameters."""
    imgs = []
    paths = image_paths if (max_images is None or max_images >= len(image_paths)) else image_paths[:max_images]
    for img_path in paths:
        try:
            imgs.append(load_image_file_as_base64(img_path, input_size, img_format, img_quality))
        except Exception as exc:
            print(f"  ❌ Error encoding image {img_path}: {exc}")
    return imgs


def request_with_fallback(system_prompt, audio_text, image_paths, base_url, request_cfg, config):
    """Attempt to send a chat completion request; on persistent failures reduce payload
    (fewer images and/or lower JPEG quality) and retry.

    Returns the JSON response on success or raises the last exception on failure.
    """
    vp = config.get("VIDEO_PROCESSING", {})
    input_size = vp.get("INPUT_SIZE", 224)
    configured_format = vp.get("IMAGE_FORMAT", "JPEG")
    configured_quality = int(vp.get("IMAGE_QUALITY", 80))
    max_images_cfg = int(vp.get("MAX_IMAGES_PER_REQUEST", min(6, len(image_paths))))

    # Build candidate (num_images, quality) pairs to try.
    qualities = [configured_quality]
    # if configured_quality > 60:
    #     qualities += [60, 40]
    # elif configured_quality > 40:
    #     qualities += [40, 20]
    # # ensure uniqueness and keep reasonable bounds
    # qualities = [max(10, min(95, q)) for q in dict.fromkeys(qualities)]

    num_images_list = []
    n = max_images_cfg
    while n >= 1:
        num_images_list.append(n)
        n = n // 2
    if 1 not in num_images_list:
        num_images_list.append(1)

    last_exc = None
    attempt = 0
    for quality in qualities:
        for num_images in num_images_list:
            attempt += 1
            print(f"  🔁 Attempt {attempt}: num_images={num_images}, quality={quality}")
            images_b64 = encode_images_for_request(image_paths, input_size, configured_format, quality, max_images=num_images)
            content = build_request_content(system_prompt, audio_text, images_b64, img_format=configured_format)
            chat_data = {
                "model": config["MODEL"]["MODEL_NAME"],
                "messages": [{"role": "user", "content": content}],
                "max_tokens": request_cfg.get("MAX_TOKENS", 2048),
                "temperature": request_cfg.get("TEMPERATURE", 0.7)
            }
            try:
                return send_chat_completion(chat_data, base_url, request_cfg)
            except Exception as exc:
                print(f"  ⚠️  Attempt failed: {exc}")
                last_exc = exc
                # continue to next fallback
    # If we reach here, all fallbacks failed
    raise last_exc if last_exc is not None else RuntimeError("All request attempts failed")


def build_request_content(system_prompt, audio_text, images_b64, img_format="PNG"):
    content = [{
        "type": "text",
        "text": f"{system_prompt.strip()}\n\nAudio Transcript:\n{audio_text.strip()}"
    }]

    mime = "jpeg" if str(img_format).upper() == "JPEG" else "png"
    for img_b64 in images_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/{mime};base64,{img_b64}"}
        })
    return content


def send_chat_completion(chat_data, base_url, request_cfg):
    endpoint = f"{base_url}/chat/completions"
    retries = request_cfg.get("MAX_RETRIES", 3)
    retry_sleep = request_cfg.get("RETRY_SLEEP_SECONDS", 5)
    # Allow long-running model generations by default (1 hour), but
    # allow this to be overridden via the config REQUEST.TIMEOUT_SECONDS
    timeout = request_cfg.get("TIMEOUT_SECONDS", 600)

    # lightweight diagnostic: number of message parts and number of images
    try:
        msgs = chat_data.get("messages", [])
        parts = 0
        images = 0
        if msgs:
            # messages[0].content is a list of content parts in our caller
            content = msgs[0].get("content")
            if isinstance(content, list):
                parts = len(content)
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "image_url":
                        images += 1
        payload_size = len(json.dumps(chat_data))
        print(f"  ℹ️  Sending request: message_parts={parts}, images={images}, approx_payload={payload_size} bytes, timeout={timeout}s")
    except Exception:
        # best-effort diagnostics; never fail the request because of this
        pass

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


def start_docker_container(config, model_name, log_folder=None):
    docker_cfg = config.get("DOCKER", {})
    # Prefer top-level keys, but fall back to model-level values if present. This
    # prevents losing important settings when resolve_model_config replaces
    # config['MODEL'] with the selected sub-config.
    image_tar = config.get("DOCKER_IMAGE_TAR") or config.get("MODEL", {}).get("DOCKER_IMAGE_TAR")
    if image_tar:
        if not os.path.exists(image_tar):
            raise FileNotFoundError(f"Docker image tar not found: {image_tar}")
        print("📦 Loading Docker image from tar…")
        # capture docker load output and save to log file if requested
        load_cmd = ["docker", "load", "-i", image_tar, "-q"]
        load_ok = run_cmd(load_cmd)
        try:
            if log_folder:
                os.makedirs(log_folder, exist_ok=True)
                with open(os.path.join(log_folder, "docker_load.log"), "w", encoding="utf-8") as lf:
                    subprocess.run(load_cmd, stdout=lf, stderr=lf, text=True)
        except Exception:
            # non-fatal: we still continue based on load_ok
            pass
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

    # Prefer top-level HF_HOME but accept model-scoped HF_HOME as a fallback
    hf_home = config.get("HF_HOME") or config.get("MODEL", {}).get("HF_HOME")
    if hf_home:
        cmd.extend(["-v", f"{hf_home}:/root/.cache/huggingface"])
    else:
        default_hf = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        cmd.extend(["-v", f"{default_hf}:/root/.cache/huggingface"])

    cmd.append("-d")
    cmd.append(docker_cfg.get("IMAGE_NAME", "vllm/vllm-openai:latest"))

    model_cfg = config.get("MODEL", {})
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

    # InternVL models need limit-mm-per-prompt to handle multiple images properly
    limit_mm = model_cfg.get("LIMIT_MM_PER_PROMPT")
    if limit_mm:
        cmd.extend(["--limit-mm-per-prompt", limit_mm])

    # Allow disabling chunked prefill (can help with some models)
    if model_cfg.get("DISABLE_CHUNKED_PREFILL", False):
        cmd.append("--disable-chunked-prefill")

    # Allow setting enforce_eager for debugging/compatibility
    if model_cfg.get("ENFORCE_EAGER", False):
        cmd.append("--enforce-eager")

    extra_args = model_cfg.get("EXTRA_ARGS", [])
    if extra_args:
        cmd.extend(extra_args)

    run_cmd(cmd)

    # give the container a short moment to start, then capture docker logs
    time.sleep(docker_cfg.get("POST_RUN_SLEEP_SECONDS", 20))
    try:
        if log_folder:
            os.makedirs(log_folder, exist_ok=True)
            log_path = os.path.join(log_folder, f"docker_{container_name}.log")
            with open(log_path, "w", encoding="utf-8") as lf:
                proc = subprocess.run(["docker", "logs", container_name], stdout=lf, stderr=lf, text=True, timeout=30)
    except Exception as exc:
        print(f"⚠️  Could not save docker logs for {container_name}: {exc}")


def ensure_vllm_server(config, model_name, log_folder=None):
    docker_cfg = config.get("DOCKER", {})
    base_url = f"http://localhost:{docker_cfg.get('HOST_PORT', 8800)}/v1"
    start_time = time.time()
    if wait_for_server(base_url, 1, 1):
        print("✅ Reusing existing vLLM server")
        return base_url, 0.0, True

    start_docker_container(config, model_name, log_folder=log_folder)

    retries = docker_cfg.get("HEALTHCHECK_RETRIES", 20)
    sleep_seconds = docker_cfg.get("HEALTHCHECK_SLEEP_SECONDS", 5)
    if not wait_for_server(base_url, retries, sleep_seconds):
        # capture logs one more time before failing
        try:
            if log_folder:
                os.makedirs(log_folder, exist_ok=True)
                container_name = docker_cfg.get("CONTAINER_NAME", "vllm-container")
                with open(os.path.join(log_folder, f"docker_{container_name}_final.log"), "w", encoding="utf-8") as lf:
                    subprocess.run(["docker", "logs", container_name], stdout=lf, stderr=lf, text=True, timeout=30)
        except Exception:
            pass
        raise RuntimeError("vLLM server did not become ready in time")

    load_time = time.time() - start_time
    return base_url, load_time, False


def process_videos(config):
    """
    Main entry point for video processing.
    Wraps the actual processing in try/finally to ensure Docker container cleanup.
    """
    # Resolve model config early so we have DOCKER settings available for cleanup
    model_cfg = resolve_model_config(config)
    
    try:
        run_video_processing(config)
    except Exception as exc:
        print(f"\n❌ Fatal error during video processing: {exc}")
        # Try to capture docker logs for debugging
        stats_folder = config.get("_stats_folder")
        if stats_folder:
            capture_docker_logs_on_failure(config, stats_folder)
        raise
    finally:
        # Always attempt to stop the Docker container
        stop_docker_container_if_needed(config)


def stop_docker_container_if_needed(config):
    """Stop the Docker container based on config. Safe to call multiple times."""
    stop_docker_container = config.get("STOP_DOCKER_CONTAINER_AFTER_INFERENCE", True)
    if stop_docker_container:
        docker_cfg = config.get("DOCKER", {})
        container_name = docker_cfg.get("CONTAINER_NAME", "vllm-container")
        print(f"🛑 Stopping Docker container {container_name}…")
        run_cmd(["docker", "stop", container_name])
    else:
        print("⚠️  Docker container left running after inference.")


def capture_docker_logs_on_failure(config, log_folder):
    """Capture Docker logs when a failure occurs for debugging."""
    try:
        docker_cfg = config.get("DOCKER", {})
        container_name = docker_cfg.get("CONTAINER_NAME", "vllm-container")
        if log_folder:
            os.makedirs(log_folder, exist_ok=True)
            log_path = os.path.join(log_folder, f"docker_{container_name}_error.log")
            with open(log_path, "w", encoding="utf-8") as lf:
                subprocess.run(
                    ["docker", "logs", "--tail", "200", container_name],
                    stdout=lf, stderr=lf, text=True, timeout=30
                )
            print(f"📝 Docker logs saved to {log_path}")
            # Also print last few lines to console
            result = subprocess.run(
                ["docker", "logs", "--tail", "20", container_name],
                capture_output=True, text=True, timeout=30
            )
            if result.stdout or result.stderr:
                print("  Docker logs:")
                print(result.stdout or result.stderr)
    except Exception as exc:
        print(f"⚠️  Could not capture docker logs: {exc}")


def run_video_processing(config):
    """Main video processing logic, wrapped by process_videos for cleanup."""
    total_start = time.time()
    model_cfg = resolve_model_config(config)
    model_name = model_cfg["MODEL_NAME"]

    # create output folders early so we can save docker logs and other
    # diagnostics into the stats folder while the container starts.
    output_root, json_folder, csv_folder, stats_folder = setup_output_folders(config)
    # Store stats_folder in config so it can be accessed in finally block
    config["_stats_folder"] = stats_folder

    base_url, model_load_time, reused_server = ensure_vllm_server(config, model_name, log_folder=stats_folder)
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
        image_paths = collect_frame_images(video_id, config)
        if not image_paths:
            print(f"  ⚠️  No frames found for {video_id}, skipping.")
            continue

        audio_text = get_audio_transcript(video_id, config)
        load_time = time.time() - load_start

        # Try sending the request with automatic fallbacks (fewer images / lower quality)
        inf_start = time.time()
        try:
            result = request_with_fallback(system_prompt, audio_text, image_paths, base_url, request_cfg, config)
        except Exception as exc:
            print(f"  ❌ Request failed for {video_id}: {exc}")
            # Capture docker logs for debugging
            capture_docker_logs_on_failure(config, stats_folder)
            continue

        latency = time.time() - inf_start
        # vLLM returns structured choices
        summary = result["choices"][0]["message"]["content"]
        cleaned = summary.replace("```json", "").replace("```", "").strip()

        try:
            summary_dict = json.loads(cleaned)
        except json.JSONDecodeError:
            print(f"  ⚠️  Failed to parse JSON for {video_id}, storing raw text.")
            summary_dict = {"summary": cleaned, "category": "Unknown"}

        summary_dict.setdefault("summary", cleaned)
        summary_dict.setdefault("category", "Unknown")
        summary_dict.update({
            "video_id": video_id
        })

        responses.append(summary_dict)
        video_load_times.append(load_time)
        inference_times.append(latency)

        json_path = os.path.join(json_folder, f"{video_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2)

        print(f"  ✅ Loaded in {load_time:.2f}s | Inference {latency:.2f}s | Frames {len(image_paths)}")

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
            common = set(ground_truth.video_id.astype(str)) & set(df['video_id'].astype(str))
            if not common:
                print("⚠️  No overlapping video IDs for evaluation.")
            else:
                gt = ground_truth[ground_truth.video_id.astype(str).isin(common)].sort_values("video_id").reset_index(drop=True)
                df_eval = df[df['video_id'].astype(str).isin(common)].sort_values("video_id").reset_index(drop=True)
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