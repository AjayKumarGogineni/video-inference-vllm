import torch
from vllm import LLM, SamplingParams


def load_internvl_model(config):

    model_name = config["MODEL"]["MODEL_NAME"]

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=16384,
        gpu_memory_utilization=0.85,
        dtype=config["MODEL"]["DTYPE"],
        limit_mm_per_prompt={"image": config["VIDEO_PROCESSING"]["NUM_SEGMENTS"]},
        tensor_parallel_size=torch.cuda.device_count(),
    )

    sampling = SamplingParams(
        temperature=config["MODEL"]["TEMPERATURE"],
        top_p=config["MODEL"]["TOP_P"],
        max_tokens=config["MODEL"]["MAX_TOKENS"],
        repetition_penalty=config["MODEL"]["REPETITION_PENALTY"]
    )

    return llm, sampling