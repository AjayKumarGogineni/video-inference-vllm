import torch
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from transformers import AutoProcessor
from typing import Optional
from pydantic import BaseModel

class OutputJson(BaseModel):
    """Schema for structured JSON output."""
    summary: Optional[str] = None
    category: Optional[str] = None
    color_composition: Optional[str] = None
    keywords: Optional[str] = None
    creator_demographic: Optional[str] = None
    political_leaning: Optional[str] = None
    misinformation_indicators: Optional[str] = None
    sentiment: Optional[str] = None
    tone: Optional[str] = None
    hateful_content: Optional[str] = None
    hateful_explanation: Optional[str] = None
    hateful_target: Optional[str] = None
    hateful_severity: Optional[str] = None
    graphic_content: Optional[str] = None
    threatening_content: Optional[str] = None
    illicit_content: Optional[str] = None
    self_harm: Optional[str] = None

def load_model(config):
    """Initialize VLM model and sampling parameters."""
    num_segments = config['VIDEO_PROCESSING']['NUM_SEGMENTS']
    model_name = config['MODEL']['MODEL_NAME']
    
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        limit_mm_per_prompt={"image": num_segments},
        tensor_parallel_size=2,#torch.cuda.device_count(),
        dtype=config['MODEL']['DTYPE'],
        max_model_len = 64000#32768
    )
    
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Setup JSON schema and sampling params
    json_schema = OutputJson.model_json_schema()
    guided_decoding_params = GuidedDecodingParams(json=json_schema)
    
    sampling_params = SamplingParams(
        temperature=config['MODEL']['TEMPERATURE'],
        top_p=config['MODEL']['TOP_P'],
        repetition_penalty=config['MODEL']['REPETITION_PENALTY'],
        max_tokens=config['MODEL']['MAX_TOKENS'],
        guided_decoding=guided_decoding_params
    )
    
    return llm, processor, sampling_params