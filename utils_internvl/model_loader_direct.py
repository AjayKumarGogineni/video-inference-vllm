import torch
import math
from transformers import AutoModel, AutoTokenizer


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    name = model_name.split('/')[-1]
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36,
        'InternVL2_5-8B': 32, 'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64,
        'InternVL2_5-78B': 80
    }[name]
    
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
    
    return device_map


def load_internvl_direct(config):
    model_name = config['MODEL']['MODEL_NAME']
    load_in_8bit = config['MODEL'].get('LOAD_IN_8BIT', False)
    use_flash_attn = config['MODEL'].get('USE_FLASH_ATTN', True)
    
    kwargs = {
        'low_cpu_mem_usage': True,
        'use_flash_attn': use_flash_attn,
        'trust_remote_code': True,
    }
    
    if load_in_8bit:
        kwargs['load_in_8bit'] = True
        kwargs['device_map'] = 'auto'
    else:
        dtype_str = config['MODEL'].get('DTYPE', 'half')
        if dtype_str == 'half':
            kwargs['torch_dtype'] = torch.float16
        elif dtype_str == 'bfloat16':
            kwargs['torch_dtype'] = torch.bfloat16
        
        if torch.cuda.device_count() > 1:
            kwargs['device_map'] = split_model(model_name)
        else:
            kwargs['device_map'] = 'auto'
    
    model = AutoModel.from_pretrained(model_name, **kwargs).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False
    )
    
    return model, tokenizer