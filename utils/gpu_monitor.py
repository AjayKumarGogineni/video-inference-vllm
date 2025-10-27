import pynvml

def get_gpu_memory():
    """Get total GPU memory usage across all devices in GB."""
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    total_used = 0
    
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_used += mem_info.used
    
    pynvml.nvmlShutdown()
    return total_used / (1024 ** 3)