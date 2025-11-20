# import json
# import os

# def load_config(config_path='config.json'):
#     """Load configuration from JSON file."""
#     with open(config_path, 'r') as f:
#         config = json.load(f)
    
#     # Set output paths based on mode
#     mode = config['OUTPUT']['MODE']
#     mode_config = config['OUTPUT']['MODES'][mode]
#     config['OUTPUT']['OUTPUT_FOLDER'] = mode_config['OUTPUT_FOLDER']
    
#     # Create necessary directories
#     folders = [
#         config['OUTPUT']['OUTPUT_FOLDER'],
#         config['PATHS']['AUDIO_TRANSCRIPT_FOLDER'],
#         config['PATHS']['AUDIO_FOLDER']
#     ]
    
#     if config['FEATURES']['EXTRACT_KEY_FRAMES']:
#         folders.append(config['PATHS']['KEY_FRAMES_FOLDER'])
    
#     for folder in folders:
#         os.makedirs(folder, exist_ok=True)
    
#     return config


## V2
import json
import os

def load_config(config_path='config.json'):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # folders = [
    #     config['OUTPUT']['OUTPUT_FOLDER'],
    #     config['PATHS']['AUDIO_TRANSCRIPT_FOLDER'],
    #     # config['PATHS']['AUDIO_FOLDER']
    # ]
    
    # if config['FEATURES']['EXTRACT_KEY_FRAMES']:
    #     folders.append(config['PATHS']['KEY_FRAMES_FOLDER'])
    
    # for folder in folders:
    #     os.makedirs(folder, exist_ok=True)
    
    return config

## V3

# import json
# import os

# def load_config(config_path='config.json'):
#     with open(config_path, 'r') as f:
#         config = json.load(f)

#     # ✅ Restore the missing logic
#     mode = config['OUTPUT']['MODE']
#     mode_config = config['OUTPUT']['MODES'][mode]
#     config['OUTPUT']['OUTPUT_FOLDER'] = mode_config['OUTPUT_FOLDER']
    
#     folders = [
#         config['OUTPUT']['OUTPUT_FOLDER'],
#         config['PATHS']['AUDIO_TRANSCRIPT_FOLDER'],
#         config['PATHS']['AUDIO_FOLDER']
#     ]
    
#     if config['FEATURES']['EXTRACT_KEY_FRAMES']:
#         folders.append(config['PATHS']['KEY_FRAMES_FOLDER'])
    
#     for folder in folders:
#         os.makedirs(folder, exist_ok=True)
    
#     return config
