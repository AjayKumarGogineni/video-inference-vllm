import json
import os

def load_config(config_path='config.json'):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Set output paths based on mode
    mode = config['OUTPUT']['MODE']
    mode_config = config['OUTPUT']['MODES'][mode]
    config['OUTPUT']['OUTPUT_FOLDER'] = mode_config['OUTPUT_FOLDER']
    config['OUTPUT']['STATISTICS_FILE'] = mode_config['STATISTICS_FILE']
    
    # Create necessary directories
    folders = [
        config['OUTPUT']['OUTPUT_FOLDER'],
        config['PATHS']['AUDIO_TRANSCRIPT_FOLDER'],
        config['PATHS']['AUDIO_FOLDER']
    ]
    
    if config['FEATURES']['EXTRACT_KEY_FRAMES']:
        folders.append(config['PATHS']['KEY_FRAMES_FOLDER'])
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    
    return config