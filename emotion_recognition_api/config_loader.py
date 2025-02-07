import yaml

CONFIG = None

def load_config(file_path="config.yaml"):
    global CONFIG
    if CONFIG is None:
        with open(file_path, "r") as f:
            CONFIG = yaml.safe_load(f)
    return CONFIG

def get_device():
    return CONFIG.get("device") if isinstance(CONFIG, dict) else None

def get_qwen_model_path():
    return CONFIG.get("qwen").get("qwen_model_path") if isinstance(CONFIG, dict) else None

def get_janus_model_path():
    return CONFIG.get("janus").get("janus_model_path") if isinstance(CONFIG, dict) else None

def get_use_flash_attention():
    return CONFIG.get("qwen").get("use_flash_attention", False) if isinstance(CONFIG, dict) else None

def get_use_janus():
    return CONFIG.get("janus").get("use_janus", False) if isinstance(CONFIG, dict) else None

def get_temp_dir():
    return CONFIG.get("temp_dir") if isinstance(CONFIG, dict) else None

load_config()
