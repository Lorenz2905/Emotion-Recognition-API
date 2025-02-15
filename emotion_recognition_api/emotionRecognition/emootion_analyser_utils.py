import requests

import config_loader as config

from console_logging import log_info, log_error


def generate_janus_content(image_paths: list, prompt: str) -> str:
    content = ""

    for i in range(len(image_paths)):
        content += f"This is image_{i}: <image_placeholder>\n"

    content += prompt

    return content


def check_service():
    url = config.get_api_base_url()
    try:
        response = requests.get(url)
        if response.status_code == 200:
            log_info("vLLM service available")
        else:
            log_error(f"vLLM service connection error: {response.status_code}")
            log_error(response.text)
    except requests.exceptions.ConnectionError:
        log_error("vLLM service not available")
