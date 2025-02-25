import base64

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


def stream_generator(response_stream):
    for chunk in response_stream:
        if chunk.choices:
            yield chunk.choices[0].delta.content


def qwen_massage_generator(images_path: list[str], text_message: str, system_prompt: str):
    content = []

    for image_path in images_path:
        with open(image_path, "rb") as f:
            encoded_image = base64.b64encode(f.read())
        encoded_image_text = encoded_image.decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
        content.append({"type": "image_url", "image_url": {"url": base64_qwen}})

    content.append({"type": "text", "text": text_message})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
    return messages