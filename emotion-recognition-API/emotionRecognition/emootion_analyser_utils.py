def generate_janus_content(image_paths: list, prompt: str) -> str:
    content = ""

    for i in range(len(image_paths)):
        content += f"This is frame_{i}: <image_placeholder>\n"

    content += prompt

    return content
