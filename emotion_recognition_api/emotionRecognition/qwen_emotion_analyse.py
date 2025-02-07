from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, StoppingCriteriaList, StoppingCriteria
from qwen_vl_utils import process_vision_info
import torch

import config_loader as config
from emotionRecognition.emotion_analyser import EmotionAnalyzer


class QwenEmotionAnalyzer(EmotionAnalyzer):
    def __init__(self):
        """
        Initialisiert das Modell und den Prozessor.
        """
        model_path = config.get_qwen_model_path()
        if config.get_use_flash_attention():
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype="auto", device_map="auto"
            )

        self.processor = AutoProcessor.from_pretrained(model_path)

    def analyze_video_emotions(self, images_path: list[str], text_message: str, system_prompt: str, streaming: bool = False):

        content = [{"type": "image", "image": f"file://{path}"} for path in images_path]
        content.append({"type": "text", "text": text_message})


        messages = [
            {
                "role": "user",
                "content": content,
            },
            {"role": "system", "content": system_prompt},

        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(config.get_device())

        if streaming:
            stop_criteria = StoppingCriteriaList([StoppingCriteria(max_length=128)])

            def stream_output():
                stream_generated_ids = self.model.generate(**inputs, max_new_tokens=128, stopping_criteria=stop_criteria)
                for token_id in stream_generated_ids[0]:
                    stream_output_text = self.processor.decode([token_id], skip_special_tokens=True)
                    yield stream_output_text

            return stream_output()
        else:
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            raw_output = output_text[0]
            return raw_output