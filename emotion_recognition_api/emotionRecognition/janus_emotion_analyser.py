from typing import List, Generator
import torch
import config_loader as config
from transformers import AutoModelForCausalLM
from Janus.janus.models import MultiModalityCausalLM, VLChatProcessor
from Janus.janus.utils.io import load_pil_images
from emotionRecognition.emootion_analyser_utils import generate_janus_content
from emotionRecognition.emotion_analyser import EmotionAnalyzer
from fastapi.responses import StreamingResponse


class JanusEmotionAnalyzer(EmotionAnalyzer):
    def __init__(self):
        self.model_path = config.get_janus_model_path()
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(self.model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer
        self.vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True
        ).to(torch.bfloat16).cuda().eval()

    def _prepare_inputs(self, images_path: List[str], text_message: str, system_prompt: str):
        content = generate_janus_content(images_path, text_message)

        conversation = [
            {"role": "<|User|>", "content": content, "images": images_path},
            {"role": "<|Assistant|>", "content": system_prompt},
        ]

        pil_images = load_pil_images(conversation)
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(self.vl_gpt.device)

        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        return prepare_inputs, inputs_embeds


    def analyze_video_emotions(self, images_path: List[str], text_message: str, system_prompt: str) -> str:
        prepare_inputs, inputs_embeds = self._prepare_inputs(images_path, text_message, system_prompt)
        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )
        return self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)


    def analyze_stream_video_emotions(self, images_path: List[str], text_message: str, system_prompt: str) -> StreamingResponse:
        def stream_generate_response(inputs_embeds_stream, attention_mask) -> Generator[str, None, None]:
            output = self.vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds_stream,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
                output_scores=True,
            )

            for token_id in output:
                yield self.tokenizer.decode(token_id.cpu().tolist(), skip_special_tokens=True)

        prepare_inputs, inputs_embeds = self._prepare_inputs(images_path, text_message, system_prompt)
        return StreamingResponse(stream_generate_response(prepare_inputs.attention_mask, inputs_embeds), media_type="text/plain")
