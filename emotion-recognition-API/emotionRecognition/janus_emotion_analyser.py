import torch
import config_loader as config
from transformers import AutoModelForCausalLM
from Janus.janus.models import MultiModalityCausalLM, VLChatProcessor
from Janus.janus.utils.io import load_pil_images
from console_logging import log_warning
from emotionRecognition.emootion_analyser_utils import generate_janus_content
from emotionRecognition.emotion_analyser import EmotionAnalyzer




class JanusEmotionAnalyzer(EmotionAnalyzer):
    def __init__(self):
        """
        Initialisiert das Modell und den Prozessor.
        """
        self.model_path = config.get_janus_model_path()
        vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(self.model_path)
        self.tokenizer = vl_chat_processor.tokenizer
        vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    def analyze_video_emotions(self, images_path: list[str], text_message: str, system_prompt: str,  streaming: bool = False):
        if streaming:
            log_warning("Janus can not be used for streaming analyse. Please use Qwen2.5-VL")
            return

        content = generate_janus_content(images_path, text_message)
        print("Content:")
        print(content)

        conversation = [
            {
                "role": "<|User|>",
                "content": content,
                "images": images_path,
            },
            {"role": "<|Assistant|>", "content": system_prompt},
        ]

        # load images and prepare for inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(self.vl_gpt.device)

        # # run image encoder to get the image embeddings
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # # run the model to get the response
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

        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        print(f"{prepare_inputs['sft_format'][0]}", answer)
