from transformers import AutoModelForCausalLM, AutoTokenizer, LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import torch
from PIL import Image
import os

# Definiujemy ścieżkę na dysku dodatkowym
CACHE_PATH = "/mnt/DyskDodatkowy/huggingface_cache"

class VLMModel:
    """Klasa bazowa, aby łatwo dodawać kolejne modele."""
    def predict(self, image_path, prompt):
        raise NotImplementedError

class MoondreamWrapper(VLMModel):
    def __init__(self, config):
        # Dodano cache_dir
        self.model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2", 
            trust_remote_code=True,
            torch_dtype=torch.float16 if config.DEVICE == "cuda" else torch.float32,
            cache_dir=CACHE_PATH
        ).to(config.DEVICE)
        
        # Dodano cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(
            "vikhyatk/moondream2",
            cache_dir=CACHE_PATH
        )

    def predict(self, image_path, prompt):
        image = Image.open(image_path).convert("RGB")
        enc_image = self.model.encode_image(image)
        return self.model.answer_question(enc_image, prompt, self.tokenizer)

class LlavaWrapper(VLMModel):
    def __init__(self, config):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
        
        # Dodano cache_dir
        self.processor = AutoProcessor.from_pretrained(
            "llava-hf/llava-1.5-7b-hf",
            cache_dir=CACHE_PATH
        )
        
        # Dodano cache_dir
        self.model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf", 
            quantization_config=bnb_config, 
            device_map="auto",
            cache_dir=CACHE_PATH
        )

    def predict(self, image_path, prompt):
        image = Image.open(image_path).convert("RGB")
        formatted_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
        inputs = self.processor(text=formatted_prompt, images=image, return_tensors="pt").to("cuda", torch.float16)
        output = self.model.generate(**inputs, max_new_tokens=50)
        decoded = self.processor.batch_decode(output, skip_special_tokens=True)[0]
        return decoded.split("ASSISTANT:")[-1].strip()