from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    LlavaForConditionalGeneration, 
    Qwen2VLForConditionalGeneration, # Import dla nowego modelu
    AutoProcessor, 
    BitsAndBytesConfig
)
import torch
from PIL import Image
import os

# Prosta klasa konfiguracyjna (możesz ją dostosować lub przekazywać własny config)
class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class VLMModel:
    """Klasa bazowa, aby łatwo dodawać kolejne modele."""
    def predict(self, image_path, prompt):
        raise NotImplementedError

class MoondreamWrapper(VLMModel):
    """
    Model: vikhyatk/moondream2
    Zalety: Bardzo mały, szybki, działa na słabszym sprzęcie.
    """
    def __init__(self, config):
        self.model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2", 
            trust_remote_code=True,
            torch_dtype=torch.float16 if config.DEVICE == "cuda" else torch.float32
        ).to(config.DEVICE)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "vikhyatk/moondream2"
        )

    def predict(self, image_path, prompt):
        image = Image.open(image_path).convert("RGB")
        enc_image = self.model.encode_image(image)
        return self.model.answer_question(enc_image, prompt, self.tokenizer)

class LlavaWrapper(VLMModel):
    """
    Model: llava-hf/llava-1.5-7b-hf
    Zalety: Klasyk, dobre ogólne rozumienie obrazu. Używa kwantyzacji 4-bit dla oszczędności pamięci.
    """
    def __init__(self, config):
        # Konfiguracja kwantyzacji (zmniejsza zużycie VRAM)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
        
        self.processor = AutoProcessor.from_pretrained(
            "llava-hf/llava-1.5-7b-hf"
        )
        
        self.model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf", 
            quantization_config=bnb_config, 
            device_map="auto" # Automatycznie rozkłada model na GPU/CPU
        )

    def predict(self, image_path, prompt):
        image = Image.open(image_path).convert("RGB")
        formatted_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
        
        inputs = self.processor(text=formatted_prompt, images=image, return_tensors="pt")
        # Przenosimy inputy na to samo urządzenie co model
        inputs = inputs.to(self.model.device) 
        if self.model.device.type == 'cuda':
            inputs = inputs.to(torch.float16)

        output = self.model.generate(**inputs, max_new_tokens=50)
        decoded = self.processor.batch_decode(output, skip_special_tokens=True)[0]
        return decoded.split("ASSISTANT:")[-1].strip()

class Qwen2VLWrapper(VLMModel):
    """
    Model: Qwen/Qwen2-VL-2B-Instruct
    Zalety: Nowoczesna architektura, świetny OCR (czytanie tekstu), mały rozmiar (2B), wysoka inteligencja.
    """
    def __init__(self, config):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype=torch.float16 if config.DEVICE == "cuda" else torch.float32,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    def predict(self, image_path, prompt):
        image = Image.open(image_path).convert("RGB")
        
        # Qwen2-VL wymaga struktury wiadomości (chat format)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Przygotowanie promptu
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Przetwarzanie wejścia
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        # Generowanie
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        
        # Wycinanie promptu z odpowiedzi
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0]