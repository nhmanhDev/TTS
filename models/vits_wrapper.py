import os
import time
from pathlib import Path
import torch
import gc
from models.base_model import BaseTTSModel

class MMSVitsWrapper(BaseTTSModel):
    def __init__(self, device: str = "cuda"):
        super().__init__(
            model_name="Meta MMS VITS (facebook/mms-tts-vie)",
            device=device
        )
        self.cache_dir = Path("checkpoints/vits")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_id = "facebook/mms-tts-vie"
        
    def load(self):
        # We don't load the model on init, we load it dynamically in synthesize
        pass

    def get_voices(self) -> list:
        return [
            {
                'id': 'mms_vietnamese',
                'name': 'Meta MMS Vietnamese Speaker',
                'gender': 'Male/Female (Đơn giọng)',
                'region': 'Standard (Toàn dân)'
            }
        ]

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> tuple:
        # Record VRAM before loading
        start_vram = 0
        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            start_vram = torch.cuda.memory_allocated(0)
            
        from transformers import VitsModel, AutoTokenizer
        
        start_time = time.time()
        
        # Load model and tokenizer from local cache or download there
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=str(self.cache_dir))
        self.model = VitsModel.from_pretrained(self.model_id, cache_dir=str(self.cache_dir)).to(self.device).eval()
        
        # Record VRAM after loading
        mid_vram = 0
        if torch.cuda.is_available() and self.device == "cuda":
            mid_vram = torch.cuda.memory_allocated(0)
            
        # Synthesize speech
        inputs = tokenizer(text, return_tensors="pt").to(self.device)
        
        # Set seed for reproducibility if needed, but MMS TTS has slight random variance
        # We can adjust speed via speed parameter if the model architecture supports it
        # Note: MMS VITS uses config.speaking_rate. We can scale it or run standard.
        # Speaking rate is typically set on the model config.
        # Let's adjust speaking rate
        if hasattr(self.model.config, 'speaking_rate'):
            # default speaking rate is usually 1.0, lower speed means slower (speaking_rate = 1.0 / speed)
            # wait, let's keep speaking_rate as 1.0 * speed if it is positive.
            # actually, standard speed = 1.0. If speed is controlled, we can scale speaking_rate
            # speaking_rate default is 1.0. Let's see: speed = 1.0, speaking_rate = 1.0.
            # if user wants 1.2x speed, we can set speaking_rate = 1.2
            self.model.config.speaking_rate = float(speed)
            
        with torch.no_grad():
            outputs = self.model(**inputs)
            waveform = outputs.waveform.cpu().numpy().flatten()
            
        end_time = time.time()
        
        sample_rate = self.model.config.sampling_rate
        duration = len(waveform) / float(sample_rate)
        inference_time = end_time - start_time
        rtf = inference_time / duration if duration > 0 else 0
        
        # Record peak VRAM
        peak_vram = 0
        if torch.cuda.is_available() and self.device == "cuda":
            peak_vram = torch.cuda.max_memory_allocated(0)
            
        # Unload model
        self.unload()
        
        # Calculate memory delta (in MB)
        vram_allocated_mb = (mid_vram - start_vram) / (1024 * 1024)
        vram_peak_mb = (peak_vram - start_vram) / (1024 * 1024)
        
        stats = {
            'inference_time': inference_time,
            'audio_duration': duration,
            'rtf': rtf,
            'vram_usage': max(0.0, vram_allocated_mb),
            'vram_peak': max(0.0, vram_peak_mb),
            'sample_rate': sample_rate,
            'phonemes': 'N/A'
        }
        
        return waveform, sample_rate, stats
