import os
import time
from pathlib import Path
import torch
import gc
from models.base_model import BaseTTSModel

class VoxCPMWrapper(BaseTTSModel):
    # We design custom synthetic voices using the Voice Design prompt feature of VoxCPM2
    VOICES = {
        'giong_nu_mien_nam': {
            'name': 'Giọng Nữ Miền Nam (Voice Design)',
            'prompt': '(A sweet, warm, and clear Southern Vietnamese female voice) ',
            'gender': 'Female (Nữ)',
            'region': 'Southern (Miền Nam)'
        },
        'giong_nam_mien_bac': {
            'name': 'Giọng Nam Miền Bắc (Voice Design)',
            'prompt': '(A deep, natural, and confident Northern Vietnamese male voice) ',
            'gender': 'Male (Nam)',
            'region': 'Northern (Miền Bắc)'
        },
        'giong_nu_mien_bac': {
            'name': 'Giọng Nữ Miền Bắc (Voice Design)',
            'prompt': '(A gentle, soft, and expressive Northern Vietnamese female voice) ',
            'gender': 'Female (Nữ)',
            'region': 'Northern (Miền Bắc)'
        },
        'giong_nam_mien_nam': {
            'name': 'Giọng Nam Miền Nam (Voice Design)',
            'prompt': '(A friendly, warm, and clear Southern Vietnamese male voice) ',
            'gender': 'Male (Nam)',
            'region': 'Southern (Miền Nam)'
        }
    }

    def __init__(self, device: str = "cuda"):
        super().__init__(
            model_name="OpenBMB VoxCPM2 (openbmb/VoxCPM2)",
            device=device
        )
        self.cache_dir = Path("checkpoints/voxcpm")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_id = "openbmb/VoxCPM2"
        
    def load(self):
        # We don't load the model on init, we load it dynamically in synthesize
        pass

    def get_voices(self) -> list:
        voices_list = []
        for vid, meta in self.VOICES.items():
            voices_list.append({
                'id': vid,
                'name': meta['name'],
                'gender': meta['gender'],
                'region': meta['region']
            })
        return voices_list

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> tuple:
        if voice not in self.VOICES:
            raise ValueError(f"Voice {voice} is not supported by VoxCPM2.")
            
        # Record VRAM before loading
        start_vram = 0
        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            start_vram = torch.cuda.memory_allocated(0)
            
        from voxcpm import VoxCPM
        
        start_time = time.time()
        
        # Load model dynamically into the specified checkpoints directory
        print(f"Loading VoxCPM2 model into checkpoints/voxcpm (2B parameters, may take a moment)...")
        # Set load_denoiser=False to save VRAM and download time, optimize=False to avoid torch.compile errors on Windows
        self.model = VoxCPM.from_pretrained(
            self.model_id, 
            cache_dir=str(self.cache_dir), 
            load_denoiser=False,
            optimize=False,
            device=self.device
        )
        
        # Record VRAM after loading
        mid_vram = 0
        if torch.cuda.is_available() and self.device == "cuda":
            mid_vram = torch.cuda.memory_allocated(0)
            
        # Prepare text with Voice Design prompt
        prompt = self.VOICES[voice]['prompt']
        full_text = prompt + text
        
        # Synthesize speech
        # cfg_value scales speech variety (default 2.0). timesteps=10 (fast inference)
        # Note: VoxCPM speed control can be done via speaking rate or length scaling if exposed.
        # We run the standard generate.
        with torch.no_grad():
            audio_data = self.model.generate(
                text=full_text,
                cfg_value=2.0,
                inference_timesteps=10
            )
            
        end_time = time.time()
        
        sample_rate = self.model.tts_model.sample_rate
        duration = len(audio_data) / float(sample_rate)
        inference_time = end_time - start_time
        rtf = inference_time / duration if duration > 0 else 0
        
        # Record peak VRAM
        peak_vram = 0
        if torch.cuda.is_available() and self.device == "cuda":
            peak_vram = torch.cuda.max_memory_allocated(0)
            
        # Unload model immediately to save VRAM (crucial for 6GB card)
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
        
        return audio_data, sample_rate, stats
