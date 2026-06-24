import os
import time
from pathlib import Path
import torch
import gc
from huggingface_hub import hf_hub_download
from models.base_model import BaseTTSModel

class KokoroVietnameseWrapper(BaseTTSModel):
    # Metadata for all voices
    VOICES_METADATA = {
        'diem_trinh': {'name': 'Diễm Trinh', 'gender': 'Female (Nữ)', 'region': 'Southern (Miền Nam)'},
        'hung_thinh': {'name': 'Hưng Thịnh', 'gender': 'Male (Nam)', 'region': 'Southern (Miền Nam)'},
        'mai_linh': {'name': 'Mai Linh', 'gender': 'Female (Nữ)', 'region': 'Southern (Miền Nam)'},
        'mai_loan': {'name': 'Mai Loan', 'gender': 'Female (Nữ)', 'region': 'Southern (Miền Nam)'},
        'manh_dung': {'name': 'Mạnh Dũng', 'gender': 'Male (Nam)', 'region': 'Northern (Miền Bắc)'},
        'my_yen': {'name': 'Mỹ Yến', 'gender': 'Female (Nữ)', 'region': 'Southern (Miền Nam)'},
        'ngoc_huyen': {'name': 'Ngọc Huyền', 'gender': 'Female (Nữ)', 'region': 'Northern (Miền Bắc)'},
        'phat_tai': {'name': 'Phát Tài', 'gender': 'Male (Nam)', 'region': 'Southern (Miền Nam)'},
        'thanh_dat': {'name': 'Thành Đạt', 'gender': 'Male (Nam)', 'region': 'Southern (Miền Nam)'},
        'thuc_trinh': {'name': 'Thục Trinh', 'gender': 'Female (Nữ)', 'region': 'Southern (Miền Nam)'},
        'tuan_ngoc': {'name': 'Tuấn Ngọc', 'gender': 'Male (Nam)', 'region': 'Northern (Miền Bắc)'},
        'duc_an': {'name': 'Đức An', 'gender': 'Male (Nam)', 'region': 'Northern (Miền Bắc)'},
        'duc_duy': {'name': 'Đức Duy', 'gender': 'Male (Nam)', 'region': 'Southern (Miền Nam)'},
        'storyvert': {'name': 'Storyvert', 'gender': 'Female (Nữ)', 'region': 'Southern (Miền Nam)'},
    }

    def __init__(self, device: str = "cuda"):
        super().__init__(
            model_name="Kokoro-Vietnamese (contextboxai/Kokoro-Vietnamese)",
            device=device
        )
        self.local_dir = Path("checkpoints/kokoro")
        self.local_dir.mkdir(parents=True, exist_ok=True)
        (self.local_dir / "voicepacks").mkdir(parents=True, exist_ok=True)
        self.repo_id = "contextboxai/Kokoro-Vietnamese"
        
    def load(self):
        # We don't load the model on init, we will load it in synthesize dynamically
        pass
        
    def _download_if_needed(self, voice: str):
        """Helper to download config, model weights and specific voicepack to checkpoints/ folder"""
        config_path = self.local_dir / "config.json"
        if not config_path.exists():
            print(f"Downloading config.json to {config_path}...")
            hf_hub_download(
                repo_id=self.repo_id, 
                filename="config.json", 
                local_dir=str(self.local_dir)
            )
            
        model_path = self.local_dir / "kokoro_vi.pth"
        if not model_path.exists():
            print(f"Downloading kokoro_vi.pth (344MB) to {model_path}...")
            hf_hub_download(
                repo_id=self.repo_id, 
                filename="kokoro_vi.pth", 
                local_dir=str(self.local_dir)
            )
            
        voice_path = self.local_dir / "voicepacks" / f"{voice}.pt"
        if not voice_path.exists():
            print(f"Downloading voicepack {voice}.pt to {voice_path}...")
            hf_hub_download(
                repo_id=self.repo_id, 
                filename=f"voicepacks/{voice}.pt", 
                local_dir=str(self.local_dir)
            )
            
        return str(model_path), str(config_path), str(voice_path)

    def get_voices(self) -> list:
        voices_list = []
        for vid, meta in self.VOICES_METADATA.items():
            voices_list.append({
                'id': vid,
                'name': meta['name'],
                'gender': meta['gender'],
                'region': meta['region']
            })
        return voices_list

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> tuple:
        if voice not in self.VOICES_METADATA:
            raise ValueError(f"Voice {voice} is not supported by Kokoro-Vietnamese.")
            
        # 1. Download model/voice files locally if not exist
        model_path, config_path, voicepack_path = self._download_if_needed(voice)
        
        # 2. Record VRAM before loading
        start_vram = 0
        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            start_vram = torch.cuda.memory_allocated(0)
            
        # 3. Load model dynamically
        from kokoro_vietnamese import KokoroVietnamese
        
        start_time = time.time()
        
        # Initialize
        self.model = KokoroVietnamese(
            model_path=model_path,
            config_path=config_path,
            voicepack_path=voicepack_path,
            device=self.device
        )
        
        # 4. Record VRAM after loading
        mid_vram = 0
        if torch.cuda.is_available() and self.device == "cuda":
            mid_vram = torch.cuda.memory_allocated(0)
            
        # 5. Synthesize speech
        audio_data, phonemes = self.model.synthesize(text, speed=speed)
        
        end_time = time.time()
        duration = len(audio_data) / 24000.0  # Kokoro sample rate is 24kHz
        inference_time = end_time - start_time
        rtf = inference_time / duration if duration > 0 else 0
        
        # 6. Record peak VRAM
        peak_vram = 0
        if torch.cuda.is_available() and self.device == "cuda":
            peak_vram = torch.cuda.max_memory_allocated(0)
            
        # 7. Unload model to free VRAM
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
            'sample_rate': 24000,
            'phonemes': phonemes
        }
        
        return audio_data, 24000, stats
