import os
import time
from pathlib import Path
import numpy as np
import gc
from models.base_model import BaseTTSModel

class VieNeuWrapper(BaseTTSModel):
    """Wrapper for VieNeu-TTS v3 Turbo model using ONNX backend (CPU).
    
    VieNeu-TTS supports bilingual (Vietnamese-English) code-switching,
    emotion control, and 48kHz studio-quality output.
    Uses ONNX backend for fast CPU inference on Windows.
    
    API usage:
      tts = Vieneu(mode='v3turbo', backend='onnx', device='cpu')
      voice_dict = tts.get_preset_voice('Ngọc Lan')
      audio = tts.infer(text, voice=voice_dict)
    """
    
    # 10 preset voices from vieneu v3turbo
    VOICES_METADATA = {
        'ngoc_lan': {'name': 'Ngọc Lan', 'gender': 'Nữ', 'region': 'Giọng dịu dàng', 'preset': 'Ngọc Lan'},
        'gia_bao': {'name': 'Gia Bảo', 'gender': 'Nam', 'region': 'Giọng mượt mà', 'preset': 'Gia Bảo'},
        'thai_son': {'name': 'Thái Sơn', 'gender': 'Nam', 'region': 'Giọng chắc khỏe', 'preset': 'Thái Sơn'},
        'duc_tri': {'name': 'Đức Trí', 'gender': 'Nam', 'region': 'Giọng rõ ràng', 'preset': 'Đức Trí'},
        'my_duyen': {'name': 'Mỹ Duyên', 'gender': 'Nữ', 'region': 'Giọng mượt mà', 'preset': 'Mỹ Duyên'},
        'truc_ly': {'name': 'Trúc Ly', 'gender': 'Nữ', 'region': 'Giọng trẻ trung', 'preset': 'Trúc Ly'},
        'xuan_vinh': {'name': 'Xuân Vĩnh', 'gender': 'Nam', 'region': 'Giọng vui tươi', 'preset': 'Xuân Vĩnh'},
        'trong_huu': {'name': 'Trọng Hữu', 'gender': 'Nam', 'region': 'Giọng uyên bác', 'preset': 'Trọng Hữu'},
        'binh_an': {'name': 'Bình An', 'gender': 'Nam', 'region': 'Giọng điềm đạm', 'preset': 'Bình An'},
        'ngoc_linh': {'name': 'Ngọc Linh', 'gender': 'Nữ', 'region': 'Giọng tươi sáng', 'preset': 'Ngọc Linh'},
    }

    def __init__(self, device: str = "cpu"):
        # VieNeu ONNX always runs on CPU
        super().__init__(
            model_name="VieNeu-TTS v3 Turbo (pnnbao97/vieneu)",
            device="cpu"
        )
        
    def load(self):
        # Model is loaded dynamically in synthesize to save memory
        pass

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
            raise ValueError(f"Voice {voice} is not supported by VieNeu-TTS.")
            
        from vieneu import Vieneu
        
        start_time = time.time()
        
        # Initialize VieNeu with ONNX backend (CPU-only, works on Windows)
        tts = Vieneu(mode='v3turbo', backend='onnx', device='cpu')
        
        # Get preset voice dict using the correct API
        preset_name = self.VOICES_METADATA[voice]['preset']
        voice_dict = tts.get_preset_voice(preset_name)
        
        # Synthesize speech with voice parameter
        audio_data = tts.infer(text, voice=voice_dict, apply_watermark=False)
        
        end_time = time.time()
        
        # VieNeu v3turbo sample rate from the engine
        sample_rate = getattr(tts, 'sample_rate', 24000)
        
        # Normalize audio to [-1, 1] range if needed
        if isinstance(audio_data, np.ndarray):
            max_val = np.max(np.abs(audio_data))
            if max_val > 0 and max_val > 1.0:
                audio_data = audio_data / max_val
        
        duration = len(audio_data) / float(sample_rate)
        inference_time = end_time - start_time
        rtf = inference_time / duration if duration > 0 else 0
        
        # VieNeu on ONNX CPU doesn't use VRAM
        stats = {
            'inference_time': inference_time,
            'audio_duration': duration,
            'rtf': rtf,
            'vram_usage': 0.0,
            'vram_peak': 0.0,
            'sample_rate': sample_rate,
            'phonemes': 'N/A'
        }
        
        # Clean up
        del tts
        gc.collect()
        
        return audio_data, sample_rate, stats
