import time
import numpy as np
import torch
import gc
from models.base_model import BaseTTSModel

# Patch transformers to expose HiggsAudioV2TokenizerModel needed by omnivoice
try:
    import transformers as _tf
    from transformers.models.higgs_audio_v2_tokenizer.modeling_higgs_audio_v2_tokenizer import (
        HiggsAudioV2TokenizerModel as _HiggsAudioV2TokenizerModel,
    )
    _tf.HiggsAudioV2TokenizerModel = _HiggsAudioV2TokenizerModel
except Exception:
    pass


class OmniVoiceWrapper(BaseTTSModel):
    VOICES_METADATA = {
        'design_nu_mien_nam': {
            'name': 'Lan Anh (Nữ, Miền Nam)',
            'gender': 'Female (Nữ)',
            'region': 'Southern (Miền Nam)',
            'type': 'design',
            'prompt': 'A sweet, warm, and naturally clear Southern Vietnamese female voice with gentle tone and smooth pronunciation'
        },
        'design_nam_mien_bac': {
            'name': 'Minh Tuấn (Nam, Miền Bắc)',
            'gender': 'Male (Nam)',
            'region': 'Northern (Miền Bắc)',
            'type': 'design',
            'prompt': 'A deep, confident, and articulate Northern Vietnamese male voice with authoritative and professional tone'
        },
        'design_nu_mien_bac': {
            'name': 'Hồng Nhung (Nữ, Miền Bắc)',
            'gender': 'Female (Nữ)',
            'region': 'Northern (Miền Bắc)',
            'type': 'design',
            'prompt': 'A gentle, soft, and expressive Northern Vietnamese female voice with clear pronunciation and elegant tone'
        },
        'design_nam_mien_nam': {
            'name': 'Trọng Khải (Nam, Miền Nam)',
            'gender': 'Male (Nam)',
            'region': 'Southern (Miền Nam)',
            'type': 'design',
            'prompt': 'A friendly, warm, and naturally flowing Southern Vietnamese male voice with relaxed and approachable tone'
        },
        'clone_diem_trinh': {
            'name': 'Diễm Trinh (Nhân Bản)',
            'gender': 'Female (Nữ)',
            'region': 'Southern (Miền Nam)',
            'type': 'clone',
            'ref_audio': 'assets/samples/kokoro/diem_trinh.wav',
            'ref_text': 'Xin chào, đây là giọng đọc mẫu AI hỗ trợ tiếng Việt.'
        },
    }

    def __init__(self, device: str = "cuda"):
        super().__init__(
            model_name="OmniVoice (k2-fsa/OmniVoice)",
            device=device
        )

    def load(self):
        pass

    def get_voices(self) -> list:
        return [
            {'id': vid, 'name': meta['name'], 'gender': meta['gender'], 'region': meta['region']}
            for vid, meta in self.VOICES_METADATA.items()
        ]

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> tuple:
        if voice not in self.VOICES_METADATA:
            raise ValueError(f"Voice {voice} is not supported by OmniVoice.")

        meta = self.VOICES_METADATA[voice]

        start_vram = 0
        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            start_vram = torch.cuda.memory_allocated(0)

        from omnivoice import OmniVoice

        start_time = time.time()
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = OmniVoice.from_pretrained(
            'k2-fsa/OmniVoice', device_map=self.device, dtype=dtype
        )

        mid_vram = 0
        if torch.cuda.is_available() and self.device == "cuda":
            mid_vram = torch.cuda.memory_allocated(0)

        gen_kwargs = {'text': text, 'lang': 'vi'}
        if meta['type'] == 'design':
            gen_kwargs['voice_design'] = meta['prompt']
        else:
            gen_kwargs['ref_audio'] = meta['ref_audio']
            gen_kwargs['ref_text'] = meta['ref_text']

        result = self.model.generate(**gen_kwargs)

        end_time = time.time()

        audio_data = result[0].cpu().numpy().flatten() if hasattr(result[0], 'cpu') else np.array(result[0]).flatten()
        sample_rate = 24000
        audio_duration = len(audio_data) / float(sample_rate)
        inference_time = end_time - start_time
        rtf = inference_time / audio_duration if audio_duration > 0 else 0

        peak_vram = 0
        if torch.cuda.is_available() and self.device == "cuda":
            peak_vram = torch.cuda.max_memory_allocated(0)

        self.unload()

        vram_allocated_mb = (mid_vram - start_vram) / (1024 * 1024)
        vram_peak_mb = (peak_vram - start_vram) / (1024 * 1024)

        stats = {
            'inference_time': inference_time,
            'audio_duration': audio_duration,
            'rtf': rtf,
            'vram_usage': max(0.0, vram_allocated_mb),
            'vram_peak': max(0.0, vram_peak_mb),
            'sample_rate': sample_rate,
            'phonemes': 'N/A'
        }

        return audio_data, sample_rate, stats
