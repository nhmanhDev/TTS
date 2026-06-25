import time
import numpy as np
import torch
import torchaudio
import soundfile as sf
import gc
from models.base_model import BaseTTSModel

# Monkey-patch torchaudio.load to use soundfile (avoids torchcodec/ffmpeg dependency)
_original_torchaudio_load = torchaudio.load
def _soundfile_load(filepath, *args, **kwargs):
    data, sr = sf.read(str(filepath), dtype='float32')
    if data.ndim == 1:
        data = data[None, :]
    else:
        data = data.T
    return torch.tensor(data), sr
torchaudio.load = _soundfile_load


class MOSSTTSLocalTransformerWrapper(BaseTTSModel):
    VOICES_METADATA = {
        'default': {
            'name': 'Giọng Mặc Định',
            'gender': 'Unknown',
            'region': 'Đa Ngôn Ngữ',
            'type': 'default',
        },
        'clone_diem_trinh': {
            'name': 'Diễm Trinh (Nhân Bản)',
            'gender': 'Female (Nữ)',
            'region': 'Southern (Miền Nam)',
            'type': 'clone',
            'ref_audio': 'assets/samples/kokoro/diem_trinh.wav',
        },
    }

    def __init__(self, device: str = "cuda"):
        super().__init__(
            model_name="MOSS-TTS Local-Transformer v1.5 (OpenMOSS-Team)",
            device=device
        )
        self.pretrained = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"

    def load(self):
        pass

    def get_voices(self) -> list:
        return [
            {'id': vid, 'name': meta['name'], 'gender': meta['gender'], 'region': meta['region']}
            for vid, meta in self.VOICES_METADATA.items()
        ]

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> tuple:
        if voice not in self.VOICES_METADATA:
            raise ValueError(f"Voice {voice} is not supported.")

        meta = self.VOICES_METADATA[voice]

        start_vram = 0
        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            start_vram = torch.cuda.memory_allocated(0)

        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

        from transformers import AutoModel, AutoProcessor

        start_time = time.time()

        processor = AutoProcessor.from_pretrained(self.pretrained, trust_remote_code=True)
        processor.audio_tokenizer = processor.audio_tokenizer.to(self.device)
        self.model = AutoModel.from_pretrained(
            self.pretrained, trust_remote_code=True,
            attn_implementation="sdpa", torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.model.eval()

        mid_vram = 0
        if torch.cuda.is_available() and self.device == "cuda":
            mid_vram = torch.cuda.memory_allocated(0)

        msg_kwargs = {'text': text, 'language': 'Vietnamese'}
        if meta['type'] == 'clone':
            msg_kwargs['reference'] = [meta['ref_audio']]

        conversations = [[processor.build_user_message(**msg_kwargs)]]

        with torch.no_grad():
            batch = processor(conversations, mode="generation")
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            outputs = self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=4096, audio_temperature=1.7, audio_top_p=0.8, audio_top_k=25,
            )

        end_time = time.time()

        audio_tensor = None
        for message in processor.decode(outputs):
            audio_tensor = message.audio_codes_list[0].cpu().numpy()

        if audio_tensor is None:
            raise RuntimeError("No audio output from MOSS-TTS")

        # Stereo output: (2, samples) -> mono by averaging channels
        if audio_tensor.ndim == 2 and audio_tensor.shape[0] == 2:
            audio_data = audio_tensor.mean(axis=0)
        elif audio_tensor.ndim == 2:
            audio_data = audio_tensor[0]
        else:
            audio_data = audio_tensor

        sample_rate = processor.model_config.sampling_rate
        audio_duration = len(audio_data) / float(sample_rate)
        inference_time = end_time - start_time
        rtf = inference_time / audio_duration if audio_duration > 0 else 0

        peak_vram = 0
        if torch.cuda.is_available() and self.device == "cuda":
            peak_vram = torch.cuda.max_memory_allocated(0)

        del processor
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


class MOSSTTSWrapper(BaseTTSModel):
    VOICES_METADATA = {
        'default': {
            'name': 'Giọng Mặc Định',
            'gender': 'Unknown',
            'region': 'Đa Ngôn Ngữ',
            'type': 'default',
        },
        'clone_diem_trinh': {
            'name': 'Diễm Trinh (Nhân Bản)',
            'gender': 'Female (Nữ)',
            'region': 'Southern (Miền Nam)',
            'type': 'clone',
            'ref_audio': 'assets/samples/kokoro/diem_trinh.wav',
        },
    }

    def __init__(self, device: str = "cuda"):
        super().__init__(
            model_name="MOSS-TTS v1.5 (OpenMOSS-Team)",
            device=device
        )
        self.pretrained = "OpenMOSS-Team/MOSS-TTS-v1.5"

    def load(self):
        pass

    def get_voices(self) -> list:
        return [
            {'id': vid, 'name': meta['name'], 'gender': meta['gender'], 'region': meta['region']}
            for vid, meta in self.VOICES_METADATA.items()
        ]

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> tuple:
        if voice not in self.VOICES_METADATA:
            raise ValueError(f"Voice {voice} is not supported.")

        meta = self.VOICES_METADATA[voice]

        start_vram = 0
        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            start_vram = torch.cuda.memory_allocated(0)

        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

        from transformers import AutoModel, AutoProcessor

        start_time = time.time()

        processor = AutoProcessor.from_pretrained(self.pretrained, trust_remote_code=True)
        processor.audio_tokenizer = processor.audio_tokenizer.to(self.device)
        self.model = AutoModel.from_pretrained(
            self.pretrained, trust_remote_code=True,
            attn_implementation="sdpa", torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.model.eval()

        mid_vram = 0
        if torch.cuda.is_available() and self.device == "cuda":
            mid_vram = torch.cuda.memory_allocated(0)

        msg_kwargs = {'text': text, 'language': 'Vietnamese'}
        if meta['type'] == 'clone':
            msg_kwargs['reference'] = [meta['ref_audio']]

        conversations = [[processor.build_user_message(**msg_kwargs)]]

        with torch.no_grad():
            batch = processor(conversations, mode="generation")
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            outputs = self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=4096, audio_temperature=1.7, audio_top_p=0.8, audio_top_k=25,
            )

        end_time = time.time()

        audio_tensor = None
        for message in processor.decode(outputs):
            audio_tensor = message.audio_codes_list[0].cpu().numpy()

        if audio_tensor is None:
            raise RuntimeError("No audio output from MOSS-TTS")

        if audio_tensor.ndim == 2:
            audio_data = audio_tensor.mean(axis=0) if audio_tensor.shape[0] == 2 else audio_tensor[0]
        else:
            audio_data = audio_tensor

        sample_rate = processor.model_config.sampling_rate
        audio_duration = len(audio_data) / float(sample_rate)
        inference_time = end_time - start_time
        rtf = inference_time / audio_duration if audio_duration > 0 else 0

        peak_vram = 0
        if torch.cuda.is_available() and self.device == "cuda":
            peak_vram = torch.cuda.max_memory_allocated(0)

        del processor
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
