import os
import sys
import time
import subprocess
import numpy as np
import torch
import soundfile as sf
import gc
from pathlib import Path
from models.base_model import BaseTTSModel

HIGGS_VENV = Path(".venv-higgs")
HIGGS_REPO = Path("checkpoints/higgs-tts-v3-transformers")
CUDART13 = "/home/manh/VideoMarketingAI/tts/.venv-kokoro/lib/python3.12/site-packages/nvidia/cu13/lib"

_HIGGS_GEN_SCRIPT = """
import sys, json, torch, soundfile as sf, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

args = json.loads(sys.argv[1])
repo = args['repo']
text = args['text']
out_path = args['out_path']
ref_audio = args.get('ref_audio')
ref_sr = args.get('ref_sr')
ref_text = args.get('ref_text')
temperature = args.get('temperature', 0.8)
top_p = args.get('top_p', 0.95)

tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    repo, trust_remote_code=True, dtype=torch.bfloat16
).to('cuda').eval()

sr = model.config.sample_rate
gen_kwargs = {'temperature': temperature, 'top_p': top_p}

if ref_audio:
    ref_data, file_sr = sf.read(ref_audio, dtype='float32')
    if ref_data.ndim > 1:
        ref_data = ref_data.mean(axis=1)
    ref_tensor = torch.tensor(ref_data).unsqueeze(0)
    gen_kwargs['reference_audio'] = ref_tensor
    gen_kwargs['reference_sample_rate'] = ref_sr or file_sr
    if ref_text:
        gen_kwargs['reference_text'] = ref_text

wav = model.generate_speech(text, tokenizer, **gen_kwargs)
sf.write(out_path, wav.cpu().numpy(), sr)
result = {'sample_rate': sr, 'num_samples': len(wav)}
print(json.dumps(result))
"""


class HiggsAudioV3Wrapper(BaseTTSModel):
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
            'ref_text': 'Xin chào, đây là giọng đọc mẫu AI hỗ trợ tiếng Việt.',
        },
    }

    def __init__(self, device: str = "cuda"):
        super().__init__(
            model_name="Higgs Audio TTS v3 4B (bosonai/higgs-audio-v3-tts-4b)",
            device=device
        )
        self.higgs_python = str(HIGGS_VENV / "bin" / "python3")

    def load(self):
        pass

    def get_voices(self) -> list:
        return [
            {'id': vid, 'name': meta['name'], 'gender': meta['gender'], 'region': meta['region']}
            for vid, meta in self.VOICES_METADATA.items()
        ]

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> tuple:
        if voice not in self.VOICES_METADATA:
            raise ValueError(f"Voice {voice} is not supported by Higgs Audio v3.")

        meta = self.VOICES_METADATA[voice]

        import tempfile, json

        start_vram = 0
        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            start_vram = torch.cuda.memory_allocated(0)

        start_time = time.time()

        with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as script_f:
            script_f.write(_HIGGS_GEN_SCRIPT)
            script_path = script_f.name

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as out_f:
            out_path = out_f.name

        try:
            args_dict = {
                'repo': str(HIGGS_REPO),
                'text': text,
                'out_path': out_path,
                'temperature': 0.8,
                'top_p': 0.95,
            }
            if meta['type'] == 'clone':
                args_dict['ref_audio'] = meta['ref_audio']
                args_dict['ref_text'] = meta.get('ref_text', '')

            env = os.environ.copy()
            if CUDART13:
                env['LD_LIBRARY_PATH'] = CUDART13 + ':' + env.get('LD_LIBRARY_PATH', '')

            result = subprocess.run(
                [self.higgs_python, script_path, json.dumps(args_dict)],
                capture_output=True, text=True, check=True, env=env
            )
            info = json.loads(result.stdout.strip().splitlines()[-1])
            sample_rate = info['sample_rate']

            audio_data, _ = sf.read(out_path, dtype='float32')
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
        finally:
            os.unlink(script_path)
            try:
                os.unlink(out_path)
            except Exception:
                pass

        end_time = time.time()

        mid_vram = 0
        peak_vram = 0
        if torch.cuda.is_available() and self.device == "cuda":
            mid_vram = torch.cuda.memory_allocated(0)
            peak_vram = torch.cuda.max_memory_allocated(0)

        audio_duration = len(audio_data) / float(sample_rate)
        inference_time = end_time - start_time
        rtf = inference_time / audio_duration if audio_duration > 0 else 0
        vram_allocated_mb = max(0.0, (mid_vram - start_vram) / (1024 * 1024))
        vram_peak_mb = max(0.0, (peak_vram - start_vram) / (1024 * 1024))

        stats = {
            'inference_time': inference_time,
            'audio_duration': audio_duration,
            'rtf': rtf,
            'vram_usage': vram_allocated_mb,
            'vram_peak': vram_peak_mb,
            'sample_rate': sample_rate,
            'phonemes': 'N/A',
        }

        return audio_data, sample_rate, stats
