import os
import time
from pathlib import Path
import torch
import gc
import numpy as np
from models.base_model import BaseTTSModel

class ViXTTSWrapper(BaseTTSModel):
    """Wrapper for viXTTS model (fine-tuned XTTS-v2 for Vietnamese).
    
    viXTTS uses voice cloning from a reference WAV file (6s sample).
    Uses Coqui TTS library (idiap/coqui-ai-TTS fork).
    """
    
    # We provide a default reference voice. Users can also upload their own.
    VOICES_METADATA = {
        'vixtts_default': {
            'name': 'viXTTS Default',
            'gender': 'Nữ',
            'region': 'Giọng Việt tự nhiên',
        },
    }

    def __init__(self, device: str = "cuda"):
        super().__init__(
            model_name="viXTTS (capleaf/viXTTS)",
            device=device
        )
        self.ref_audio_dir = Path("assets/ref_audio")
        self.ref_audio_dir.mkdir(parents=True, exist_ok=True)
        
    def load(self):
        # Model is loaded dynamically in synthesize
        pass

    def _get_ref_audio(self, voice: str) -> str:
        """Get reference audio path for voice cloning."""
        ref_path = self.ref_audio_dir / f"{voice}.wav"
        if ref_path.exists():
            return str(ref_path)
        
        # Generate a simple reference audio using Meta MMS VITS if no ref exists
        default_ref = self.ref_audio_dir / "vixtts_default.wav"
        if not default_ref.exists():
            try:
                self._generate_default_ref(str(default_ref))
            except Exception as e:
                raise FileNotFoundError(
                    f"Reference audio not found at {ref_path}. "
                    f"viXTTS requires a reference WAV file for voice cloning. "
                    f"Please place a 6-second WAV file at: {ref_path}"
                ) from e
        return str(default_ref)

    def _generate_default_ref(self, output_path: str):
        """Generate a default reference audio using Meta MMS VITS."""
        from transformers import VitsModel, AutoTokenizer
        import soundfile as sf
        
        tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
        model = VitsModel.from_pretrained("facebook/mms-tts-vie").to("cpu").eval()
        
        ref_text = "Xin chào, tôi là giọng đọc mẫu cho hệ thống chuyển văn bản thành giọng nói tiếng Việt."
        inputs = tokenizer(ref_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            waveform = outputs.waveform.cpu().numpy().flatten()
        
        sample_rate = model.config.sampling_rate
        sf.write(output_path, waveform, sample_rate, subtype='PCM_16')
        
        del model, tokenizer
        gc.collect()

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
            raise ValueError(f"Voice {voice} is not supported by viXTTS.")
        
        # Record VRAM before loading
        start_vram = 0
        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            start_vram = torch.cuda.memory_allocated(0)
        # Auto-agree to Coqui CPML license terms
        os.environ["COQUI_TOS_AGREED"] = "1"
        from huggingface_hub import snapshot_download
        from TTS.api import TTS as CoquiTTS
        
        # Download capleaf/viXTTS checkpoint locally to avoid symlink/permission errors on Windows
        checkpoint_dir = snapshot_download(
            repo_id="capleaf/viXTTS",
            local_dir="checkpoints/vixtts",
            local_dir_use_symlinks=False
        )
        
        start_time = time.time()
        
        # Load viXTTS model via local checkpoint directory
        self.model = CoquiTTS(
            model_path=checkpoint_dir,
            config_path=os.path.join(checkpoint_dir, "config.json"),
            gpu=(self.device == "cuda")
        )
        
        # Record VRAM after loading
        mid_vram = 0
        if torch.cuda.is_available() and self.device == "cuda":
            mid_vram = torch.cuda.memory_allocated(0)
        
        # Get reference audio for voice cloning
        ref_audio_path = self._get_ref_audio(voice)
        
        # Synthesize using voice cloning
        import tempfile
        import soundfile as sf
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            self.model.tts_to_file(
                text=text,
                file_path=tmp_path,
                speaker_wav=ref_audio_path,
                language="vi",
                speed=speed
            )
            
            # Read the generated audio
            audio_data, sample_rate = sf.read(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        end_time = time.time()
        
        duration = len(audio_data) / float(sample_rate)
        inference_time = end_time - start_time
        rtf = inference_time / duration if duration > 0 else 0
        
        # Record peak VRAM
        peak_vram = 0
        if torch.cuda.is_available() and self.device == "cuda":
            peak_vram = torch.cuda.max_memory_allocated(0)
            
        # Unload model to free VRAM
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
