"""Generate voice sample previews for VieNeu-TTS model (10 preset voices)."""
import sys
import os
import time
import numpy as np
import soundfile as sf
from pathlib import Path

# Setup Windows console encoding
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

SAMPLE_TEXT = "Xin chào, tôi là giọng đọc mẫu tiếng Việt. Hôm nay trời đẹp quá!"
SAMPLES_DIR = Path("assets/samples")
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

def generate_vieneu_samples():
    """Generate samples for all 10 VieNeu preset voices."""
    print("\n" + "="*60)
    print("🎤 Generating VieNeu-TTS v3 Turbo samples...")
    print("="*60)
    
    from vieneu import Vieneu
    tts = Vieneu(mode='v3turbo', backend='onnx', device='cpu')
    
    voices = {
        'ngoc_lan': 'Ngọc Lan',
        'gia_bao': 'Gia Bảo',
        'thai_son': 'Thái Sơn',
        'duc_tri': 'Đức Trí',
        'my_duyen': 'Mỹ Duyên',
        'truc_ly': 'Trúc Ly',
        'xuan_vinh': 'Xuân Vĩnh',
        'trong_huu': 'Trọng Hữu',
        'binh_an': 'Bình An',
        'ngoc_linh': 'Ngọc Linh',
    }
    
    sr = getattr(tts, 'sample_rate', 24000)
    print(f"  Sample rate: {sr} Hz")
    
    for voice_id, preset_name in voices.items():
        out_path = SAMPLES_DIR / f"vieneu_{voice_id}.wav"
        if out_path.exists():
            print(f"  ⏩ Skipping {voice_id} (already exists)")
            continue
            
        try:
            print(f"  🔊 Generating: {preset_name} ({voice_id})...", end=" ", flush=True)
            start = time.time()
            
            # Correct API: get_preset_voice returns a dict, pass it to infer(voice=...)
            voice_dict = tts.get_preset_voice(preset_name)
            audio = tts.infer(SAMPLE_TEXT, voice=voice_dict, apply_watermark=False)
            
            # Normalize
            max_val = np.max(np.abs(audio))
            if max_val > 0 and max_val > 1.0:
                audio = audio / max_val
            
            audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            sf.write(str(out_path), audio_int16, sr, subtype='PCM_16')
            elapsed = time.time() - start
            print(f"✅ ({elapsed:.1f}s)")
        except Exception as e:
            import traceback
            print(f"❌ Error: {e}")
            traceback.print_exc()
    
    del tts
    print("\n✅ VieNeu samples done!")

if __name__ == "__main__":
    generate_vieneu_samples()
    print("\n✅ All sample generation complete!")
