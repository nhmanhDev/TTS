import os
import sys
import numpy as np
import soundfile as sf
from pathlib import Path

# Fix Windows console encoding issues with Vietnamese characters
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add project root to path
sys.path.append("D:\\WorkSpace\\Text2Speech")

from models.kokoro_wrapper import KokoroVietnameseWrapper
from models.vits_wrapper import MMSVitsWrapper
from models.voxcpm_wrapper import VoxCPMWrapper

# Create output folder
samples_dir = Path("assets/samples")
samples_dir.mkdir(parents=True, exist_ok=True)

print("Initializing Wrappers...")
kokoro = KokoroVietnameseWrapper(device="cuda")
vits = MMSVitsWrapper(device="cuda")
voxcpm = VoxCPMWrapper(device="cuda")

# 1. Generate Kokoro voice samples
print("\n--- Generating Kokoro Samples ---")
for voice_info in kokoro.get_voices():
    vid = voice_info['id']
    name = voice_info['name']
    region = voice_info['region']
    gender = voice_info['gender']
    
    text = f"Xin chào, tôi là {name}, đây là giọng đọc mẫu của tôi."
    output_path = samples_dir / f"kokoro_{vid}.wav"
    
    if output_path.exists():
        print(f"Sample for Kokoro {vid} already exists. Skipping.")
        continue
        
    print(f"Generating sample for Kokoro: {vid} ({name})...")
    try:
        audio, sr, _ = kokoro.synthesize(text, vid)
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        sf.write(str(output_path), audio_int16, sr, subtype='PCM_16')
    except Exception as e:
        print(f"Failed for Kokoro {vid}: {e}")

# 2. Generate VITS sample
print("\n--- Generating VITS Sample ---")
v_output_path = samples_dir / "vits_mms_vietnamese.wav"
if not v_output_path.exists():
    try:
        text = "Xin chào, đây là giọng đọc mẫu từ mô hình VITS tiếng Việt của Meta."
        audio, sr, _ = vits.synthesize(text, "mms_vietnamese")
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        sf.write(str(v_output_path), audio_int16, sr, subtype='PCM_16')
        print("Generated VITS sample successfully.")
    except Exception as e:
        print(f"Failed for VITS: {e}")
else:
    print("VITS sample already exists.")

# 3. Generate VoxCPM2 samples
print("\n--- Generating VoxCPM2 Samples ---")
for voice_info in voxcpm.get_voices():
    vid = voice_info['id']
    name = voice_info['name']
    
    text = f"Xin chào, đây là giọng mẫu được thiết kế tự động từ mô hình VoxCPM."
    output_path = samples_dir / f"voxcpm_{vid}.wav"
    
    if output_path.exists():
        print(f"Sample for VoxCPM2 {vid} already exists. Skipping.")
        continue
        
    print(f"Generating sample for VoxCPM2: {vid}... (This might take a moment to load model)")
    try:
        audio, sr, _ = voxcpm.synthesize(text, vid)
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        sf.write(str(output_path), audio_int16, sr, subtype='PCM_16')
    except Exception as e:
        print(f"Failed for VoxCPM2 {vid}: {e}")

print("\nAll samples generated successfully!")
