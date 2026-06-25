import os
import sys
import argparse
import json
import time
import gc
import torch
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

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.resolve()))

# Import our wrappers
from models.kokoro_wrapper import KokoroVietnameseWrapper
from models.vits_wrapper import MMSVitsWrapper
from models.voxcpm_wrapper import VoxCPMWrapper
from models.vieneu_wrapper import VieNeuWrapper
from models.vixtts_wrapper import ViXTTSWrapper

def main():
    parser = argparse.ArgumentParser(description="Vietnamese TTS Command Line Interface")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["kokoro", "vits", "voxcpm", "vieneu", "vixtts"], 
                        help="Model type to use")
    parser.add_argument("--voice", type=str, required=True, help="Voice ID")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--speed", type=float, default=1.0, help="Speaking speed")
    
    args = parser.parse_args()
    
    # Initialize appropriate wrapper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.model == "kokoro":
        wrapper = KokoroVietnameseWrapper(device=device)
    elif args.model == "vits":
        wrapper = MMSVitsWrapper(device=device)
    elif args.model == "voxcpm":
        wrapper = VoxCPMWrapper(device=device)
    elif args.model == "vieneu":
        wrapper = VieNeuWrapper(device="cpu")  # VieNeu ONNX only on CPU
    elif args.model == "vixtts":
        wrapper = ViXTTSWrapper(device=device)
    else:
        print(json.dumps({"success": False, "error": f"Unknown model: {args.model}"}))
        return

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    # Record starting RAM usage
    try:
        import psutil
        process = psutil.Process(os.getpid())
        start_ram = process.memory_info().rss / (1024 * 1024)
    except Exception:
        start_ram = 0

    gc.collect()

    try:
        # Run synthesis
        audio_data, sample_rate, stats = wrapper.synthesize(args.text, args.voice, args.speed)
        
        # Save output file (int16 PCM for universal compatibility)
        output_filename = f"{args.model}_{args.voice}_{int(time.time())}.wav"
        output_path = outputs_dir / output_filename
        
        audio_int16 = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
        sf.write(str(output_path), audio_int16, sample_rate, subtype='PCM_16')
        
        # Calculate RAM delta
        try:
            end_ram = process.memory_info().rss / (1024 * 1024)
            ram_delta = max(0.0, end_ram - start_ram)
        except Exception:
            ram_delta = 0.0
            
        stats['ram_usage'] = ram_delta
        
        response = {
            "success": True,
            "audio_path": f"outputs/{output_filename}",
            "stats": stats
        }
        
        print(json.dumps(response, ensure_ascii=False))
        
    except Exception as e:
        import traceback
        response = {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print(json.dumps(response, ensure_ascii=False))

if __name__ == "__main__":
    main()
