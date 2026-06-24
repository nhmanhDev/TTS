import os
import gc
import time
import psutil
import torch
import soundfile as sf
import numpy as np
import gradio as gr
from pathlib import Path

# Import our wrappers
from models.kokoro_wrapper import KokoroVietnameseWrapper
from models.vits_wrapper import MMSVitsWrapper
from models.voxcpm_wrapper import VoxCPMWrapper

# Create necessary directories
outputs_dir = Path("outputs")
outputs_dir.mkdir(exist_ok=True)

# Device selection (defaults to GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device.upper()}")

# Instantiate wrappers (they lazy-load, so lightweight init)
kokoro_wrapper = KokoroVietnameseWrapper(device=device)
vits_wrapper = MMSVitsWrapper(device=device)
voxcpm_wrapper = VoxCPMWrapper(device=device)

# Available models dict
MODELS = {
    "Kokoro-Vietnamese (contextboxai/Kokoro-Vietnamese)": kokoro_wrapper,
    "Meta MMS VITS (facebook/mms-tts-vie)": vits_wrapper,
    "OpenBMB VoxCPM2 (openbmb/VoxCPM2)": voxcpm_wrapper
}

# Default sample text
DEFAULT_TEXT = "Giữa một buổi chiều yên tĩnh, cô ấy kể lại câu chuyện bằng một giọng nói ấm áp và chậm rãi. Tiếng Việt là một ngôn ngữ rất đẹp và truyền cảm."

def get_process_memory():
    """Get current RAM usage of this python process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_single_tts(model_name, voice_str, text, speed):
    """Run TTS for a single model and return the audio path and stats."""
    if not text.strip():
        return None, "Vui lòng nhập văn bản cần đọc."
        
    # Extract model wrapper
    wrapper = MODELS[model_name]
    
    # Extract voice id from voice_str e.g. "diem_trinh (Diễm Trinh - ...)" -> "diem_trinh"
    voice_id = voice_str.split(" ")[0]
    
    # Track RAM before loading
    gc.collect()
    start_ram = get_process_memory()
    
    try:
        # Run synthesis (wrapper will load, synthesize, then unload)
        audio_data, sample_rate, stats = wrapper.synthesize(text, voice_id, speed)
        
        # Save output audio (Convert float32 to 16-bit PCM for universal browser audio player support)
        output_filename = f"{voice_id}_{int(time.time())}.wav"
        output_path = outputs_dir / output_filename
        audio_int16 = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
        sf.write(str(output_path), audio_int16, sample_rate, subtype='PCM_16')
        
        # Calculate RAM change
        end_ram = get_process_memory()
        ram_delta = max(0.0, end_ram - start_ram)
        
        # Format HTML stats table
        stats_html = f"""
        <div style="background: rgba(30, 41, 59, 0.7); padding: 15px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.1); margin-top: 10px;">
            <h4 style="margin: 0 0 10px 0; color: #38bdf8; font-size: 1.1rem;">⚡ Thông số Benchmark:</h4>
            <table style="width: 100%; border-collapse: collapse; font-family: sans-serif; font-size: 0.95rem; text-align: left;">
                <tr>
                    <td style="padding: 6px 0; font-weight: bold; color: #94a3b8; width: 45%;">Tốc độ xử lý (RTF):</td>
                    <td style="padding: 6px 0; font-weight: bold; color: #4ade80;">{stats['rtf']:.4f} <span style="font-size: 0.8rem; font-weight: normal; color: #94a3b8;">({"(Cực nhanh, nhỏ hơn 1.0 là nhanh hơn thời gian thực)" if stats['rtf'] < 1 else ""})</span></td>
                </tr>
                <tr>
                    <td style="padding: 6px 0; font-weight: bold; color: #94a3b8;">Thời gian chạy model:</td>
                    <td style="padding: 6px 0; color: #e2e8f0;">{stats['inference_time']:.2f} giây</td>
                </tr>
                <tr>
                    <td style="padding: 6px 0; font-weight: bold; color: #94a3b8;">Độ dài file âm thanh:</td>
                    <td style="padding: 6px 0; color: #e2e8f0;">{stats['audio_duration']:.2f} giây</td>
                </tr>
                <tr>
                    <td style="padding: 6px 0; font-weight: bold; color: #94a3b8;">Độ lớn VRAM GPU nạp thêm:</td>
                    <td style="padding: 6px 0; color: #f43f5e; font-weight: bold;">{stats['vram_usage']:.2f} MB</td>
                </tr>
                <tr>
                    <td style="padding: 6px 0; font-weight: bold; color: #94a3b8;">VRAM Đỉnh GPU (Peak VRAM):</td>
                    <td style="padding: 6px 0; color: #f43f5e; font-weight: bold;">{stats['vram_peak']:.2f} MB</td>
                </tr>
                <tr>
                    <td style="padding: 6px 0; font-weight: bold; color: #94a3b8;">RAM Hệ thống nạp thêm:</td>
                    <td style="padding: 6px 0; color: #fbbf24; font-weight: bold;">{ram_delta:.2f} MB</td>
                </tr>
                <tr>
                    <td style="padding: 6px 0; font-weight: bold; color: #94a3b8;">Tần số lấy mẫu (Sample Rate):</td>
                    <td style="padding: 6px 0; color: #e2e8f0;">{stats['sample_rate']} Hz</td>
                </tr>
            </table>
        </div>
        """
        
        # Return audio path, stats, and phonemes
        phonemes_info = f"**Phonemes:** {stats['phonemes']}" if stats['phonemes'] != 'N/A' else "*(Mô hình này không xuất ra phoneme)*"
        
        return str(output_path), stats_html, phonemes_info
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in single synthesis: {error_details}")
        error_html = f"""
        <div style="background: rgba(239, 68, 68, 0.2); padding: 15px; border-radius: 10px; border: 1px solid rgba(239, 68, 68, 0.4); margin-top: 10px; color: #f87171;">
            <h4 style="margin: 0 0 5px 0;">❌ Lỗi trong quá trình tạo giọng nói:</h4>
            <p style="margin: 0; font-size: 0.9rem; font-family: monospace;">{str(e)}</p>
        </div>
        """
        return None, error_html, ""

def run_comparison(text, speed):
    """Run both models sequentially to benchmark and compare side-by-side."""
    if not text.strip():
        return None, "", "", None, "", ""
        
    # 1. Run Kokoro
    k_audio, k_stats, k_phonemes = run_single_tts(
        "Kokoro-Vietnamese (contextboxai/Kokoro-Vietnamese)",
        "diem_trinh (Diễm Trinh - Female (Nữ) - Southern (Miền Nam))",
        text,
        speed
    )
    
    # 2. Run Meta MMS VITS
    v_audio, v_stats, v_phonemes = run_single_tts(
        "Meta MMS VITS (facebook/mms-tts-vie)",
        "mms_vietnamese (Meta MMS Vietnamese Speaker - Male/Female (Đơn giọng) - Standard (Toàn dân))",
        text,
        speed
    )
    
    return k_audio, k_stats, k_phonemes, v_audio, v_stats, v_phonemes

def get_voice_choices(model_name):
    """Get list of formatted voice options for the model."""
    wrapper = MODELS[model_name]
    choices = []
    for v in wrapper.get_voices():
        choices.append(f"{v['id']} ({v['name']} - {v['gender']} - {v['region']})")
    return gr.Dropdown(choices=choices, value=choices[0])

def update_voice_sample(model_name, voice_str):
    """Get the path to the pre-generated sample file if it exists."""
    if not voice_str:
        return None
    voice_id = voice_str.split(" ")[0]
    if "Kokoro-Vietnamese" in model_name:
        path = f"assets/samples/kokoro_{voice_id}.wav"
    elif "Meta MMS VITS" in model_name:
        path = "assets/samples/vits_mms_vietnamese.wav"
    elif "VoxCPM2" in model_name:
        path = f"assets/samples/voxcpm_{voice_id}.wav"
    else:
        return None
        
    if os.path.exists(path):
        return os.path.abspath(path)
    return None

def on_model_change(model_name):
    """Update voices choices and load the default voice sample path."""
    choices = []
    wrapper = MODELS[model_name]
    for v in wrapper.get_voices():
        choices.append(f"{v['id']} ({v['name']} - {v['gender']} - {v['region']})")
    default_voice = choices[0]
    sample_path = update_voice_sample(model_name, default_voice)
    return gr.Dropdown(choices=choices, value=default_voice), sample_path

# Gradio Styling & UI Layout
custom_css = """
body {
    background-color: #0b1329 !important;
}
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}
#title_header {
    text-align: center;
    background: linear-gradient(135deg, #1e293b, #0f172a);
    padding: 30px;
    border-radius: 15px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    margin-bottom: 25px;
}
#title_header h1 {
    font-family: 'Outfit', 'Inter', sans-serif !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px !important;
    font-size: 2.5rem !important;
}
#title_header p {
    color: #94a3b8 !important;
    font-size: 1.1rem !important;
}
.compare-card {
    background: #0f172a !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
}
.btn-primary {
    background: linear-gradient(90deg, #3b82f6, #6366f1) !important;
    border: none !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4) !important;
}
.btn-primary:hover {
    background: linear-gradient(90deg, #2563eb, #4f46e5) !important;
}
"""

with gr.Blocks() as demo:
    # Title Header
    with gr.Column(elem_id="title_header"):
        gr.Markdown("# 🇻🇳 Vietnamese TTS Benchmark & Testing Hub")
        gr.Markdown("Hệ thống đánh giá và so sánh chất lượng, tốc độ và lượng RAM/VRAM tiêu hao của các mô hình Text-To-Speech tiếng Việt tốt nhất hiện nay.")
        
    with gr.Tabs():
        # TAB 1: Single Model Test
        with gr.TabItem("🎯 Chạy Từng Mô Hình"):
            with gr.Row():
                with gr.Column(scale=3):
                    # Input controls
                    model_choice = gr.Dropdown(
                        label="1. Chọn mô hình TTS",
                        choices=list(MODELS.keys()),
                        value=list(MODELS.keys())[0]
                    )
                    
                    # Populate default voices to prevent Gradio validation error on cached reload
                    default_model_name = list(MODELS.keys())[0]
                    default_voices = [f"{v['id']} ({v['name']} - {v['gender']} - {v['region']})" for v in MODELS[default_model_name].get_voices()]
                    
                    voice_choice = gr.Dropdown(
                        label="2. Chọn Giọng mẫu (Pre-trained Voice)",
                        choices=default_voices,
                        value=default_voices[0]
                    )
                    
                    quick_voice_sample = gr.Audio(
                        label="🎵 Nghe nhanh giọng mẫu có sẵn (Pre-recorded Demo - Phát tức thì không cần chạy model)",
                        type="filepath",
                        interactive=False,
                        visible=True
                    )
                    
                    text_input = gr.Textbox(
                        label="3. Văn bản tiếng Việt cần đọc",
                        lines=5,
                        placeholder="Nhập đoạn văn bản ở đây...",
                        value=DEFAULT_TEXT
                    )
                    
                    speed_slider = gr.Slider(
                        label="Tốc độ đọc (Speed)",
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1
                    )
                    
                    btn_run = gr.Button("🚀 Tạo Giọng Nói & Đánh Giá", variant="primary", elem_classes="btn-primary")
                    
                with gr.Column(scale=2):
                    # Output display
                    gr.Markdown("### 🔊 Kết quả đầu ra:")
                    audio_output = gr.Audio(label="Audio", type="filepath")
                    phoneme_output = gr.Markdown(label="Phonemes Preview")
                    stats_output = gr.HTML(label="Stats Table", value="<p style='color: #64748b;'>Chưa có kết quả. Nhấp nút chạy để xem benchmark.</p>")

            # When model choice changes, update the voice dropdown and the sample player
            model_choice.change(
                fn=on_model_change,
                inputs=[model_choice],
                outputs=[voice_choice, quick_voice_sample]
            )
            
            # When voice choice changes, update the sample player
            voice_choice.change(
                fn=update_voice_sample,
                inputs=[model_choice, voice_choice],
                outputs=[quick_voice_sample]
            )
            
            def on_initial_load(model_name):
                choices = []
                wrapper = MODELS[model_name]
                for v in wrapper.get_voices():
                    choices.append(f"{v['id']} ({v['name']} - {v['gender']} - {v['region']})")
                default_voice = choices[0]
                sample_path = update_voice_sample(model_name, default_voice)
                return gr.Dropdown(choices=choices, value=default_voice), sample_path

            # Initial load of voices and voice sample
            demo.load(
                fn=on_initial_load,
                inputs=[model_choice],
                outputs=[voice_choice, quick_voice_sample]
            )
            
            btn_run.click(
                fn=run_single_tts,
                inputs=[model_choice, voice_choice, text_input, speed_slider],
                outputs=[audio_output, stats_output, phoneme_output]
            )

        # TAB 2: Side-by-Side Comparison
        with gr.TabItem("⚖️ So Sánh Song Song (Side-by-Side)"):
            gr.Markdown("So sánh trực tiếp chất lượng âm thanh và hiệu suất tài nguyên phần cứng giữa hai mô hình trên cùng một câu văn.")
            with gr.Row():
                text_input_comp = gr.Textbox(
                    label="Văn bản tiếng Việt chung để test",
                    lines=3,
                    placeholder="Nhập đoạn văn bản cần so sánh...",
                    value="Chào anh! Đây là chế độ so sánh song song giữa mô hình Kokoro và mô hình Meta VITS."
                )
            with gr.Row():
                speed_slider_comp = gr.Slider(
                    label="Tốc độ đọc chung (Speed)",
                    minimum=0.5,
                    maximum=2.0,
                    value=1.0,
                    step=0.1
                )
            
            btn_compare = gr.Button("⚖️ Bắt đầu Chạy Benchmark So Sánh", variant="primary", elem_classes="btn-primary")
            
            with gr.Row():
                # Kokoro Card
                with gr.Column(elem_classes="compare-card"):
                    gr.Markdown("### 🌟 1. Kokoro-Vietnamese (82M)")
                    gr.Markdown("*(Giọng mẫu: Diễm Trinh - Southern Female)*")
                    k_audio_out = gr.Audio(label="Kokoro Audio", type="filepath")
                    k_phoneme_out = gr.Markdown(label="Kokoro Phonemes")
                    k_stats_out = gr.HTML(label="Kokoro Stats", value="<p style='color: #64748b;'>Chờ chạy benchmark...</p>")
                    
                # VITS Card
                with gr.Column(elem_classes="compare-card"):
                    gr.Markdown("### 🏷️ 2. Meta MMS VITS")
                    gr.Markdown("*(Giọng mẫu: Meta Standard Speaker)*")
                    v_audio_out = gr.Audio(label="MMS VITS Audio", type="filepath")
                    v_phoneme_out = gr.Markdown(label="VITS Phonemes")
                    v_stats_out = gr.HTML(label="MMS VITS Stats", value="<p style='color: #64748b;'>Chờ chạy benchmark...</p>")

            btn_compare.click(
                fn=run_comparison,
                inputs=[text_input_comp, speed_slider_comp],
                outputs=[k_audio_out, k_stats_out, k_phoneme_out, v_audio_out, v_stats_out, v_phoneme_out]
            )

# Run the app
if __name__ == "__main__":
    try:
        try:
            print("Starting Gradio on port 7860...")
            demo.launch(
                server_name="127.0.0.1", 
                server_port=7860, 
                share=False,
                css=custom_css,
                theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
            )
        except OSError as oe:
            print(f"Port 7860 is busy or in TimeWait ({oe}). Falling back to another available port...")
            demo.launch(
                server_name="127.0.0.1", 
                share=False,
                css=custom_css,
                theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
            )
    except Exception as e:
        import traceback
        with open("crash_log.txt", "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        raise e
