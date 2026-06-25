# Báo Cáo Dự Án: Vietnamese TTS Benchmark

## Tổng Quan

Dự án xây dựng một ứng dụng benchmark so sánh các mô hình Text-to-Speech (TTS) đa ngôn ngữ hiện đại, tập trung vào khả năng tổng hợp tiếng Việt kết hợp với các từ/thuật ngữ tiếng Anh (code-switching). Giao diện Gradio cho phép người dùng chọn model, chọn giọng, nhập văn bản, và so sánh output trực tiếp.

---

## Tiêu Chí Lựa Chọn Model

Dự án chỉ giữ lại các model thỏa mãn **toàn bộ** tiêu chí sau:

| Tiêu chí | Lý do |
|----------|-------|
| **Multilingual LLM-based** | Hỗ trợ code-switching Anh–Việt tự nhiên |
| **Zero-shot / voice cloning** | Không cần fine-tune cho tiếng Việt |
| **Open weights** | Self-host được, không phụ thuộc API |
| **Chất lượng đủ dùng** | Nghe được với văn bản tiếng Việt thực tế |

**Các model đã loại bỏ:**

| Model | Lý do loại |
|-------|-----------|
| Kokoro-Vietnamese | Thuần Việt, đọc "AI" thành "ai" (tiếng Việt), không code-switch |
| Meta MMS VITS | Phoneme-based thuần Việt, chất lượng thấp |
| Supertonic-3 | Multilingual nhưng `lang="vi"` force Vietnamese phonemes |
| Fish Audio S2-Pro | Code-switch không ổn định, pipeline phức tạp (3-step DAC) |

---

## Các Model Hiện Tại (5 models)

### 1. OpenBMB VoxCPM2
- **Repo:** `openbmb/VoxCPM2`
- **Kích thước:** 2B parameters + AudioVAE
- **Checkpoint:** `checkpoints/voxcpm/` (~4.7 GB)
- **Kiến trúc:** LLM backbone + AudioVAE decoder
- **Giọng:** Voice Design (mô tả bằng văn bản), 4 preset giọng Việt
- **Sample rate:** 24 kHz mono
- **Điểm mạnh:** Voice design linh hoạt, có thể tạo giọng từ mô tả ngôn ngữ tự nhiên

**Voices:**

| Voice ID | Tên hiển thị | Giới tính | Vùng |
|----------|-------------|-----------|------|
| `giong_nu_mien_nam` | Hà Linh | Nữ | Miền Nam |
| `giong_nam_mien_bac` | Quang Minh | Nam | Miền Bắc |
| `giong_nu_mien_bac` | Thu Hà | Nữ | Miền Bắc |
| `giong_nam_mien_nam` | Bảo Long | Nam | Miền Nam |

---

### 2. OmniVoice
- **Repo:** `k2-fsa/OmniVoice`
- **Kích thước:** 0.6B parameters
- **Kiến trúc:** Compact LLM, hỗ trợ voice design + voice cloning
- **Giọng:** 4 voice design + 1 voice clone
- **Sample rate:** 24 kHz mono
- **Điểm mạnh:** Nhẹ nhất trong bộ (0.6B), hỗ trợ cả voice design lẫn cloning
- **Dependency đặc biệt:** Cần monkey-patch `transformers` để expose `HiggsAudioV2TokenizerModel` (class có trong transformers 5.12.1 nhưng không được export ra `__init__.py`)

**Voices:**

| Voice ID | Tên hiển thị | Giới tính | Vùng |
|----------|-------------|-----------|------|
| `design_nu_mien_nam` | Lan Anh | Nữ | Miền Nam |
| `design_nam_mien_bac` | Minh Tuấn | Nam | Miền Bắc |
| `design_nu_mien_bac` | Hồng Nhung | Nữ | Miền Bắc |
| `design_nam_mien_nam` | Trọng Khải | Nam | Miền Nam |
| `clone_diem_trinh` | Diễm Trinh (Nhân Bản) | Nữ | Miền Nam |

---

### 3. MOSS-TTS Local-Transformer v1.5
- **Repo:** `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5`
- **Kích thước:** 5B parameters
- **Kiến trúc:** Local-Transformer với audio codebook
- **Giọng:** Giọng mặc định + voice clone
- **Sample rate:** 48 kHz stereo (averaged to mono khi xử lý)
- **Điểm mạnh:** Sample rate cao nhất (48 kHz), chất lượng âm thanh tốt
- **Prompting:** `language='Vietnamese'` bắt buộc qua `processor.build_user_message()`

**Voices:**

| Voice ID | Tên hiển thị |
|----------|-------------|
| `default` | Giọng Mặc Định |
| `clone_diem_trinh` | Diễm Trinh (Nhân Bản) |

---

### 4. MOSS-TTS v1.5 8B
- **Repo:** `OpenMOSS-Team/MOSS-TTS-v1.5`
- **Kích thước:** 8B parameters
- **Kiến trúc:** LLM 8B với audio head, cùng family với MOSS-TTS LT
- **Giọng:** Giọng mặc định + voice clone
- **Sample rate:** 48 kHz stereo
- **Điểm mạnh:** Model lớn nhất, khả năng ngữ nghĩa tốt nhất trong MOSS family

**Voices:**

| Voice ID | Tên hiển thị |
|----------|-------------|
| `default` | Giọng Mặc Định |
| `clone_diem_trinh` | Diễm Trinh (Nhân Bản) |

---

### 5. Higgs Audio TTS v3 4B
- **Repo:** `bosonai/higgs-audio-v3-tts-4b` (chạy qua transformers port `multimodalart/higgs-audio-v3-tts-4b-transformers`)
- **Kích thước:** 4B parameters (Qwen3 backbone)
- **Checkpoint:** `checkpoints/higgs-tts-v3-transformers/` (~8.7 GB)
- **Kiến trúc:** Qwen3-4B + 8-codebook audio head + HiggsAudioV2Tokenizer
- **Giọng:** Zero-shot + voice clone
- **Sample rate:** 24 kHz mono
- **Điểm mạnh:** Tốt nhất về code-switching, hỗ trợ 100+ ngôn ngữ, có hệ thống tag kiểm soát cảm xúc/prosody
- **Venv riêng:** `.venv-higgs` (torch 2.11.0 + vllm 0.22.0, cần `libcudart.so.13`)

**Voices:**

| Voice ID | Tên hiển thị |
|----------|-------------|
| `default` | Giọng Mặc Định |
| `clone_diem_trinh` | Diễm Trinh (Nhân Bản) |

**Control Tags (43 tags, đặt đầu câu):**
- Cảm xúc: `<|emotion:elation|>`, `<|emotion:contentment|>`, `<|emotion:sadness|>`, ...
- Phong cách: `<|style:whispering|>`, `<|style:shouting|>`, `<|style:singing|>`
- Tốc độ: `<|prosody:speed_slow|>`, `<|prosody:speed_fast|>`, ...
- Âm thanh inline: `<|sfx:laughter|>Haha...`, `<|sfx:sigh|>Haah...`

---

## Kiến Trúc Kỹ Thuật

### Cấu Trúc Dự Án

```
TTS/
├── app.py                          # Gradio UI, model routing
├── models/
│   ├── base_model.py               # BaseTTSModel ABC
│   ├── voxcpm_wrapper.py
│   ├── omnivoice_wrapper.py
│   ├── moss_tts_wrapper.py
│   ├── higgs_audio_wrapper.py
│   └── __init__.py
├── checkpoints/
│   ├── voxcpm/                     # VoxCPM2 weights (~4.7 GB)
│   ├── higgs-tts-v3-4b/            # Higgs vllm-omni weights (~8.7 GB)
│   ├── higgs-tts-v3-transformers/  # Higgs transformers port (~8.7 GB)
│   ├── higgs-audio-v2-tokenizer/   # Audio tokenizer (~769 MB)
│   └── moss-tts-repo/              # MOSS repo clone
├── assets/samples/
│   ├── voxcpm/                     # 4 WAV files
│   ├── omnivoice/                  # 5 WAV files
│   ├── moss_lt/                    # 2 WAV files
│   ├── moss_v15/                   # 2 WAV files
│   └── higgs_v3/                   # 2 WAV files
├── outputs/                        # Generated audio files
├── .venv/                          # Main venv (torch 2.9.1+cu128)
└── .venv-higgs/                    # Higgs venv (torch 2.11.0+cu128)
```

### Pattern Lazy-Load

Tất cả wrapper đều implement pattern load-on-demand để tránh OOM khi chạy nhiều model:

```python
def synthesize(self, text, voice, speed=1.0):
    self.load()           # load model vào VRAM
    audio = self.model(text, ...)
    self.unload()         # xóa model khỏi VRAM ngay sau khi xong
    return audio, sr, stats
```

### Subprocess Isolation

Một số model yêu cầu dependency không tương thích với main venv nên được chạy qua subprocess:

| Model | Venv | Lý do |
|-------|------|-------|
| Higgs Audio v3 | `.venv-higgs` | Cần torch 2.11.0 + vllm 0.22.0 + `libcudart.so.13` |

### Dependency Đặc Biệt

**OmniVoice — Monkey-patch transformers:**
```python
import transformers
from transformers.models.higgs_audio_v2_tokenizer.modeling_higgs_audio_v2_tokenizer import (
    HiggsAudioV2TokenizerModel,
)
transformers.HiggsAudioV2TokenizerModel = HiggsAudioV2TokenizerModel
```
Cần thiết vì `omnivoice` import class này từ `transformers` ở module-level, nhưng transformers 5.x không export nó trong `__init__.py`.

**MOSS-TTS — torchaudio monkey-patch:**
```python
torchaudio.load = _soundfile_load  # thay thế torchcodec bằng soundfile
```
Cần thiết vì MOSS dùng `torchaudio.load` để đọc reference audio, nhưng không có ffmpeg/torchcodec.

---

## Benchmark Setup

### Text Mẫu Chuẩn
```
Xin chào, đây là giọng đọc mẫu AI hỗ trợ tiếng Việt.
```
Câu này được chọn vì chứa từ "AI" để kiểm tra khả năng code-switching Anh–Việt.

### Metrics Thu Thập

| Metric | Ý nghĩa |
|--------|---------|
| **RTF** (Real-Time Factor) | Thời gian xử lý / độ dài audio. RTF < 1.0 = nhanh hơn thời gian thực |
| **Thời gian chạy** | Tổng thời gian từ lúc bắt đầu đến khi có audio |
| **Độ dài audio** | Độ dài file output (giây) |
| **VRAM allocated** | VRAM tăng thêm sau khi load model (MB) |
| **VRAM peak** | VRAM đỉnh trong quá trình inference (MB) |
| **RAM hệ thống** | RAM process tăng thêm (MB) |

### Pre-generated Samples

Mỗi model có sẵn file WAV mẫu trong `assets/samples/{model}/` để preview không cần chạy model:

```
assets/samples/
├── voxcpm/     Hà Linh, Quang Minh, Thu Hà, Bảo Long
├── omnivoice/  Lan Anh, Minh Tuấn, Hồng Nhung, Trọng Khải, Diễm Trinh (Nhân Bản)
├── moss_lt/    Giọng Mặc Định, Diễm Trinh (Nhân Bản)
├── moss_v15/   Giọng Mặc Định, Diễm Trinh (Nhân Bản)
└── higgs_v3/   Giọng Mặc Định, Diễm Trinh (Nhân Bản)
```

---

## Môi Trường

| Thành phần | Phiên bản |
|-----------|----------|
| GPU | NVIDIA RTX PRO 6000 Blackwell (~96 GB VRAM) |
| CUDA | 12.8 (libcudart.so.12) + 13.x (libcudart.so.13 cho Higgs) |
| Python | 3.12 |
| PyTorch (main venv) | 2.9.1+cu128 |
| PyTorch (higgs venv) | 2.11.0+cu128 |
| transformers | 5.12.1 |

---

## Chạy Ứng Dụng

```bash
cd /home/manh/TestText2Speech/TTS
source .venv/bin/activate
python app.py
```

Gradio UI sẽ khởi động tại `http://localhost:7860`.

---

## So Sánh Nhanh

| Model | Size | Code-switch | Voice Design | Voice Clone | Sample Rate |
|-------|------|-------------|--------------|-------------|-------------|
| VoxCPM2 | 2B | ✅ | ✅ | ❌ | 24 kHz |
| OmniVoice | 0.6B | ✅ | ✅ | ✅ | 24 kHz |
| MOSS-TTS LT | 5B | ✅ | ❌ | ✅ | **48 kHz** |
| MOSS-TTS v1.5 | 8B | ✅ | ❌ | ✅ | **48 kHz** |
| Higgs Audio v3 | 4B | ✅✅ | ❌ | ✅ | 24 kHz |
