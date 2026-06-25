const express = require('express');
const cors = require('cors');
const path = require('path');
const { execFile } = require('child_process');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3000;

// Enable CORS and JSON parser
app.use(cors());
app.use(express.json());

// Serve static web pages from public/ folder
app.use(express.static(path.join(__dirname, 'public')));

// Serve audio outputs and assets
app.use('/outputs', express.static(path.join(__dirname, 'outputs')));
app.use('/assets', express.static(path.join(__dirname, 'assets')));

// Metadata definitions for models and voices
const MODELS = {
  kokoro: {
    name: "Kokoro-Vietnamese (contextboxai/Kokoro-Vietnamese)",
    description: "Mô hình siêu nhẹ (82M params) tốc độ nhanh và có chất giọng biểu cảm tự nhiên nhất hiện nay.",
    voices: [
      { id: 'diem_trinh', name: 'Diễm Trinh', gender: 'Nữ', region: 'Miền Nam' },
      { id: 'hung_thinh', name: 'Hưng Thịnh', gender: 'Nam', region: 'Miền Nam' },
      { id: 'mai_linh', name: 'Mai Linh', gender: 'Nữ', region: 'Miền Nam' },
      { id: 'mai_loan', name: 'Mai Loan', gender: 'Nữ', region: 'Miền Nam' },
      { id: 'manh_dung', name: 'Mạnh Dũng', gender: 'Nam', region: 'Miền Bắc' },
      { id: 'my_yen', name: 'Mỹ Yến', gender: 'Nữ', region: 'Miền Nam' },
      { id: 'ngoc_huyen', name: 'Ngọc Huyền', gender: 'Nữ', region: 'Miền Bắc' },
      { id: 'phat_tai', name: 'Phát Tài', gender: 'Nam', region: 'Miền Nam' },
      { id: 'thanh_dat', name: 'Thành Đạt', gender: 'Nam', region: 'Miền Nam' },
      { id: 'thuc_trinh', name: 'Thục Trinh', gender: 'Nữ', region: 'Miền Nam' },
      { id: 'tuan_ngoc', name: 'Tuấn Ngọc', gender: 'Nam', region: 'Miền Bắc' },
      { id: 'duc_an', name: 'Đức An', gender: 'Nam', region: 'Miền Bắc' },
      { id: 'duc_duy', name: 'Đức Duy', gender: 'Nam', region: 'Miền Nam' },
      { id: 'storyvert', name: 'Storyvert', gender: 'Nữ', region: 'Miền Nam' }
    ]
  },
  vits: {
    name: "Meta MMS VITS (facebook/mms-tts-vie)",
    description: "Mô hình TTS của Meta chạy siêu nhanh trên cả CPU và GPU, chuẩn hóa từ ngữ tốt.",
    voices: [
      { id: 'mms_vietnamese', name: 'Meta MMS Vietnamese Speaker', gender: 'Nam/Nữ', region: 'Toàn dân' }
    ]
  },
  voxcpm: {
    name: "OpenBMB VoxCPM2 (openbmb/VoxCPM2)",
    description: "Mô hình ngôn ngữ giọng nói lớn 2B tham số, hỗ trợ chất lượng phòng thu Studio (48kHz) và thiết kế giọng đọc.",
    voices: [
      { id: 'giong_nu_mien_nam', name: 'Giọng Nữ Miền Nam', gender: 'Nữ', region: 'Miền Nam' },
      { id: 'giong_nam_mien_bac', name: 'Giọng Nam Miền Bắc', gender: 'Nam', region: 'Miền Bắc' },
      { id: 'giong_nu_mien_bac', name: 'Giọng Nữ Miền Bắc', gender: 'Nữ', region: 'Miền Bắc' },
      { id: 'giong_nam_mien_nam', name: 'Giọng Nam Miền Nam', gender: 'Nam', region: 'Miền Nam' }
    ]
  },
  vieneu: {
    name: "VieNeu-TTS v3 Turbo (pnnbao97/vieneu)",
    description: "Mô hình TTS 48kHz hỗ trợ song ngữ Anh-Việt, cảm xúc, và 10 giọng đọc mẫu. Chạy ONNX trên CPU siêu nhanh.",
    voices: [
      { id: 'ngoc_lan', name: 'Ngọc Lan', gender: 'Nữ', region: 'Giọng dịu dàng' },
      { id: 'gia_bao', name: 'Gia Bảo', gender: 'Nam', region: 'Giọng mượt mà' },
      { id: 'thai_son', name: 'Thái Sơn', gender: 'Nam', region: 'Giọng chắc khỏe' },
      { id: 'duc_tri', name: 'Đức Trí', gender: 'Nam', region: 'Giọng rõ ràng' },
      { id: 'my_duyen', name: 'Mỹ Duyên', gender: 'Nữ', region: 'Giọng mượt mà' },
      { id: 'truc_ly', name: 'Trúc Ly', gender: 'Nữ', region: 'Giọng trẻ trung' },
      { id: 'xuan_vinh', name: 'Xuân Vĩnh', gender: 'Nam', region: 'Giọng vui tươi' },
      { id: 'trong_huu', name: 'Trọng Hữu', gender: 'Nam', region: 'Giọng uyên bác' },
      { id: 'binh_an', name: 'Bình An', gender: 'Nam', region: 'Giọng điềm đạm' },
      { id: 'ngoc_linh', name: 'Ngọc Linh', gender: 'Nữ', region: 'Giọng tươi sáng' }
    ]
  },
  vixtts: {
    name: "viXTTS (capleaf/viXTTS)",
    description: "Fine-tuned từ XTTS-v2, hỗ trợ nhân bản giọng nói (voice cloning) tiếng Việt chỉ từ 6 giây audio mẫu.",
    voices: [
      { id: 'vixtts_default', name: 'viXTTS Default', gender: 'Nữ', region: 'Giọng Việt tự nhiên' }
    ]
  }
};

// GET /api/models: Get list of models, their voices and local pre-recorded sample URLs
app.get('/api/models', (req, res) => {
  const result = {};
  
  for (const [modelKey, modelData] of Object.entries(MODELS)) {
    result[modelKey] = {
      name: modelData.name,
      description: modelData.description,
      voices: modelData.voices.map(voice => {
        const sampleFilename = `${modelKey}_${voice.id}.wav`;
        
        const sampleRelativePath = path.join('assets', 'samples', sampleFilename);
        const sampleAbsolutePath = path.join(__dirname, sampleRelativePath);
        const sampleUrl = fs.existsSync(sampleAbsolutePath) ? `/${sampleRelativePath.replace(/\\/g, '/')}` : null;
        
        return {
          ...voice,
          sampleUrl
        };
      })
    };
  }
  
  res.json(result);
});

// POST /api/synthesize: Call python script to synthesize speech and get stats
app.post('/api/synthesize', (req, res) => {
  const { model, voice, text, speed } = req.body;
  
  if (!model || !voice || !text) {
    return res.status(400).json({ success: false, error: 'Thiếu tham số bắt buộc (model, voice, text)' });
  }
  
  const pythonPath = path.join(__dirname, '.venv', 'Scripts', 'python.exe');
  const cliScriptPath = path.join(__dirname, 'tts_cli.py');
  
  const args = [
    cliScriptPath,
    '--model', model,
    '--voice', voice,
    '--text', text,
    '--speed', (speed || 1.0).toString()
  ];
  
  console.log(`Executing: ${pythonPath} ${args.join(' ')}`);
  
  // Increase timeout to 5 minutes for large models like viXTTS
  execFile(pythonPath, args, { maxBuffer: 1024 * 1024 * 10, timeout: 300000 }, (error, stdout, stderr) => {
    if (error) {
      console.error(`Exec error: ${error}`);
      console.error(`Stderr: ${stderr}`);
      return res.status(500).json({ 
        success: false, 
        error: 'Lỗi hệ thống khi gọi Python CLI', 
        details: error.message,
        stderr: stderr 
      });
    }
    
    try {
      const responseJson = JSON.parse(stdout);
      res.json(responseJson);
    } catch (parseError) {
      console.error(`JSON Parse error: ${parseError}`);
      console.error(`Raw Stdout: ${stdout}`);
      res.status(500).json({ 
        success: false, 
        error: 'Lỗi parse JSON đầu ra của Python', 
        rawOutput: stdout 
      });
    }
  });
});

// Fallback to serve index.html for undefined frontend routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Start the server
app.listen(PORT, () => {
  console.log(`==================================================`);
  console.log(`🚀 Node.js Server is running on port ${PORT}`);
  console.log(`👉 Access URL: http://localhost:${PORT}`);
  console.log(`==================================================`);
});
