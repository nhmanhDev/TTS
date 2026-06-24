// Globals
let modelMetadata = {};
let currentChart = null;

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initSliders();
  fetchModels();
  initSingleSynthesis();
  initCompareSynthesis();
});

// 1. Tab Navigation logic
function initTabs() {
  const singleBtn = document.getElementById('tab-single-btn');
  const compareBtn = document.getElementById('tab-compare-btn');
  const singlePane = document.getElementById('tab-single');
  const comparePane = document.getElementById('tab-compare');

  singleBtn.addEventListener('click', () => {
    singleBtn.classList.add('active');
    compareBtn.classList.remove('active');
    singlePane.classList.add('active');
    comparePane.classList.remove('active');
  });

  compareBtn.addEventListener('click', () => {
    compareBtn.classList.add('active');
    singleBtn.classList.remove('active');
    comparePane.classList.add('active');
    singlePane.classList.remove('active');
    
    // Resize chart if populated to prevent canvas layout bugs
    if (currentChart) {
      currentChart.resize();
    }
  });
}

// 2. Speed Slider displays
function initSliders() {
  const speedSlider = document.getElementById('slider-speed');
  const speedVal = document.getElementById('speed-value');
  const resetBtn = document.getElementById('btn-reset-speed');

  speedSlider.addEventListener('input', (e) => {
    speedVal.textContent = parseFloat(e.target.value).toFixed(1) + 'x';
  });

  resetBtn.addEventListener('click', () => {
    speedSlider.value = 1.0;
    speedVal.textContent = '1.0x';
  });

  const compSpeedSlider = document.getElementById('compare-slider-speed');
  const compSpeedVal = document.getElementById('compare-speed-value');

  compSpeedSlider.addEventListener('input', (e) => {
    compSpeedVal.textContent = parseFloat(e.target.value).toFixed(1) + 'x';
  });
}

// 3. Fetch Models & populate selections
async function fetchModels() {
  try {
    const response = await fetch('/api/models');
    modelMetadata = await response.json();
    
    const modelSelect = document.getElementById('select-model');
    modelSelect.innerHTML = '';
    
    for (const [key, data] of Object.entries(modelMetadata)) {
      const option = document.createElement('option');
      option.value = key;
      option.textContent = data.name;
      modelSelect.appendChild(option);
    }
    
    // Initial triggers
    onModelChange(modelSelect.value);
    
    modelSelect.addEventListener('change', (e) => {
      onModelChange(e.target.value);
    });
    
    document.getElementById('select-voice').addEventListener('change', () => {
      updateVoiceSample();
    });
    
  } catch (error) {
    console.error('Failed to fetch models:', error);
    showErrorSingle('Lỗi kết nối đến server API. Vui lòng kiểm tra cổng mạng Node.js.');
  }
}

function onModelChange(modelKey) {
  const data = modelMetadata[modelKey];
  if (!data) return;
  
  // Update description
  document.getElementById('model-desc').textContent = data.description;
  
  // Populate voices select
  const voiceSelect = document.getElementById('select-voice');
  voiceSelect.innerHTML = '';
  
  data.voices.forEach(voice => {
    const option = document.createElement('option');
    option.value = voice.id;
    option.textContent = `${voice.name} (${voice.gender} - ${voice.region})`;
    voiceSelect.appendChild(option);
  });
  
  // Trigger update voice sample
  updateVoiceSample();
}

function updateVoiceSample() {
  const modelKey = document.getElementById('select-model').value;
  const voiceId = document.getElementById('select-voice').value;
  const data = modelMetadata[modelKey];
  if (!data) return;
  
  const voice = data.voices.find(v => v.id === voiceId);
  const sampleBox = document.getElementById('voice-sample-box');
  const demoPlayer = document.getElementById('audio-demo-player');
  
  if (voice && voice.sampleUrl) {
    demoPlayer.src = voice.sampleUrl;
    demoPlayer.load();
    sampleBox.classList.remove('hidden');
  } else {
    demoPlayer.src = '';
    sampleBox.classList.add('hidden');
  }
}

// 4. Single Synthesis Trigger
function initSingleSynthesis() {
  const btn = document.getElementById('btn-synthesize');
  const textInput = document.getElementById('input-text');
  
  btn.addEventListener('click', async () => {
    const model = document.getElementById('select-model').value;
    const voice = document.getElementById('select-voice').value;
    const text = textInput.value.trim();
    const speed = parseFloat(document.getElementById('slider-speed').value);
    
    if (!text) {
      alert('Vui lòng nhập văn bản cần đọc.');
      return;
    }
    
    // Set UI to loading state
    setLoadingState(true);
    hideErrorSingle();
    
    try {
      const response = await fetch('/api/synthesize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, voice, text, speed })
      });
      
      const result = await response.json();
      
      if (result.success) {
        displaySingleResult(model, result.audio_path, result.stats);
      } else {
        showErrorSingle(result.error + (result.details ? `\nDetails: ${result.details}` : ''));
      }
    } catch (error) {
      console.error('Synthesis request failed:', error);
      showErrorSingle('Không kết nối được với server API hoặc quá trình xử lý bị ngắt kết nối.');
    } finally {
      setLoadingState(false);
    }
  });
}

function setLoadingState(isLoading) {
  const btn = document.getElementById('btn-synthesize');
  const btnText = btn.querySelector('.btn-text');
  const spinner = btn.querySelector('.loading-spinner');
  
  if (isLoading) {
    btn.disabled = true;
    btnText.classList.add('hidden');
    spinner.classList.remove('hidden');
  } else {
    btn.disabled = false;
    btnText.classList.remove('hidden');
    spinner.classList.add('hidden');
  }
}

function displaySingleResult(model, audioPath, stats) {
  const resultArea = document.getElementById('result-area');
  const placeholder = document.getElementById('result-placeholder');
  
  // Set audio player source
  const player = document.getElementById('audio-output-player');
  player.src = audioPath;
  player.load();
  
  // Display phonemes if available (only Kokoro has phonemes)
  const phonemesBox = document.getElementById('phonemes-box');
  const phonemesText = document.getElementById('phonemes-text');
  if (model === 'kokoro' && stats.phonemes && stats.phonemes !== 'N/A') {
    phonemesText.textContent = stats.phonemes;
    phonemesBox.classList.remove('hidden');
  } else {
    phonemesBox.classList.add('hidden');
  }
  
  // Update Benchmark Stats
  document.getElementById('stat-rtf').textContent = stats.rtf.toFixed(4);
  document.getElementById('stat-latency').textContent = stats.inference_time.toFixed(2) + 's';
  document.getElementById('stat-vram-peak').textContent = stats.vram_peak.toFixed(2) + ' MB';
  document.getElementById('stat-ram').textContent = stats.ram_usage.toFixed(2) + ' MB';
  
  // RTF description text
  const rtfDesc = document.getElementById('rtf-efficiency');
  if (stats.rtf < 1.0) {
    rtfDesc.innerHTML = `<span class="text-emerald"><i class="fa-solid fa-bolt"></i> Nhanh hơn thời gian thực (${(1.0 / stats.rtf).toFixed(1)}x)</span>`;
  } else {
    rtfDesc.innerHTML = `<span class="text-rose"><i class="fa-solid fa-hourglass-half"></i> Chậm hơn thời gian thực (${stats.rtf.toFixed(1)}x)</span>`;
  }
  
  // Metadata table
  document.getElementById('meta-duration').textContent = stats.audio_duration.toFixed(2) + ' giây';
  document.getElementById('meta-sample-rate').textContent = stats.sample_rate + ' Hz';
  document.getElementById('meta-vram-allocated').textContent = stats.vram_usage.toFixed(2) + ' MB';
  
  // Show result box
  placeholder.classList.add('hidden');
  resultArea.classList.remove('hidden');
}

function showErrorSingle(msg) {
  const banner = document.getElementById('error-banner');
  const message = document.getElementById('error-message');
  message.textContent = msg;
  banner.classList.remove('hidden');
  document.getElementById('result-area').classList.add('hidden');
  document.getElementById('result-placeholder').classList.add('hidden');
}

function hideErrorSingle() {
  document.getElementById('error-banner').classList.add('hidden');
}

// 5. Compare Tab logic (Side-by-Side)
function initCompareSynthesis() {
  const btn = document.getElementById('btn-compare');
  const textInput = document.getElementById('compare-text-input');
  
  btn.addEventListener('click', async () => {
    const text = textInput.value.trim();
    const speed = parseFloat(document.getElementById('compare-slider-speed').value);
    
    if (!text) {
      alert('Vui lòng nhập văn bản so sánh chung.');
      return;
    }
    
    // UI Loading state
    btn.disabled = true;
    btn.querySelector('.btn-text').classList.add('hidden');
    btn.querySelector('.loading-spinner').classList.remove('hidden');
    
    // Clear and display loaders
    document.getElementById('comp-k-placeholder').textContent = 'Đang chạy mô hình Kokoro...';
    document.getElementById('comp-v-placeholder').textContent = 'Đang chạy mô hình MMS VITS...';
    document.getElementById('comp-k-area').classList.add('hidden');
    document.getElementById('comp-v-area').classList.add('hidden');
    document.getElementById('comp-k-placeholder').classList.remove('hidden');
    document.getElementById('comp-v-placeholder').classList.remove('hidden');
    
    let k_success = false;
    let v_success = false;
    let k_data = null;
    let v_data = null;
    
    // 1. Run Kokoro
    try {
      const response = await fetch('/api/synthesize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: 'kokoro', voice: 'diem_trinh', text, speed })
      });
      const result = await response.json();
      if (result.success) {
        k_success = true;
        k_data = result;
        // Display Kokoro card
        document.getElementById('comp-k-audio').src = result.audio_path;
        document.getElementById('comp-k-phonemes').textContent = result.stats.phonemes;
        document.getElementById('comp-k-time').textContent = result.stats.inference_time.toFixed(2) + 's';
        document.getElementById('comp-k-rtf').textContent = result.stats.rtf.toFixed(4);
        document.getElementById('comp-k-vram').textContent = result.stats.vram_peak.toFixed(2) + ' MB';
        document.getElementById('comp-k-ram').textContent = result.stats.ram_usage.toFixed(2) + ' MB';
        
        document.getElementById('comp-k-placeholder').classList.add('hidden');
        document.getElementById('comp-k-area').classList.remove('hidden');
      } else {
        document.getElementById('comp-k-placeholder').textContent = `Lỗi Kokoro: ${result.error}`;
      }
    } catch (err) {
      document.getElementById('comp-k-placeholder').textContent = 'Kết nối thất bại khi gọi Kokoro.';
    }
    
    // 2. Run Meta MMS VITS
    try {
      const response = await fetch('/api/synthesize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: 'vits', voice: 'mms_vietnamese', text, speed })
      });
      const result = await response.json();
      if (result.success) {
        v_success = true;
        v_data = result;
        // Display VITS card
        document.getElementById('comp-v-audio').src = result.audio_path;
        document.getElementById('comp-v-time').textContent = result.stats.inference_time.toFixed(2) + 's';
        document.getElementById('comp-v-rtf').textContent = result.stats.rtf.toFixed(4);
        document.getElementById('comp-v-vram').textContent = result.stats.vram_peak.toFixed(2) + ' MB';
        document.getElementById('comp-v-ram').textContent = result.stats.ram_usage.toFixed(2) + ' MB';
        
        document.getElementById('comp-v-placeholder').classList.add('hidden');
        document.getElementById('comp-v-area').classList.remove('hidden');
      } else {
        document.getElementById('comp-v-placeholder').textContent = `Lỗi MMS VITS: ${result.error}`;
      }
    } catch (err) {
      document.getElementById('comp-v-placeholder').textContent = 'Kết nối thất bại khi gọi VITS.';
    }
    
    // 3. Render visual comparisons Chart
    if (k_success && v_success) {
      renderCompareChart(k_data.stats, v_data.stats);
    }
    
    // Reset Compare Button state
    btn.disabled = false;
    btn.querySelector('.btn-text').classList.remove('hidden');
    btn.querySelector('.loading-spinner').classList.add('hidden');
  });
}

function renderCompareChart(k_stats, v_stats) {
  const ctx = document.getElementById('benchmarkChart').getContext('2d');
  
  if (currentChart) {
    currentChart.destroy();
  }
  
  // Styling rules for Chart.js to fit dark mode theme
  Chart.defaults.color = '#94a3b8';
  Chart.defaults.font.family = 'Inter';
  
  currentChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Độ trễ (Inference Time - s)', 'Tốc độ (RTF - Ratio)', 'Peak VRAM GPU (MB)'],
      datasets: [
        {
          label: '🌟 Kokoro-Vietnamese',
          data: [k_stats.inference_time, k_stats.rtf, k_stats.vram_peak],
          backgroundColor: 'rgba(59, 130, 246, 0.65)',
          borderColor: '#3b82f6',
          borderWidth: 2,
          borderRadius: 6
        },
        {
          label: '🏷️ Meta MMS VITS',
          data: [v_stats.inference_time, v_stats.rtf, v_stats.vram_peak],
          backgroundColor: 'rgba(139, 92, 246, 0.65)',
          borderColor: '#8b5cf6',
          borderWidth: 2,
          borderRadius: 6
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          type: 'logarithmic', // Logarithmic scale is perfect because Peak VRAM (e.g. 380MB) is way larger than Latency (e.g. 1.5s)
          grid: {
            color: 'rgba(255, 255, 255, 0.05)'
          },
          border: {
            dash: [4, 4]
          }
        },
        x: {
          grid: {
            display: false
          }
        }
      },
      plugins: {
        legend: {
          position: 'top',
          labels: {
            font: {
              family: 'Outfit',
              weight: 'bold',
              size: 13
            }
          }
        },
        tooltip: {
          backgroundColor: '#0f172a',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          titleFont: {
            family: 'Outfit',
            weight: 'bold'
          },
          bodyFont: {
            family: 'Inter'
          }
        }
      }
    }
  });
}
