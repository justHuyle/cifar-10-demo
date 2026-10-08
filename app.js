/**
 * ─────────────────────────────────────────────────────────────
 * CIFAR-10 Image Classifier • ONNX Runtime Web Client
 * Standalone bundle for Vercel / GitHub Pages deployment
 * ─────────────────────────────────────────────────────────────
 */

// 1. CIFAR-10 Classes Specification
const CIFAR_CLASSES = [
  { id: 'airplane', vi: 'Máy bay', en: 'airplane', emoji: '✈️' },
  { id: 'automobile', vi: 'Ô tô', en: 'automobile', emoji: '🚗' },
  { id: 'bird', vi: 'Chim', en: 'bird', emoji: '🐦' },
  { id: 'cat', vi: 'Mèo', en: 'cat', emoji: '🐱' },
  { id: 'deer', vi: 'Hươu', en: 'deer', emoji: '🦌' },
  { id: 'dog', vi: 'Chó', en: 'dog', emoji: '🐶' },
  { id: 'frog', vi: 'Ếch', en: 'frog', emoji: '🐸' },
  { id: 'horse', vi: 'Ngựa', en: 'horse', emoji: '🐴' },
  { id: 'ship', vi: 'Tàu thuyền', en: 'ship', emoji: '🚢' },
  { id: 'truck', vi: 'Xe tải', en: 'truck', emoji: '🚚' }
];

// Normalization Hyperparameters (CIFAR-10 standard)
const MEAN = [0.4914, 0.4822, 0.4465];
const STD = [0.2470, 0.2435, 0.2616];

// State
let ortSession = null;
let isInferring = false;
let currentSampleClass = null;

// DOM Elements
const modelStatusIndicator = document.getElementById('modelStatusIndicator');
const statusSpinner = document.getElementById('statusSpinner');
const statusText = document.getElementById('statusText');
const inferenceTimeTag = document.getElementById('inferenceTimeTag');

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const sampleGrid = document.getElementById('sampleGrid');

const imagePreview = document.getElementById('imagePreview');
const previewPlaceholder = document.getElementById('previewPlaceholder');
const imageMetaText = document.getElementById('imageMetaText');
const imageNameText = document.getElementById('imageNameText');

const topClassEmoji = document.getElementById('topClassEmoji');
const topClassVi = document.getElementById('topClassVi');
const topClassEn = document.getElementById('topClassEn');
const topConfidenceVal = document.getElementById('topConfidenceVal');
const topConfidenceFill = document.getElementById('topConfidenceFill');

const probList = document.getElementById('probList');
const preprocessCanvas = document.getElementById('preprocessCanvas');
const ctx = preprocessCanvas.getContext('2d', { willReadFrequently: true });

// ─────────────────────────────────────────────────────────────
// Initialization
// ─────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  renderSampleCards();
  renderInitialProbList();
  setupEventListeners();
  await loadModel();
});

// Render the 10-sample gallery cards
function renderSampleCards() {
  sampleGrid.innerHTML = '';
  CIFAR_CLASSES.forEach((cls) => {
    const card = document.createElement('div');
    card.className = 'sample-card';
    card.id = `sample-${cls.id}`;
    card.title = `Thử nghiệm ảnh mẫu: ${cls.vi} (${cls.en})`;

    // Path to sample image in samples folder
    const samplePath = `samples/${cls.id}.png`;

    card.innerHTML = `
      <div class="sample-thumb-wrapper">
        <img src="${samplePath}" alt="${cls.en}" class="sample-thumb" loading="lazy">
      </div>
      <span class="sample-card-name">${cls.emoji} ${cls.vi}</span>
    `;

    card.addEventListener('click', () => {
      selectSample(cls, samplePath);
    });

    sampleGrid.appendChild(card);
  });
}

// Initial placeholder probability bars (0%)
function renderInitialProbList() {
  probList.innerHTML = '';
  CIFAR_CLASSES.forEach((cls) => {
    const row = document.createElement('div');
    row.className = 'prob-row';
    row.id = `prob-row-${cls.id}`;

    row.innerHTML = `
      <span class="prob-emoji">${cls.emoji}</span>
      <div class="prob-names">
        <span class="prob-name-vi">${cls.vi}</span>
        <span class="prob-name-en">${cls.en}</span>
      </div>
      <div class="prob-bar-track">
        <div class="prob-bar-fill" id="prob-bar-${cls.id}" style="width: 0%;"></div>
      </div>
      <span class="prob-percent" id="prob-pct-${cls.id}">0.0%</span>
    `;

    probList.appendChild(row);
  });
}

// Setup Drag & Drop and File Input listeners
function setupEventListeners() {
  // Drop Zone drag events
  ['dragenter', 'dragover'].forEach(evt => {
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(evt => {
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('dragover');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleImageFile(files[0]);
    }
  });

  // File input change
  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleImageFile(e.target.files[0]);
    }
  });
}

// ─────────────────────────────────────────────────────────────
// Load ONNX Model
// ─────────────────────────────────────────────────────────────
async function loadModel() {
  try {
    statusText.textContent = 'Đang tải model.onnx (WASM)...';
    
    // Check if ONNX Runtime Web library is available
    if (typeof ort === 'undefined') {
      throw new Error('Thư viện ONNX Runtime Web chưa được tải. Hãy kiểm tra kết nối mạng.');
    }

    // Configure ONNX Runtime Web environment
    if (ort.env && ort.env.wasm) {
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.simd = true;
    }

    // Create session
    ortSession = await ort.InferenceSession.create('./model.onnx', {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all'
    });

    // Update UI status
    statusSpinner.style.display = 'none';
    modelStatusIndicator.classList.add('ready');
    statusText.textContent = 'Mô hình ONNX sẵn sàng';
    inferenceTimeTag.textContent = 'WASM Ready';

    // Auto-select first sample for instant demo
    const defaultClass = CIFAR_CLASSES[0];
    selectSample(defaultClass, `samples/${defaultClass.id}.png`);

  } catch (err) {
    console.error('Error loading ONNX model:', err);
    statusSpinner.style.display = 'none';
    modelStatusIndicator.classList.add('error');
    statusText.textContent = 'Lỗi nạp model.onnx';
    alert(`Không thể tải mô hình ONNX: ${err.message}\n\nHãy đảm bảo bạn đang mở trang qua một HTTP server (ví dụ: Vercel hoặc python -m http.server 8000) và tệp model.onnx đã được triển khai.`);
  }
}

// ─────────────────────────────────────────────────────────────
// Image Handling & Preprocessing
// ─────────────────────────────────────────────────────────────
function handleImageFile(file) {
  if (!file.type.startsWith('image/')) {
    alert('Vui lòng chọn một tệp hình ảnh hợp lệ (PNG, JPG, JPEG, WEBP).');
    return;
  }

  // Clear active state of sample cards
  clearActiveSamples();

  const reader = new FileReader();
  reader.onload = (e) => {
    const img = new Image();
    img.onload = () => {
      displayPreview(img, file.name, `${img.naturalWidth} × ${img.naturalHeight}`);
      classifyImage(img);
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

function selectSample(cls, samplePath) {
  clearActiveSamples();
  const card = document.getElementById(`sample-${cls.id}`);
  if (card) card.classList.add('active');

  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.onload = () => {
    displayPreview(img, `Sample: ${cls.vi}`, `32 × 32`);
    classifyImage(img);
  };
  img.src = samplePath;
}

function clearActiveSamples() {
  document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('active'));
}

function displayPreview(img, name, dimensions) {
  imagePreview.src = img.src;
  imagePreview.classList.remove('hidden');
  previewPlaceholder.style.display = 'none';
  imageNameText.textContent = name;
  imageMetaText.textContent = `Kích thước: ${dimensions}`;
}

// ─────────────────────────────────────────────────────────────
// Preprocess to Float32 Tensor: (1, 3, 32, 32)
// ─────────────────────────────────────────────────────────────
function preprocessImage(img) {
  // 1. Draw onto 32x32 canvas
  ctx.clearRect(0, 0, 32, 32);
  ctx.drawImage(img, 0, 0, 32, 32);

  // 2. Extract RGBA pixel data
  const imageData = ctx.getImageData(0, 0, 32, 32);
  const rgba = imageData.data;

  // 3. Construct Float32Array in CHW order (channel first: R plane, G plane, B plane)
  const float32Data = new Float32Array(3 * 32 * 32);
  const planeSize = 32 * 32;

  for (let i = 0; i < planeSize; i++) {
    // Normalize pixel values to [0.0, 1.0] and apply CIFAR-10 mean & std
    const r = rgba[i * 4 + 0] / 255.0;
    const g = rgba[i * 4 + 1] / 255.0;
    const b = rgba[i * 4 + 2] / 255.0;

    float32Data[0 * planeSize + i] = (r - MEAN[0]) / STD[0];
    float32Data[1 * planeSize + i] = (g - MEAN[1]) / STD[1];
    float32Data[2 * planeSize + i] = (b - MEAN[2]) / STD[2];
  }

  return new ort.Tensor('float32', float32Data, [1, 3, 32, 32]);
}

// ─────────────────────────────────────────────────────────────
// Softmax Calculation
// ─────────────────────────────────────────────────────────────
function softmax(logits) {
  const maxLogit = Math.max(...logits);
  const exps = logits.map(v => Math.exp(v - maxLogit));
  const sumExps = exps.reduce((acc, v) => acc + v, 0);
  return exps.map(v => v / sumExps);
}

// ─────────────────────────────────────────────────────────────
// Real-time Inference Pipeline
// ─────────────────────────────────────────────────────────────
async function classifyImage(img) {
  if (!ortSession) {
    alert('Mô hình ONNX chưa hoàn tất nạp. Vui lòng thử lại sau giây lát.');
    return;
  }
  if (isInferring) return;

  try {
    isInferring = true;
    inferenceTimeTag.textContent = 'Đang suy luận...';

    // Preprocessing on canvas
    const inputTensor = preprocessImage(img);

    // Forward inference
    const t0 = performance.now();
    const feeds = { input: inputTensor };
    const results = await ortSession.run(feeds);
    const elapsed = (performance.now() - t0).toFixed(1);

    // Extract raw logits from model output
    const outputTensor = results.output || results[ortSession.outputNames[0]];
    const rawLogits = Array.from(outputTensor.data);

    // Apply Softmax
    const probabilities = softmax(rawLogits);

    // Update UI
    updatePredictionUI(probabilities, elapsed);

  } catch (err) {
    console.error('Inference error:', err);
    inferenceTimeTag.textContent = 'Lỗi suy luận';
  } finally {
    isInferring = false;
  }
}

// ─────────────────────────────────────────────────────────────
// UI Results Update
// ─────────────────────────────────────────────────────────────
function updatePredictionUI(probabilities, elapsedMs) {
  inferenceTimeTag.textContent = `${elapsedMs} ms`;

  // Find Top-1 Class
  let maxIdx = 0;
  let maxProb = probabilities[0];
  for (let i = 1; i < probabilities.length; i++) {
    if (probabilities[i] > maxProb) {
      maxProb = probabilities[i];
      maxIdx = i;
    }
  }

  const topClass = CIFAR_CLASSES[maxIdx];
  const confidencePercent = (maxProb * 100).toFixed(2);

  // Update Top-1 Card
  topClassEmoji.textContent = topClass.emoji;
  topClassVi.textContent = topClass.vi;
  topClassEn.textContent = topClass.en;
  topConfidenceVal.textContent = `${confidencePercent}%`;
  topConfidenceFill.style.width = `${confidencePercent}%`;

  // Update 10-Class Probability Bars
  CIFAR_CLASSES.forEach((cls, idx) => {
    const prob = probabilities[idx];
    const pct = (prob * 100).toFixed(1);

    const row = document.getElementById(`prob-row-${cls.id}`);
    const bar = document.getElementById(`prob-bar-${cls.id}`);
    const pctLabel = document.getElementById(`prob-pct-${cls.id}`);

    if (row && bar && pctLabel) {
      bar.style.width = `${pct}%`;
      pctLabel.textContent = `${pct}%`;

      if (idx === maxIdx) {
        row.classList.add('top-rank');
      } else {
        row.classList.remove('top-rank');
      }
    }
  });
}
