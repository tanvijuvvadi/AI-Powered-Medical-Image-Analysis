const input = document.getElementById('imageInput');
const dropZone = document.getElementById('dropZone');
const previewWrap = document.getElementById('previewWrap');
const previewImage = document.getElementById('previewImage');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const analyzeBtn = document.getElementById('analyzeBtn');
const demoNotice = document.getElementById('demoNotice');
const resultPlaceholder = document.getElementById('resultPlaceholder');
const resultOutput = document.getElementById('resultOutput');
const resultStatus = document.getElementById('resultStatus');
let selectedFile = null;
let objectUrl = null;

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}
function setFile(file) {
  if (!file) return;
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
    demoNotice.textContent = 'Please choose a JPG, PNG, or WEBP image.';
    return;
  }
  if (file.size > 12 * 1024 * 1024) {
    demoNotice.textContent = 'Please choose an image smaller than 12 MB.';
    return;
  }
  selectedFile = file;
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  previewImage.src = objectUrl;
  fileName.textContent = file.name;
  fileSize.textContent = formatBytes(file.size);
  previewWrap.classList.remove('hidden');
  dropZone.classList.add('hidden');
  analyzeBtn.disabled = false;
  demoNotice.textContent = 'Image selected locally. No file is uploaded to a server.';
  resetResult();
}
function resetResult() {
  resultPlaceholder.classList.remove('hidden');
  resultOutput.classList.add('hidden');
  resultStatus.innerHTML = '<i></i> Awaiting image';
}
input.addEventListener('change', event => setFile(event.target.files[0]));
['dragenter', 'dragover'].forEach(type => dropZone.addEventListener(type, event => {
  event.preventDefault(); dropZone.classList.add('dragging');
}));
['dragleave', 'drop'].forEach(type => dropZone.addEventListener(type, event => {
  event.preventDefault(); dropZone.classList.remove('dragging');
}));
dropZone.addEventListener('drop', event => setFile(event.dataTransfer.files[0]));
document.getElementById('removeImage').addEventListener('click', clearImage);
document.getElementById('resetDemo').addEventListener('click', () => { clearImage(); demoNotice.textContent = 'Demo reset. Choose an image to begin.'; });
function clearImage() {
  selectedFile = null;
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = null;
  input.value = '';
  previewImage.removeAttribute('src');
  previewWrap.classList.add('hidden');
  dropZone.classList.remove('hidden');
  analyzeBtn.disabled = true;
  demoNotice.textContent = 'Upload an image to enable the demo.';
  resetResult();
}
analyzeBtn.addEventListener('click', () => {
  if (!selectedFile) return;
  // This is intentionally a UI simulation, not an inference from a trained model.
  const reader = new Image();
  reader.onload = () => {
    const confidence = Math.floor(70 + Math.random() * 25);
    const illustrativeLabels = ['Illustrative class A', 'Illustrative class B'];
    const label = illustrativeLabels[Math.floor(Math.random() * illustrativeLabels.length)];
    document.getElementById('classLabel').textContent = label;
    document.getElementById('confidenceValue').textContent = `${confidence}%`;
    document.getElementById('confidenceBar').style.width = `${confidence}%`;
    document.getElementById('imageDimensions').textContent = `${reader.naturalWidth} × ${reader.naturalHeight} px`;
    resultPlaceholder.classList.add('hidden');
    resultOutput.classList.remove('hidden');
    resultStatus.innerHTML = '<i style="background:#e4ac4b"></i> Simulated output';
    demoNotice.textContent = 'Simulation complete. The result is random and has no diagnostic meaning.';
  };
  reader.onerror = () => { demoNotice.textContent = 'Could not read this image. Please try another file.'; };
  reader.src = objectUrl;
});

const mobileMenu = document.getElementById('mobileMenu');
const sidebar = document.getElementById('sidebar');
mobileMenu.addEventListener('click', () => sidebar.classList.toggle('open'));
document.querySelectorAll('.nav-link').forEach(link => link.addEventListener('click', () => {
  document.querySelectorAll('.nav-link').forEach(item => item.classList.remove('active'));
  link.classList.add('active');
  sidebar.classList.remove('open');
}));
