const recordBtn = document.getElementById('record-btn');
const stopBtn = document.getElementById('stop-btn');
const saveBtn = document.getElementById('save-btn');
const transcribeBtn = document.getElementById('transcribe-btn');
const player = document.getElementById('player');
const clipDurationEl = document.getElementById('clip-duration');
const transcriptEl = document.getElementById('transcript');
const wordTableBody = document.querySelector('#word-table tbody');
const statusEl = document.getElementById('health-status');
const logEl = document.getElementById('log');
const hotkeyIndicators = {
  dictate: document.querySelector('[data-hotkey="dictate"]'),
};

const API_BASE = `${window.location.origin}/api`;

let mediaRecorder;
let mediaStream;
let audioChunks = [];
let wavBlob = null;
let currentRecordingId = null;
let latestDuration = 0;
let healthCheckTimer;
const hotkeyState = { dictate: false };
const pressedCodes = new Set();
const controlCodes = ['ControlLeft', 'ControlRight'];
const altCodes = ['AltLeft', 'AltRight'];
const altGrCodes = ['AltRight'];
const triggerCodes = ['KeyP'];

init();

function init() {
  attachListeners();
  runHealthCheck();
  healthCheckTimer = setInterval(runHealthCheck, 10000);
}

function attachListeners() {
  recordBtn.addEventListener('click', startRecording);
  stopBtn.addEventListener('click', stopRecording);
  saveBtn.addEventListener('click', saveRecording);
  transcribeBtn.addEventListener('click', transcribeRecording);
  window.addEventListener('keydown', handleKeyDown);
  window.addEventListener('keyup', handleKeyUp);
  window.addEventListener('blur', resetHotkeys);
}

async function startRecording() {
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('Your browser does not support audio capture.');
    }

    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: 48000,
        noiseSuppression: true,
        echoCancellation: true,
      },
      video: false,
    });

    const mimeType = getSupportedMimeType();
    const options = mimeType ? { mimeType } : undefined;
    mediaRecorder = new MediaRecorder(mediaStream, options);
    audioChunks = [];

    mediaRecorder.addEventListener('dataavailable', (event) => {
      if (event.data && event.data.size > 0) {
        audioChunks.push(event.data);
      }
    });

    mediaRecorder.addEventListener('stop', handleRecordingStop);

    mediaRecorder.start();
    setStatus('recording', 'Recording…');
    logEvent('Recording started');
    recordBtn.disabled = true;
    stopBtn.disabled = false;
    saveBtn.disabled = true;
    transcribeBtn.disabled = true;
    wavBlob = null;
    currentRecordingId = null;
    clipDurationEl.textContent = '';
    player.removeAttribute('src');
  } catch (error) {
    handleError(error);
  }
}

function getSupportedMimeType() {
  const candidates = [
    'audio/webm;codecs=opus',
    'audio/ogg;codecs=opus',
    'audio/webm',
  ];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || '';
}

function stopRecording() {
  try {
    if (!mediaRecorder || mediaRecorder.state !== 'recording') {
      return;
    }
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
    recordBtn.disabled = false;
    stopBtn.disabled = true;
    setStatus('processing', 'Processing clip…');
    logEvent('Recording stopped, processing audio…');
  } catch (error) {
    handleError(error);
  }
}

async function handleRecordingStop() {
  try {
    if (!audioChunks.length) {
      throw new Error('No audio data captured. Try recording again.');
    }

    const rawBlob = new Blob(audioChunks, { type: mediaRecorder?.mimeType || 'audio/webm' });
    const { blob: convertedBlob, duration } = await convertToWav(rawBlob, 16000);
    wavBlob = convertedBlob;
    latestDuration = duration;

    if (player.src) {
      URL.revokeObjectURL(player.src);
    }
    player.src = URL.createObjectURL(wavBlob);
    player.load();
    clipDurationEl.textContent = `Duration: ${duration.toFixed(2)} s`;

    mediaRecorder = null;
    saveBtn.disabled = false;
    setStatus('ready', 'Ready to save clip');
    logEvent(`Clip ready (${duration.toFixed(2)}s)`);
  } catch (error) {
    handleError(error);
  }
}

async function convertToWav(blob, targetSampleRate) {
  const arrayBuffer = await blob.arrayBuffer();
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer.slice(0));
  const duration = audioBuffer.duration;

  const renderBuffer = await resampleToMono(audioBuffer, targetSampleRate);
  await audioCtx.close();

  const pcm = renderBuffer.getChannelData(0);
  const wavBuffer = encodeWav(pcm, targetSampleRate);
  const wavBlob = new Blob([wavBuffer], { type: 'audio/wav' });
  return { blob: wavBlob, duration };
}

async function resampleToMono(audioBuffer, targetSampleRate) {
  const offline = new OfflineAudioContext(1, Math.ceil(audioBuffer.duration * targetSampleRate), targetSampleRate);
  const source = offline.createBufferSource();
  const monoBuffer = offline.createBuffer(1, audioBuffer.length, audioBuffer.sampleRate);

  const channelData = monoBuffer.getChannelData(0);
  const tmp = new Float32Array(audioBuffer.length);

  for (let channel = 0; channel < audioBuffer.numberOfChannels; channel += 1) {
    audioBuffer.copyFromChannel(tmp, channel);
    for (let i = 0; i < tmp.length; i += 1) {
      channelData[i] += tmp[i] / audioBuffer.numberOfChannels;
    }
  }

  source.buffer = monoBuffer;
  source.connect(offline.destination);
  source.start(0);
  return offline.startRendering();
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  let offset = 0;

  writeString(view, offset, 'RIFF'); offset += 4;
  view.setUint32(offset, 36 + samples.length * 2, true); offset += 4;
  writeString(view, offset, 'WAVE'); offset += 4;
  writeString(view, offset, 'fmt '); offset += 4;
  view.setUint32(offset, 16, true); offset += 4; // PCM chunk size
  view.setUint16(offset, 1, true); offset += 2; // PCM format
  view.setUint16(offset, 1, true); offset += 2; // mono channel
  view.setUint32(offset, sampleRate, true); offset += 4;
  view.setUint32(offset, sampleRate * 2, true); offset += 4; // byte rate
  view.setUint16(offset, 2, true); offset += 2; // block align
  view.setUint16(offset, 16, true); offset += 2; // bits per sample
  writeString(view, offset, 'data'); offset += 4;
  view.setUint32(offset, samples.length * 2, true); offset += 4;

  const volume = 0x7fff;
  for (let i = 0; i < samples.length; i += 1) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * volume, true);
    offset += 2;
  }

  return buffer;
}

function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i += 1) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

function handleKeyDown(event) {
  pressedCodes.add(event.code);
  if (event.key === 'AltGraph') {
    pressedCodes.add('AltGraph');
  }

  if (isDictateComboActive()) {
    activateHotkey('dictate');
  }
}

function handleKeyUp(event) {
  pressedCodes.delete(event.code);
  if (event.key === 'AltGraph') {
    pressedCodes.delete('AltGraph');
  }

  if (!isDictateComboActive()) {
    deactivateHotkey('dictate');
  }
}

function isDictateComboActive() {
  const hasAltGr = altGrCodes.some((code) => pressedCodes.has(code)) || pressedCodes.has('AltGraph');
  const hasCtrlAlt =
    controlCodes.some((code) => pressedCodes.has(code)) && altCodes.some((code) => pressedCodes.has(code));
  const hasTrigger = triggerCodes.some((code) => pressedCodes.has(code));

  return (hasAltGr || hasCtrlAlt) && hasTrigger;
}

function activateHotkey(type) {
  if (!hotkeyIndicators[type] || hotkeyState[type]) {
    return;
  }
  hotkeyIndicators[type].classList.add('active');
  hotkeyState[type] = true;
}

function deactivateHotkey(type) {
  if (!hotkeyIndicators[type] || !hotkeyState[type]) {
    return;
  }
  hotkeyIndicators[type].classList.remove('active');
  hotkeyState[type] = false;
}

function resetHotkeys() {
  deactivateHotkey('dictate');
}

async function saveRecording() {
  try {
    if (!wavBlob) {
      throw new Error('Record something first.');
    }

    setStatus('uploading', 'Uploading clip…');
    logEvent('Uploading clip to backend…');

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const formData = new FormData();
    formData.append('file', wavBlob, `clip-${timestamp}.wav`);

    const response = await fetch(`${API_BASE}/recordings`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const message = await response.text();
      throw new Error(message || 'Failed to save recording.');
    }

    const data = await response.json();
    currentRecordingId = data.recording_id;
    setStatus('saved', 'Clip saved locally');
    logEvent(`Clip saved (id: ${currentRecordingId})`);
    transcribeBtn.disabled = false;
  } catch (error) {
    handleError(error);
  }
}

async function transcribeRecording() {
  try {
    if (!currentRecordingId) {
      throw new Error('Save a clip first.');
    }

    setStatus('transcribing', 'Running transcription…');
    transcribeBtn.disabled = true;
    logEvent('Transcribing clip with local model…');

    const response = await fetch(`${API_BASE}/transcribe`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ recording_id: currentRecordingId, with_timestamps: true }),
    });

    if (!response.ok) {
      const message = await response.text();
      throw new Error(message || 'Transcription failed.');
    }

    const data = await response.json();
    updateTranscript(data);
    setStatus('ready', 'Transcription complete');
    logEvent('Transcription complete');
    transcribeBtn.disabled = false;
  } catch (error) {
    handleError(error);
    transcribeBtn.disabled = false;
  }
}

function updateTranscript(payload) {
  const text = payload.text || 'No text returned.';
  transcriptEl.textContent = text;
  renderWordTable(payload.words || []);
  copyTranscriptToClipboard(text);
}

function renderWordTable(words) {
  wordTableBody.innerHTML = '';
  if (!words.length) {
    const emptyRow = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 4;
    cell.textContent = 'No word-level timestamps available.';
    emptyRow.appendChild(cell);
    wordTableBody.appendChild(emptyRow);
    return;
  }

  for (const word of words) {
    const row = document.createElement('tr');

    const textCell = document.createElement('td');
    textCell.textContent = word.word || word.text || '';
    row.appendChild(textCell);

    const startCell = document.createElement('td');
    startCell.textContent = formatTime(word.start);
    row.appendChild(startCell);

    const endCell = document.createElement('td');
    endCell.textContent = formatTime(word.end);
    row.appendChild(endCell);

    const confCell = document.createElement('td');
    confCell.textContent = word.confidence !== undefined ? `${(word.confidence * 100).toFixed(1)}%` : '—';
    row.appendChild(confCell);

    wordTableBody.appendChild(row);
  }
}

function formatTime(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '—';
  }
  return value.toFixed(2);
}

async function runHealthCheck() {
  try {
    const response = await fetch(`${API_BASE}/health`);
    const data = await response.json();
    if (response.ok && data.status === 'ok') {
      setStatus('ready', 'Ready');
    } else {
      setStatus('starting', 'Model loading…');
    }
  } catch (error) {
    setStatus('error', 'Offline');
  }
}

function setStatus(state, label) {
  statusEl.textContent = label;
  statusEl.dataset.state = state;
}

function logEvent(message) {
  const entry = document.createElement('li');
  const timestamp = new Date().toLocaleTimeString();
  entry.textContent = `[${timestamp}] ${message}`;
  logEl.prepend(entry);
  while (logEl.children.length > 50) {
    logEl.removeChild(logEl.lastChild);
  }
}

function handleError(error) {
  console.error(error);
  setStatus('error', error.message || 'Unexpected error');
  logEvent(`⚠️ ${error.message || error}`);
}

function copyTranscriptToClipboard(text) {
  if (!text || !text.trim()) {
    logEvent('Transcript empty; clipboard not updated.');
    return;
  }
  if (!navigator.clipboard || typeof navigator.clipboard.writeText !== 'function') {
    logEvent('Clipboard API unavailable; copy skipped.');
    return;
  }

  navigator.clipboard
    .writeText(text)
    .then(() => logEvent('Transcript copied to clipboard.'))
    .catch((error) => logEvent(`⚠️ Clipboard copy failed: ${error.message || error}`));
}

window.addEventListener('beforeunload', () => {
  if (healthCheckTimer) {
    clearInterval(healthCheckTimer);
  }
  if (player.src) {
    URL.revokeObjectURL(player.src);
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
  }
  resetHotkeys();
});
