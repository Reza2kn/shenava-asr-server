const $ = id => document.getElementById(id);
const start = $('start'), stop = $('stop'), device = $('device'), status = $('status'), transcript = $('transcript');
let stream, context, source, capture, socket, timer;
let queue = [], pending = false, finishing = false, flushed = false, finishSent = false, active = false;
let available = false;

async function devices() {
  const selected = device.value;
  const inputs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'audioinput');
  device.replaceChildren(new Option('System default microphone', ''));
  for (const input of inputs) device.add(new Option(input.label || `Microphone ${device.options.length}`, input.deviceId));
  if ([...device.options].some(o => o.value === selected)) device.value = selected;
}

function releaseMic() {
  stream?.getTracks().forEach(track => track.stop());
  stream = null;
  source?.disconnect(); capture?.disconnect();
  source = capture = null;
  if (context) { context.close().catch(() => {}); context = null; }
}
function cleanup() {
  active = false;
  clearTimeout(timer);
  releaseMic();
  if (socket) { socket.onclose = socket.onerror = socket.onmessage = null; socket.close(); socket = null; }
  queue = []; pending = false;
  start.disabled = !available; stop.disabled = true; device.disabled = false;
}
function fail(message) { cleanup(); status.textContent = message; }
function watchdog() {
  clearTimeout(timer);
  timer = setTimeout(() => fail('The server stopped responding. Recording stopped; reconnect and try again.'), 45000);
}
function pump() {
  if (pending || socket?.readyState !== WebSocket.OPEN) return;
  if (queue.length) {
    socket.send(queue.shift()); pending = true; watchdog();
  } else if (finishing && flushed && !finishSent) {
    finishSent = true; socket.send(JSON.stringify({type: 'finish'})); watchdog();
  }
}

start.onclick = async () => {
  active = true; start.disabled = true; device.disabled = true;
  finishing = flushed = finishSent = false; queue = []; pending = false;
  transcript.value = ''; $('copy').disabled = true;
  status.textContent = 'Requesting microphone access…';
  try {
    stream = await navigator.mediaDevices.getUserMedia({audio: {
      deviceId: device.value ? {exact: device.value} : undefined,
      channelCount: 1, echoCancellation: true, noiseSuppression: true,
    }});
    await devices();
    context = new AudioContext({sampleRate: 16000});
    if (context.sampleRate !== 16000) throw new Error('This browser cannot capture at 16 kHz. Try Chrome or Edge.');
    await context.audioWorklet.addModule('/mic-worklet.js');
    await context.resume();
    source = context.createMediaStreamSource(stream);
    capture = new AudioWorkletNode(context, 'mic-capture');
    capture.port.onmessage = ({data}) => {
      if (!active) return;
      if (data.type === 'audio') {
        queue.push(data.packet);
        if (queue.length > 25) { fail('The server is over 5 seconds behind. Recording stopped; queued audio was not transcribed. Try a faster backend.'); return; }
      } else if (data.type === 'flushed') {
        flushed = true; releaseMic();
      }
      pump();
    };
    status.textContent = 'Connecting to Shenava…';
    socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/stream`);
    watchdog();
    socket.onmessage = ({data}) => {
      try {
        const message = JSON.parse(data);
        if (message.type === 'error') { fail(message.message); return; }
        clearTimeout(timer);
        if (message.type === 'ready') {
          source.connect(capture); capture.connect(context.destination);
          stop.disabled = false;
          status.textContent = `Listening · ${message.backend} · text updates about every 1.12 seconds, plus processing time`;
          stream.getAudioTracks()[0].onended = () => { if (!finishing) stop.click(); };
          return;
        }
        if (typeof message.text === 'string') { transcript.value = message.text; $('copy').disabled = !message.text; }
        if (message.type === 'final') { cleanup(); status.textContent = 'Recording stopped. Transcript complete.'; return; }
        if (message.type === 'ack') { pending = false; pump(); }
      } catch (error) { fail(`Invalid server response: ${error.message}`); }
    };
    socket.onerror = () => fail('Could not open live transcription. The server may be busy or streaming is not enabled.');
    socket.onclose = () => { if (active) fail('Connection closed before the final transcript. Recording stopped.'); };
  } catch (error) { fail(error.name === 'NotAllowedError' ? 'Microphone permission was denied. Allow microphone access in your browser and try again.' : error.message); }
};
stop.onclick = () => {
  if (!active || finishing) return;
  finishing = true; stop.disabled = true; status.textContent = 'Finishing transcript…';
  capture.port.postMessage('finish'); watchdog();
};
$('copy').onclick = async () => {
  try { await navigator.clipboard.writeText(transcript.value); status.textContent = 'Transcript copied.'; }
  catch { transcript.select(); status.textContent = 'Select and copy the transcript manually.'; }
};
window.addEventListener('pagehide', cleanup);

try {
  if (!window.isSecureContext || !navigator.mediaDevices) throw new Error('Microphone access requires HTTPS or localhost. Open this server through HTTPS, or use http://localhost:3000.');
  const response = await fetch('/mic-config');
  if (!response.ok) throw new Error('Could not read microphone configuration.');
  const config = await response.json();
  available = config.available;
  status.textContent = available ? 'Ready. Press Start microphone to allow access.' : config.message;
  start.disabled = !available;
  await devices();
  navigator.mediaDevices.addEventListener('devicechange', () => devices().catch(() => {}));
} catch (error) { fail(error.message); }
