// AudioContext performs device-rate conversion to 16 kHz. Transfer PCM buffers,
// avoiding base64, WAV headers, main-thread sample loops, and recording to disk.
class MicCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.packet = new ArrayBuffer(6400);
    this.view = new DataView(this.packet);
    this.used = 0;
    this.active = true;
    this.port.onmessage = ({data}) => {
      if (data === 'finish') {
        this.active = false;
        if (this.used) this.emit();
        this.port.postMessage({type: 'flushed'});
      }
    };
  }
  emit() {
    const packet = this.packet.slice(0, this.used * 2);
    this.port.postMessage({type: 'audio', packet}, [packet]);
    this.used = 0;
  }
  process(inputs) {
    const channels = inputs[0];
    if (!this.active || !channels?.length) return true;
    for (let i = 0; i < channels[0].length; i++) {
      let sample = 0;
      for (const channel of channels) sample += channel[i];
      sample = Math.max(-1, Math.min(1, sample / channels.length));
      this.view.setInt16(this.used++ * 2, Math.round(sample < 0 ? sample * 32768 : sample * 32767), true);
      if (this.used === 3200) this.emit();
    }
    return true;
  }
}
registerProcessor('mic-capture', MicCapture);
