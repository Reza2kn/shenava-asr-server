// Run with node --test tests/mic-worklet.test.cjs. No browser packages required.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');

test('worklet mixes, clips, batches PCM16 LE and flushes the tail once', () => {
  let Processor;
  const messages = [];
  class AudioWorkletProcessor { constructor() { this.port = {postMessage: m => messages.push(m)}; } }
  vm.runInNewContext(fs.readFileSync('web/mic-worklet.js', 'utf8'), {
    AudioWorkletProcessor, registerProcessor: (_, p) => { Processor = p; },
  });
  const capture = new Processor();
  const a = new Float32Array(3203).fill(2);
  const b = new Float32Array(3203).fill(-0.5);
  capture.process([[a, b]]);
  assert.equal(messages.length, 1);
  assert.equal(messages[0].packet.byteLength, 6400);
  assert.equal(new DataView(messages[0].packet).getInt16(0, true), Math.round(0.75 * 32767));
  capture.port.onmessage({data: 'finish'});
  assert.equal(messages[1].packet.byteLength, 6);
  assert.equal(messages[2].type, 'flushed');
  capture.process([[a, b]]);
  assert.equal(messages.length, 3);
});
