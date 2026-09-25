"""Real-model WebSocket regression: pip install requests websockets.
Run: python tests/live_smoke.py --url http://127.0.0.1:3000 --audio speech.wav
The audio must be mono PCM16 at 16 kHz; no physical microphone is accessed.
"""
import argparse
import asyncio
import json
import time
import wave

import requests
import websockets

async def run(url, audio):
    with wave.open(audio, 'rb') as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
        pcm = wav.readframes(wav.getnframes())
    with open(audio, 'rb') as file:
        response = requests.post(url + '/transcribe', files={'file': file}, data={'mode': 'streaming'}, timeout=180)
    response.raise_for_status()
    reference = response.json()['greedy']
    ws_url = url.replace('http://', 'ws://').replace('https://', 'wss://') + '/stream'
    results = []
    for packet_size in [6400, 638]:
        started = time.monotonic()
        partials = []
        async with websockets.connect(ws_url, max_size=2**20) as ws:
            assert json.loads(await ws.recv())['type'] == 'ready'
            for offset in range(0, len(pcm), packet_size):
                await ws.send(pcm[offset:offset + packet_size])
                message = json.loads(await ws.recv())
                assert message['type'] == 'ack', message
                if 'text' in message:
                    partials.append(message['text'])
            await ws.send(json.dumps({'type': 'finish'}))
            final = json.loads(await ws.recv())
            assert final['type'] == 'final', final
            assert final['text'] == reference, (final, reference)
            assert final['audio_ms'] == len(pcm) // 2 * 1000 // 16000
        results.append({'packet_bytes': packet_size, 'partials': len(partials), 'elapsed_s': time.monotonic() - started})
    # Caches must be isolated, and closing one session must free its slot.
    async with websockets.connect(ws_url) as first, websockets.connect(ws_url) as second:
        await first.recv()
        await second.recv()
        try:
            async with websockets.connect(ws_url):
                raise AssertionError('Unbounded third session accepted')
        except websockets.exceptions.InvalidStatus as error:
            assert error.response.status_code == 503
        await first.send(pcm[:6400])
        assert json.loads(await first.recv())['type'] == 'ack'
        await second.send('{"type":"finish"}')
        second_final = json.loads(await second.recv())
        assert second_final['text'] == '' and second_final['audio_ms'] == 0
        await first.send('{"type":"finish"}')
        assert json.loads(await first.recv())['audio_ms'] == min(6400, len(pcm)) // 32
    for invalid in [b'\x00', '{"type":"unknown"}']:
        async with websockets.connect(ws_url) as ws:
            await ws.recv()
            await ws.send(invalid)
            assert json.loads(await ws.recv())['type'] == 'error'
    async with websockets.connect(ws_url) as ws:
        await ws.recv()
        await ws.send('{"type": "finish"}')
        final = json.loads(await ws.recv())
        assert final['text'] == '' and final['audio_ms'] == 0
    try:
        async with websockets.connect(ws_url, origin='https://unrelated.example'):
            raise AssertionError('Cross-origin request was accepted')
    except websockets.exceptions.InvalidStatus as error:
        assert error.response.status_code == 403
    print(json.dumps({'reference': reference, 'audio_s': len(pcm) / 32000, 'runs': results, 'malformed_packets': 'passed', 'empty_finish': 'passed', 'origin_check': 'passed', 'session_isolation_and_limit': 'passed'}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:3000')
    parser.add_argument('--audio', required=True)
    args = parser.parse_args()
    asyncio.run(run(args.url.rstrip('/'), args.audio))
