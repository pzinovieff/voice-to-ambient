#!/usr/bin/env python3
"""
Local MusicGen generation script.
Takes an input audio file + text prompt, outputs a generated WAV file.
Uses Meta's audiocraft library (same model as Replicate's MusicGen API).
"""

import sys
import os
import json
import argparse
import torch
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# Cache the model globally so repeated calls don't reload weights
_model = None

def get_model(model_name='melody'):
    global _model
    if _model is None:
        print(f'[MusicGen] Loading model: {model_name}...', file=sys.stderr)
        _model = MusicGen.get_pretrained(f'facebook/musicgen-{model_name}')
        print('[MusicGen] Model loaded.', file=sys.stderr)
    return _model

def generate(input_audio_path, output_path, prompt, duration=30):
    # Select device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print(f'[MusicGen] Device: {device}', file=sys.stderr)

    model = get_model('melody')
    model.set_generation_params(duration=duration)

    # Convert input to WAV if needed (torchaudio can't read webm)
    wav_input_path = input_audio_path
    if not input_audio_path.endswith('.wav'):
        import av
        import numpy as np
        wav_input_path = input_audio_path.rsplit('.', 1)[0] + '_converted.wav'
        container = av.open(input_audio_path)
        audio_stream = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(format='s16', layout='mono', rate=32000)
        samples = []
        for frame in container.decode(audio_stream):
            frame = resampler.resample(frame)[0]
            arr = frame.to_ndarray().flatten()
            samples.append(arr)
        container.close()
        all_samples = np.concatenate(samples)
        # Save as WAV
        tensor = torch.from_numpy(all_samples).float() / 32768.0  # s16 -> float
        torchaudio.save(wav_input_path, tensor.unsqueeze(0), 32000)
        print(f'[MusicGen] Converted input to WAV: {wav_input_path}', file=sys.stderr)

    # Load the input audio (voice recording) for melody conditioning
    melody, sr = torchaudio.load(wav_input_path)

    # Generate music conditioned on the melody
    print(f'[MusicGen] Generating {duration}s with prompt: "{prompt[:60]}..."', file=sys.stderr)
    wav = model.generate_with_chroma(
        descriptions=[prompt],
        melody_wavs=melody.unsqueeze(0),
        melody_sample_rate=sr,
        progress=True,
    )

    # Save output
    # audio_write expects (batch, channels, samples) -> we take first batch item
    audio_write(
        output_path,  # audio_write appends .wav automatically
        wav[0].cpu(),
        model.sample_rate,
        strategy='peak',
        loudness_compressor=True,
    )

    final_path = output_path + '.wav'
    print(f'[MusicGen] Saved to {final_path}', file=sys.stderr)

    # Output the path as JSON for the Node server to read
    print(json.dumps({'path': final_path, 'duration': duration, 'sample_rate': model.sample_rate}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate music with MusicGen')
    parser.add_argument('--input', required=True, help='Path to input audio file')
    parser.add_argument('--output', required=True, help='Output path (without .wav extension)')
    parser.add_argument('--prompt', required=True, help='Text description of desired music')
    parser.add_argument('--duration', type=int, default=30, help='Duration in seconds')
    args = parser.parse_args()

    generate(args.input, args.output, args.prompt, args.duration)
