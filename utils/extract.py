import argparse
import os

import torchaudio


def extract_clip(audio_path: str, start_time: float, clip_seconds: float):
    waveform, sr = torchaudio.load(audio_path, backend='ffmpeg')

    if start_time < 0:
        raise ValueError('start_time must be non-negative.')
    if clip_seconds <= 0:
        raise ValueError('clip_seconds must be positive.')

    start_sample = int(start_time * sr)
    clip_num_samples = int(clip_seconds * sr)
    end_sample = start_sample + clip_num_samples

    total_samples = waveform.shape[1]
    total_duration = total_samples / sr

    if start_sample >= total_samples:
        raise ValueError(
            f'start_time={start_time}s is outside audio duration ({total_duration:.3f}s).'
        )

    if end_sample > total_samples:
        raise ValueError(
            f'Requested clip [{start_time:.3f}s, {start_time + clip_seconds:.3f}s] '
            f'exceeds audio duration ({total_duration:.3f}s).'
        )

    clip = waveform[:, start_sample:end_sample]
    return clip, sr


def default_output_path(audio_path: str, start_time: float, clip_seconds: float) -> str:
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    return f'{stem}_start{start_time:.3f}_T{clip_seconds:.3f}.wav'


def main() -> None:
    parser = argparse.ArgumentParser(description='Extract a contiguous T-second audio clip from an input file.')
    parser.add_argument('--audio-path', type=str, required=True, help='Path to input audio file.')
    parser.add_argument('--start-time', type=float, required=True, help='Start time in seconds.')
    parser.add_argument('--clip-seconds', type=float, required=True, help='Length T of extracted clip in seconds.')
    parser.add_argument('--output-path', type=str, default=None, help='Path to output clip .wav file.')
    args = parser.parse_args()

    output_path = args.output_path or default_output_path(
        args.audio_path,
        args.start_time,
        args.clip_seconds,
    )

    clip, sr = extract_clip(args.audio_path, args.start_time, args.clip_seconds)
    torchaudio.save(output_path, clip, sample_rate=sr, backend='ffmpeg')

    print(f'Input file: {args.audio_path}')
    print(f'Start time: {args.start_time}s')
    print(f'Clip length (T): {args.clip_seconds}s')
    print(f'Output file: {output_path}')
    print(f'Output shape: {tuple(clip.shape)}, sample rate: {sr}')


if __name__ == '__main__':
    main()
