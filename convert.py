#!/usr/bin/env python3
"""Convert using one source utterance and multiple target utterances."""

import warnings
from datetime import datetime
from pathlib import Path
from copy import deepcopy
from shutil import copyfile

import torch
from torch.nn.functional import sigmoid
import numpy as np
import soundfile as sf
from jsonargparse import ArgumentParser, ActionConfigFile

import sox

from data import load_wav, log_mel_spectrogram, plot_mel, plot_attn
from models import load_pretrained_wav2vec


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("source_path", type=str)
    parser.add_argument("target_paths", type=str, nargs="+")
    parser.add_argument("-w", "--wav2vec_path", type=str, required=True)
    parser.add_argument("-c", "--ckpt_path", type=str, required=True)
    parser.add_argument("-v", "--vocoder_path", type=str, required=True)
    parser.add_argument("-o", "--output_path", type=str, default="output.wav")

    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--preemph", type=float, default=0.97)
    parser.add_argument("--hop_len", type=int, default=326)
    parser.add_argument("--win_len", type=int, default=1304)
    parser.add_argument("--n_fft", type=int, default=1304)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--f_min", type=int, default=80)
    parser.add_argument("--audio_config", action=ActionConfigFile)

    return vars(parser.parse_args())


def main(
    source_path,
    target_paths,
    wav2vec_path,
    ckpt_path,
    vocoder_path,
    output_path,
    sample_rate,
    preemph,
    hop_len,
    win_len,
    n_fft,
    n_mels,
    f_min,
    **kwargs,
):
    """Main function."""

    begin_time = step_moment = datetime.now()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wav2vec = load_pretrained_wav2vec(wav2vec_path).to(device)
    print("[INFO] Wav2Vec is loaded from", wav2vec_path)

    model = torch.jit.load(ckpt_path).to(device).eval()
    print("[INFO] FragmentVC is loaded from", ckpt_path)

    vocoder = torch.jit.load(vocoder_path).to(device).eval()
    print("[INFO] Vocoder is loaded from", vocoder_path)

    elaspe_time = datetime.now() - step_moment
    step_moment = datetime.now()
    print("[INFO] elasped time", elaspe_time.total_seconds())

    tfm = sox.Transformer()
    tfm.vad(location=1)
    tfm.vad(location=-1)

    src_wav = load_wav(source_path, sample_rate)
    src_wav = deepcopy(tfm.build_array(
        input_array=src_wav, sample_rate_in=sample_rate))
    src_mel = log_mel_spectrogram(
        src_wav, preemph, sample_rate, n_mels, n_fft, hop_len, win_len, f_min
    )
    src_wav = torch.FloatTensor(src_wav).unsqueeze(0).to(device)
    print("[INFO] source waveform shape:", src_wav.shape)

    tgt_mels = []
    copyfile(source_path, "./src.wav")
    copyfile(target_paths[0], "./tgt.wav")
    for tgt_path in target_paths:
        tgt_wav = load_wav(tgt_path, sample_rate)
        tgt_wav = tfm.build_array(
            input_array=tgt_wav, sample_rate_in=sample_rate)
        tgt_wav = deepcopy(tgt_wav)
        tgt_mel = log_mel_spectrogram(
            tgt_wav, preemph, sample_rate, n_mels, n_fft, hop_len, win_len, f_min
        )
        tgt_mels.append(tgt_mel)

    ref_mel = np.concatenate(tgt_mels[1:], axis=0)
    ref_mel = torch.FloatTensor(ref_mel.T).unsqueeze(0).to(device)

    with torch.no_grad():
        src_feat = wav2vec.extract_features(src_wav, None)[0]
        print("[INFO] source Wav2Vec feature shape:", src_feat.shape)

        elaspe_time = datetime.now() - step_moment
        step_moment = datetime.now()
        print("[INFO] elasped time", elaspe_time.total_seconds())

        # tgt = torch.FloatTensor(tgt_mels[0].T).unsqueeze(0).to(device)
        # _, out_post, attn, attns, _ = model(src_feat, ref_mel, tgt)

        tgt = torch.zeros(1, 80, 1).to(device)
        for _ in range(1000):
            out, out_post, attn, attns, stop_pred = model(
                src_feat, ref_mel, tgt)
            if tgt is None:
                tgt = out
            else:
                tgt = torch.cat((tgt, out[:, :, -1:].detach()), 2)
            if torch.sum(sigmoid(stop_pred[0][-1]) > 0.5):
                print("[INFO] stop by stop token")
                break
        out_mel = out_post.transpose(1, 2).squeeze(0)
        print("[INFO] converted spectrogram shape:", out_mel.shape)
        attns = [attn] + attns
        elaspe_time = datetime.now() - step_moment
        step_moment = datetime.now()
        print("[INFO] elasped time", elaspe_time.total_seconds())

        out_wav = vocoder.generate([out_mel])[0]
        out_wav = out_wav.cpu().numpy()
        print("[INFO] generated waveform shape:", out_wav.shape)

        elaspe_time = datetime.now() - step_moment
        step_moment = datetime.now()
        print("[INFO] elasped time", elaspe_time.total_seconds())

    wav_path = Path(output_path)
    sf.write(wav_path, out_wav, sample_rate)
    print("[INFO] generated waveform is saved to", wav_path)

    mel_path = wav_path.with_suffix(".mel.png")
    plot_mel(out_mel, filename=mel_path)
    mel_path = wav_path.with_suffix(".src.mel.png")
    plot_mel(torch.from_numpy(src_mel), filename=mel_path)
    mel_path = wav_path.with_suffix(".tgt.mel.png")
    plot_mel(torch.from_numpy(tgt_mels[0]), filename=mel_path)
    print("[INFO] mel-spectrogram plot is saved to", mel_path)

    attn_path = wav_path.with_suffix(".attn.png")
    plot_attn(attns, filename=attn_path)
    print("[INFO] attention plot is saved to", attn_path)

    elaspe_time = datetime.now() - begin_time
    print("[INFO] Overall elasped time", elaspe_time.total_seconds())


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main(**parse_args())
