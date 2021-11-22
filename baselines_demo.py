import argparse
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import cog
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.utils.data as utils
from tqdm import tqdm

from models.MMUB import MIMO_UNet_Beamforming
#from models.FaSNet import FaSNet_origin
from models.SELDNet import Seldnet_augmented
from utility_functions import (gen_submission_list_task2, load_model,
                               spectrum_fast)


class Predictor(cog.Predictor):
    def setup(self):
        """Load the model"""
        use_cuda = False
        gpu_id = 0
        if use_cuda:
            self.device = "cuda:" + str(gpu_id)
        else:
            self.device = "cpu"
        task1_pretrained_path = "pretrained/baseline_task1_checkpoint"
        task2_pretrained_path = "pretrained/baseline_task2_checkpoint"
        self.model_task1 = MIMO_UNet_Beamforming(
            fft_size=512,
            hop_size=128,
            input_channel=4)

        self.model_task2 = Seldnet_augmented(
            time_dim=2400,
            freq_dim=256,
            input_channels=4,
            output_classes=14,
            pool_size=[[8, 2], [8, 2], [2, 2], [1, 1]],
            pool_time=True,
            rnn_size=256,
            n_rnn=3,
            fc_size=1024,
            dropout_perc=0.3,
            cnn_filters=[64, 128, 256, 512],
            class_overlaps=3,
            verbose=False,
        )
        self.model_task1 = self.model_task1.to(self.device)
        self.model_task2 = self.model_task2.to(self.device)

        print("loading models")
        _ = load_model(self.model_task1, None, task1_pretrained_path, use_cuda)
        _ = load_model(self.model_task2, None, task2_pretrained_path, use_cuda)
        self.model_task1.eval()
        self.model_task2.eval()

    @cog.input(
        "input",
        type=Path,
        help="Input 1st order Ambisonics sound path"
        )
    @cog.input(
        "task", type=int, help="Task to evaluate", default=1, options=[1, 2]
    )
    @cog.input(
        "output_type",
        type=str,
        help="Can be 'data' or 'plot'",
        default="plot",
        options=["data", "plot"],
    )
    def predict(self, input, task, output_type):
        """Compute prediction"""
        # preprocessing
        sr_task1 = 16000
        sr_task2 = 32000
        input = str(input)

        if task == 1:
            x_in, sr = librosa.load(input, sr_task1, mono=False)
            x = torch.tensor(x_in).float().unsqueeze(0)
            with torch.no_grad():
                y = enhance_sound(x, self.model_task1, self.device, 76672, 0.5)
                y = np.squeeze(y)
                # write output
                if output_type == "data":
                    output_path_wav = Path(tempfile.mkdtemp()) / "output.wav"
                    output_path_mp3 = Path(tempfile.mkdtemp()) / "output.mp3"
                    sf.write(output_path_wav, y, sr_task1)
                    subprocess.check_output(
                        [
                            "ffmpeg",
                            "-i",
                            str(output_path_wav),
                            "-ab",
                            "320k",
                            str(output_path_mp3),
                        ],
                    )
                    np.save("test_input/prova_in.npx", x_in)
                    np.save("test_input/prova_out.npx", y)

                    return output_path_mp3

                elif output_type == "plot":
                    output_path_png = Path(tempfile.mkdtemp()) / "output.png"
                    plot_task1(output_path_png, x_in, y)

                    return output_path_png

        elif task == 2:
            x, sr = librosa.load(input, sr_task2, mono=False)
            x = spectrum_fast(
                x, nperseg=512, noverlap=112, window="hamming", output_phase=False
            )
            x = torch.tensor(x).float().unsqueeze(0)
            with torch.no_grad():
                sed, doa = self.model_task2(x)
            sed = sed.cpu().numpy().squeeze()
            doa = doa.cpu().numpy().squeeze()

            # write output
            if output_type == "data":
                seld = gen_submission_list_task2(
                    sed, doa, max_overlaps=3, max_loc_value=1
                )

                return seld

            elif output_type == "plot":
                output_path_png = Path(tempfile.mkdtemp()) / "output.png"
                plot_task2(output_path_png, sed, doa)

                return output_path_png


def enhance_sound(predictors, model, device, length, overlap):
    """
    Compute enhanced waveform using a trained model,
    applying a sliding crossfading window
    """

    def pad(x, d):
        # zeropad to desired length
        pad = torch.zeros((x.shape[0], x.shape[1], d))
        pad[:, :, : x.shape[-1]] = x
        return pad

    def xfade(x1, x2, fade_samps, exp=1.0):
        # simple linear/exponential crossfade and concatenation
        out = []
        fadein = np.arange(fade_samps) / fade_samps
        fadeout = np.arange(fade_samps, 0, -1) / fade_samps
        fade_in = fadein * exp
        fade_out = fadeout * exp
        x1[:, :, -fade_samps:] = x1[:, :, -fade_samps:] * fadeout
        x2[:, :, :fade_samps] = x2[:, :, :fade_samps] * fadein
        left = x1[:, :, :-fade_samps]
        center = x1[:, :, -fade_samps:] + x2[:, :, :fade_samps]
        end = x2[:, :, fade_samps:]
        return np.concatenate((left, center, end), axis=-1)

    overlap_len = int(length * overlap)  # in samples
    total_len = predictors.shape[-1]
    starts = np.arange(0, total_len, overlap_len)  # points to cut
    # iterate the sliding frames
    for i in range(len(starts)):
        start = starts[i]
        end = starts[i] + length
        if end < total_len:
            cut_x = predictors[:, :, start:end]
        else:
            # zeropad the last frame
            end = total_len
            cut_x = pad(predictors[:, :, start:end], length)

        # compute model's output
        cut_x = cut_x.to(device)
        predicted_x = model(cut_x, torch.tensor([0.0]))
        predicted_x = predicted_x.cpu().numpy()

        # reconstruct sound crossfading segments
        if i == 0:
            recon = predicted_x
        else:
            recon = xfade(recon, predicted_x, overlap_len)

    # undo final pad
    recon = recon[:, :, :total_len]

    return recon


def plot_task1(output_path, x, y):
    x = np.mean(x, axis=0)

    plt.figure(1)
    plt.suptitle("TASK 1 BASELINE MODEL", fontweight="bold")
    plt.subplot(211)
    plt.title("Noisy input (mono sum)")
    plt.specgram(x, NFFT=512, Fs=16000, mode="psd", scale="dB")
    plt.ylabel("Frequency (Hz)")
    plt.subplot(212)
    plt.title("Enhanced output")
    plt.specgram(y, NFFT=512, Fs=16000, mode="psd", scale="dB")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(output_path, format="png", dpi=300)


def plot_task2(output_path, sed, doa):
    n = sed.shape[1]
    x = doa[:, :n]
    y = doa[:, n : n * 2]
    z = doa[:, n * 2 :]
    positions = np.arange(0, sed.shape[0] + 1, 50)
    labels = np.array(positions / 10, dtype="int32")

    plt.figure(1)
    plt.suptitle("TASK 2 BASELINE MODEL", fontweight="bold")

    plt.subplot(221)
    plt.title("Sound activations")
    plt.pcolormesh(sed.T)
    plt.ylabel("Sound class")
    plt.xticks(positions, labels)
    plt.subplot(222)
    plt.title("X axis")
    plt.pcolormesh(x.T)
    plt.xticks(positions, labels)
    plt.subplot(223)
    plt.title("Y axis")
    plt.pcolormesh(y.T)
    plt.ylabel("Sound class")
    plt.xlabel("Time")
    plt.xticks(positions, labels)
    plt.subplot(224)
    plt.title("Z axis")
    plt.pcolormesh(z.T)
    plt.xticks(positions, labels)
    plt.xlabel("Time")

    plt.tight_layout()
    plt.savefig(output_path, format="png", dpi=300)
