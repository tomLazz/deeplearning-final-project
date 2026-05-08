import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparationLoss(nn.Module):
    def __init__(self, n_fft=2048, hop_length=512, win_length=2048):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, pred, target):

        b, stems, channels, samples = pred.shape

        pred_flat = pred.reshape(b * stems * channels, samples)
        target_flat = target.reshape(b * stems * channels, samples)

        window = torch.hann_window(self.win_length, device=pred.device)

        pred_stft = torch.stft(
            pred_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True
        )

        target_stft = torch.stft(
            target_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True
        )

        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)

        spectral_loss = F.l1_loss(pred_mag, target_mag)

        return spectral_loss
