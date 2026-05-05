import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from frequency_transformer import load_bs_roformer

STEMS = ['vocals', 'drums', 'bass', 'guitar', 'other']

class StemDecoder(nn.Module):
    def __init__(self, config_path, checkpoint_path, device='cpu'):
        super().__init__()
        model, _ = load_bs_roformer(config_path, checkpoint_path, device=device)

        # Reuse first 5 of the pretrained MaskEstimators, one per stem
        self.heads = nn.ModuleList([
            copy.deepcopy(model.mask_estimators[i]) for i in range(5)
        ])
        self.stft_kwargs    = dict(model.stft_kwargs)
        self.freq_pad       = model.freq_pad
        self.audio_channels = model.audio_channels
        self.stft_window_fn = model.stft_window_fn

        del model

    def forward(self, x, stft_repr):
        """
        x:         (batch, time, bands, 256)        - fused features
        stft_repr: (batch, freq*channels, time, 2)  - real-view STFT from encoder
        Returns:   (batch, 5, channels, samples)
        """
        device = x.device

        # Each head: (b, t, bands, 256) -> (b, t, 4100)
        masks = torch.stack([head(x) for head in self.heads], dim=1)  # (b, 5, t, 4100)
        masks = rearrange(masks, 'b n t (f c) -> b n f t c', c=2)     # (b, 5, 2050, t, 2)

        stft = rearrange(stft_repr, 'b f t c -> b 1 f t c')
        stft  = torch.view_as_complex(stft.contiguous())   # (b, 1, 2050, t)
        masks = torch.view_as_complex(masks.contiguous())  # (b, 5, 2050, t)
        stft  = stft * masks                               # (b, 5, 2050, t) complex

        stft_window = self.stft_window_fn(device=device)
        stft = rearrange(stft, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)
        stft = F.pad(stft, (0, 0, *self.freq_pad))

        audio = torch.istft(stft, **self.stft_kwargs, window=stft_window, return_complex=False)
        audio = rearrange(audio, '(b n s) t -> b n s t', s=self.audio_channels, n=len(self.heads))

        return audio  # (b, 5, channels, samples)


if __name__ == "__main__":
    CONFIG_FILE     = "weights\\roformer-model-bs-roformer-sw-by-jarredou\\BS-Rofo-SW-Fixed.yaml"
    CHECKPOINT_FILE = "weights\\roformer-model-bs-roformer-sw-by-jarredou\\BS-Rofo-SW-Fixed.ckpt"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Shapes from the established pipeline:
    #   BandSplit output: (batch, time=1151, bands=62, dim=256)
    #   STFT repr:        (batch, freq*channels=2050, time=1151, 2)
    batch, T, bands, dim = 1, 1151, 62, 256
    freq_ch = 2050  # 1025 freq bins * 2 stereo channels

    dummy_fused = torch.randn(batch, T, bands, dim).to(DEVICE)
    dummy_stft  = torch.randn(batch, freq_ch, T, 2).to(DEVICE)

    try:
        decoder = StemDecoder(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)
        decoder.eval()

        with torch.no_grad():
            audio = decoder(dummy_fused, dummy_stft)

        print(f"[+] Fused input:  {dummy_fused.shape}")
        print(f"[+] STFT input:   {dummy_stft.shape}")
        print(f"[+] Audio output: {audio.shape}  (batch, stems=5, channels=2, samples)")
        assert audio.shape[0] == batch
        assert audio.shape[1] == 5
        assert audio.shape[2] == 2
        print("[*] SUCCESS: StemDecoder works correctly.")

    except Exception as e:
        import traceback
        print(f"[!] Execution failed: {e}")
        traceback.print_exc()
