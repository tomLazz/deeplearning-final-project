import copy
import torch
import torch.nn as nn
from einops import rearrange
from frequency_transformer import load_bs_roformer
from time_module import TemporalBranch
from fusion_module import GatedFusion

STEMS = ['vocals', 'drums', 'bass', 'guitar', 'other']


class StemSeparator(nn.Module):
    def __init__(
        self,
        config_path,
        checkpoint_path,
        temporal_dim=256,
        temporal_kernel=3,
        temporal_layers=4,
        temporal_dilation=2,
        device='cpu'
    ):
        super().__init__()

        # Load BSRoformer once and extract all pretrained components
        full_model, _ = load_bs_roformer(config_path, checkpoint_path, device=device)

        # Pretrained: STFT encoder + BandSplit
        self.stft_kwargs    = dict(full_model.stft_kwargs)
        self.stft_window_fn = full_model.stft_window_fn
        self.audio_channels = full_model.audio_channels
        self.band_split     = copy.deepcopy(full_model.band_split)

        # Pretrained: frequency transformer (layers[0][1])
        self.freq_transformer = copy.deepcopy(full_model.layers[0][1])

        # Random init: temporal branch + gated fusion
        self.temporal_branch = TemporalBranch(
            dim=temporal_dim,
            kernel_size=temporal_kernel,
            num_layers=temporal_layers,
            base_dilation=temporal_dilation
        )
        self.gated_fusion = GatedFusion(dim=temporal_dim)

        # Pretrained: decoder heads (first 5 MaskEstimators)
        self.decoder_heads = nn.ModuleList([
            copy.deepcopy(full_model.mask_estimators[i]) for i in range(5)
        ])

        del full_model
        self.to(device)

    def _encode(self, audio):
        """(b, 2, samples) -> x: (b, T, 62, 256), stft_repr: (b, f*channels, T, 2)"""
        device = audio.device
        b, s, _ = audio.shape

        stft_window = self.stft_window_fn(device=device)
        stft = torch.stft(
            audio.reshape(b * s, -1),
            **self.stft_kwargs, window=stft_window, return_complex=True
        )
        stft = torch.view_as_real(stft)                                     # (b*s, f, T, 2)
        f, T = stft.shape[1], stft.shape[2]
        stft = stft.reshape(b, s, f, T, 2)                                  # (b, s, f, T, 2)
        stft_repr = rearrange(stft, 'b s f t c -> b (f s) t c')            # (b, f*s, T, 2)

        x = self.band_split(rearrange(stft_repr, 'b f t c -> b t (f c)'))  # (b, T, 62, 256)
        return x, stft_repr

    def _freq_transform(self, x):
        """(b, T, f, d) -> same shape, attention over frequency bands"""
        b, t, f, d = x.shape
        x = x.reshape(b * t, f, d)
        out = self.freq_transformer(x)
        x = out[0] if isinstance(out, tuple) else out
        return x.reshape(b, t, f, d)

    def _decode(self, fused, stft_repr):
        """(b, T, 62, 256) + stft_repr -> (b, 5, channels, samples)"""
        device = fused.device

        masks = torch.stack([head(fused) for head in self.decoder_heads], dim=1)  # (b, 5, T, 4100)
        masks = rearrange(masks, 'b n t (f c) -> b n f t c', c=2)

        stft  = torch.view_as_complex(rearrange(stft_repr, 'b f t c -> b 1 f t c').contiguous())
        masks = torch.view_as_complex(masks.contiguous())
        stft  = stft * masks  # (b, 5, f*s, T) complex

        stft_window = self.stft_window_fn(device=device)
        stft = rearrange(stft, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)

        audio = torch.istft(stft, **self.stft_kwargs, window=stft_window, return_complex=False)
        return rearrange(audio, '(b n s) t -> b n s t', s=self.audio_channels, n=5)

    def forward(self, audio):
        """
        audio: (batch, 2, samples)
        Returns: (batch, 5, 2, samples) - one stereo waveform per stem
        """
        x, stft_repr = self._encode(audio)

        freq_out = self._freq_transform(x)
        temp_out = self.temporal_branch(x)
        fused    = self.gated_fusion(freq_out, temp_out)

        return self._decode(fused, stft_repr)


if __name__ == "__main__":
    CONFIG_FILE     = "weights\\roformer-model-bs-roformer-sw-by-jarredou\\BS-Rofo-SW-Fixed.yaml"
    CHECKPOINT_FILE = "weights\\roformer-model-bs-roformer-sw-by-jarredou\\BS-Rofo-SW-Fixed.ckpt"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        model = StemSeparator(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)
        model.eval()
        print(f"[+] Model built on {DEVICE}")

        dummy_audio = torch.randn(1, 2, 588800).to(DEVICE)
        print(f"[*] Input:  {dummy_audio.shape}  (batch, channels, samples)")

        with torch.no_grad():
            stems = model(dummy_audio)

        print(f"[+] Output: {stems.shape}  (batch, stems, channels, samples)")
        assert stems.shape == (1, 5, 2, 588800), f"Unexpected shape: {stems.shape}"
        print(f"[*] Stems: {STEMS}")
        print("[*] SUCCESS: Full pipeline works end-to-end.")

    except Exception as e:
        import traceback
        print(f"[!] Execution failed: {e}")
        traceback.print_exc()
