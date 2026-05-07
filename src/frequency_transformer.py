import copy
import torch
import yaml
from ml_collections import ConfigDict
from bs_roformer import get_model_from_config
import torch.nn as nn

# 1. Custom Loader to handle !!python/tuple in the YAML
class SafeLoaderWithTuple(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))

SafeLoaderWithTuple.add_constructor(
    'tag:yaml.org,2002:python/tuple', 
    SafeLoaderWithTuple.construct_python_tuple
)

def load_bs_roformer(config_path, checkpoint_path, device='cpu'):
    """Loads the BS-Roformer model with the provided config and weights."""
    print(f"[*] Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        raw_config = yaml.load(f, Loader=SafeLoaderWithTuple)
    
    config = ConfigDict(raw_config)
    
    print(f"[*] Initializing model architecture...")
    model = get_model_from_config("bs_roformer", config)
    
    print(f"[*] Loading weights from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print("[+] Model loaded successfully.")
    return model, config

class FrequencyTransformer(nn.Module):
    def __init__(self, config_path, checkpoint_path, device='cpu'):
        super().__init__()
        model, _ = load_bs_roformer(config_path, checkpoint_path, device=device)
        # layers[0][0] is the frequency-dimension Transformer from the first BSRoformer layer.
        # Deepcopy so the full 12-layer model can be GC'd.
        self.transformer = copy.deepcopy(model.layers[0][1])
        self.transformer.to(device)
        del model

    def forward(self, x):
        # x: (batch, time, bands, dim=256)
        # The Transformer block expects (batch, seq, dim), so merge batch and time dims
        b, t, f, d = x.shape
        x = x.reshape(b * t, f, d)
        x, *_ = self.transformer(x)
        return x.reshape(b, t, f, d)


if __name__ == "__main__":
    CONFIG_FILE = "weights\\roformer-model-bs-roformer-sw-by-jarredou\\BS-Rofo-SW-Fixed.yaml"
    CHECKPOINT_FILE = "weights\\roformer-model-bs-roformer-sw-by-jarredou\\BS-Rofo-SW-Fixed.ckpt"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        # Load full model to build the BandSplit input pipeline
        full_model, config = load_bs_roformer(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)

        batch_size   = 1
        num_channels = config.audio.num_channels    # 2
        chunk_size   = config.audio.chunk_size      # 588800
        n_fft        = config.model.stft_n_fft      # 2048
        hop_length   = config.model.stft_hop_length # 512
        win_length   = config.model.stft_win_length # 2048

        dummy_input = torch.randn(batch_size, num_channels, chunk_size).to(DEVICE)
        print(f"[*] Dummy waveform shape: {dummy_input.shape}")

        with torch.no_grad():
            audio_reshaped = dummy_input.reshape(-1, chunk_size)
            spec = torch.stft(
                audio_reshaped,
                n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                window=torch.hann_window(win_length).to(DEVICE),
                center=True, pad_mode='reflect',
                normalized=config.model.stft_normalized,
                onesided=True, return_complex=True
            )
            spec = spec.reshape(batch_size, num_channels, spec.shape[-2], spec.shape[-1])
            x_in = torch.view_as_real(spec).permute(0, 3, 1, 2, 4)
            x_in = x_in.reshape(batch_size, spec.shape[-1], -1)

            x = full_model.band_split(x_in)
            print(f"[+] BandSplit output (FrequencyTransformer input) shape: {x.shape}")

        # Instantiate FrequencyTransformer with pretrained weights
        freq_transformer = FrequencyTransformer(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)
        freq_transformer.eval()

        with torch.no_grad():
            output = freq_transformer(x)

        print(f"[+] FrequencyTransformer output shape: {output.shape}")
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
        print("[*] SUCCESS: FrequencyTransformer with pretrained weights works correctly.")

    except Exception as e:
        import traceback
        print(f"[!] Execution failed: {e}")
        traceback.print_exc()
