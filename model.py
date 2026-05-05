import torch
import yaml
from ml_collections import ConfigDict
from bs_roformer import get_model_from_config

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

if __name__ == "__main__":
    CONFIG_FILE = "models\\roformer-model-bs-roformer-sw-by-jarredou\\BS-Rofo-SW-Fixed.yaml" 
    CHECKPOINT_FILE = "models\\roformer-model-bs-roformer-sw-by-jarredou\\BS-Rofo-SW-Fixed.ckpt" 
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        model, config = load_bs_roformer(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)
        
        # Audio parameters from YAML [cite: 67, 72]
        batch_size = 1
        num_channels = config.audio.num_channels   # 2
        chunk_size = config.audio.chunk_size       # 588800
        n_fft = config.model.stft_n_fft            # 2048
        hop_length = config.model.stft_hop_length   # 512
        win_length = config.model.stft_win_length   # 2048

        # 1. Create Dummy Input
        print(f"[*] Input waveform: {batch_size}x{num_channels}x{chunk_size}")
        dummy_input = torch.randn(batch_size, num_channels, chunk_size).to(DEVICE)

        with torch.no_grad():
            # 2. STEP 1: Manual STFT 
            # Output: (Batch, Channels, Freqs, Time) Complex
            print("[*] STEP 1: Calculating STFT...")
            audio_reshaped = dummy_input.reshape(-1, chunk_size)
            spec = torch.stft(
                audio_reshaped,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=torch.hann_window(win_length).to(DEVICE),
                center=True,
                pad_mode='reflect',
                normalized=config.model.stft_normalized,
                onesided=True,
                return_complex=True
            )
            spec = spec.reshape(batch_size, num_channels, spec.shape[-2], spec.shape[-1])
            print(f"[+] Complex Spectrogram shape: {spec.shape}")

            # 3. STEP 2: Feature Engineering for BandSplit 
            # Convert Complex to Real: (B, C, F, T) -> (B, C, F, T, 2)
            spec_real = torch.view_as_real(spec)
            
            # Permute to (Batch, Time, Channels, Freq, Real/Imag)
            spec_real = spec_real.permute(0, 3, 1, 2, 4)
            
            # Flatten into (Batch, Time, 4100) where 4100 = 2 channels * 1025 freqs * 2
            x_in = spec_real.reshape(batch_size, spec.shape[-1], -1)
            print(f"[+] Reshaped input for BandSplit: {x_in.shape}")

            # 4. STEP 3: Run BandSplit Encoder [cite: 51, 52]
            print("[*] STEP 2: Running BandSplit Encoder...")
            # FIXED: Removed the extra unpacking variable as your model returns only x
            x = model.band_split(x_in)
            print(f"[+] Transformer latent input shape: {x.shape}")

            # 5. STEP 4: Isolate and Run First Transformer Block 
            # model.layers[0] is the first of 12 layers; index 0 is the first sub-transformer.
            print("[*] STEP 3: Running Isolated Transformer Block...")
            first_block = model.layers[0][0] 
            block_output = first_block(x)
            
            print(f"[+] Block Output Shape: {block_output.shape}")
            
            if block_output.shape == x.shape:
                print("[*] SUCCESS: Block 0 executed and preserved sequence dimensions.")

    except Exception as e:
        import traceback
        print(f"[!] Execution failed: {e}")
        traceback.print_exc()