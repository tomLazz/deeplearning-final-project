import torch
import torch.nn as nn
from frequency_transformer import FrequencyTransformer
from time_module import TemporalBranch


class GatedFusion(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        # Projects concatenated freq+temporal features to a gate in [0,1]
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, freq, temp):
        # freq, temp: (batch, time, bands, dim)
        g = self.gate(torch.cat([freq, temp], dim=-1))
        return g * freq + (1 - g) * temp


CONFIG_FILE     = "weights\\roformer-model-bs-roformer-sw-by-jarredou\\BS-Rofo-SW-Fixed.yaml"
CHECKPOINT_FILE = "weights\\roformer-model-bs-roformer-sw-by-jarredou\\BS-Rofo-SW-Fixed.ckpt"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dummy input matching BandSplit output: (batch, time, bands, dim)
dummy = torch.randn(1, 1151, 62, 256).to(DEVICE)

if __name__ == "__main__":
    print(f"[*] Device: {DEVICE}")
    print(f"[*] Dummy input shape: {dummy.shape}\n")

    # --- FrequencyTransformer ---
    print("[*] Testing FrequencyTransformer...")
    freq_transformer = FrequencyTransformer(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)
    freq_transformer.eval()
    with torch.no_grad():
        freq_out = freq_transformer(dummy)
    assert freq_out.shape == dummy.shape, f"Shape mismatch: {freq_out.shape}"
    print(f"[+] FrequencyTransformer output: {freq_out.shape}\n")

    # --- TemporalBranch ---
    print("[*] Testing TemporalBranch...")
    temporal_branch = TemporalBranch(dim=256, kernel_size=3, num_layers=4, base_dilation=2).to(DEVICE)
    temporal_branch.eval()
    with torch.no_grad():
        temp_out = temporal_branch(dummy)
    assert temp_out.shape == dummy.shape, f"Shape mismatch: {temp_out.shape}"
    print(f"[+] TemporalBranch output: {temp_out.shape}\n")

    # --- GatedFusion ---
    print("[*] Testing GatedFusion...")
    gated_fusion = GatedFusion(dim=256).to(DEVICE)
    gated_fusion.eval()
    with torch.no_grad():
        fused = gated_fusion(freq_out, temp_out)
    assert fused.shape == dummy.shape, f"Shape mismatch: {fused.shape}"
    print(f"[+] GatedFusion output: {fused.shape}\n")

    print("[*] SUCCESS: All modules produce correct output shapes.")
