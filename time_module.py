import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(self, dim, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(dim, dim, kernel_size, dilation=dilation, padding=padding)
        self.norm = nn.GroupNorm(1, dim)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (batch, dim, time)
        return x + self.act(self.norm(self.conv(x)))


class TemporalBranch(nn.Module):
    def __init__(self, dim=256, kernel_size=3, num_layers=4, base_dilation=2):
        super().__init__()
        # Dilations: 1, base_dilation, base_dilation^2, ...
        dilations = [base_dilation ** i for i in range(num_layers)]
        self.blocks = nn.ModuleList([
            TemporalBlock(dim, kernel_size, d) for d in dilations
        ])

    def forward(self, x):
        # x: (batch, time, bands, dim)
        b, t, f, d = x.shape
        # Process each band independently over time
        x = x.permute(0, 2, 3, 1).reshape(b * f, d, t)  # (b*f, dim, time)
        for block in self.blocks:
            x = block(x)
        x = x.reshape(b, f, d, t).permute(0, 3, 1, 2)   # (b, time, bands, dim)
        return x


if __name__ == "__main__":
    branch = TemporalBranch(dim=256, kernel_size=3, num_layers=4, base_dilation=2)
    branch.eval()

    dummy = torch.randn(1, 1151, 62, 256)
    with torch.no_grad():
        out = branch(dummy)

    assert out.shape == dummy.shape, f"Shape mismatch: {out.shape} != {dummy.shape}"
    print(f"[+] Input:  {dummy.shape}")
    print(f"[+] Output: {out.shape}")
    print("[*] SUCCESS: TemporalBranch works correctly.")
