import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append('/content/drive/MyDrive/CMSC472Final')
from model import StemSeparator
from training_collection import collect_stems, RemixDataset, STEMS

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
MANIFEST_PATH = '/content/manifest.csv'
RAWSTEMS_DIR  = '/content/rawstems'


# -----------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------
def spectral_loss(pred, target, n_ffts=(2048, 1024, 512)):
    b, n, c, t = pred.shape
    pred_flat   = pred.reshape(b * n * c, t)
    target_flat = target.reshape(b * n * c, t)

    loss = 0.0
    for n_fft in n_ffts:
        window = torch.hann_window(n_fft).to(pred.device)
        pred_mag = torch.stft(pred_flat,   n_fft=n_fft, hop_length=n_fft // 4,
                              window=window, return_complex=True).abs()
        tgt_mag  = torch.stft(target_flat, n_fft=n_fft, hop_length=n_fft // 4,
                              window=window, return_complex=True).abs()
        loss += F.l1_loss(pred_mag, tgt_mag)

    return loss / len(n_ffts)


# -----------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------
def _set_pretrained_frozen(model, frozen):
    for p in (list(model.band_split.parameters()) +
              list(model.freq_transformer.parameters()) +
              list(model.decoder_heads.parameters())):
        p.requires_grad = not frozen


def train(
    config_path,
    checkpoint_path,
    save_dir='/content/drive/MyDrive/CMSC472Final/checkpoints',
    epochs=50,
    warmup_epochs=5,
    batch_size=2,
    lr_pretrained=1e-5,
    lr_new=1e-4,
    lr_warmup_new=1e-6,   # LR for new components during warmup
    num_workers=0,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[*] Training on {device}  |  warmup: {warmup_epochs} epochs")
    os.makedirs(save_dir, exist_ok=True)

    model = StemSeparator(config_path, checkpoint_path, device=device)

    # Freeze pretrained components for warmup phase
    _set_pretrained_frozen(model, frozen=True)

    # Optimizer covers all params; pretrained group LR is irrelevant while frozen
    optimizer = torch.optim.AdamW([
        {'params': list(model.band_split.parameters()) +
                   list(model.freq_transformer.parameters()) +
                   list(model.decoder_heads.parameters()),
         'lr': lr_pretrained},
        {'params': list(model.temporal_branch.parameters()) +
                   list(model.gated_fusion.parameters()),
         'lr': lr_warmup_new},
    ])

    # Cosine annealing runs over the post-warmup epochs only
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs)
    )

    train_set    = RemixDataset(collect_stems(MANIFEST_PATH, split='train', rawstems_dir=RAWSTEMS_DIR))
    val_set      = RemixDataset(collect_stems(MANIFEST_PATH, split='test',  rawstems_dir=RAWSTEMS_DIR))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=(device == 'cuda'))
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=(device == 'cuda'))

    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

    best_val_loss = float('inf')
    history = {'train': [], 'val': []}

    for epoch in range(1, epochs + 1):

        # Transition from warmup to main training
        if epoch == warmup_epochs + 1:
            _set_pretrained_frozen(model, frozen=False)
            optimizer.param_groups[0]['lr'] = lr_pretrained
            optimizer.param_groups[1]['lr'] = lr_new
            print(f"[*] Warmup complete — unfreezing pretrained components")

        # ---- Train ----
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{epochs} [Train]", leave=False)
        for mix, stems in pbar:
            mix   = mix.to(device)
            stems = stems.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                pred = model(mix)
                loss = spectral_loss(pred, stems)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        train_loss /= len(train_loader)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mix, stems in tqdm(val_loader, desc=f"Epoch {epoch:3d}/{epochs} [Val]  ", leave=False):
                mix   = mix.to(device)
                stems = stems.to(device)
                with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                    val_loss += spectral_loss(model(mix), stems).item()
        val_loss /= len(val_loader)
        if epoch > warmup_epochs:
            scheduler.step()

        history['train'].append(train_loss)
        history['val'].append(val_loss)
        print(f"Epoch {epoch:3d} | train: {train_loss:.4f} | val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
            print(f"         -> New best saved (val: {val_loss:.4f})")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'epoch_{epoch:03d}.pt'))

    return history


# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------
def si_sdr(pred, target, eps=1e-8):
    target = target - target.mean(dim=-1, keepdim=True)
    pred   = pred   - pred.mean(dim=-1, keepdim=True)
    alpha  = (pred * target).sum(dim=-1, keepdim=True) / (target.pow(2).sum(dim=-1, keepdim=True) + eps)
    proj   = alpha * target
    noise  = pred - proj
    return 10 * torch.log10((proj.pow(2).sum(dim=-1) + eps) / (noise.pow(2).sum(dim=-1) + eps))


# -----------------------------------------------------------------------
# Test loop
# -----------------------------------------------------------------------
def test(model, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()
    test_set    = RemixDataset(collect_stems(MANIFEST_PATH, split='test', rawstems_dir=RAWSTEMS_DIR))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    stem_sisdrs     = {s: [] for s in STEMS}
    total_spec_loss = 0.0

    with torch.no_grad():
        for mix, stems in tqdm(test_loader, desc='[Test]'):
            mix   = mix.to(device)
            stems = stems.to(device)
            pred  = model(mix)
            total_spec_loss += spectral_loss(pred, stems).item()
            for i, name in enumerate(STEMS):
                stem_sisdrs[name].append(si_sdr(pred[:, i], stems[:, i]).mean().item())

    n = len(test_loader)
    metrics = {
        'spec_loss': total_spec_loss / n,
        'si_sdr':    {s: float(np.mean(v)) for s, v in stem_sisdrs.items()},
    }
    print(f"\n[Test] spectral loss: {metrics['spec_loss']:.4f}")
    for s, v in metrics['si_sdr'].items():
        print(f"       SI-SDR {s:>7s}: {v:.2f} dB")
    return metrics


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------
def plot_metrics(train_losses, val_losses, test_metrics=None, save_path=None):
    n_plots = 2 if test_metrics else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    ax = axes[0]
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Train')
    ax.plot(epochs, val_losses,   label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Spectral Loss')
    ax.set_title('Training / Validation Loss')
    ax.legend()
    ax.grid(True)

    if test_metrics:
        ax  = axes[1]
        sdr = test_metrics['si_sdr']
        ax.bar(list(sdr.keys()), list(sdr.values()))
        ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
        ax.set_ylabel('SI-SDR (dB)')
        ax.set_title('Per-Stem SI-SDR (Test Set)')
        ax.grid(axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[*] Plot saved to {save_path}")
    else:
        plt.show()


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    CONFIG_FILE     = "/content/drive/MyDrive/CMSC472Final/weights/roformer-model-bs-roformer-sw-by-jarredou/BS-Rofo-SW-Fixed.yaml"
    CHECKPOINT_FILE = "/content/drive/MyDrive/CMSC472Final/weights/roformer-model-bs-roformer-sw-by-jarredou/BS-Rofo-SW-Fixed.ckpt"

    history = train(CONFIG_FILE, CHECKPOINT_FILE)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = StemSeparator(CONFIG_FILE, CHECKPOINT_FILE, device=device)
    model.load_state_dict(torch.load(
        '/content/drive/MyDrive/CMSC472Final/checkpoints/best_model.pt',
        map_location='cpu'
    ))

    test_metrics = test(model)
    plot_metrics(history['train'], history['val'], test_metrics,
                 save_path='/content/drive/MyDrive/CMSC472Final/metrics.png')
