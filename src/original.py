from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from dataset import RawStemsDataset
from frequency_transformer import load_bs_roformer
from losses import SeparationLoss


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "rawstems" / "RawStems_1"

CONFIG_FILE = PROJECT_ROOT / "src" / "weights" / "roformer-model-bs-roformer-sw-by-jarredou" / "BS-Rofo-SW-Fixed.yaml"
CHECKPOINT_FILE = PROJECT_ROOT / "src" / "weights" / "roformer-model-bs-roformer-sw-by-jarredou" / "BS-Rofo-SW-Fixed.ckpt"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Using device: {device}")

    dataset = RawStemsDataset(DATA_DIR)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size


    _, val_dataset = random_split(
        dataset,
        [train_size, val_size]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    model, _ = load_bs_roformer(
        str(CONFIG_FILE),
        str(CHECKPOINT_FILE),
        device=device
    )

    model.eval()
    loss_fn = SeparationLoss()

    val_loss = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            mixture = batch["mixture"].to(device)
            stems = batch["stems"].to(device)

            pred = model(mixture)

            if isinstance(pred, (tuple, list)):
                pred = pred[0]

            if pred.shape[1] > stems.shape[1]:
                pred = pred[:, :stems.shape[1], :, :]

            min_len = min(pred.shape[-1], stems.shape[-1])
            pred = pred[..., :min_len]
            stems = stems[..., :min_len]

            loss = loss_fn(pred, stems)
            val_loss += loss.item()

            print(
                f"Batch [{batch_idx + 1}/{len(val_loader)}], "
                f"Original Pretrained Val Loss: {loss.item():.10f}"
            )

    val_loss /= len(val_loader)

    print("=" * 50)
    print(f"Original pretrained model Val Loss = {val_loss:.10f}")


if __name__ == "__main__":
    main()