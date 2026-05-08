from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from model import StemSeparator
from dataset import RawStemsDataset
from losses import SeparationLoss


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "rawstems" / "RawStems_1"

CONFIG_FILE = "weights\\roformer-model-bs-roformer-sw-by-jarredou\\BS-Rofo-SW-Fixed.yaml"
CHECKPOINT_FILE = "weights\\roformer-model-bs-roformer-sw-by-jarredou\\BS-Rofo-SW-Fixed.ckpt"


def freeze_pretrained_parts(model):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.temporal_branch.parameters():
        param.requires_grad = True

    for param in model.gated_fusion.parameters():
        param.requires_grad = True

    for param in model.decoder_heads.parameters():
        param.requires_grad = True


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Using device: {device}")

    dataset = RawStemsDataset(DATA_DIR)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    model = StemSeparator(
        config_path=str(CONFIG_FILE),
        checkpoint_path=str(CHECKPOINT_FILE),
        device=device
    )

    freeze_pretrained_parts(model)

    loss_fn = SeparationLoss()

    optimizer = torch.optim.AdamW([
    {
        "params": list(model.temporal_branch.parameters()) +
                  list(model.gated_fusion.parameters()),
        "lr": 1e-4
    },
    {
        "params": list(model.decoder_heads.parameters()) +
                  list(model.freq_transformer.parameters()),
        "lr": 1e-6
    }
], weight_decay=1e-4)

    best_val_loss = float("inf")
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            mixture = batch["mixture"].to(device)
            stems = batch["stems"].to(device)

            optimizer.zero_grad()

            pred = model(mixture)
            loss = loss_fn(pred, stems)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()

            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Batch [{batch_idx + 1}/{len(train_loader)}], "
                f"Loss: {loss.item():.10f}"
            )

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                mixture = batch["mixture"].to(device)
                stems = batch["stems"].to(device)

                pred = model(mixture)
                loss = loss_fn(pred, stems)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] finished: "
            f"Train Loss = {train_loss:.4f}, "
            f"Val Loss = {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = PROJECT_ROOT / "best_model.pt"
            torch.save(model.state_dict(), save_path)
            print(f"[+] Saved best model to {save_path}")


if __name__ == "__main__":
    main()