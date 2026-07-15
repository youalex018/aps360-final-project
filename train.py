"""Train the LSTM, checkpoint the best epoch, and report test accuracy."""
import torch
import torch.nn as nn

import config
from dataset import get_dataloaders
from model import ToxicChatLSTM


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over `loader`; trains when an optimizer is given, else evaluates.

    Returns (avg_loss, accuracy). Sharing this between train/val keeps the metric
    computation identical, so the numbers are comparable epoch to epoch.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for x, lengths, y in loader:
            x, y = x.to(device), y.to(device)
            probs = model(x, lengths)
            loss = criterion(probs, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * y.size(0)
            correct += ((probs > 0.5).float() == y).sum().item()
            total += y.size(0)

    return total_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_dl, val_dl, test_dl, vocab = get_dataloaders()
    model = ToxicChatLSTM(vocab_size=len(vocab)).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    config.CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Track val loss (not accuracy) for selection: it's a smoother signal on
    # small/imbalanced sets where accuracy plateaus in coarse steps.
    best_val_loss = float("inf")

    for epoch in range(1, config.EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_dl, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_dl, criterion, device)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.CKPT_PATH)
            marker = "  <- saved"

        print(
            f"Epoch {epoch:02d}/{config.EPOCHS} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}{marker}"
        )

    # Evaluate the checkpointed best model, not the (possibly overfit) last one.
    model.load_state_dict(torch.load(config.CKPT_PATH, map_location=device))
    _, test_acc = run_epoch(model, test_dl, criterion, device)
    print(f"\nBest model test accuracy: {test_acc:.3f}")


if __name__ == "__main__":
    main()
