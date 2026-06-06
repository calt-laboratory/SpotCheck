"""
Training script for SpotCheck: Binary classification of skin moles (nevus vs melanoma).
Uses a pretrained EfficientNet model, fine-tuned on HAM10000 dataset.
"""

import time
import torch
from dataclasses import dataclass
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch import nn, optim
from pathlib import Path

from src.data_preparation.data_preparation import split_datasets


# Set seed for reproducibility
torch.manual_seed(42)


@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 0.001
    patience: int = 3
    num_workers: int = 16
    model_save_path: Path = Path("models") / "efficient_b0.pth"


def _initialize_model(device: torch.device) -> nn.Module:
    """Initialize EfficientNet-B0 with pretrained weights and binary classifier."""
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)

    # Freeze all parameters (weights + biases) of all layers
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2)

    return model.to(device)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: CrossEntropyLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    non_blocking: bool,
) -> tuple[float, float]:
    """Run one training epoch, return (avg_loss, avg_accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def _validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: CrossEntropyLoss,
    device: torch.device,
) -> tuple[float, float]:
    """Run one validation epoch, return (avg_loss, avg_accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def train_model(config: TrainingConfig) -> None:
    """Main training function with early stopping and model checkpointing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    non_blocking = True if device.type == "cuda" else False

    # Initialize data loaders
    train_dataset, validation_dataset, _ = split_datasets()
    pin_memory = True if device.type == "cuda" else False

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    # Initialize model, loss, and optimizer
    model = _initialize_model(device)
    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Ensure models directory exists
    config.model_save_path.parent.mkdir(parents=True, exist_ok=True)

    # Training state tracking
    best_val_acc = 0.0
    no_improvement_count = 0
    start_time = time.time()

    for epoch in range(config.epochs):
        # Training phase
        train_loss, train_acc = _train_epoch(
            model, train_loader, criterion, optimizer, device, non_blocking
        )

        # Validation phase
        validation_loss, validation_acc = _validate_epoch(
            model, validation_loader, criterion, device
        )

        # Check for best model and save
        if validation_acc > best_val_acc:
            best_val_acc = validation_acc
            no_improvement_count = 0
            torch.save(model.state_dict(), config.model_save_path)
            print(f"New best model saved! Val Acc: {best_val_acc:.2f}%")
        else:
            no_improvement_count += 1
            print(f"No improvement for {no_improvement_count}/{config.patience} epochs")

        print(
            f"Epoch {epoch + 1}/{config.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {validation_loss:.4f} | Val Acc: {validation_acc:.2f}%"
        )

        # Early stopping check
        if no_improvement_count >= config.patience:
            print(
                f"Early stopping triggered after {epoch + 1} epochs (no improvement for {config.patience} epochs)"
            )
            break

    # Print training summary
    training_time = time.time() - start_time
    minutes, seconds = divmod(int(training_time), 60)
    print(f"Training complete. Best Val Acc: {best_val_acc:.2f}%")
    print(f"Total training time: {minutes}m {seconds}s")
    print(f"Model saved to {config.model_save_path}")


if __name__ == "__main__":
    config = TrainingConfig()
    train_model(config)
