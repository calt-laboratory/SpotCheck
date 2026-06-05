import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch import nn, optim
from pathlib import Path
from typing import Final

from src.data_preparation.data_preparation import split_datasets


NUM_CPU_WORKERS: Final[int] = 16


def train_model(
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    model_save_path: Path = Path("models/efficient_b0.pth"),
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset, validation_dataset, _ = split_datasets()

    pin_memory = True if device.type == "cuda" else False

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_CPU_WORKERS,
        pin_memory=pin_memory,
    )

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_CPU_WORKERS,
        pin_memory=pin_memory,
    )

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)

    # Freeze all parameters (weights + biases) of all layers
    for param in model.parameters():
        param.requires_grad = False

    num_in_features = model.classifier[1].in_features
    # Replace final classifier layer w/ new binary classifier (nevus and melanoma)
    model.classifier[1] = nn.Linear(in_features=num_in_features, out_features=2)

    model = model.to(device)

    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

    non_blocking = True if device.type == "cuda" else False

    for epoch in range(epochs):
        model.train()  # Activate training modus
        train_loss = 0.0
        correct_train_predictions = 0
        total_num_train_samples = 0

        for images, labels in train_loader:
            # Transfer data to target device (GPU/CPU) asynchronously so CPU can load/transform next batch while GPU
            # computes gradients on curr batch
            images = images.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)

            # Reset gradients to zero because Pytorch accumulates gradients by default
            optimizer.zero_grad()

            # Forward pass: compute model predictions
            outputs = model(images)

            # Calculate loss btw predictions and labels
            loss = criterion(outputs, labels)

            # Backward pass: compute gradients of loss w.r.t. all params
            loss.backward()

            # Update model weights using gradients
            optimizer.step()

            # Accumulate loss for epoch average weighted by batch size
            train_loss += loss.item() * images.size(0)

            # Get predicted class indices by selecting the class w/ the highest logit for each sample
            # dim=1: class dimension (max over all classes for each sample in batch)
            # Note: Logits are the raw model outputs - Cross Entropy expects logits, not probabilities
            _, predicted = torch.max(outputs.data, dim=1)

            # Count total samples in curr batch
            total_num_train_samples += labels.size(0)

            correct_train_predictions += (predicted == labels).sum().item()

        # Compute average training loss per sample for the entire epoch
        train_loss /= total_num_train_samples
        train_acc = 100 * correct_train_predictions / total_num_train_samples

        # Switch to evaluation mode
        model.eval()

        validation_loss = 0.0
        correct_validation_predictions = 0
        total_num_validation_samples = 0

        with torch.no_grad():
            for images, labels in validation_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                validation_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_num_validation_samples += labels.size(0)
                correct_validation_predictions += (predicted == labels).sum().item()

        validation_loss /= total_num_validation_samples
        validation_acc = (
            100 * correct_validation_predictions / total_num_validation_samples
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {validation_loss:.4f} | Val Acc: {validation_acc:.2f}%"
        )

        print(f"Training complete. Saving model to {model_save_path}")
        torch.save(model.state_dict(), model_save_path)
        print("Model saved successfully!")


if __name__ == "__main__":
    train_model(
        epochs=10,
        batch_size=256,
        learning_rate=0.001,
    )
