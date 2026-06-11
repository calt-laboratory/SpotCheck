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
from torchvision.models import (
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_B3_Weights,
    EfficientNet_B4_Weights,
)
from typing import Callable
from torch import nn, optim
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from typing import Final, Any
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from src.data_preparation.data_preparation import (
    split_datasets,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


# Set seed for reproducibility
torch.manual_seed(42)


# Model registry: name -> (model_fn, weights, input_size)
MODEL_REGISTRY: Final[dict[str, tuple[Callable, Any, int]]] = {
    "efficientnet_b0": (efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1, 224),
    "efficientnet_b1": (efficientnet_b1, EfficientNet_B1_Weights.IMAGENET1K_V1, 240),
    "efficientnet_b2": (efficientnet_b2, EfficientNet_B2_Weights.IMAGENET1K_V1, 260),
    "efficientnet_b3": (efficientnet_b3, EfficientNet_B3_Weights.IMAGENET1K_V1, 300),
    "efficientnet_b4": (efficientnet_b4, EfficientNet_B4_Weights.IMAGENET1K_V1, 384),
}


# Top-level class for applying transforms to datasets
class TransformDataset(Dataset):
    """Wrapper to apply transforms to an existing dataset"""

    def __init__(
        self, base_dataset: Dataset, transform: Callable | None = None
    ) -> None:
        self._base_dataset = base_dataset
        self._transform = transform

    def __len__(self) -> int:
        return len(self._base_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img, label = self._base_dataset[idx]
        if self._transform:
            img = self._transform(img)
        return img, label


@dataclass
class TrainingConfig:
    model_name: str = "efficientnet_b0"
    epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 0.001
    patience: int = 3
    num_workers: int = 16
    model_save_path: Path = Path("models") / "efficient_b0.pth"


def _get_input_size(model_name: str) -> int:
    """Get the expected input size for the specified model"""
    return MODEL_REGISTRY[model_name][2]


def get_transforms(model_name: str) -> transforms.Compose:
    """Get transforms with size matching the model's expected input"""
    input_size = _get_input_size(model_name)
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def initialize_model(device: torch.device, model_name: str) -> nn.Module:
    """Initialize model from registry with pretrained weights and binary classifier"""
    model_fn, weights, _ = MODEL_REGISTRY[model_name]
    model = model_fn(weights=weights)

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
    """Run one training epoch, return (avg_loss, avg_accuracy)"""
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


def validate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: CrossEntropyLoss,
    device: torch.device,
) -> tuple[float, float, float]:
    """Run one validation epoch, return (avg_loss, avg_accuracy, sensitivity)"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / total
    accuracy = 100 * correct / total

    tn, fp, fn, tp = confusion_matrix(
        y_true=all_labels,
        y_pred=all_predictions,
        labels=[0, 1],
    ).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return avg_loss, accuracy, sensitivity


def compute_class_weights(dataset: Dataset) -> torch.Tensor:
    all_labels = []
    for _, label in dataset:
        all_labels.append(label)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(all_labels),
        y=all_labels,
    )

    return torch.tensor(class_weights, dtype=torch.float32)


def train_model(config: TrainingConfig) -> tuple[float, float]:
    """Main training fn w/ early stopping and model checkpointing"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model: {config.model_name}")

    non_blocking = True if device.type == "cuda" else False

    # Initialize data loaders w/ model-specific transforms
    pin_memory = True if device.type == "cuda" else False
    transform = get_transforms(config.model_name)

    train_dataset, validation_dataset, _ = split_datasets(
        test_size=0.2,
        validation_size=0.1,
        transform=transform,
    )

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
    model = initialize_model(device, config.model_name)
    class_weights = compute_class_weights(dataset=train_dataset).to(device)
    criterion = CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Ensure models directory exists
    config.model_save_path.parent.mkdir(parents=True, exist_ok=True)

    # Training state tracking
    best_validation_sensitivity = 0.0
    best_validation_accuracy = 0.0
    no_improvement_count = 0
    start_time = time.time()

    for epoch in range(config.epochs):
        # Training phase
        train_loss, train_acc = _train_epoch(
            model, train_loader, criterion, optimizer, device, non_blocking
        )

        # Validation phase
        validation_loss, validation_accuracy, validation_sensitivity = validate_epoch(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        # Check for best model and save
        if validation_sensitivity > best_validation_sensitivity:
            best_validation_sensitivity = validation_sensitivity
            best_validation_accuracy = validation_accuracy
            no_improvement_count = 0
            torch.save(model.state_dict(), config.model_save_path)
            print(
                f"New best model saved! "
                f"Val Sensitivity: {validation_sensitivity:.4f} | Val Acc: {validation_accuracy:.2f}%"
            )
        else:
            no_improvement_count += 1
            print(f"No improvement for {no_improvement_count}/{config.patience} epochs")

        print(
            f"Epoch {epoch + 1}/{config.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {validation_loss:.4f} | Val Acc: {validation_accuracy:.2f}% | "
            f"Val Sensitivity: {validation_sensitivity:.4f}"
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
    print(
        f"Training complete. Best Val Sensitivity: {best_validation_sensitivity:.4f} | Best Val Acc: {best_validation_accuracy:.2f}%"
    )
    print(f"Total training time: {minutes}m {seconds}s")
    print(f"Model saved to {config.model_save_path}")

    return best_validation_accuracy, best_validation_sensitivity
