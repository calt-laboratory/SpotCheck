from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from src.data_preparation.data_preparation import split_datasets
from src.training.training import initialize_model, validate_epoch, get_transforms


def test(model_path: Path, model_name: str) -> tuple[float, float, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = initialize_model(device, model_name)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    transform = get_transforms(model_name)

    _, _, test_dataset = split_datasets(
        test_size=0.2,
        validation_size=0.1,
        transform=transform,
    )

    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
    )

    criterion = CrossEntropyLoss()
    return validate_epoch(
        model=model, data_loader=test_data_loader, criterion=criterion, device=device
    )


if __name__ == "__main__":
    for model_name in [
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2",
        "efficientnet_b3",
        "efficientnet_b4",
    ]:
        model_path = Path("models") / f"{model_name}.pth"
        test_loss, test_acc, test_sensitivity = test(model_path, model_name)
        print(
            f"{model_name}: Test Acc = {test_acc:.2f}%, Test Loss = {test_loss:.4f}, Sensitivity = {test_sensitivity:.4f}"
        )
