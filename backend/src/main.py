from pathlib import Path

from src.training.training import train_model, TrainingConfig
from src.training.evaluation import test

MODELS = [
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
]

BATCH_SIZES = {
    "efficientnet_b0": 256,
    "efficientnet_b1": 256,
    "efficientnet_b2": 256,
    "efficientnet_b3": 256,
    "efficientnet_b4": 128,
}


def main() -> None:
    print("Starting training...")
    for model_name in MODELS:
        print(f"\n📊 Training {model_name}...")
        config = TrainingConfig(
            model_name=model_name,
            batch_size=BATCH_SIZES[model_name],
            model_save_path=Path("models") / f"{model_name}.pth",
        )
        train_model(config)

    print("\n🔍 Starting evaluation...")
    for model_name in MODELS:
        model_path = Path("models") / f"{model_name}.pth"
        print(f"\n📈 Evaluating {model_name}...")
        test_loss, test_acc, test_sensitivity = test(model_path, model_name)
        print(
            f"  {model_name}: Test Acc = {test_acc:.2f}%, "
            f"Test Loss = {test_loss:.4f}, Sensitivity = {test_sensitivity:.4f}"
        )


if __name__ == "__main__":
    main()
