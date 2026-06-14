
import torch

from torch.utils.data import DataLoader

from dataset import MelanomaDataset
from model import FedMelanomaModel

from focal_loss import (
    compute_class_weights,
    ClassWeightedFocalLoss
)


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def main():

    train_dataset = MelanomaDataset(
        "metadata_train.csv",
       "CityGeneral_Multispecialty_Hospital/images/Train"
    )

    test_dataset = MelanomaDataset(
       "metadata_test.csv"
,
       "CityGeneral_Multispecialty_Hospital/images/Test"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    model = FedMelanomaModel().to(
        DEVICE
    )

    criterion = ClassWeightedFocalLoss(
        gamma_f=2.0,
        reduction="mean"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    train_targets = []

    for label in train_dataset.df[
        "diagnosis_1"
    ]:

        train_targets.append(
            1 if label == "Malignant"
            else 0
        )

    class_weights = compute_class_weights(
        torch.tensor(train_targets)
    ).to(DEVICE)

    best_acc = 0

    for epoch in range(5):

        model.train()

        train_loss = 0

        for images, meta, labels in train_loader:

            images = images.to(DEVICE)
            meta = meta.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(
                images,
                meta
            )

            loss = criterion(
                logits,
                labels,
                class_weights=class_weights
            )

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images, meta, labels in test_loader:

                images = images.to(DEVICE)
                meta = meta.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(
                    images,
                    meta
                )

                preds = torch.argmax(
                    logits,
                    dim=1
                )

                correct += (
                    preds == labels
                ).sum().item()

                total += labels.size(0)

        accuracy = (
            100 * correct / total
        )

        print(
            f"Epoch {epoch+1}"
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Test Accuracy: {accuracy:.2f}%"
        )

        if accuracy > best_acc:

            best_acc = accuracy

            torch.save(
                model.state_dict(),
                "best_fedmelanoma_model.pth"
            )

    print(
        f"Best Accuracy: {best_acc:.2f}%"
    )


if __name__ == "__main__":
    main()