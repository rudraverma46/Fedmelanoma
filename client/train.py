import torch
from torch.utils.data import DataLoader

from dataset import MelanomaDataset
from model import FedMelanomaModel
from losses import (
    compute_class_weights,
    ClassWeightedFocalLoss
)

# ==========================================
# CONFIGURATION
# ==========================================

TRAIN_CSV = "metadata_train.csv"
TEST_CSV = "metadata_test.csv"

TRAIN_IMAGE_DIR = "CityGeneral_Multispecialty_Hospital/images/Train"
TEST_IMAGE_DIR = "CityGeneral_Multispecialty_Hospital/images/Test"

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using Device : {DEVICE}")

# ==========================================
# DATASETS
# ==========================================

train_dataset = MelanomaDataset(
    TRAIN_CSV,
    TRAIN_IMAGE_DIR
)

test_dataset = MelanomaDataset(
    TEST_CSV,
    TEST_IMAGE_DIR
)

# ==========================================
# DATALOADERS
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ==========================================
# CLASS WEIGHTS
# ==========================================

train_labels = []

for diagnosis in train_dataset.df["diagnosis_1"]:

    if diagnosis == "Malignant":
        train_labels.append(1)
    else:
        train_labels.append(0)

class_weights = compute_class_weights(
    train_labels
).to(DEVICE)

print("Class Weights:")
print(class_weights)

# ==========================================
# MODEL
# ==========================================

model = FedMelanomaModel().to(
    DEVICE
)

criterion = ClassWeightedFocalLoss(
    alpha=class_weights,
    gamma=2
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# ==========================================
# TRAINING LOOP
# ==========================================

best_test_acc = 0

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0

    train_correct = 0
    train_total = 0

    for images, metadata, labels in train_loader:

        images = images.to(DEVICE)
        metadata = metadata.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        logits = model(
            images,
            metadata
        )

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        predictions = torch.argmax(
            logits,
            dim=1
        )

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)

    train_accuracy = (
        100 * train_correct / train_total
    )

    # ======================================
    # TESTING
    # ======================================

    model.eval()

    test_correct = 0
    test_total = 0

    with torch.no_grad():

        for images, metadata, labels in test_loader:

            images = images.to(DEVICE)
            metadata = metadata.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(
                images,
                metadata
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            test_correct += (
                predictions == labels
            ).sum().item()

            test_total += labels.size(0)

    test_accuracy = (
        100 * test_correct / test_total
    )

    print(
        f"Epoch [{epoch+1}/{EPOCHS}]"
    )

    print(
        f"Train Loss : {train_loss:.4f}"
    )

    print(
        f"Train Accuracy : {train_accuracy:.2f}%"
    )

    print(
        f"Test Accuracy : {test_accuracy:.2f}%"
    )

    print("-" * 50)

    # ======================================
    # SAVE BEST MODEL
    # ======================================

    if test_accuracy > best_test_acc:

        best_test_acc = test_accuracy

        torch.save(
            model.state_dict(),
            "best_fedmelanoma_model.pth"
        )

        print(
            f"Best Model Saved "
            f"(Test Acc = {test_accuracy:.2f}%)"
        )

print(
    f"\nTraining Finished"
)

print(
    f"Best Test Accuracy : "
    f"{best_test_acc:.2f}%"
)