import os
import cv2
import torch
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms


class MelanomaDataset(Dataset):

    def __init__(self, csv_file, image_dir):

        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def encode_metadata(self, row):

        skin_types = ["I", "II", "III", "IV", "V", "VI"]

        image_types = [
            "dermoscopic",
            "clinical: close-up",
            "clinical: overview"
        ]

        confirm_types = [
            "histopathology",
            "serial imaging showing no change",
            "single image expert consensus",
            "single contributor clinical assessment"
        ]

        skin = [0] * 6
        image = [0] * 3
        confirm = [0] * 4

        try:
            skin[
                skin_types.index(
                    str(row["fitzpatrick_skin_type"]).strip()
                )
            ] = 1
        except:
            pass

        try:
            image[
                image_types.index(
                    str(row["image_type"]).strip()
                )
            ] = 1
        except:
            pass

        try:
            confirm[
                confirm_types.index(
                    str(row["diagnosis_confirm_type"]).strip()
                )
            ] = 1
        except:
            pass

        metadata = skin + image + confirm

        return torch.tensor(
            metadata,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = os.path.join(
            self.image_dir,
            row["isic_id"] + ".jpg"
        )

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(image_path)

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = self.transform(image)

        metadata = self.encode_metadata(row)

        label = (
            1
            if str(row["diagnosis_1"]).lower()
            == "malignant"
            else 0
        )

        return (
            image,
            metadata,
            torch.tensor(label, dtype=torch.long)
        )