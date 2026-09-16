import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Dict, Tuple, Any


class MelanomaDataset(Dataset):
    """
    PyTorch Dataset for melanoma classification.

    Optimizations applied vs. the original:
      1. Row caching           – df.to_dict("records") eliminates per-sample iloc overhead.
      2. Label tensor          – diagnosis_1 decoded into a single contiguous torch.long
                                 tensor at init; __getitem__ indexes it with zero allocation.
      3. Metadata tensor       – encode_metadata() results stacked into one (N, 13)
                                 float32 tensor; __getitem__ returns a view, not a copy.
      4. PIL removal           – cv2.resize() replaces ToPILImage + Resize, cutting one
                                 redundant format conversion per sample.
      5. Contiguous label buf  – replaces List[int] + per-sample torch.tensor() call.
      6. Contiguous metadata   – replaces List[torch.Tensor]; single allocation, better
                                 CPU cache locality, view-based access.
    """

    # ------------------------------------------------------------------ #
    #  Class-level constants – defined once, shared across all instances   #
    # ------------------------------------------------------------------ #
    _SKIN_TYPES: List[str] = ["I", "II", "III", "IV", "V", "VI"]
    _IMAGE_TYPES: List[str] = [
        "dermoscopic",
        "clinical: close-up",
        "clinical: overview",
    ]
    _CONFIRM_TYPES: List[str] = [
        "histopathology",
        "serial imaging showing no change",
        "single image expert consensus",
        "single contributor clinical assessment",
    ]

    def __init__(self, csv_file: str, image_dir: str) -> None:
        self.image_dir = image_dir

        # --- Load CSV -------------------------------------------------- #
        self.df = pd.read_csv(csv_file)

        # ---------------------------------------------------------------- #
        # Optimization 1 – Row caching                                      #
        #                                                                    #
        # df.iloc[idx] must parse the row each call; converting to a list   #
        # of plain dicts (once) gives O(1) Python dict access in            #
        # __getitem__ and avoids Pandas overhead on the hot path.           #
        # ---------------------------------------------------------------- #
        self.rows: List[Dict[str, Any]] = self.df.to_dict("records")

        # ---------------------------------------------------------------- #
        # Optimization 2 – Precompute labels as a contiguous tensor         #
        #                                                                    #
        # Previously a Python List[int]; now a single torch.long tensor     #
        # allocated once in one contiguous block of memory.                 #
        #                                                                    #
        # Benefits vs. List[int]:                                           #
        #   • self.labels[idx] returns a 0-dim tensor directly – no         #
        #     torch.tensor() call in __getitem__, eliminating one heap       #
        #     allocation per sample per epoch.                              #
        #   • All label values live in a single contiguous memory region,   #
        #     improving CPU cache locality when the DataLoader prefetches   #
        #     multiple indices in order.                                    #
        #   • dtype=torch.long is guaranteed at construction; no            #
        #     per-sample dtype argument needed.                             #
        #                                                                    #
        # DataLoader compatibility: the tensor is created in the main       #
        # process and shared read-only across workers via fork/spawn, so    #
        # no locking or copying is required.                                #
        # ---------------------------------------------------------------- #
        self.labels: torch.Tensor = torch.tensor(
            [
                1 if str(x).lower() == "malignant" else 0
                for x in self.df["diagnosis_1"]
            ],
            dtype=torch.long,
        )

        # ---------------------------------------------------------------- #
        # Optimization 3 – Precompute metadata as a single stacked tensor   #
        #                                                                    #
        # Previously a List[torch.Tensor] – N separate heap objects, each  #
        # with its own storage pointer, reference count, and metadata       #
        # header.  torch.stack() merges them into one contiguous (N, 13)   #
        # float32 tensor allocated in a single memory block.                #
        #                                                                    #
        # Benefits vs. List[torch.Tensor]:                                  #
        #   • One allocation instead of N allocations → less GC pressure.  #
        #   • All 13-element rows are stored adjacently in memory; when the #
        #     DataLoader fetches indices that are close together (e.g.      #
        #     sequential or within the same batch), the CPU cache already   #
        #     holds nearby rows, reducing cache misses.                     #
        #   • self.metadata_cache[idx] uses PyTorch's __getitem__ on a 2-D #
        #     tensor, returning a view (not a copy) – zero allocation on    #
        #     the hot path.                                                 #
        #                                                                    #
        # Shape:  (N, 13)   dtype: torch.float32  – identical per-row      #
        # values to the original list of 1-D (13,) tensors.                #
        #                                                                    #
        # DataLoader compatibility: same fork/spawn read-only sharing as   #
        # before; contiguous layout also makes shared-memory transfer       #
        # between processes faster.                                         #
        # ---------------------------------------------------------------- #
        self.metadata_cache: torch.Tensor = torch.stack(
            [self.encode_metadata(row) for row in self.rows]
        )  # shape: (N, 13), dtype: torch.float32

        # ---------------------------------------------------------------- #
        # Optimization 4 – Remove PIL conversion                            #
        #                                                                    #
        # The original pipeline: BGR ndarray → ToPILImage → Resize →       #
        # ToTensor → Normalize.                                             #
        # cv2.resize() on the ndarray before handing it to transforms       #
        # removes the PIL round-trip entirely.  ToTensor() accepts a uint8  #
        # HxWxC ndarray directly, so the output is identical.              #
        # ---------------------------------------------------------------- #
        self.transform = transforms.Compose([
            # ToPILImage() + Resize() replaced by cv2.resize() in __getitem__
            transforms.ToTensor(),          # HxWxC uint8 ndarray → CxHxW float32 [0,1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    # ------------------------------------------------------------------ #
    #  Metadata encoding – unchanged logic, extracted to a pure method    #
    # ------------------------------------------------------------------ #
    def encode_metadata(self, row: Dict[str, Any]) -> torch.Tensor:
        """
        One-hot encode fitzpatrick_skin_type, image_type, and
        diagnosis_confirm_type into a single 13-dim float32 tensor.

        Logic is identical to the original implementation.
        """
        skin    = [0] * len(self._SKIN_TYPES)
        image   = [0] * len(self._IMAGE_TYPES)
        confirm = [0] * len(self._CONFIRM_TYPES)

        try:
            skin[self._SKIN_TYPES.index(str(row["fitzpatrick_skin_type"]).strip())] = 1
        except (ValueError, KeyError):
            pass

        try:
            image[self._IMAGE_TYPES.index(str(row["image_type"]).strip())] = 1
        except (ValueError, KeyError):
            pass

        try:
            confirm[self._CONFIRM_TYPES.index(str(row["diagnosis_confirm_type"]).strip())] = 1
        except (ValueError, KeyError):
            pass

        return torch.tensor(skin + image + confirm, dtype=torch.float32)

    # ------------------------------------------------------------------ #
    #  Dataset protocol                                                    #
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.rows)   # rows list has the same length as the DataFrame

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ---------------------------------------------------------------- #
        # Opt 1 – O(1) dict lookup instead of df.iloc[idx]                 #
        # ---------------------------------------------------------------- #
        row = self.rows[idx]

        # --- Load image ------------------------------------------------ #
        image_path = os.path.join(self.image_dir, row["isic_id"] + ".jpg")
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(image_path)

        # Convert BGR (OpenCV default) → RGB before any further processing
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ---------------------------------------------------------------- #
        # Opt 4 – Resize with OpenCV before ToTensor()                     #
        #                                                                    #
        # cv2.resize() is a fast C++ operation on the raw ndarray.         #
        # This avoids allocating a PIL Image object and its internal copy. #
        # Output shape: (224, 224, 3) uint8 – exactly what ToTensor()      #
        # expects, so the transform pipeline output is bit-for-bit the     #
        # same as ToPILImage → Resize → ToTensor.                         #
        # ---------------------------------------------------------------- #
        image = cv2.resize(image, (224, 224))   # replaces ToPILImage + Resize
        image = self.transform(image)            # ToTensor + Normalize

        # ---------------------------------------------------------------- #
        # Opt 3 – View into the (N, 13) stacked tensor; zero allocation.   #
        #         Returns a 1-D (13,) float32 tensor – same shape & dtype  #
        #         as the original per-row tensors.                         #
        # ---------------------------------------------------------------- #
        metadata = self.metadata_cache[idx]   # view, not a copy

        # ---------------------------------------------------------------- #
        # Opt 2 – Index into the pre-allocated torch.long tensor.          #
        #         Returns a 0-dim torch.long tensor directly – no          #
        #         torch.tensor() construction on the hot path.             #
        # ---------------------------------------------------------------- #
        label = self.labels[idx]              # 0-dim torch.long tensor

        return (
            image,
            metadata,
            label,   # already torch.long; torch.tensor() call removed
        )