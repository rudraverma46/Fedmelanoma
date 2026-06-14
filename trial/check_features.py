import os
import numpy as np

ROOT_DIR = r"E:\Hospital_partitions"

for hospital in os.listdir(ROOT_DIR):

    hospital_path = os.path.join(ROOT_DIR, hospital)

    if not os.path.isdir(hospital_path):
        continue

    print("\n", "="*50)
    print(hospital)

    for file in os.listdir(hospital_path):

        if file.endswith(".npy"):

            path = os.path.join(hospital_path, file)

            arr = np.load(path, allow_pickle=True)

            print(
                f"{file:35s}",
                arr.shape
            )