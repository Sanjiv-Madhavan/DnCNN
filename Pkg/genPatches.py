# import cv2
# import numpy as np
# import h5py
# import glob
# import os
#
#
# def extract_patches(img, patch_size=60, stride=10):
#     """
#     Extracts image patches of size (patch_size x patch_size) with a given stride.
#     """
#     h, w, c = img.shape
#     patches = []
#
#     for i in range(0, h - patch_size + 1, stride):
#         for j in range(0, w - patch_size + 1, stride):
#             patch = img[i:i + patch_size, j:j + patch_size]
#             patches.append(patch)
#
#     return np.array(patches)
#
#
# # Define dataset paths
# train_images_path = "../BSDS500_RR/Coloured/trainsmall/*.jpg"
# val_images_path = "../BSDS500_RR/Coloured/valsmall/*.jpg"
#
# # Load all training images
# train_images = glob.glob(train_images_path)
# val_images = glob.glob(val_images_path)
#
# X_train, Y_train = [], []
# X_val, Y_val = [], []
#
# # Process training dataset
# for img_path in train_images:
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
#     patches = extract_patches(img)  # Extract patches
#     X_train.extend(patches)
#     Y_train.extend(patches)  # Labels are clean images
#
# # Process validation dataset
# for img_path in val_images:
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
#     patches = extract_patches(img)
#     X_val.extend(patches)
#     Y_val.extend(patches)
#
# # Convert to NumPy arrays
# X_train = np.array(X_train, dtype=np.float32) / 255.0  # Normalize to [0,1]
# Y_train = np.array(Y_train, dtype=np.float32) / 255.0
#
# X_val = np.array(X_val, dtype=np.float32) / 255.0
# Y_val = np.array(Y_val, dtype=np.float32) / 255.0
#
# print(f"Training patches: {X_train.shape}, Validation patches: {X_val.shape}")
#
# # Save preprocessed data to HDF5
# # with h5py.File("../Datapy/dataset.h5", "w") as h5f:
# #     h5f.create_dataset("X_train", data=X_train)
# #     h5f.create_dataset("Y_train", data=Y_train)
# #     h5f.create_dataset("X_val", data=X_val)
# #     h5f.create_dataset("Y_val", data=Y_val)
# #
# # print("Preprocessed dataset saved in HDF5 format!")

import cv2
import numpy as np
import h5py
import glob
import os


def extract_patches(img, patch_size=60, stride=10):
    """
    Extracts image patches of size (patch_size x patch_size) with a given stride.
    """
    h, w, c = img.shape
    patches = []

    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = img[i:i + patch_size, j:j + patch_size]
            patches.append(patch)

    return np.array(patches)


# Define dataset paths
train_images_path = "../BSDS500_RR/Coloured/trainsmall/*.jpg"
val_images_path = "../BSDS500_RR/Coloured/valsmall/*.jpg"

# Load all training images
train_images = glob.glob(train_images_path)
val_images = glob.glob(val_images_path)

X_train, Y_train = [], []
X_val, Y_val = [], []

# Process training dataset
for img_path in train_images:
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Warning: Could not load {img_path}")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    patches = extract_patches(img)  # Extract patches

    X_train.extend(patches)
    Y_train.extend(patches)  # Labels are clean images

# Process validation dataset
for img_path in val_images:
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Warning: Could not load {img_path}")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    patches = extract_patches(img)

    X_val.extend(patches)
    Y_val.extend(patches)

# Convert to NumPy arrays
X_train = np.array(X_train, dtype=np.float32) / 255.0  # Normalize to [0,1]
Y_train = np.array(Y_train, dtype=np.float32) / 255.0

X_val = np.array(X_val, dtype=np.float32) / 255.0
Y_val = np.array(Y_val, dtype=np.float32) / 255.0

print(f"✅ Training patches: {X_train.shape}, Validation patches: {X_val.shape}")

# Ensure directory exists
os.makedirs("DataPy", exist_ok=True)

# ✅ Save extracted patches to HDF5
with h5py.File("DataPy/dataset.h5", "w") as h5f:
    h5f.create_dataset("X_train", data=X_train)
    h5f.create_dataset("Y_train", data=Y_train)
    h5f.create_dataset("X_val", data=X_val)
    h5f.create_dataset("Y_val", data=Y_val)

print("✅ Preprocessed dataset saved in HDF5 format (DataPy/dataset.h5)!")
