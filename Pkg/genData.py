import glob

train_images_path = "../BSDS500_RR/Coloured/train/*.jpg"
val_images_path = "../BSDS500_RR/Coloured/val/*.jpg"

train_images = glob.glob(train_images_path)
val_images = glob.glob(val_images_path)

print(f"Training images found: {len(train_images)}")
print(f"Validation images found: {len(val_images)}")
