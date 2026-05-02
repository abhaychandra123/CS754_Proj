from datasets import load_dataset
import os
from PIL import Image
from tqdm import tqdm
import random

SAVE_DIR = "dataset"
IMG_SIZE = (128, 128)
SAMPLES_PER_DATASET = 1000

random.seed(42)

folders = {
    "faces": os.path.join(SAVE_DIR, "faces"),
    "scenes": os.path.join(SAVE_DIR, "scenes"),
    "textures": os.path.join(SAVE_DIR, "textures"),
}

for f in folders.values():
    os.makedirs(f, exist_ok=True)


def save_images(dataset_name, split, folder, max_samples):
    print(f"\nProcessing {dataset_name} → {folder}")

    dataset = load_dataset(dataset_name, split=split, streaming=True)
    dataset = dataset.shuffle(buffer_size=10000)

    count = 0

    for sample in tqdm(dataset):
        try:
            img = sample["image"]

            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)

            if img.mode != "RGB":
                img = img.convert("RGB")

            img = img.resize(IMG_SIZE)
            img.save(os.path.join(folder, f"{count}.png"))

            count += 1
            if count >= max_samples:
                break

        except Exception:
            continue

    print(f"Saved {count} images")


# ✅ FACES (FIXED)
save_images(
    dataset_name="nielsr/CelebA-faces",
    split="train",
    folder=folders["faces"],
    max_samples=SAMPLES_PER_DATASET
)

# ✅ SCENES (works)
save_images(
    dataset_name="scene_parse_150",
    split="train",
    folder=folders["scenes"],
    max_samples=SAMPLES_PER_DATASET
)

# ✅ TEXTURES (works)
save_images(
    dataset_name="dtd",
    split="train",
    folder=folders["textures"],
    max_samples=SAMPLES_PER_DATASET
)

print("DONE ✅")