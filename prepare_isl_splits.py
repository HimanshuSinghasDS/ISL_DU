from pathlib import Path
import random
import shutil

random.seed(42)

source_dir = Path(r"C:\Users\himan\Downloads\archive\original_images")
target_dir = Path(r"C:\Users\himan\Downloads\isl_cnn_kan_project\isl_cnn_kan\data\isl")

train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
class_dirs = sorted(class_dirs, key=lambda x: x.name)

for class_dir in class_dirs:
    images = [p for p in class_dir.iterdir() if p.suffix.lower() in image_exts]
    random.shuffle(images)

    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    for split_name, split_imgs in [
        ("train", train_imgs),
        ("val", val_imgs),
        ("test", test_imgs),
    ]:
        split_class_dir = target_dir / split_name / class_dir.name
        split_class_dir.mkdir(parents=True, exist_ok=True)

        for img_path in split_imgs:
            shutil.copy2(img_path, split_class_dir / img_path.name)

    print(f"{class_dir.name}: total={n}, train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")

print("Dataset split completed.")