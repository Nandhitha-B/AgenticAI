import os
import h5py
import numpy as np
import cv2
import shutil
import random

mat_dir = "brain_tumor_data"   
dataset_dir = "dataset"          
train_dir = os.path.join(dataset_dir, "train")
test_dir = os.path.join(dataset_dir, "test")


class_map = {1: "meningioma", 2: "glioma", 3: "pituitary"}

os.makedirs(dataset_dir, exist_ok=True)
for cls_name in class_map.values():
    os.makedirs(os.path.join(dataset_dir, cls_name), exist_ok=True)

for fname in os.listdir(mat_dir):
    if not fname.endswith(".mat"):
        continue

    fpath = os.path.join(mat_dir, fname)
    with h5py.File(fpath, "r") as f:
        cjdata = f["cjdata"]
        image = np.array(cjdata["image"]).T
        label = int(np.array(cjdata["label"])[0][0])


    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min() + 1e-6)
    image = (image * 255).astype(np.uint8)

    class_name = class_map[label]
    out_path = os.path.join(dataset_dir, class_name,
                            fname.replace(".mat", ".png"))
    cv2.imwrite(out_path, image)

split_ratio = 0.8

for cls_name in class_map.values():
    cls_path = os.path.join(dataset_dir, cls_name)
    images = os.listdir(cls_path)
    random.shuffle(images)

    train_cls_dir = os.path.join(train_dir, cls_name)
    test_cls_dir = os.path.join(test_dir, cls_name)
    os.makedirs(train_cls_dir, exist_ok=True)
    os.makedirs(test_cls_dir, exist_ok=True)

    split_idx = int(len(images) * split_ratio)
    for img in images[:split_idx]:
        shutil.move(os.path.join(cls_path, img),
                    os.path.join(train_cls_dir, img))
    for img in images[split_idx:]:
        shutil.move(os.path.join(cls_path, img),
                    os.path.join(test_cls_dir, img))

print("Conversion and split completed!")
