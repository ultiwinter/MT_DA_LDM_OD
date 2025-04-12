import os
import shutil
from tqdm import tqdm

def merge_yolo_datasets(baseline_path, augmented_path, combined_path):

    paths = {
        "images": "images/train",
        "labels": "labels/train"
    }

    for subdir in paths.values():
        os.makedirs(os.path.join(combined_path, subdir), exist_ok=True)

    def copy_files(src_path, dst_path, suffix, desc):
        files = [f for f in os.listdir(src_path) if f.endswith(".txt") or f.endswith(('.jpg', '.png'))]
        for filename in tqdm(files, desc=desc):
            name, ext = filename.rsplit('.', 1)
            new_filename = f"{name}{suffix}.{ext}"
            shutil.copy(os.path.join(src_path, filename), os.path.join(dst_path, new_filename))

    for key, subdir in paths.items():
        src = os.path.join(baseline_path, subdir)
        dst = os.path.join(combined_path, subdir)
        copy_files(src, dst, "_baseline", f"Copying baseline {key} files")

    for key, subdir in paths.items():
        src = os.path.join(augmented_path, subdir)
        dst = os.path.join(combined_path, subdir)
        copy_files(src, dst, "_augmented", f"Copying augmented {key} files")

    for split in ["valid", "test"]:
        for key in ["images", "labels"]:
            src = os.path.join(baseline_path, key, split)
            dst = os.path.join(combined_path, key, split)
            os.makedirs(dst, exist_ok=True)
            files = [f for f in os.listdir(src) if f.endswith(".txt") or f.endswith(('.jpg', '.png'))]
            for filename in tqdm(files, desc=f"Copying {split} {key} files"):
                shutil.copy(os.path.join(src, filename), os.path.join(dst, filename))


baseline_dataset = "/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/baseline_orig/baseline_orig"
augmented_dataset = "/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/nonObj_negPrompt/nonObj_negPrompt"
combined_dataset = "/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/TEST8_MERGED_nonobj/TEST8_MERGED_nonobj"


merge_yolo_datasets(baseline_dataset, augmented_dataset, combined_dataset)
print(f"Datasets combined and saved to {combined_dataset}")
