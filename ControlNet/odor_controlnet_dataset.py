import json
import cv2
import numpy as np

from torch.utils.data import Dataset


class ODORDatasetCN(Dataset):
    def __init__(self):
        self.data = []
        with open('/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/seg2img_controlnet_train_all_data/seg2img_odor_train_all_data.jsonl', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread(source_filename)
        target = cv2.imread(target_filename)

        source = cv2.resize(source, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        target = cv2.resize(target, (256, 256), interpolation=cv2.INTER_LANCZOS4)

        if source is None:
            print(f"Warning: Failed to load source image at {source_filename}")
            return None
        if target is None:
            print(f"Warning: Failed to load target image at {target_filename}")
            return None

        # do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

