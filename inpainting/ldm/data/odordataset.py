from PIL import Image, ImageDraw
import numpy as np
import os
from torchvision import transforms
from torch.utils.data import Dataset

class ODORBaseConfig(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 ):
        self.data_root = data_root
        with open(txt_file, "r") as f:
            self.image_paths = f.read().splitlines()

        self._length = len(self.image_paths)
        self.size = size
        self.interpolation = {"bilinear": Image.BILINEAR,
                              "bicubic": Image.BICUBIC,
                              "lanczos": Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def _load_image(self, image_path):
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize the image if size is specified
        if self.size:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        # Apply horizontal flip
        image = self.flip(image)

        # Convert to numpy array
        image = np.array(image).astype(np.float32)

        # Create mask
        mask = self._create_center_mask(image.shape[0], image.shape[1])

        # Create masked image
        masked_image = image * (mask / 255)

        return image, mask, masked_image

    def _create_center_mask(self, height, width):
        mask = Image.new("L", (width, height), 0) 
        draw = ImageDraw.Draw(mask)
        
        
        mask_width = int(width * 0.5) 
        mask_height = int(height * 0.5)
        
        left = (width - mask_width) // 2
        top = (height - mask_height) // 2
        right = left + mask_width
        bottom = top + mask_height

        draw.rectangle([left, top, right, bottom], fill='white')

        mask = np.array(mask).astype(np.float32)

        return mask

    def __getitem__(self, i):
        image_name = os.path.splitext(os.path.basename(self.image_paths[i]))[0]
        image_path = os.path.join(self.data_root, self.image_paths[i])

        caption = image_name.split("_")[-3]

        image, mask, masked_image = self._load_image(image_path)

        example = {"image": image, "mask": mask, "masked_image": masked_image, "txt": caption}

        # normalize the images and mask
        for k in example:
            if k != "txt": 
                example[k] = example[k] * 2.0 / 255.0 - 1.0

        return example

class OdorTrainImagesDatasetConfig(ODORBaseConfig):
    def __init__(self, **kwargs):
        super().__init__(txt_file="/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/train_images.txt",
                         data_root="/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/bbox_images_train",
                         **kwargs)
