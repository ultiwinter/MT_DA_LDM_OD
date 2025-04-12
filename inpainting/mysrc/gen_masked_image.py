import cv2
import numpy as np

image_path = "/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/images_train_train_modified/e792b9ca-1ea3-0b5d-cb41-c5b9051a00da.jpg"  # Change this to your image path
mask_path = "/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/novel_masking_masks/e792b9ca-1ea3-0b5d-cb41-c5b9051a00da_donkey_7642_mask.png"    # Change this to your mask path
output_path = "/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/misc/masked_img.png"

image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

mask = cv2.bitwise_not(mask)

_, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

if len(image.shape) == 3:
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

white_background = np.full_like(image, 255)

masked_image = np.where(mask == 255, image, white_background)

cv2.imwrite(output_path, masked_image)

print(f"Masked image saved as {output_path}")
