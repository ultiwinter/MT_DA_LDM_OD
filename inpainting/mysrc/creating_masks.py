import json
import os
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm
import random
from matplotlib import pyplot as plt
import cv2 as cv
from skimage.feature import hog
from skimage import exposure
from skimage.filters import rank
from skimage.morphology import disk
import glob
from skimage.filters.rank import entropy
from skimage.morphology import disk

class MaskCreator:
    def __init__(self, json_path, output_dir):
        self.json_path = json_path
        self.output_dir = output_dir
        self.data = self.load_json()
        self.image_id_to_file_name = self.create_image_id_to_file_name_dict()
        self.category_id_to_name = self.create_category_id_to_name_dict()

        os.makedirs(self.output_dir, exist_ok=True)

    def load_json(self):
        try:
            with open(self.json_path, 'r') as file:
                data = json.load(file)
            return data
        except FileNotFoundError:
            print(f"Error: The file {self.json_path} was not found")
            exit(1)
        except json.JSONDecodeError:
            print(f"Error: The file {self.json_path} is not a valid JSON file")
            exit(1)
    
    def create_image_id_to_file_name_dict(self):
        try:
            return {image['id']: image['file_name'] for image in self.data['images']}
        except KeyError as e:
            print(f"Error: Missing key {e} in the images data.")
            exit(1)

    def create_category_id_to_name_dict(self):
        try:
            return {category['id']: category['name'] for category in self.data['categories']}
        except KeyError as e:
            print(f"Error: Missing key {e} in the categories data.")
            exit(1)
    
    def create_whole_mask(self, image_size, bbox):
        try:
            mask = Image.new('L', (image_size[1], image_size[0]), 0)
            draw = ImageDraw.Draw(mask)
            x0, y0, w, h = bbox
            x1, y1 = x0 + w, y0 + h
            if x0 < 0 or y0 < 0 or x1 > image_size[1] or y1 > image_size[0]:
                print("Error: Bounding box is out of image bounds.")
            draw.rectangle([x0, y0, x1, y1], outline='white', fill='white')
            return mask
        except Exception as e:
            print(f"Error while creating mask: {e}")
            return None
    
    def create_partial_mask(self, image_size, bbox, factor):
        try:
            mask = Image.new('L', (image_size[1], image_size[0]), 0)
            draw = ImageDraw.Draw(mask)
            
            partial_width = bbox[2] * factor
            partial_height = bbox[3] * factor

            partial_width = max(1, int(partial_width))
            partial_height = max(1, int(partial_height))
            
            max_x_offset = bbox[2] - partial_width
            max_y_offset = bbox[3] - partial_height
            max_x_offset = max(0, int(max_x_offset))
            max_y_offset = max(0, int(max_y_offset))
            x_offset = random.randint(0, max_x_offset)
            y_offset = random.randint(0, max_y_offset)

            x0=bbox[0] + x_offset
            y0=bbox[1] + y_offset
            x1=x0+partial_width
            y1=y0+partial_width

            draw.rectangle([x0, y0, x1, y1], outline='white', fill='white')
            return mask
        except Exception as e:
            print(f"Error while creating partial mask: {e}")
            return None
        
    def create_partial_pixelwise_mask(self, image_size, bbox, factor):
        try:
            mask = Image.new('L', (int(image_size[1]), int(image_size[0])), 0)
            draw = ImageDraw.Draw(mask)

            total_pixels = int(bbox[2]) * int(bbox[3])
            num_pixels_to_mask = int(total_pixels * factor)

            num_pixels_to_mask = max(1, num_pixels_to_mask)

            all_pixels = [(x, y) for x in range(int(bbox[0]), int(bbox[0]) + int(bbox[2])) for y in range(int(bbox[1]), int(bbox[1]) + int(bbox[3]))]

            masked_pixels = random.sample(all_pixels, num_pixels_to_mask)

            for pixel in masked_pixels:
                draw.point(pixel, fill='white')

            return mask
        except Exception as e:
            print(f"Error while creating partial pixelwise mask: {e}")
            return None
    
    
    def edge_detection_mask(self, image_path, bbox):
        image_rgb = cv.imread(image_path)
        
        x, y, w, h = map(int, bbox)
        bbox_img = image_rgb[y:y+h, x:x+w]

        grayscale_image = cv.cvtColor(bbox_img, cv.COLOR_BGR2GRAY)

        def gaussian(x, y, sigma):
            return np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)

        def gaussian_kernel(size, sigma):
            kernel = np.zeros((size, size))
            center = size // 2
            for i in range(size):
                for j in range(size):
                    kernel[i, j] = gaussian(i - center, j - center, sigma)
            return kernel / np.sum(kernel)

        def convolution(image, kernel):
            height, width = image.shape
            kernel_size = kernel.shape[0]
            padding = kernel_size // 2
            padded_image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
            smoothed_image = np.zeros_like(image, dtype=np.float32)
            for y in range(height):
                for x in range(width):
                    smoothed_image[y, x] = np.sum(padded_image[y:y+kernel_size, x:x+kernel_size] * kernel)
            return smoothed_image.astype(np.uint8)

        kernel_size = 3
        sigma = 2
        gaussian_filter = gaussian_kernel(kernel_size, sigma)

        smoothed_image = convolution(grayscale_image, gaussian_filter)

        gradient_x = cv.Sobel(smoothed_image, cv.CV_64F, 1, 0, ksize=3)
        gradient_y = cv.Sobel(smoothed_image, cv.CV_64F, 0, 1, ksize=3)

        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

        threshold_value = 100
        edges = np.uint8(gradient_magnitude > threshold_value) * 255

        mask = np.zeros((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)
        mask[y:y+h, x:x+w] = edges

        return mask
    
    def saliency_detection_mask(self, image_path, bbox):
        img = cv.imread(image_path)
        x, y, w, h = map(int, bbox)
        bbox_img = img[y:y+h, x:x+w]
        saliency = cv.saliency.StaticSaliencyFineGrained_create()
        (success, saliencyMap) = saliency.computeSaliency(bbox_img)
        if not success:
            raise RuntimeError("Saliency detection failed")
        mask = np.zeros_like(img[:, :, 0], dtype=np.uint8)
        mask_bbox = (saliencyMap * 255).astype("uint8")
        _, mask[y:y+h, x:x+w] = cv.threshold(mask_bbox, 128, 255, cv.THRESH_BINARY)
        return mask
    
    def reversed_saliency_detection_mask(self, image_path, bbox):
        img = cv.imread(image_path)
        x, y, w, h = map(int, bbox)
        bbox_img = img[y:y+h, x:x+w]
        saliency = cv.saliency.StaticSaliencyFineGrained_create()
        (success, saliencyMap) = saliency.computeSaliency(bbox_img)
        if not success:
            raise RuntimeError("Saliency detection failed")

        mask_bbox = (saliencyMap * 255).astype("uint8")
        _, binary_mask_bbox = cv.threshold(mask_bbox, 128, 255, cv.THRESH_BINARY)

        inverted_mask_bbox = cv.bitwise_not(binary_mask_bbox)

        mask = np.zeros_like(img[:, :, 0], dtype=np.uint8)
        mask[y:y+h, x:x+w] = inverted_mask_bbox  #apply the inverted mask to the bbox region

        return mask

    
    def gradient_based_mask(self, image_path, bbox):
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        x, y, w, h = map(int, bbox)
        bbox_img = img[y:y+h, x:x+w]
        grad_x = cv.Sobel(bbox_img, cv.CV_64F, 1, 0, ksize=5)
        grad_y = cv.Sobel(bbox_img, cv.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = cv.magnitude(grad_x, grad_y)
        mean_gradient = np.mean(gradient_magnitude)
        _, mask_bbox = cv.threshold(gradient_magnitude, mean_gradient, 255, cv.THRESH_BINARY)
        mask = np.zeros_like(img, dtype=np.uint8)
        mask[y:y+h, x:x+w] = mask_bbox.astype(np.uint8)
        mask[mask > 0] = 255
        return mask
        
    def hog_based_mask(self, image_path, bbox):
        img = cv.imread(image_path)
        x, y, w, h = map(int, bbox)
        bbox_img = img[y:y+h, x:x+w]

        fd, hog_image = hog(bbox_img, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualize=True, channel_axis=-1)

        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, hog_image.max()))
        mean_intensity = np.mean(bbox_img)
        _, mask_bbox = cv.threshold(hog_image_rescaled, mean_intensity, 255, cv.THRESH_BINARY)
        mask = np.zeros_like(img[:, :, 0], dtype=np.uint8)
        mask[y:y+h, x:x+w] = mask_bbox.astype(np.uint8)
        mask[mask > 0] = 255
        return mask
    
    def entropy_nonbinary_mask(self, image_path, bbox):
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        
        x, y, w, h = map(int, bbox)
        
        bbox_img = img[y:y+h, x:x+w]
        
        entropy_map = rank.entropy(bbox_img, disk(5))
        normalized_entropy_map = ((entropy_map - entropy_map.min()) / 
                                (entropy_map.max() - entropy_map.min()) * 255).astype(np.uint8)
        
        mask = np.zeros_like(img, dtype=np.uint8)
        mask[y:y+h, x:x+w] = normalized_entropy_map
        
        return mask

    
    def entropy_based_mask(self, image_path, bbox):
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        x, y, w, h = map(int, bbox)
        bbox_img = img[y:y+h, x:x+w]
        entropy_map = rank.entropy(bbox_img, disk(5))
        threshold = np.median(entropy_map)#  + np.std(entropy_map)
        _, mask_bbox = cv.threshold(entropy_map, threshold, 255, cv.THRESH_BINARY)
        mask = np.zeros_like(img, dtype=np.uint8)
        mask[y:y+h, x:x+w] = mask_bbox.astype(np.uint8)
        mask[mask > 0] = 255 
        return mask

    def reversed_entropy_based_mask(self, image_path, bbox):
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        x, y, w, h = map(int, bbox)
        bbox_img = img[y:y+h, x:x+w]
        entropy_map = rank.entropy(bbox_img, disk(5))
        threshold = np.median(entropy_map)#  + np.std(entropy_map)
        _, mask_bbox = cv.threshold(entropy_map, threshold, 255, cv.THRESH_BINARY)

        inverted_mask_bbox = cv.bitwise_not(mask_bbox.astype(np.uint8))

        mask = np.zeros_like(img, dtype=np.uint8)
        mask[y:y+h, x:x+w] = inverted_mask_bbox
        return mask
    
    def novel_masking(self, image_path, bbox):

        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        x, y, w, h = map(int, bbox)
        bbox_img = img[y:y+h, x:x+w]

        entropy_map = entropy(bbox_img, disk(5))
        threshold = np.median(entropy_map)#  + np.std(entropy_map)
        _, mask_bbox = cv.threshold(entropy_map, threshold, 255, cv.THRESH_BINARY)

        center_x, center_y = w // 2, h // 2
        center_radius_x, center_radius_y = w // 10, h // 10
        center_region = mask_bbox[
            center_y - center_radius_y : center_y + center_radius_y,
            center_x - center_radius_x : center_x + center_radius_x,
        ]

        # # whether to reverse or not
        # majority_color = np.mean(center_region)
        # if majority_color < 128:  # Majority black
        #     mask_bbox = cv.bitwise_not(mask_bbox.astype(np.uint8))

        mask = np.zeros_like(img, dtype=np.uint8)
        mask[y:y+h, x:x+w] = mask_bbox.astype(np.uint8)

        if w > h: 
            left_region = mask_bbox[:, :w // 2]
            right_region = mask_bbox[:, w // 2:]
            if np.sum(left_region) > np.sum(right_region):
                mask[y:y+h, x:x+w//2] = 255
                mask[y:y+h, x+w//2:x+w] = 0
            else:
                mask[y:y+h, x+w//2:x+w] = 255
                mask[y:y+h, x:x+w//2] = 0
        else:
            top_region = mask_bbox[:h // 2, :]
            bottom_region = mask_bbox[h // 2:, :]
            if np.sum(top_region) > np.sum(bottom_region):
                mask[y:y+h//2, x:x+w] = 255
                mask[y+h//2:y+h, x:x+w] = 0
            else:
                mask[y+h//2:y+h, x:x+w] = 255
                mask[y:y+h//2, x:x+w] = 0
                
        return mask
    
    def refined_smart_entropy_based_mask(self, image_path, bbox):
            img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
            x, y, w, h = map(int, bbox)
            
            bbox_img = img[y:y+h, x:x+w]
            
            entropy_map = entropy(bbox_img, disk(5))
            threshold = np.median(entropy_map)
            _, mask_bbox = cv.threshold(entropy_map, threshold, 255, cv.THRESH_BINARY)
            
            inverted_mask_bbox = cv.bitwise_not(mask_bbox.astype(np.uint8))
            
            mask = np.zeros_like(img, dtype=np.uint8)
            mask[y:y+h, x:x+w] = inverted_mask_bbox
            
            num_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
            
            if num_labels <= 1:
                return np.zeros_like(mask, dtype=np.uint8)
            
            largest_label = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA]) 
            
            largest_mask = np.zeros_like(mask, dtype=np.uint8)
            largest_mask[labels == largest_label] = 255
            
            # Post-process the mask for smoothing:
            
            # 1. gaussian nlur to smooth edges
            blurred_mask = cv.GaussianBlur(largest_mask, (7, 7), 0)
            
            # 2. threshold again to ensure binary result after blurring
            _, smoothed_mask = cv.threshold(blurred_mask, 127, 255, cv.THRESH_BINARY)
            
            # 3. morphological closing to fill small gaps
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
            closed_mask = cv.morphologyEx(smoothed_mask, cv.MORPH_CLOSE, kernel)
            
            # 4. contour smoothing
            contours, _ = cv.findContours(closed_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            final_mask = np.zeros_like(closed_mask)
            cv.drawContours(final_mask, contours, -1, (255), thickness=cv.FILLED)
            
            return final_mask

            
    def segment_and_apply_mask(self, image_name, bbox):
        try:
            img = cv.imread(image_name)

            try:
                x, y, w, h = map(int, bbox)
            except ValueError as e:
                print(f"Error converting bbox coordinates to integers: {e}")
                return None

            roi = img[y:y+h, x:x+w]

            image_contours = np.zeros((roi.shape[0], roi.shape[1], 1), np.uint8)
            image_binary = np.zeros((roi.shape[0], roi.shape[1], 1), np.uint8)

            for channel in range(roi.shape[2]):
                ret, image_thresh = cv.threshold(roi[:, :, channel], 38, 255, cv.THRESH_BINARY)
                contours = cv.findContours(image_thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0]
                cv.drawContours(image_contours, contours, -1, (255, 255, 255), 3)

            contours = cv.findContours(image_contours, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0]

            cv.drawContours(image_binary, [max(contours, key=cv.contourArea)], -1, (255, 255, 255), -1)

            full_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
            full_mask[y:y+h, x:x+w] = image_binary[:, :, 0]

            mask = Image.fromarray(full_mask)

            return mask

        except Exception as e:
            print(f"Error while segmenting and applying mask: {e}")
            return None
        
    def object_border_mask(self, image_shape, bboxes):
        final_mask = np.zeros(image_shape, dtype=np.uint8)
        
        for bbox in bboxes:
            x, y, w, h = map(int, bbox)

            left_space = x
            right_space = image_shape[1] - (x + w)
            top_space = y
            bottom_space = image_shape[0] - (y + h)
            max_thickness = min(left_space, right_space, top_space, bottom_space)
            thickness = int(min(w, h) * 0.25)
            thickness = min(thickness, max_thickness)

            x_out = max(x - thickness, 0)
            y_out = max(y - thickness, 0)
            w_out = min(w + 2 * thickness, image_shape[1] - x_out)
            h_out = min(h + 2 * thickness, image_shape[0] - y_out)
            cv.rectangle(final_mask, (x_out, y_out), (x_out + w_out, y_out + h_out), color=255, thickness=-1)

        for bbox in bboxes:
            x, y, w, h = map(int, bbox)
            cv.rectangle(final_mask, (x, y), (x + w, y + h), color=0, thickness=-1)

        return final_mask

    def nonobject_random_mask(self, image_shape, bboxes):
        non_bbox_mask = np.ones(image_shape, dtype=np.uint8) * 255
        for bbox in bboxes:
            x, y, w, h = map(int, bbox)
            cv.rectangle(non_bbox_mask, (x, y), (x + w, y + h), color=0, thickness=-1)

        mask = np.zeros(image_shape, dtype=np.uint8)

        h, w = mask.shape
        for _ in range(8):
            mask_width = random.randint(int(0.25 * w), int(0.35 * w))
            mask_height = random.randint(int(0.25 * h), int(0.35 * h))
            x = random.randint(0, w - mask_width)
            y = random.randint(0, h - mask_height)

    
            random_mask = np.zeros_like(mask)
            cv.rectangle(random_mask, (x, y), (x + mask_width, y + mask_height), color=255, thickness=-1)

        
            random_mask = cv.bitwise_and(random_mask, non_bbox_mask)

            mask = cv.bitwise_or(mask, random_mask)

        return mask

    

    def display_images(self, original_image, mask):
        original_image = original_image.convert("RGB")
        mask = mask.convert("RGB")
        original_np = np.array(original_image)
        mask_np = np.array(mask)

        plt.ion()
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(mask_np, cmap='gray')
        axes[1].set_title('Mask')
        axes[1].axis('off')

        plt.show()

        plt.pause(0.001)  
        while plt.fignum_exists(fig.number):
            plt.pause(0.1)

                    
    def process_masks(self,Whole_Obj_RIO=False, use_partial=False, use_pixelwise=False, factor=0.3, mode=""):
        for annotation in tqdm(self.data['annotations'], desc="Processing masks for annotations"):
            try:
                image_name = self.image_id_to_file_name[annotation['image_id']]
                category_name = self.category_id_to_name[annotation['category_id']]
                bbox = annotation['bbox']
                
                image_info = next((image for image in self.data['images'] if image['id'] == annotation['image_id']), None)
                if image_info is None:
                    print(f"Warning: No matching image found for annotation {annotation['id']}. Skipping!")
                    continue
                
                if 'height' not in image_info or 'width' not in image_info:
                    print(f"Warning: Missing height or width for image {image_info['file_name']}. Skipping!")
                    continue
                
                image_height = image_info['height']
                image_width = image_info['width']
                image_size = (image_height, image_width)
                
                if mode=="segment":
                    base_path = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/images/'
                    image_path = base_path + image_name
                    mask = self.segment_and_apply_mask(image_path, bbox)
                    # original_image = Image.open(image_name).convert("RGB")
                    # self.display_images(original_image, mask)
                elif mode=="edge_detection":
                        base_path = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/images/'
                        image_path = base_path + image_name
                        mask = self.edge_detection_mask(image_path, bbox)
                elif mode=="saliency_detection":
                        base_path = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/images/'
                        image_path = base_path + image_name
                        mask = self.saliency_detection_mask(image_path, bbox)
                elif mode=="gradient_based":
                        base_path = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/images/'
                        image_path = base_path + image_name
                        mask = self.gradient_based_mask(image_path, bbox)
                elif mode=="hog_based":
                        base_path = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/images/'
                        image_path = base_path + image_name
                        mask = self.hog_based_mask(image_path, bbox)
                elif mode=="entropy_based":
                        base_path = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/images/'
                        image_path = base_path + image_name
                        mask = self.entropy_based_mask(image_path, bbox)
                elif mode=="reversed_entropy_based":
                        base_path = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/images/'
                        image_path = base_path + image_name
                        mask = self.reversed_entropy_based_mask(image_path, bbox)
                elif mode=="refined_smart_entropy_based":
                        base_path = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/images/'
                        image_path = base_path + image_name
                        mask = self.refined_smart_entropy_based_mask(image_path, bbox)
                elif mode=="reversed_saliency":
                        base_path = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/images/'
                        image_path = base_path + image_name
                        mask = self.reversed_saliency_detection_mask(image_path, bbox)
                elif mode=="object_border":
                        base_path = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/images/'
                        image_path = base_path + image_name
                        mask = self.object_border_mask(image_path, bbox)
                elif mode=="novel_masking":
                        base_path = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/images/'
                        image_path = base_path + image_name
                        mask = self.novel_masking(image_path, bbox)
                elif mode=="entropy_nonbinary":
                        base_path = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/images/'
                        image_path = base_path + image_name
                        mask = self.entropy_nonbinary_mask(image_path, bbox)
                elif mode=="superpixel":
                        base_path = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/images/'
                        image_path = base_path + image_name
                elif mode=="random_partial":
                        base_path = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/images/'
                        image_path = base_path + image_name
                        mask = self.create_partial_mask(image_size, bbox, factor)
                elif mode=="whole":
                        base_path = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/images/'
                        image_path = base_path + image_name
                        mask = self.create_whole_mask(image_size, bbox)

                if category_name=="reptile/amphibia":
                     category_name="reptile-amphibia"
                mask_filename = f"{image_name.rsplit('.', 1)[0]}_{category_name}_{annotation['id']}_mask.png"
                if isinstance(mask, np.ndarray):
                    mask = Image.fromarray(mask)
                save_path = os.path.join(self.output_dir, mask_filename)
                mask.save(save_path)

            except KeyError as e:
                print(f"Error: Missing key {e} in the annotations data.")
            except Exception as e:
                print(f"Error processing annotation {annotation['id']}: {e}")
        
        if mode=="object_border_nonoverlap":
            image_id_to_info = {image['id']: image for image in self.data['images']}

            image_bboxes = {}
            for annotation in tqdm(self.data['annotations'],desc="Collecting bboxes"):
                image_id = annotation['image_id']
                bbox = annotation['bbox']  # COCO format: [x, y, width, height]
                
                if image_id not in image_bboxes:
                    image_bboxes[image_id] = []
                image_bboxes[image_id].append(bbox)

            for image_info in tqdm(self.data['images'], desc="Processing border masks"):
                image_id = image_info['id']
                file_name = image_info['file_name']
                base_path = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/images/'
                image_path = base_path + file_name

                # Load the image to get its dimensions
                image = cv.imread(str(image_path))
                if image is None:
                    print(f"Image {file_name} not found, skipping.")
                    continue

                image_shape = image.shape[:2]  # Get (height, width)
                
                bboxes = image_bboxes.get(image_id, [])
                final_mask = self.object_border_mask(image_shape, bboxes)

                # Save the final mask
                mask_output_path = self.output_dir + f"{file_name.rsplit('.', 1)[0]}_border_mask.png"
                cv.imwrite(str(mask_output_path), final_mask)
            
        if mode == "nonobject_random_mask":
            image_bboxes = {}
            for annotation in tqdm(self.data['annotations'],desc="Collecting bboxes"):
                image_id = annotation['image_id']
                bbox = annotation['bbox']  # COCO format: [x, y, width, height]
                
                if image_id not in image_bboxes:
                    image_bboxes[image_id] = []
                image_bboxes[image_id].append(bbox)

            for image_info in tqdm(self.data['images'], desc="Processing border masks"):
                image_id = image_info['id']
                file_name = image_info['file_name']
                base_path = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/images/'
                image_path = base_path + file_name

                # Load the image to get its dimensions
                image = cv.imread(str(image_path))
                if image is None:
                    print(f"Image {file_name} not found, skipping.")
                    continue

                image_shape = image.shape[:2]  # Get (height, width)
                
                bboxes = image_bboxes.get(image_id, [])
                final_mask = self.nonobject_random_mask(image_shape, bboxes)

                # Save the final mask
                mask_output_path = self.output_dir + f"{file_name.rsplit('.', 1)[0]}_border_mask.png"
                cv.imwrite(str(mask_output_path), final_mask)
            
        if mode == "FINAL_FINAL_blank_CN_mask_outside_bboxes":
            image_bboxes = {}
            for annotation in tqdm(self.data['annotations'], desc="Collecting bboxes"):
                image_id = annotation['image_id']
                bbox = annotation['bbox']  # COCO format: [x, y, width, height]

                if image_id not in image_bboxes:
                    image_bboxes[image_id] = []
                image_bboxes[image_id].append(bbox)

            for image_info in tqdm(self.data['images'], desc="Processing mask outside bboxes"):
                
                image_id = image_info['id']
                file_name = image_info['file_name']

                # Process only images that start with "oversampled"
                if not file_name.startswith("blank"):
                    continue
                base_path = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/FINAL_FINAL_overlayed_imgs_controlnet_hed_f20_smin256max512_withClassBalance_random_overlaying_blank/'
                image_path = base_path + file_name

                # Load the image to get its dimensions
                image = cv.imread(str(image_path))
                if image is None:
                    print(f"Image {file_name} not found, skipping.")
                    continue

                image_shape = image.shape[:2]  # Get (height, width)
                bboxes = image_bboxes.get(image_id, [])

                # Create an initial white mask
                mask = np.ones(image_shape, dtype=np.uint8) * 255  # White background

                # Draw black rectangles for each bounding box
                for bbox in bboxes:
                    x, y, width, height = map(int, bbox)
                    cv.rectangle(mask, (x, y), (x + width, y + height), 0, -1)  # Fill the bbox with black

                # Save the final mask
                mask_output_path = self.output_dir + f"{file_name.rsplit('.', 1)[0]}_mask.png"
                cv.imwrite(str(mask_output_path), mask)
        
             

        print("Mask images created successfully.")

if __name__ == "__main__":
    masking_mode="FINAL_FINAL_blank_CN_mask_outside_bboxes"  # edge_detection, saliency_detection, gradient_based, hog_based, entropy_based, object_border_nonoverlap
    print(f"Creating masks on {masking_mode} masking mode.")
    json_path = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/FINAL_FINAL_my_modified_train_plus_withClassBalance_added_hed_blank.json'
    output_dir = f'/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/{masking_mode}_masks/'
    os.makedirs(output_dir, exist_ok=True)

    mask_creator = MaskCreator(json_path, output_dir)
    mask_creator.process_masks(mode=masking_mode)
