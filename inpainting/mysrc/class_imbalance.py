import json
from collections import defaultdict
from tqdm import tqdm

class ImageAugmentationPrioritizer:
    def __init__(self, annotation_file, max_augmentations=10):
        self.annotation_file = annotation_file
        self.max_augmentations = max_augmentations
        self.coco_data = self._load_annotations()
        self.class_counts = self._count_annotations_per_class()
        self.class_ratings = self._generate_class_ratings()
        self.image_ratings = self._rate_images()
        self.image_augmentation_counts = self._calculate_augmentation_counts()
        self.sum_augmentations_counts = self.sum_all_augmentations()

    def _load_annotations(self):
        with open(self.annotation_file) as f:
            return json.load(f)

    def _count_annotations_per_class(self):
        class_counts = defaultdict(int)
        for ann in self.coco_data['annotations']:
            class_counts[ann['category_id']] += 1
        return class_counts

    def _get_class_rating(self, count):
        if count > 1000:
            return 0
        elif count > 500:
            return 1
        elif count > 100:
            return 2
        elif count > 10:
            return 3
        else:
            return 4 

    def _generate_class_ratings(self):
        return {class_id: self._get_class_rating(count) for class_id, count in self.class_counts.items()}

    def _rate_images(self):
        image_ratings = {}
        for image in self.coco_data['images']:
            image_id = image['id']
            file_name = image['file_name']
            image_classes = set(ann['category_id'] for ann in self.coco_data['annotations'] if ann['image_id'] == image_id)
            
            total_rating = sum(self.class_ratings[class_id] for class_id in image_classes)
            image_ratings[file_name] = total_rating
        return image_ratings

    def _calculate_augmentation_counts(self):
        sorted_image_ratings = sorted(self.image_ratings.items(), key=lambda x: x[1], reverse=True)
        image_augmentation_counts = {}
        for file_name, rating in sorted_image_ratings:
            aug_count = int(1 + (rating - 1) / (40 - 1) * (self.max_augmentations - 1))
            image_augmentation_counts[file_name] = aug_count
        return image_augmentation_counts
    
    def sum_all_augmentations(self):
        count=0
        for v in self.image_augmentation_counts.values():
            count+=v

        self.sum_augmentations_counts = count
        return self.sum_augmentations_counts

    def save_image_ratings(self, output_file='image_ratings.txt'):
        sorted_image_ratings = sorted(self.image_ratings.items(), key=lambda x: x[1], reverse=True)
        with open(output_file, 'w') as f:
            for file_name, total_rating in sorted_image_ratings:
                f.write(f'File Name: {file_name}, Rating: {total_rating}\n')
        print(f"Image ratings have been saved to {output_file}")

    def save_augmentation_counts(self, output_file='image_augmentation_counts.txt'):
        with open(output_file, 'w') as f:
            for file_name, aug_count in self.image_augmentation_counts.items():
                f.write(f'File Name: {file_name}, Augmentations: {aug_count}\n')
        print(f"Image augmentation counts have been saved to {output_file}")

    def get_class_ratings(self):
        return self.class_ratings

    def get_image_augmentation_counts(self):
        return self.image_augmentation_counts
    
    def calculate_total_objects_augmented(self):

        total_objects_augmented = {}
        for image in self.coco_data['images']:
            image_id = image['id']
            file_name = image['file_name']

            num_objects = sum(1 for ann in self.coco_data['annotations'] if ann['image_id'] == image_id)
            
            aug_count = self.image_augmentation_counts.get(file_name, 0)

            total_objects_augmented[file_name] = num_objects * aug_count

        return total_objects_augmented


prioritizer = ImageAugmentationPrioritizer('/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/my_modified_train_split.json')
prioritizer.save_image_ratings()
prioritizer.save_augmentation_counts()
class_ratings = prioritizer.get_class_ratings()
augmentation_counts = prioritizer.get_image_augmentation_counts()
print(f"Summ all augs: {prioritizer.sum_augmentations_counts}")
print(f"sum ALLLLL images: {sum(prioritizer.calculate_total_objects_augmented().values())}")
