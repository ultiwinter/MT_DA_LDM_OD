import json
from collections import defaultdict

class ClassAugmentationPrioritizer:
    def __init__(self, annotation_file):
        self.annotation_file = annotation_file
        self.coco_data = self._load_annotations()
        self.category_names = self._extract_category_names()
        self.class_counts = self._count_annotations_per_class()
        self.class_augmentation_counts = self._calculate_class_augmentation_counts()

    def _load_annotations(self):
        with open(self.annotation_file) as f:
            return json.load(f)

    def _extract_category_names(self):
        return {category['id']: category['name'] for category in self.coco_data['categories']}

    def _count_annotations_per_class(self):
        class_counts = defaultdict(int)
        for ann in self.coco_data['annotations']:
            class_counts[ann['category_id']] += 1
        return class_counts

    def _get_augmentation_count(self, instance_count):
        if instance_count < 5:
            return 335
        elif 5 <= instance_count < 10:
            return 130
        elif 10 <= instance_count < 20:
            return 80
        elif 20 <= instance_count < 30:
            return 45
        elif 30 <= instance_count < 50:
            return 35
        elif 50 <= instance_count < 75:
            return 20
        elif 75 <= instance_count < 100:
            return 15
        elif 100 <= instance_count < 150:
            return 10
        elif 150 <= instance_count < 250:
            return 7
        elif 250 <= instance_count < 500:
            return 3
        elif 500 <= instance_count < 1000:
            return 2
        elif instance_count > 3000:
            return 0
        else:
            return 1

    def _calculate_class_augmentation_counts(self):
        class_augmentation_counts = {}
        for class_id, count in self.class_counts.items():
            aug_count = self._get_augmentation_count(count)
            class_augmentation_counts[class_id] = aug_count
        return class_augmentation_counts

    def get_category_augmentation_dict(self):
        category_augmentation = {}
        for class_id, aug_count in self.class_augmentation_counts.items():
            category_name = self.category_names.get(class_id, "Unknown")
            category_augmentation[category_name] = aug_count
        return category_augmentation
    
    def get_total_augmentations(self):
        total_augmentations = 0
        for class_id, count in self.class_counts.items():
            augmentation_count = self.class_augmentation_counts[class_id]
            total_augmentations += count * augmentation_count
        return total_augmentations

    def get_total_instances_count(self):
        total_instances = 0
        for class_id, count in self.class_counts.items():
            total_instances += count
        return total_instances


    def save_class_augmentation_counts(self, output_file='class_augmentation_counts.txt'):
        with open(output_file, 'w') as f:
            f.write("Class ID\tClass Name\tCounts\tAugmentations\n")
            for class_id, aug_count in self.class_augmentation_counts.items():
                class_name = self.category_names.get(class_id, "Unknown")
                count = self.class_counts[class_id]
                f.write("{}\t{}\t{}\t{}\n".format(class_id, class_name, count, aug_count))
        print("Class augmentation counts have been saved to {}".format(output_file))

prioritizer = ClassAugmentationPrioritizer(
    annotation_file='/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/my_modified_train_split.json' # my_modified_train_split.json  instances_train.json
)

category_augmentation_dict = prioritizer.get_category_augmentation_dict()
print(category_augmentation_dict)
total_augs = prioritizer.get_total_augmentations()
print(total_augs)
total_instances = prioritizer.get_total_instances_count()
print(total_instances)


