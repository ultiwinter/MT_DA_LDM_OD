import json
import argparse
from collections import defaultdict

class CocoAnalyzer:
    def __init__(self, json_file_path):
        with open(json_file_path, 'r') as file:
            self.data = json.load(file)
        self.categories = self.data.get('categories', [])
        self.annotations = self.data.get('annotations', [])
        self.category_summary = None
        self.hypercategories = None
        self.images = self.data.get('images', [])
        self.my_classes_num = 0

    def get_category_info(self):
        category_info = defaultdict(int)
        for annotation in self.annotations:
            category_id = annotation['category_id']
            category_info[category_id] += 1

        categories = {category['id']: category['name'] for category in self.categories}
        self.category_summary = {categories[cat_id]: count for cat_id, count in category_info.items()}
        self.category_summary = dict(sorted(self.category_summary.items(), key=lambda item: item[1], reverse=True))
        return self.category_summary

    def get_hypercategories_info(self):
        self.hypercategories = defaultdict(set)
        for category in self.categories:
            if 'supercategory' in category:
                self.hypercategories[category['supercategory']].add(category['name'])

        return self.hypercategories
    
    def get_all_class_names(self):
        return [category['name'] for category in self.categories]
    
    def get_image_names(self):
        return [image['file_name'] for image in self.images]
    
    def get_category_ids(self):
        return [category['id'] for category in self.categories]

    def print_summary(self):
        self.category_summary = self.get_category_info()
        self.hypercategories_summary = self.get_hypercategories_info()

        print("Category Summary:")
        for category, count in self.category_summary.items():
            print(f"{category}: {count} annotations")

        print("\nHypercategories Summary:")
        for hypercategory, categories in self.hypercategories_summary.items():
            print(f"{hypercategory}: {', '.join(categories)} categories")
        
        print(f"Total number of categories (classes): {len(self.category_summary)}")

if __name__ == "__main__":
    file = "/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/my_modified_val_split75.json"  # my_val_split.json
    # "/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/instances_train.json"
    analyzer = CocoAnalyzer(file)
    analyzer.print_summary()
