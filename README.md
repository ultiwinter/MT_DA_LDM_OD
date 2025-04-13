## 🖌️ Inpainting Pipeline

This module focuses on augmenting images in the ODOR dataset by applying various inpainting techniques to address class imbalance. Below is a breakdown of the workflow and the responsibilities of each script.

### 📁 `inpainting/mysrc/`

- **`class_imbalance.py`**  
  Determines the number of augmentations required per image based on a class rating score. Images containing rarer classes receive higher scores, resulting in more augmentations.

- **`data_analysis.py`**  
  Analyzes the ODOR dataset, reporting:
  - Number of annotations per class  
  - Supercategory classification for each class

- **`split_train_val_modified.py`**  
  Splits the ODOR training dataset into training and validation sets (80-20 split). It ensures:
  - All classes are present in both sets  
  - Edge cases (e.g., classes with only 3 instances) are handled by enforcing at least one instance in the validation set

- **`creating_masks.py`**  
  Generates masks based on the selected **masking strategy**:
  - `entropy_based`: High entropy regions  
  - `reversed_entropy_based`: Low entropy regions  
  - `novel_masking`: Adaptive strategy  
  - `saliency_detection`: Saliency-based areas  
  - `reversed_saliency`: Non-salient areas  
  - `gradient_based`: High gradient regions  
  - `whole`: Masks entire object (bounding box)  
  - `nonobject_random_mask`: Object-preserving background masking

- **`automatic_inpaining.py`**  
  Runs inference with the inpainting model and generates the augmented images based on the masked regions.

- **`synthesized_annotations_generation.py`**  
  Generates COCO-style annotations for the newly synthesized inpainted images.

### 📄 Other Files

- **`combine_augs_deluxe.py`**  
  Combines all augmented objects that originated from the same source image into a single image. This avoids repetition of unaugmented objects across multiple augmented images, which helps reduce overfitting in downstream detection models.

- **`adjust_json_ann.py`**  
  Adjusts bounding box annotations if they were incorrectly scaled or formatted during the augmentation process.
