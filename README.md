# Master Thesis: Data Augmentation using Latent Diffusion Models for Object Detection

This repository contains the full source code used in the master thesis titled **"Data Augmentation using Latent Diffusion Models for Object Detection"**.

The thesis explores advanced data augmentation techniques to address class imbalance and data scarcity in object detection tasks, particularly in Object Detection for Olfactory References (ODOR) datasets. It integrates two powerful generative pipelines—**Inpainting** and **ControlNet**—built on top of Latent Diffusion Models. The effectiveness of these augmentations is evaluated using a YOLO-based object detector.

**Core Components:**
- **Inpainting Pipeline:** Multiple object-aware and background augmentation strategies using various masking modes.
- **ControlNet Pipeline:** Guided synthesis and reintegration of objects through finetuned diffusion models.
- **YOLO Evaluation:** Preparation, merging, and training of YOLO datasets to evaluate the effect of augmentations on detection performance.

This codebase is modular and reproducible, enabling further experimentation with dataset augmentation strategies for domain-specific object detection.


## Inpainting Pipeline

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





## ControlNet Augmentation Pipeline

This module focuses on class-balanced data augmentation using a ControlNet-based diffusion pipeline. It includes dataset preparation, model finetuning, inference, and reintegration of generated objects.

### 📁 `ControlNet/`

- **`class_imbalance_ann.py`**  
  Determines the number of required augmentations for underrepresented classes. Ensures a minimum of **1000 instances** for each class across the dataset.

- **`odor_controlnet_dataset.py`**  
  Defines the ODOR dataset for ControlNet finetuning, including input preprocessing and dataset loading utilities.

- **`setup_train_json_controlnet.py`**  
  Prepares the ODOR dataset annotations and file structure specifically for finetuning with ControlNet.

- **`finetune_stable_diffusion_odor_DDP.py`**  
  Finetunes a Stable Diffusion model with ControlNet using **multi-GPU training** (Distributed Data Parallel). Supports scalable training for high-performance environments.

- **`my_auto_hed2img_DPP.py`**  
  Performs inference using the finetuned ControlNet model to generate synthesized object images. Supports parallelized processing with DDP.

- **`overlay_bbox_imgs_plus_random.py`**  
  Reinserts the generated object images back into the original artwork images. If non-overlapping placement is not feasible, it overlays them on blank canvases.  
  Also generates **COCO-style JSON annotations** for the new augmented dataset.




## YOLO & Object Detection

This module handles preparing the YOLO dataset, merging multiple data sources, and training the YOLO object detection model.

- **`inpainting/mysrc/my_yolo_setup_modified.py`**  
  Converts the ODOR dataset (including augmented images) into YOLO format. Handles bounding box conversion and directory structuring for training.

- **`YOLO/merge_yolo_datasets.py`**  
  Merges multiple YOLO-format datasets into a single unified dataset. Useful for combining original, inpainted, and ControlNet-augmented images.

- **`YOLO/my_yolo_train.py`**  
  Trains a YOLO model using the prepared and merged dataset. Includes training configuration and checkpointing logic.

