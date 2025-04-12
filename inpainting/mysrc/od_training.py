import argparse
from data_analysis import CocoAnalyzer
from myutils import get_confusion_matrix, get_metrics, tlbr2cthw, collate_fn, nms
import torch
import torch.multiprocessing as mp
from collections import defaultdict
import json
import os
import albumentations as A
from retina_network import MyRetinaModel
from torchvision.transforms import functional as F
import torch
import numpy as np
from sklearn.model_selection import train_test_split
# import nms
from pycocotools.coco import COCO
import random
from dataset import ODORTrainDataset, ODORTestDataset

import time
import datetime
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import logging
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.ERROR)



def get_cmdline_args_and_run():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m', '--mode', type=str, required=True, help='Execution mode. Options: \'train\', \'val\', \'test\'')
    parser.add_argument('-p', '--patchsize', type=int, default=512, help='Patchsize - network will use pxp patches during training and inference')
    parser.add_argument('-b', '--batchsize', type=int, default=8, help='Batchsize')
    parser.add_argument('-nt', '--npatchtrain', type=int, default=6, help='Number of patches per image during training')
    parser.add_argument('-nv', '--npatchval', type=int, default=6, help='Number of patches per image during validation')
    parser.add_argument('-ne', '--nepochs', type=int, default=80, help='Total number of epochs for training')
    parser.add_argument('-se', '--startepoch', type=int, default=0, help='Starting epoch for training (remaining number of training epochs is nepochs-startepoch)')
    parser.add_argument('-lr', '--learningrate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--logckptdir', type=str, default='./output/', help='Directory for lightning logs/checkpoints')
    # parser.add_argument('--resdir', type=str, default='./results', help='Directory for result files')
    parser.add_argument('-c', '--checkptfile', type=str, default=None, help='Path to model file (necessary for reloading/retraining)')
    parser.add_argument('-s', '--seed', type=int, default='31415', help='Seed for randomness, default=31415; set to -1 for random')

    args = parser.parse_args()

    possible_execution_modes = ['train', 'val', 'test']
    if not args.mode in possible_execution_modes:
        print('Error: Execution mode {} is unknown. Please choose one of {}'.format(args.mode, possible_execution_modes))

    if not args.seed == -1:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if not os.path.exists(args.logckptdir):
        print("Creating log directory {}".format(args.logckptdir))
        os.makedirs(args.logckptdir)

    if args.mode == 'train':
        training_val(args)

    if args.mode == 'val':
        training_val(args)

    if args.mode == 'test':
        test(args)


def my_improved_sampling_func(targets, classes, shape, level_dimensions):

    height, width = shape
    level_height, level_width = level_dimensions

    class_bboxes = defaultdict(list)

    for t in targets:
        if t['category_id'] in classes:
            class_bboxes[t['category_id']].append(t['bbox'])
    
    min_bboxes_per_class = min(len(bboxes) for bboxes in class_bboxes.values())

    sampled_bboxes = []
    for cls in classes:
        if cls in class_bboxes:
            class_bboxes_list = class_bboxes[cls]
            sampled_bboxes.extend([class_bboxes_list[i] for i in np.random.choice(len(class_bboxes_list), min_bboxes_per_class, replace=False)])


    if sampled_bboxes:
        bbox = sampled_bboxes[0]
        bbox_x_min, bbox_y_min, bbox_width, bbox_height = bbox

        # Calculate the center of the bounding box
        center_x = bbox_x_min + bbox_width // 2
        center_y = bbox_y_min + bbox_height // 2

        # Calculate the top-left corner of the sampled region ensuring it is within the image boundaries
        sample_x_min = max(center_x - width // 2, 0)
        sample_y_min = max(center_y - height // 2, 0)

        # Ensure the sampled region is within the image boundaries
        sample_x_min = min(sample_x_min, level_width - width)
        sample_y_min = min(sample_y_min, level_height - height)


    else:
        raise ValueError("No bounding boxes available for sampling")

    return sample_x_min, sample_y_min


def training_val(args):
    
    annotation_json = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/instances_train.json'

    # transformations
    tfms = None
    # tfms = A.Compose([
    #             A.HorizontalFlip(p=0.5),
    #             A.VerticalFlip(p=0.5),
    #             A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.1, p=0.5),
    #             A.Normalize(mean=[0.5, 0.58, 0.65], std=[0.2, 0.225, 0.25])],
    #             bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    # create training and validation datasets
    root = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/images'
    train_dataset = ODORTrainDataset(root=root, ann_file=annotation_json, patches_per_image=args.npatchtrain, transform=tfms, patch_size=(args.patchsize, args.patchsize), sample_func=None)
    val_dataset = ODORTrainDataset(root=root, ann_file=annotation_json, patches_per_image=args.npatchval, transform=tfms, patch_size=(args.patchsize, args.patchsize), sample_func=None)

    # split the datasets
    train_indices, val_indices = train_test_split(list(range(len(train_dataset))), train_size=0.8, random_state=args.seed)
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    # Create data loaders
    mp.set_start_method('spawn')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batchsize, shuffle=False, num_workers=0, collate_fn=collate_fn)

    print("Cuda available: {}".format(torch.cuda.is_available()), flush=True)

    cur_time = time.time()
    time_str = datetime.datetime.fromtimestamp(cur_time).strftime('%Y-%m-%d-%H-%M-%S')
    lr_monitor = LearningRateMonitor(logging_interval='step')

    checkpoint_callback = ModelCheckpoint(
        monitor='val_mAP',  # Metric to monitor for saving the best model
        dirpath=args.logckptdir,  # Directory to save the checkpoints
        filename='best-checkpoint-{epoch:02d}-{val_mAP:.2f}',  # Filename format
        save_top_k=3,  # Save only the 3 best models
        mode='max',  # Mode to determine if the monitored metric should be minimized or maximized
        verbose=True 
    )
    trainer = Trainer(
        logger=TensorBoardLogger(save_dir=args.logckptdir, name="baseline_training", version='version_lr{lr}_p{p}_b{b}_{t}'.format(
            lr=args.learningrate, p=args.patchsize, b=args.batchsize, t=time_str)),
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=args.nepochs,
        log_every_n_steps=200,
        progress_bar_refresh_rate=500,
        callbacks=[lr_monitor, checkpoint_callback],
    )
    my_detection_model = MyRetinaModel(num_classes=139, iterations_epoch=len(train_loader), lr=args.learningrate, epochs=args.nepochs)

    if torch.cuda.is_available():
        my_detection_model = my_detection_model.cuda()

    ckpt = None
    if args.checkptfile:
        ckpt = args.checkptfile
        my_detection_model = MyRetinaModel.load_from_checkpoint(ckpt)
        if torch.cuda.is_available():
            my_detection_model = my_detection_model.cuda()
        print("Model loaded!")
    if args.mode == 'train':
        trainer.fit(my_detection_model, train_loader, val_loader)  # removed ckpt_path=ckpt
    if args.mode == 'val':
        trainer.validate(my_detection_model, val_loader, ckpt_path=ckpt)

def test(args):
    annotation_json = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/instances_test.json'
    coco = COCO(annotation_json)

    test_batchsize = args.batchsize
    root = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/images'

    cur_time = time.time()
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(
        logger=None,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=args.nepochs,
        log_every_n_steps=5,
        callbacks=[lr_monitor]
    )

    ckpt = args.checkptfile
    if ckpt != None:
        my_detection_model = MyRetinaModel.load_from_checkpoint(ckpt)

    all_predictions = []
    all_gt = []

    img_ids = coco.getImgIds()

    for img_id in img_ids:
        image_id = img_id.split('.')[0]
        img_info = COCO(annotation_json).loadImgs(image_id)[0]
        ann_ids = COCO(annotation_json).getAnnIds(imgIds=img_id)
        anns = COCO(annotation_json).loadAnns(ann_ids)

        # boxes = [ann['bbox'] for ann in anns]
        # labels = [ann['category_id'] for ann in anns]

        test_dataset = ODORTestDataset(root=root, ann_file=annotation_json, patch_size=(args.patchsize, args.patchsize))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batchsize, shuffle=False, num_workers=0, collate_fn=collate_fn)


        # predictions['boxes'] = torch of bboxes (N, 4)
        # predictions['scores'] = torch of corresponding scores (confidence) (N,)
        # predictions['labels'] = torch of corresponding labels (N,)
        predictions = trainer.predict(model=my_detection_model, dataloaders=test_loader)
        image_pred = torch.empty((0, 4)).cuda()
        image_scores_raw = torch.empty((0)).cuda()
        image_labels = torch.empty((0), dtype=torch.long).cuda() #

        for batch_id, pred_batch in enumerate(predictions): # iterating through the batches, pred[0] might contain metadata or identifiers for the batch of images. For example, it could include image IDs or indices to map predictions back to the original dataset. In some implementations, this part may be empty or not used.
            for image_id, pred in enumerate(pred_batch[1]): # contains the actual predictions, bboxes, scores, labels
                # transform prediction to global coordinates
                cur_global_pred = test_dataset.local_to_global(batch_id * test_batchsize + image_id, pred['boxes'])
                image_pred = torch.cat([image_pred, cur_global_pred])
                image_scores_raw = torch.cat([image_scores_raw, pred["scores"]])
                image_labels = torch.cat([image_labels, pred["labels"]]) #

        image_pred_th = image_pred[image_scores_raw > 0.5]  # taking the predictions whose scores are more than 0.5
        image_pred_cthw = tlbr2cthw(image_pred_th)[:, :2]
        image_pred_cthw = nms(image_pred_cthw, 0.4)
        image_gt_cthw = tlbr2cthw(test_dataset.get_whole_img_labels_as_dict()['boxes'])[:, :2]

        

        # image_pred_labels = image_labels[image_scores_raw > 0.5]
        # gt_labels = test_dataset.get_whole_img_labels_as_dict()['labels']
        # gt_boxes = tlbr2cthw(test_dataset.get_whole_img_labels_as_dict()['boxes'])[:, :2]
        # all_predictions.append({'boxes': image_pred_cthw, 'labels': image_pred_labels})
        # all_gt.append({'boxes': gt_boxes, 'labels': gt_labels})
        # for img_id in test_dataset.ids:
        #     gt_data = test_dataset.get_whole_img_labels_as_dict(img_id)
        #     gt_labels = gt_data['labels']
        #     gt_boxes = tlbr2cthw(gt_data['boxes'])[:, :2]

        #     all_predictions.append({'boxes': image_pred_cthw, 'labels': image_pred_labels})
        #     all_gt.append({'boxes': gt_boxes, 'labels': gt_labels})

        all_predictions.append(image_pred_cthw)
        all_gt.append(image_gt_cthw)

    tp, fp, fn = get_confusion_matrix(all_gt, all_predictions)
    aggregates = get_metrics(tp, fp, fn)
    print("The performance on the test set for the current setting was \n" +
          "F1-score:  {:.3f}\n".format(aggregates["f1_score"]) +
          "Precision: {:.3f}\n".format(aggregates["precision"]) +
          "Recall:    {:.3f}\n".format(aggregates["recall"]))

if __name__ == "__main__":
    get_cmdline_args_and_run()