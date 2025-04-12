from myutils import get_confusion_matrix, get_metrics, tlbr2cthw

from torchmetrics.detection import MAP
MeanAveragePrecision = MAP
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.models.detection.anchor_utils import AnchorGenerator
# from torchvision.models.detection.retinanet import retinanet_resnet50_fpn_v2, RetinaNetHead
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn, RetinaNetHead
# from torchvision.models import resnet50, ResNet50_Weights
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch.optim
from typing import Any, Callable, Dict, List, Optional, Tuple, OrderedDict


class MyRetinaModel(LightningModule):

    def __init__(self, num_classes=139, iterations_epoch=100, lr=1e-4, epochs=200, detectthresh_val=0.5) -> None:
        super().__init__()
        
        self.num_classes = num_classes
        self.lr = lr
        self.epochs = epochs
        self.iterations_epoch = iterations_epoch
        self.detectthresh_val = detectthresh_val

        self.sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [4, 8, 16, 32, 64])
        self.ratios = ((1.0,),) * len(self.sizes)

        # load a model pre-trained on COCO
        # self.model = retinanet_resnet50_fpn(weights='DEFAULT', trainable_backbone_layers = 5) # unfroze 5 out of 5 backbone layers
        self.model = retinanet_resnet50_fpn(pretrained=False, progress=True, num_classes=139, pretrained_backbone=True, trainable_backbone_layers=5)
        # me: an anchor generator is a component that generates a set of bounding box proposals.
        # me: Anchors are typically pre-defined bounding boxes that are evenly distributed across the
        # me: spatial locations of the feature maps extracted from the input image by the convolutional
        # me: layers of the object detection model. The anchor generator produces anchors with different
        # me: scales and aspect ratios to handle objects of different sizes and shapes. For example, smaller
        # me: anchors may be used for smaller objects, while larger anchors may be used for larger objects

        # replace the pre-trained head with a new one and set a new anchor generator
        self.model.anchor_generator = AnchorGenerator(sizes=self.sizes, aspect_ratios=self.ratios)
        self.model.head = RetinaNetHead(self.model.backbone.out_channels, self.model.anchor_generator.num_anchors_per_location()[0], self.num_classes)
        self.val_step_outputs = []

        # Storage for mAP calculation
        self.detections = []
        self.targets = []

        # Initialize mAP metric
        self.map_metric = MeanAveragePrecision()

    def get_RetinaNet_validation_loss(self, images, targets):
        
        # transform the input
        images, targets = self.model.transform(images, targets)

        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                torch._assert(
                    False,
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}.",
                )

        # get the features from the backbone
        features = self.model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        features = list(features.values())

        # compute the retinanet heads outputs using the features
        head_outputs = self.model.head(features)

        # create the set of anchors
        anchors = self.model.anchor_generator(images, features)

        losses = {}
        detections: List[Dict[str, torch.Tensor]] = []
        if targets is None:
            torch._assert(False, "targets should not be none when in training mode")
        else:
            # compute the losses
            losses = self.model.compute_loss(targets, head_outputs, anchors)
        return losses

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch

        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # using PyTorch Lighting logging
        self.log("train_loss", losses)
        return losses

    def on_validation_epoch_end(self):
        conf_mat = torch.sum(torch.stack([v[2] for v in self.val_step_outputs]), dim=0)
        binary_metrics = get_metrics(*conf_mat)
        self.log("val_f1", binary_metrics["f1_score"])
        self.log("val_precision", binary_metrics["precision"])
        self.log("val_recall", binary_metrics["recall"])
        self.val_step_outputs.clear()
        if self.detections and self.targets:
            # Calculate mAP
            detections = [{'boxes': det['boxes'].cpu(), 'scores': det['scores'].cpu(), 'labels': det['labels'].cpu()} for det in self.detections]
            targets = [{'boxes': tgt['boxes'].cpu(), 'labels': tgt['labels'].cpu()} for tgt in self.targets]
            self.map_metric.update(detections, targets)
            mAP_score = self.map_metric.compute()
            if mAP_score:
                print(f"mAP_score = {mAP_score}")
                self.log("val_mAP", mAP_score['map'], on_epoch=True, prog_bar=True, logger=True)
            else:
                print("No detections or targets to compute mAP.")

            
        self.detections.clear()
        self.targets.clear()

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        predictions = self.model(images, targets)
        loss_dict = self.get_RetinaNet_validation_loss(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        bboxes_cat = torch.cat([t["boxes"] for t in targets])
        predictions_cat = torch.cat([p["boxes"][predictions[i]["scores"] > self.detectthresh_val] for i, p in enumerate(predictions)])
        boxes_cthw = tlbr2cthw(bboxes_cat)
        predictions_cthw = tlbr2cthw(predictions_cat)

        tp, fp, fn = get_confusion_matrix(boxes_cthw[:, :2].cpu(), predictions_cthw[:, :2].cpu())
        # Using PyTorch Lighting logging
        self.log("val_loss", losses)
        self.val_step_outputs.append([predictions, losses, torch.Tensor([tp, fp, fn])])

        for pred in predictions:
            high_conf_idx = pred['scores'] > self.detectthresh_val
            self.detections.append({
                'boxes': pred['boxes'][high_conf_idx],
                'scores': pred['scores'][high_conf_idx],
                'labels': pred['labels'][high_conf_idx]
            })
        self.targets.extend(targets)
        return losses

    def test_step(self, batch, batch_idx):
        # this is the test loop
        images, targets = batch
        # prediction = self.model.forward(images)
        prediction = self.model(images)
        loss_dict = self.get_RetinaNet_validation_loss(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        self.log("test_loss", losses)
        return {'test_loss': losses, 'preds': prediction, 'target': targets}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if len(batch) == 2:
            x, y = batch
        else:
            x = batch
        return x, self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # warmup_factor = 1.0 / min(1000.0, self.iterations_epoch)
        # warmup_iters = min(1000, self.iterations_epoch - 1)
        # warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
        #                                                         start_factor=warmup_factor,
        #                                                         total_iters=warmup_iters, verbose=True)

        cyclic_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, epochs=self.epochs,
                                                                  steps_per_epoch=self.iterations_epoch)

        # lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_lr_scheduler, cyclic_lr_scheduler],
        #                                                      milestones=[1])

        return [optimizer], [cyclic_lr_scheduler]
