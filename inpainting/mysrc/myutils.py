from evalutils.scorers import score_detection, DetectionScore
from collections import namedtuple
import torch
import numpy as np
from sklearn.neighbors import KDTree


def get_confusion_matrix(ground_truth, prediction):
    """ Provides F1 score, recall and precision for binary detection problems."""
    """ We take 30 pixels, which corresponds to approx. 7.5 microns """

    if hasattr(ground_truth, "dim") and ground_truth.dim() == 2:
        sc = score_detection(ground_truth=ground_truth, predictions=prediction, radius=30)
        tp = sc.true_positives
        fp = sc.false_positives
        fn = sc.false_negatives
    else:
        tp, fp, fn = 0, 0, 0
        for gt, pred in zip(ground_truth, prediction):
            sc = score_detection(ground_truth=gt, predictions=pred, radius=30)
            tp += sc.true_positives
            fp += sc.false_positives
            fn += sc.false_negatives
    return tp, fp, fn


def get_metrics(tp, fp, fn):
          
    aggregate_results = dict()

    aggregate_results["precision"] = tp / (tp + fp + 1e-7)
    aggregate_results["recall"] = tp / (tp + fn + 1e-7)
    aggregate_results["f1_score"] = 2 * tp / ((2 * tp) + fp + fn + 1e-7)

    return aggregate_results

def tlbr2cthw(boxes):
    """Convert top/left bottom/right format `boxes` to center/size corners.
        output: [center_x, center_y, width, height].
    """
    center = (boxes[:, :2] + boxes[:, 2:])/2  # Computes the center coordinates by averaging the top-left and bottom-right coordinates.
    sizes = boxes[:, 2:] - boxes[:, :2]  # Computes the width and height by subtracting the top-left coordinates from the bottom-right coordinates.
    return torch.cat([center, sizes], 1)  # Concatenates the center coordinates and sizes along the second dimension.


def cthw2tlbr(boxes):
    """Convert center/size format `boxes` to top/left bottom/right corners."""
    top_left = boxes[:, :2] - boxes[:, 2:]/2
    bot_right = boxes[:, :2] + boxes[:, 2:]/2
    return torch.cat([top_left, bot_right], 1)


def collate_fn(batch):
    """collate_fn receives a list of tuples if __getitem__ function from a 
        Dataset subclass returns a tuple, or just a normal list if your Dataset 
        subclass returns only one element. Its main objective is to create your 
        batch without spending much time implementing it manually"""

    images, targets = zip(*batch)
    images = list(image.cuda() for image in images)
    targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
    return images, targets


def convert_bbox_format(boxes):
    boxes = np.array(boxes)
    
    if boxes.shape[1] != 4:
        raise ValueError("Invalid bounding box format. Each box should have 4 elements.")

    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]

    return boxes



def filter_bboxes(bboxes, labels, x0, y0, w, h):
    # generate copy of boxes
    if len(bboxes) != len(labels):
         raise ValueError("lists bboxes and classes should have the same length but have length {} and {}".format(len(bboxes), len(labels)))

    if len(labels) > 0:
        bboxes = np.copy(bboxes)

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - x0
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - y0


        bb_half_widths = (bboxes[:, 2] - bboxes[:, 0]) / 2  # if half of the box is still contained in the image -- keep
        bb_half_heights = (bboxes[:, 3] - bboxes[:, 1]) / 2
        ids = ((bboxes[:, 0] + bb_half_widths) > 0) \
                & ((bboxes[:, 1] + bb_half_heights) > 0) \
                & ((bboxes[:, 2] - bb_half_widths) < w) \
                & ((bboxes[:, 3] - bb_half_heights) < h)

        bboxes = bboxes[ids]
        labels = labels[ids]

        bboxes = np.clip(bboxes, 0, max(h, w))  # not an issue for quadratic regions, but wouldn't this be problematic otherwise?
        
        invalid_indices = np.where((bboxes[:, 2] <= bboxes[:, 0]) | (bboxes[:, 3] <= bboxes[:, 1]))[0]
        if len(invalid_indices) > 0:
            print("Invalid bboxes found:")
            for idx in invalid_indices:
                print(f"Invalid bbox: {bboxes[idx]}")
                print(f"Invalid bbox: {bboxes[idx]}, Annotation ID: {labels[idx]}")
    return bboxes, labels


def filter_degenerate_bboxes(bboxes, labels):

    if len(bboxes) != len(labels):
        raise ValueError("Lists 'bboxes' and 'labels' should have the same length but have length {} and {}".format(len(bboxes), len(labels)))

    # Identify degenerate bounding boxes
    invalid_indices = np.where((bboxes[:, 2] <= bboxes[:, 0]) | (bboxes[:, 3] <= bboxes[:, 1]))[0]
    
    if len(invalid_indices) > 0:
        print("Pre-Check: Invalid bboxes found and removed:")
        for idx in invalid_indices:
            x_min, y_min, x_max, y_max = bboxes[idx]
            width = x_max - x_min
            height = y_max - y_min
            print(f"Invalid bbox: (x: {x_min}, y: {y_min}, w: {width}, h: {height}), Annotation ID: {labels[idx]}")
        
        
        # Remove degenerate bounding boxes
        bboxes = np.delete(bboxes, invalid_indices, axis=0)
        labels = np.delete(labels, invalid_indices, axis=0)
    
    return bboxes, labels


def non_max_suppression_by_distance(boxes, scores, radius: float = 25, det_thres=None):
    if det_thres is not None:  # perform thresholding
        to_keep = scores > det_thres
        boxes = boxes[to_keep]
        scores = scores[to_keep]

    #TODO: This doesn't look right... 4 instead of 6?
    if boxes.shape[-1] == 4:  # BBOXES
        center_x = boxes[:, 0] + (boxes[:, 2] - boxes[:, 0]) / 2
        center_y = boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) / 2
    else:  # if [x_min, y_min, x_max, y_max] format, then they are in [center_x, center_y] format.
        center_x = boxes[:, 0]
        center_y = boxes[:, 1]

    # stacking center vectors together, after stacking they are 3D, so we take [0] effectively to get back the 2D
    X = np.dstack((center_x, center_y))[0]
    tree = KDTree(X)

    sorted_ids = np.argsort(scores)[::-1]

    ids_to_keep = []
    ind = tree.query_radius(X, r=radius)

    while len(sorted_ids) > 0:
        # picking the id with the highest confidence score
        ids = sorted_ids[0]
        ids_to_keep.append(ids)
        # Deletes all indices from sorted_ids that correspond to the neighboring points within the specified radius of the current highest scoring box. This ensures that overlapping boxes are suppressed.
        sorted_ids = np.delete(sorted_ids, np.in1d(sorted_ids, ind[ids]).nonzero()[0])

    return boxes[ids_to_keep]


def nms(result_boxes, det_thres=None):
    arr = np.array(result_boxes)
    if arr is not None and isinstance(arr, np.ndarray) and (arr.shape[0] == 0):
        return result_boxes # if empty, return original list
    if det_thres is not None:
        before = np.sum(arr[:, -1] > det_thres)
    if arr.shape[0] > 0:
        try:
            arr = non_max_suppression_by_distance(arr, arr[:, -1], 25, det_thres)
        except:
            pass

    result_boxes = arr

    return result_boxes

